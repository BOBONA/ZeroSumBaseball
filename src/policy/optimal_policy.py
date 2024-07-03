import random
import warnings
from collections import defaultdict

import cvxpy as cp
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

try:
    # noinspection PyUnresolvedReferences
    ipython_name = get_ipython().__class__.__name__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from src.data.data_loading import BaseballData
from src.data.datasets import SwingResult, PitchDataset, PitchControlDataset
from src.distributions.batter_patience import BatterSwings, batter_patience_map
from src.distributions.pitcher_control import PitcherControl
from src.distributions.swing_outcome import SwingOutcome, map_swing_outcome
from src.model.at_bat import AtBatState, PitchResult, AtBat
from src.model.pitch import Pitch
from src.model.pitch_type import PitchType
from src.model.players import Pitcher, Batter
from src.model.zones import Zone, default


# To simplify typing, we define some types in accordance with the reference paper's notation
type A = tuple[PitchType, Zone]
type O = bool
type S = AtBatState

# These distributions can be indexed into according to the lists defined above
type SwingOutcomeDistribution = list[list[list[float]]]  # [S_i][A_i][SwingResult] -> swing result probability
type PitcherControlDistribution = list[list[float]]  # [A_i][ZONE_i] -> pitch outcome zone probability
type BatterPatienceDistribution = list[
    list[list[float]]]  # [S_i][PitchType][(BORDERLINE_)ZONE_i] -> batter swing probability


class PolicySolver:
    """
    Given a pitcher and batter, this class aims to calculate the optimal policy for the pitcher.
    The solution to the game is dependent on the setup of AtBatState and its transition rules.
    """

    # We define some type aliases to improve readability, note how they are indexed into
    type Policy = list[list[float]]  # [S_i][A_i] -> float
    type TransitionDistribution = list[list[
        list[dict[int, tuple[float, float]]]]]  # [S_i][A_i][O_i] -> {total_states_i: (next state probability, reward)}

    default_batch: int = 512

    def __init__(self, pitcher: Pitcher, batter: Batter):
        """
        Initializes the policy solver with the given pitcher and batter, and some optional parameters.
        This does not do any actual calculations, but sets up the necessary data structures and config.

        :param pitcher:
        :param batter:
        """

        # BatterPatienceDistribution indexing relies on COMBINED_ZONES = ZONES + BORDERLINE_ZONES in that order
        assert default.COMBINED_ZONES[len(default.ZONES)] == default.BORDERLINE_ZONES[0]

        self.pitcher = pitcher
        self.batter = batter

        self.pitcher_actions: list[A] = [(pitch_type, zone) for zone in default.ZONES for pitch_type in PitchType]
        self.batter_actions: list[O] = [False, True]  # Order is important!

        self.game_states: list[S] = [
            AtBatState(balls=balls, strikes=strikes, outs=outs, first=first, second=second, third=third)
            for balls in range(4) for strikes in range(3) for outs in range(3)
            for first in [False, True] for second in [False, True] for third in [False, True]
        ]

        # Sort by "lateness" of the state, to optimize the value iteration algorithm
        self.game_states.sort(key=lambda st: (st.num_outs + st.num_runs) * 100 + (st.balls + st.strikes) * 10 +
                                             int(st.third) * 4 + int(st.second) * 2 + int(st.first), reverse=True)

        # Terminal states are stored separately for easier indexing
        self.final_states: list[S] = [
            AtBatState(balls=balls, strikes=strikes, outs=3, first=first, second=second, third=third)
            for balls in range(4) for strikes in range(3)
            for first in [False, True] for second in [False, True] for third in [False, True]
        ]

        self.total_states = self.game_states + self.final_states
        self.total_states_dict = {state: i for i, state in enumerate(self.total_states)}
        
        self.transition_distribution = None
        self.raw_values: list[float] | None = None
        self.raw_policy: PolicySolver.Policy | None = None

    def initialize_distributions(self, batch_size: int = default_batch):
        """
        Initializes the transition distributions for the given pitcher and batter pair. Note that
        the individual calculate distribution methods support batching for multiple pairs, but this 
        is not currently exposed.
        
        :param batch_size: Batch size for model inference
        """
        
        self.transition_distribution = self.precalculate_transition_distribution(self.pitcher, self.batter, batch_size=batch_size)

    def precalculate_transition_distribution(self, pitcher: Pitcher, batter: Batter, batch_size: int = default_batch,
                                             batter_patience: BatterPatienceDistribution | None = None,
                                             swing_outcome: SwingOutcomeDistribution | None = None,
                                             pitcher_control: PitcherControlDistribution | None = None) -> TransitionDistribution:
        """
        Precalculates the transition probabilities for a given pitcher and batter pair.
        This method is complicated by the fact that both the pitch outcome and the swing outcome
        are stochastic.
        """

        swing_outcome = self.calculate_swing_outcome_distribution([(pitcher, batter)], batch_size=batch_size)[pitcher, batter] \
            if swing_outcome is None else swing_outcome
        pitcher_control = self.calculate_pitcher_control_distribution([pitcher], batch_size=batch_size)[pitcher] \
            if pitcher_control is None else pitcher_control
        batter_patience = self.calculate_batter_patience_distribution([batter], batch_size=batch_size)[batter] \
            if batter_patience is None else batter_patience

        transition_distribution: PolicySolver.TransitionDistribution = [[[defaultdict(PolicySolver.tuple_help) for _ in range(len(self.batter_actions))]
                                                                         for _ in range(len(self.pitcher_actions))] for _ in range(len(self.game_states))]

        for state_i, state in tqdm(enumerate(self.game_states), desc='Calculating transition distribution', total=len(self.game_states)):
            for action_i, action in enumerate(self.pitcher_actions):
                pitch_type, intended_zone = action
                for batter_swung in self.batter_actions:

                    # Given a pitcher and batter action, calculate the actual outcome distribution
                    # The pitch outcome is a distribution over zones given the intended pitch
                    for outcome_zone_i, prob in enumerate(pitcher_control[action_i]):

                        # To account for batter patience, we override the outcomes for borderline zones,
                        # to be a measure of the batter's patience. This requires another nested loop
                        swing_probs = {o: float(batter_swung == o) for o in self.batter_actions}

                        outcome_zone = default.COMBINED_ZONES[outcome_zone_i]
                        if outcome_zone.is_borderline:
                            patience = batter_patience[state_i][pitch_type][outcome_zone_i - len(default.ZONES)]
                            swing_probs[True] = patience
                            swing_probs[False] = 1 - patience

                        # Loop over the swing "outcomes". When the zone is not borderline, this part isn't
                        # stochastic and the loop is redundant (swing_prob = {True: 1/0, False: 0/1})
                        for outcome_swing, swing_prob in swing_probs.items():
                            if outcome_swing:
                                swing_results = swing_outcome[state_i][action_i]
                                for swing_result_i, result_prob in enumerate(swing_results):
                                    swing_result = SwingResult(swing_result_i)
                                    next_state = state.transition_from_pitch_result(swing_result.to_pitch_result())

                                    reward = next_state.value() - state.value()

                                    next_state_i = self.total_states_dict[next_state]
                                    current_prob = transition_distribution[state_i][action_i][batter_swung][next_state_i][0]
                                    transition_distribution[state_i][action_i][batter_swung][next_state_i] = (current_prob + prob * result_prob * swing_prob, reward)
                            else:
                                next_state = state.transition_from_pitch_result(PitchResult.CALLED_STRIKE if outcome_zone.is_strike else PitchResult.CALLED_BALL)
                                reward = next_state.value() - state.value()

                                next_state_i = self.total_states_dict[next_state]
                                current_prob = transition_distribution[state_i][action_i][batter_swung][next_state_i][0]
                                transition_distribution[state_i][action_i][batter_swung][next_state_i] = (current_prob + prob * swing_prob, reward)

        return transition_distribution

    def calculate_swing_outcome_distribution(self, matchups: list[tuple[Pitcher, Batter]], batch_size=default_batch) -> dict[tuple[Pitcher, Batter], SwingOutcomeDistribution]:
        """
        Calculates the distribution of swing outcomes for a list of pitcher and batter pairs, given the current game state.
        This method takes in a list to allow for batch processing.

        :return: A dictionary mapping a state, pitcher action and batter action to a distribution of swing outcomes
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        swing_outcome_model = SwingOutcome().to(device)
        swing_outcome_model.load_state_dict(torch.load('../../model_weights/swing_outcome.pth'))
        swing_outcome_model.eval()

        swing_outcome = {}

        for pitcher, batter in tqdm(matchups, desc='Calculating swing outcomes'):
            swing_outcome[(pitcher, batter)] = [[[] for _ in range(len(self.pitcher_actions))] for _ in
                                                range(len(self.game_states))]

            pitch_states = [(state_i, pitch_i)
                            for state_i in range(len(self.game_states)) for pitch_i in range(len(self.pitcher_actions))]
            pitch_data = [Pitch(self.game_states[state_i], AtBat(None, pitcher, batter, self.game_states[state_i]),
                                0, self.pitcher_actions[pitch_i][1], self.pitcher_actions[pitch_i][0], 0, PitchResult.HIT_SINGLE)
                          for state_i, pitch_i in pitch_states]
            pitch_dataset = PitchDataset(data_source=None, pitches=pitch_data, map_to=map_swing_outcome)
            pitch_dataloader = DataLoader(pitch_dataset, batch_size=batch_size, shuffle=False)

            for batch, (pitch_idx, data, target) in enumerate(pitch_dataloader):
                data = [d.to(device) for d in data]
                outcome_tensor = swing_outcome_model(*data, softmax=True)

                result_distributions = outcome_tensor.squeeze().tolist()
                if not isinstance(result_distributions[0], list):  # In case the batch is a single element
                    result_distributions = [result_distributions]

                for i, result_distribution in enumerate(result_distributions):
                    state_i, pitch_i = pitch_states[batch * batch_size + i]
                    swing_outcome[(pitcher, batter)][state_i][pitch_i] = result_distribution

        return swing_outcome

    def calculate_pitcher_control_distribution(self, pitchers: list[Pitcher], batch_size=default_batch) -> dict[Pitcher, PitcherControlDistribution]:
        """
        Calculates the distribution of actual pitch outcomes for a given pitcher, given the intended pitch type and zone

        :return: A dictionary mapping a pitcher action to a distribution of actual pitch outcomes over zones
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        pitcher_control_model = PitcherControl().to(device)
        pitcher_control_model.load_state_dict(torch.load('../../model_weights/pitcher_control.pth'))
        pitcher_control_model.eval()

        pitcher_type_control = defaultdict(defaultdict)

        # This dataset class automatically iterates over pitch types, empty_data is a custom flag for this purpose
        pitch_control_dataset = PitchControlDataset(data_source=None, pitchers=pitchers, empty_data=True)
        dataloader = DataLoader(pitch_control_dataset, batch_size=batch_size, shuffle=False)

        num_batches = len(dataloader)
        for i, (p_i, p_type, p_data, distribution) in tqdm(enumerate(dataloader), desc='Calculating pitcher control',
                                                           leave=False, total=num_batches):
            p_data = [d.to(device) for d in p_data]
            control_tensor = pitcher_control_model(*p_data)

            control_list = control_tensor.squeeze().tolist()
            if not isinstance(control_list[0], list):
                control_list = [control_list]

            for k, control in enumerate(control_list):
                pitcher = pitchers[p_i[k]]
                pitch_type = PitchType(p_type[k].item())
                gaussian = MultivariateNormal(torch.tensor(control[:2]), torch.tensor([[control[2], control[4]],
                                                                                       [control[4], control[3]]]))
                pitcher_type_control[pitcher][pitch_type] = gaussian

        # To make things simple, we use random sampling to find a distribution of pitch outcomes
        pitcher_control = {}
        for pitcher in tqdm(pitchers, desc='Sampling pitcher control'):
            pitcher_control[pitcher] = [[0 for _ in range(len(default.COMBINED_ZONES))] for _ in
                                        range(len(self.pitcher_actions))]

            for pitch_i, pitch in enumerate(self.pitcher_actions):
                pitch_type, intended_zone = pitch
                zone_center = intended_zone.center()
                gaussian = pitcher_type_control[pitcher][pitch_type]
                gaussian = MultivariateNormal(torch.tensor([zone_center[0], zone_center[1]]),
                                              gaussian.covariance_matrix)

                num_samples = 1000
                sample_pitches = gaussian.sample(torch.Size((num_samples,)))
                zones = default.get_zones_batched(sample_pitches[:, 0], sample_pitches[:, 1])
                for zone_i in zones:
                    pitcher_control[pitcher][pitch_i][zone_i] += 1 / num_samples

        return pitcher_control

    def calculate_batter_patience_distribution(self, batters: list[Batter], batch_size=default_batch) -> dict[Batter, BatterPatienceDistribution]:
        """
        Calculates the distribution of batter patience for a given batter, given the current game state and pitcher action

        :return: A dictionary mapping a state and pitcher action (on a borderline zone) to the probability that the batter will swing
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batter_patience_model = BatterSwings().to(device)
        batter_patience_model.load_state_dict(torch.load('../../model_weights/batter_patience.pth'))
        batter_patience_model.eval()

        batter_patience = {}
        for batter in tqdm(batters, desc='Calculating batter patience'):
            batter_patience[batter] = [
                [[0 for _ in range(len(default.BORDERLINE_ZONES))] for _ in range(len(PitchType))]
                for _ in range(len(self.game_states))]

            pitch_states = [(state_i, type_i, borderline_zone_i) for state_i in range(len(self.game_states))
                            for type_i in range(len(PitchType)) for borderline_zone_i in range(len(default.BORDERLINE_ZONES))]
            pitch_data = [Pitch(self.game_states[state_i], AtBat(None, None, batter, self.game_states[state_i]),
                                0, default.BORDERLINE_ZONES[zone_i], PitchType(type_i), 0, PitchResult.HIT_SINGLE)
                          for state_i, type_i, zone_i in pitch_states]

            patience_dataset = PitchDataset(data_source=None, pitches=pitch_data, map_to=batter_patience_map)
            dataloader = DataLoader(patience_dataset, batch_size=batch_size, shuffle=False)

            for batch, (pitch_idx, data, swing) in enumerate(dataloader):
                data = [d.to(device) for d in data]

                swing_percent = batter_patience_model(*data).squeeze().tolist()
                if not isinstance(swing_percent, list):
                    swing_percent = [swing_percent]

                for i, swing_percent in enumerate(swing_percent):
                    state_i, pitch_type_i, zone_i = pitch_states[batch * batch_size + i]
                    batter_patience[batter][state_i][pitch_type_i][zone_i] = swing_percent

        return batter_patience

    @classmethod
    def tuple_help(cls):
        """Utility for default dict"""
    
        return 0, 0

    def calculate_optimal_policy(self, print_difference: bool = False, use_ordered_iteration: bool = True,
                                 beta: float = 1e-3) -> tuple[Policy, list[float]]:
        """
        Uses value iteration algorithm to calculate the optimal policy for our model, given
        the pitcher and batter. https://doi.org/10.1016/B978-1-55860-335-6.50027-1

        A policy (or mixed strategy) defines a probability distribution over actions for each state for
        the pitcher.

        :param print_difference: This flag prints the difference between iterations, useful for debugging
        :param use_ordered_iteration: This is a flag that improves the convergence by calculating states with the values of the current iteration
        :param beta: The minimum change in value to continue iterating
        :return: The optimal pitcher policy, assigning a probability to each action in each state and the value of each state.
            These structures are indexed according to the lists defined in the class, but methods are provided to view the data more easily.
            You can ignore the return value and use get_value or get_policy instead.
        """

        # Stores the "value" of each state, indexed according to total_states
        # Last time I measured, random initialization in this manner converged 1/3 faster
        value = [random.random() for _ in self.game_states] + [0 for _ in self.final_states]

        # Stores the policy for each state, indexed according to game_states
        policy: PolicySolver.Policy = [[1 / len(self.pitcher_actions) for _ in self.pitcher_actions] for _ in self.game_states]

        transition_distribution = self.precalculate_transition_distribution(self.pitcher, self.batter) if self.transition_distribution is None else self.transition_distribution

        difference = float('inf')
        iter_num = 0
        while difference > beta:
            iter_num += 1

            # Independently optimize the policy for each state
            new_policy = []
            new_value = value.copy()

            value_src = new_value if use_ordered_iteration else value  # Faster convergence with "ordered iteration"

            # Note, this can be parallelized
            for state_i, state in tqdm(enumerate(self.game_states), f'Iterating over values, iter={iter_num}', total=len(self.game_states)):
                action_quality = [[sum([prob * (reward + value_src[next_state_i])
                                        for next_state_i, (prob, reward) in transition_distribution[state_i][a_i][o].items()])
                                   for o in self.batter_actions] for a_i in range(len(self.pitcher_actions))]

                new_state_policy, new_value[state_i] = self.update_policy(action_quality)
                new_policy.append(new_state_policy)

            # Update values
            difference = max([abs(new_value[state_i] - value[state_i]) for state_i in range(len(self.game_states))])
            policy = new_policy
            value = new_value

            if print_difference:
                print(difference)

        self.raw_values = value
        self.raw_policy = policy

        return policy, value
    
    def update_policy(self, action_quality: list[list[float]], max_pitch_percentage: float = 0.7,
                      print_warnings: bool = False) -> tuple[list[float], float]:
        """Optimizes a new policy using dynamic programming"""

        if len(self.pitcher_actions) == 1:
            max_pitch_percentage = 1.0

        policy = cp.Variable(len(self.pitcher_actions))
        policy_constraints = [policy >= 0, cp.sum(policy) == 1, policy <= max_pitch_percentage]  # Limit the maximum probability of any action
    
        # We want to minimize the maximum expected value (for the batter) of the next state
        objective = cp.Minimize(cp.maximum(*[sum([policy[a_i] * action_quality[a_i][o]
                                           for a_i in range(len(self.pitcher_actions))]) for o in self.batter_actions]))
    
        problem = cp.Problem(objective, policy_constraints)
        problem.solve()
    
        if problem.status != cp.OPTIMAL and problem.status != cp.OPTIMAL_INACCURATE:
            raise ValueError(f'Policy optimization failed, status {problem.status}')
        else:
            if problem.status == cp.OPTIMAL_INACCURATE and print_warnings:
                warnings.warn('Inaccurate optimization detected')
            new_policy = [policy[a_i].value for a_i in range(len(self.pitcher_actions))]
            return new_policy, problem.value

    def get_value(self, state: AtBatState = AtBatState()) -> float:
        """After policy has been calculated, returns the value of a state"""

        return self.raw_values[self.total_states_dict[state]]

    def get_policy(self) -> dict[AtBatState, list[tuple[A, float]]]:
        """After policy has been calculated, returns the optimal policy as a dictionary, for easier access"""

        optimal_policy = {}
        for i, state in enumerate(self.game_states):
            optimal_policy[state] = []
            for j, prob in enumerate(self.raw_policy[i]):
                if prob > 0.01:
                    optimal_policy[state].append((self.pitcher_actions[j], prob))
        return optimal_policy


def main(debug: bool = False):
    if not debug:
        bd = BaseballData.load_with_cache()

        # Note how the resulting ERA is highly dependent on the pitcher and batter chosen
        # Try pitchers[2] (obp_percentile = 0.21) and batters[0] (obp_percentile = 0.95) for a higher ERA
        # Also note that lower percentile batters often have little data and create unstable results from the models

        pitcher = list(bd.pitchers.values())[90]  # obp_percentile = 0.94
        batter = list(bd.batters.values())[513]  # obp_percentile = 0.54

        print(f'Pitcher OBP: {pitcher.obp_percentile}, Batter OBP: {batter.obp_percentile}')

        solver = PolicySolver(pitcher, batter)
        solver.initialize_distributions()
        solver.calculate_optimal_policy(print_difference=True)

        print(f'ERA {solver.get_value()}')

        torch.save(solver, 'solved_policy.pth')
    else:
        solver = torch.load('solved_policy.pth')
        raw_values, raw_policy = solver.raw_values, solver.raw_policy
        print(f'ERA {solver.get_value()}')
        pass  # Do something with the data


if __name__ == '__main__':
    main(debug=False)
