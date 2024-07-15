import random
import warnings
from collections import defaultdict
from typing import Self

import cvxpy as cp
import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

try:
    # noinspection PyUnresolvedReferences
    ipython_name = get_ipython().__class__.__name__
    from tqdm.notebook import tqdm
except NameError:
    from tqdm import tqdm

from src.data.data_loading import BaseballData, load_blosc2, save_blosc2
from src.data.datasets import SwingResult, PitchDataset, PitchControlDataset
from src.distributions.batter_patience import BatterSwings, batter_patience_map
from src.distributions.pitcher_control import PitcherControl
from src.distributions.swing_outcome import SwingOutcome, map_swing_outcome
from src.model.state import GameState, PitchResult
from src.model.pitch import Pitch
from src.model.pitch_type import PitchType
from src.model.zones import default


# To simplify typing, we define some types in accordance with the reference paper's notation
type A = tuple[PitchType, int]  # (PitchType, ZONE_i)
type O = bool
type S = GameState

# These distributions can be indexed into according to the lists defined above
type SwingOutcomeDistribution = np.ndarray  # [S_i][A_i][SwingResult] -> swing result probability
type PitcherControlDistribution = np.ndarray  # [A_i][ZONE_i] -> pitch outcome zone probability
type BatterPatienceDistribution = np.ndarray  # [S_i][PitchType][ZONE_i] -> batter swing probability


class PolicySolver:
    """
    Given a pitcher and batter, this class aims to calculate the optimal policy for the pitcher.
    The solution to the game is dependent on the setup of GameState and its transition rules.
    Depending on your use case, you will want to modify the hash function in GameState.

    Note that we're not currently using the num_runs attribute in the GameState, but it can be used to limit the number
    of runs, and the models are also trained with runs as a parameter. However, this increases the number of states.
    """

    # We define some type aliases to improve readability, note how they are indexed into
    type Policy = np.ndarray  # [S_i][A_i] -> float

    # We can define the transition distribution with reasonably sized ndarrays, since the number of states
    # that can be reached from a single state is limited
    type Transitions = np.ndarray  # [S_i][0-7] -> (next state index, reward)
    type TransitionDistribution = np.ndarray  # [S_i][A_i][O_i][0-7] -> probability
    max_transitions = 8

    default_batch: int = 512

    def __init__(self, bd: BaseballData, pitcher_id: int, batter_lineup: list[int]):
        """
        Initializes the policy solver with the given pitcher and batter, and some optional parameters.
        This does not do any actual calculations, but sets up the necessary data structures and config.
        """

        # BatterPatienceDistribution indexing relies on COMBINED_ZONES = ZONES + BORDERLINE_ZONES in that order
        assert default.COMBINED_ZONES[len(default.ZONES)] == default.BORDERLINE_ZONES[0]

        self.bd = bd
        self.pitcher_id = pitcher_id
        self.batter_lineup = batter_lineup

        self.pitcher_actions: list[A] = [(pitch_type, zone_i) for zone_i in range(len(default.ZONES)) for pitch_type in PitchType]
        self.batter_actions: list[O] = [False, True]  # Order is important!

        self.game_states: list[S] = [
            GameState(inning=inning, balls=balls, strikes=strikes, outs=outs, first=first, second=second, third=third, batter=batter)
            for inning in range(9) for balls in range(4) for strikes in range(3) for outs in range(3)
            for first in [False, True] for second in [False, True] for third in [False, True] for batter in range(9)
        ]

        # Sort by "lateness" of the state, to optimize the value iteration algorithm
        self.game_states.sort(key=lambda st: st.inning * 1000 + (st.num_outs + st.num_runs + st.batter) * 100 +
                                             (st.balls + st.strikes) * 10 + int(st.third) * 4 + int(st.second) * 2 +
                                             int(st.first), reverse=True)

        # Terminal states are stored separately for easier indexing
        self.final_states: list[S] = [
            GameState(inning=9, balls=balls, strikes=strikes, outs=outs, first=first, second=second, third=third, batter=batter)
            for balls in range(4) for strikes in range(3) for outs in range(3)
            for first in [False, True] for second in [False, True] for third in [False, True] for batter in range(9)
        ]

        self.total_states = self.game_states + self.final_states
        self.total_states_dict = {state: i for i, state in enumerate(self.total_states)}

        self.transitions = None
        self.transition_distribution = None
        self.policy_problem = None
        self.raw_values: list[float] | None = None
        self.raw_policy: PolicySolver.Policy | None = None

    @classmethod
    def from_saved(cls, path: str, bd: BaseballData | None = None) -> Self:
        """Loads a saved policy solver from a file"""

        solver: Self = load_blosc2(path)
        solver.bd = bd
        return solver

    def save(self, path: str):
        """This method deletes references to self.bd and self.policy_problem before saving, to avoid pickling issues"""

        self.bd = None  # We don't want to save the BaseballData object
        self.policy_problem = None
        save_blosc2(self, path)

    def initialize_distributions(self, batch_size: int = default_batch, save_distributions: bool = False,
                                 load_distributions: bool = False, load_transition: bool = False):
        """
        Initializes the transition distributions for the given pitcher and batter pair. Note that
        the individual calculate distribution methods support batching for multiple pairs, but this 
        is not currently exposed.
        """

        if load_distributions and load_transition:
            distributions = load_blosc2('transition_distribution.blosc2')
            self.transitions, self.transition_distribution = distributions['transitions'], distributions['transition_distribution']
            return

        distributions = defaultdict(lambda: None)
        if load_distributions:
            distributions = load_blosc2('distributions.blosc2')

        self.transitions, self.transition_distribution = self.precalculate_transition_distribution(
            batch_size=batch_size, save_distributions=save_distributions, batter_patiences=distributions['batter_patience'],
            swing_outcomes=distributions['swing_outcomes'], pitcher_control=distributions['pitcher_control']
        )

        if save_distributions:
            save_blosc2({'transition_distribution': self.transition_distribution, 'transitions': self.transitions}, 'transition_distribution.blosc2')

    def precalculate_transition_distribution(self, batch_size: int = default_batch,
                                             batter_patiences: BatterPatienceDistribution | None = None,
                                             swing_outcomes: SwingOutcomeDistribution | None = None,
                                             pitcher_control: PitcherControlDistribution | None = None,
                                             save_distributions: bool = False) -> tuple[Transitions, TransitionDistribution]:
        """
        Precalculates the transition probabilities for a given pitcher and batter pair.
        This method is complicated by the fact that both the pitch outcome and the swing outcome
        are stochastic.
        """

        swing_outcomes = self.calculate_swing_outcome_distribution([(self.pitcher_id, batter_id) for batter_id in self.batter_lineup], batch_size=batch_size) \
            if swing_outcomes is None else swing_outcomes
        pitcher_control = self.calculate_pitcher_control_distribution([self.pitcher_id], batch_size=batch_size)[self.pitcher_id] \
            if pitcher_control is None else pitcher_control
        batter_patiences = self.calculate_batter_patience_distribution(self.batter_lineup, batch_size=batch_size) \
            if batter_patiences is None else batter_patiences

        if save_distributions:
            save_blosc2({'swing_outcomes': swing_outcomes, 'pitcher_control': pitcher_control, 'batter_patience': batter_patiences}, 'distributions.blosc2')

        def map_t(t):
            return self.total_states_dict[t[0]], t[1]

        # Remove duplicates maintaining order, and pad
        def pad(s):
            s = list(dict.fromkeys(s).keys())
            return s + [(-1, 0)] * (self.max_transitions - len(s))

        transitions = np.asarray([pad([map_t(state.transition_from_pitch_result(result)) for result in PitchResult]) for state in self.game_states])
        probabilities = np.zeros((len(self.game_states), len(self.pitcher_actions), len(self.batter_actions), self.max_transitions))

        swing_to_transition_matrix = np.asarray([
            np.asarray([transitions[state_i, :, 0] == self.total_states_dict[
                self.total_states[state_i].transition_from_pitch_result(result.to_pitch_result())[0]] for result in SwingResult]).transpose()
            for state_i in range(len(self.game_states))
        ])

        borderline_mask = np.asarray([zone.is_borderline for zone in default.COMBINED_ZONES])
        strike_mask = np.asarray([zone.is_strike for zone in default.COMBINED_ZONES])

        # It's important for indexing that these are at the start
        called_ball_i = 0
        called_strike_i = 1
        assert PitchResult.CALLED_BALL == called_ball_i
        assert PitchResult.CALLED_STRIKE == called_strike_i

        # Iterate over each state
        # At the cost of readability, we use numpy operations to speed up the calculations
        for state_i, state in tqdm(enumerate(self.game_states), desc='Calculating transition distribution', total=len(self.game_states)):
            for action_i, action in enumerate(self.pitcher_actions):
                pitch_type, intended_zone_i = action
                for batter_swung in range(len(self.batter_actions)):
                    # Given an intended pitch, we get the actual outcome distribution
                    outcome_zone_probs = pitcher_control[action_i]

                    # To account for batter patience, we override the outcomes for borderline zones
                    patience = batter_patiences[self.batter_lineup[state.batter]][state_i, pitch_type]
                    swing_probs = (np.zeros(len(default.COMBINED_ZONES)) + batter_swung) * ~borderline_mask + patience * borderline_mask
                    take_probs = 1 - swing_probs

                    # Handle swing outcomes (stochastic)
                    # TODO this is problematic, we need to choose swing outcome from the outcome zones not the intended zone
                    result_probs = swing_outcomes[(self.pitcher_id, self.batter_lineup[state.batter])][state_i, action_i]
                    transition_probs = np.dot(swing_to_transition_matrix[state_i], result_probs)
                    probabilities[state_i, action_i, batter_swung] += transition_probs * np.dot(swing_probs, outcome_zone_probs)

                    # Handle take outcome (deterministic)
                    probabilities[state_i, action_i, batter_swung, called_strike_i] += np.dot(take_probs, outcome_zone_probs * strike_mask)
                    probabilities[state_i, action_i, batter_swung, called_ball_i] += np.dot(take_probs, outcome_zone_probs * ~strike_mask)

        return transitions, probabilities

    def calculate_swing_outcome_distribution(self, matchups: list[tuple[int, int]], batch_size=default_batch) -> dict[tuple[int, int], SwingOutcomeDistribution]:
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

        # We only care about calculating the results for states that are unique to the model (which does not consider every variable)
        interested_states = [self.total_states_dict[GameState(balls=balls, strikes=strikes, outs=outs, first=first, second=second, third=third)]
                             for balls in range(4) for strikes in range(3) for outs in range(3)
                             for first in [False, True] for second in [False, True] for third in [False, True]]
        interested_pitches = [(state_i, pitch_i) for state_i in interested_states for pitch_i in range(len(self.pitcher_actions))]

        for pitcher_id, batter_id in tqdm(matchups, desc='Calculating swing outcomes'):
            swing_outcome[(pitcher_id, batter_id)] = np.zeros((len(self.game_states), len(self.pitcher_actions), len(SwingResult)))

            pitch_data = [Pitch(self.game_states[state_i], batter_id=batter_id, pitcher_id=pitcher_id,
                                location=self.pitcher_actions[pitch_i][1], pitch_type=self.pitcher_actions[pitch_i][0], pitch_result=PitchResult.HIT_SINGLE)
                          for state_i, pitch_i in interested_pitches]
            pitch_dataset = PitchDataset(pitches=pitch_data, map_to=lambda idx, p: map_swing_outcome(idx, p, self.bd), valid_only=False)
            pitch_dataloader = DataLoader(pitch_dataset, batch_size=batch_size, shuffle=False)

            for batch, (pitch_idx, data, target) in enumerate(pitch_dataloader):
                data = [d.to(device) for d in data]
                outcome_tensor = swing_outcome_model(*data, softmax=True)

                result_distributions = outcome_tensor.squeeze().tolist()
                if not isinstance(result_distributions[0], list):  # In case the batch is a single element
                    result_distributions = [result_distributions]

                for i, result_distribution in enumerate(result_distributions):
                    state_i, pitch_i = interested_pitches[batch * batch_size + i]
                    swing_outcome[(pitcher_id, batter_id)][state_i][pitch_i] = result_distribution

            for state_i in range(len(self.game_states)):
                if state_i not in interested_states:
                    for pitch_i in range(len(self.pitcher_actions)):
                        state = self.game_states[state_i]
                        interested_state = self.total_states_dict[GameState(balls=state.balls, strikes=state.strikes, outs=state.num_outs,
                                                                            first=state.first, second=state.second, third=state.third)]
                        swing_outcome[(pitcher_id, batter_id)][state_i, pitch_i] = swing_outcome[(pitcher_id, batter_id)][interested_state][pitch_i]

        return swing_outcome

    def calculate_pitcher_control_distribution(self, pitchers: list[int], batch_size=default_batch) -> dict[int, PitcherControlDistribution]:
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
        pitch_control_dataset = PitchControlDataset(data_source=None,
                                                    pitchers=[self.bd.pitchers[pitcher_i] for pitcher_i in pitchers],
                                                    empty_data=True)
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
            pitcher_control[pitcher] = np.zeros((len(self.pitcher_actions), len(default.COMBINED_ZONES)))

            for pitch_i, pitch in enumerate(self.pitcher_actions):
                pitch_type, intended_zone_i = pitch
                zone_center = default.ZONES[intended_zone_i].center()
                gaussian = pitcher_type_control[pitcher][pitch_type]
                gaussian = MultivariateNormal(torch.tensor([zone_center[0], zone_center[1]]), gaussian.covariance_matrix)

                num_samples = 10000
                sample_pitches = gaussian.sample(torch.Size((num_samples,)))
                zones = default.get_zones_batched(sample_pitches[:, 0], sample_pitches[:, 1])
                for zone_i in zones:
                    pitcher_control[pitcher][pitch_i, zone_i] += 1 / num_samples

        return pitcher_control

    def calculate_batter_patience_distribution(self, batters: list[int], batch_size=default_batch) -> dict[int, BatterPatienceDistribution]:
        """
        Calculates the distribution of batter patience for a given batter, given the current game state and pitcher action

        :return: A dictionary mapping a state and pitcher action (on a borderline zone) to the probability that the batter will swing
        """

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        batter_patience_model = BatterSwings().to(device)
        batter_patience_model.load_state_dict(torch.load('../../model_weights/batter_patience.pth'))
        batter_patience_model.eval()

        # We only care about calculating the results for states that are unique to the model (which does not consider every variable)
        interested_states = [self.total_states_dict[GameState(balls=balls, strikes=strikes, outs=outs, first=first, second=second, third=third)]
                             for balls in range(4) for strikes in range(3) for outs in range(3)
                             for first in [False, True] for second in [False, True] for third in [False, True]]
        interested_pitches = [(state_i, type_i, borderline_zone_i) for state_i in interested_states
                              for type_i in range(len(PitchType)) for borderline_zone_i in range(len(default.BORDERLINE_ZONES))]

        batter_patience = {}
        for batter_i in tqdm(batters, desc='Calculating batter patience'):
            batter_patience[batter_i] = np.zeros((len(self.game_states), len(PitchType), len(default.COMBINED_ZONES)))

            pitch_data = [Pitch(self.game_states[state_i], pitcher_id=-1, batter_id=batter_i,
                                location=zone_i, pitch_type=PitchType(type_i), pitch_result=PitchResult.HIT_SINGLE)
                          for state_i, type_i, zone_i in interested_pitches]

            patience_dataset = PitchDataset(pitches=pitch_data, map_to=lambda idx, p: batter_patience_map(self.bd, idx, p), valid_only=False)
            dataloader = DataLoader(patience_dataset, batch_size=batch_size, shuffle=False)

            for batch, (pitch_idx, data, swing) in enumerate(dataloader):
                data = [d.to(device) for d in data]

                swing_percent = batter_patience_model(*data).squeeze().tolist()
                if not isinstance(swing_percent, list):
                    swing_percent = [swing_percent]

                for i, swing_percent in enumerate(swing_percent):
                    state_i, pitch_type_i, zone_i = interested_pitches[batch * batch_size + i]
                    batter_patience[batter_i][state_i, pitch_type_i, zone_i + len(default.ZONES)] = swing_percent

            for state_i in range(len(self.game_states)):
                if state_i not in interested_states:
                    for pitch_type_i in range(len(PitchType)):
                        for zone_i in range(len(default.BORDERLINE_ZONES)):
                            interested_state = self.total_states_dict[GameState(balls=self.game_states[state_i].balls, strikes=self.game_states[state_i].strikes,
                                                                                outs=self.game_states[state_i].num_outs, first=self.game_states[state_i].first,
                                                                                second=self.game_states[state_i].second, third=self.game_states[state_i].third)]
                            batter_patience[batter_i][state_i, pitch_type_i, zone_i] = batter_patience[batter_i][interested_state, pitch_type_i, zone_i]

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
        value = np.concatenate((np.random.rand(len(self.game_states)), np.zeros(len(self.final_states))))

        # Stores the policy for each state, indexed according to game_states
        policy: PolicySolver.Policy = np.ones((len(self.game_states), len(self.pitcher_actions))) / len(self.pitcher_actions)

        if self.transition_distribution is None:
            self.initialize_distributions()

        difference = float('inf')
        iter_num = 0
        while difference > beta:
            iter_num += 1

            # Independently optimize the policy for each state
            new_policy = np.zeros((len(self.total_states), len(self.pitcher_actions)))
            new_value = value.copy()

            value_src = new_value if use_ordered_iteration else value  # Faster convergence with "ordered iteration"

            # Note, this can be parallelized
            for state_i, state in tqdm(enumerate(self.game_states), f'Iterating over values, iter={iter_num}', total=len(self.game_states)):
                # The expected value (transition reward + value of next states) for each action pair
                action_quality = np.dot(self.transition_distribution[state_i], self.transitions[state_i, :, 1] + value_src[self.transitions[state_i, :, 0]])
                new_policy[state_i], new_value[state_i] = self.update_policy(action_quality)

            # Update values
            difference = np.abs(new_value - value).max()
            policy = new_policy
            value = new_value

            if print_difference:
                print(difference)

        self.raw_values = value
        self.raw_policy = policy

        return policy, value

    def initialize_policy_problem(self, max_pitch_percentage: float = 0.7):
        """We only need to initialize the policy problem once, as the constraints always the same"""

        if len(self.pitcher_actions) == 1:
            max_pitch_percentage = 1.0

        policy = cp.Variable(len(self.pitcher_actions))
        policy_constraints = [policy >= 0, cp.sum(policy) == 1, policy <= max_pitch_percentage]  # Limit the maximum probability of any action

        action_quality = [cp.Parameter(len(self.pitcher_actions)) for _ in self.batter_actions]

        objective = cp.Minimize(cp.maximum(*[cp.sum([policy[a_i] * action_quality[o][a_i]
                                                     for a_i in range(len(self.pitcher_actions))]) for o in self.batter_actions]))

        problem = cp.Problem(objective, policy_constraints)
        self.policy_problem = policy, action_quality, problem

    def update_policy(self, action_quality: np.ndarray, print_warnings: bool = False) -> tuple[np.asarray, float]:
        """Optimizes a new policy using dynamic programming"""

        if self.policy_problem is None:
            self.initialize_policy_problem()

        policy, action_quality_param, problem = self.policy_problem
        for o in range(len(self.batter_actions)):
            action_quality_param[o].value = action_quality[:, o]
        problem.solve()

        if problem.status != cp.OPTIMAL and problem.status != cp.OPTIMAL_INACCURATE:
            raise ValueError(f'Policy optimization failed, status {problem.status}')
        else:
            if problem.status == cp.OPTIMAL_INACCURATE and print_warnings:
                warnings.warn('Inaccurate optimization detected')
            new_policy = np.asarray([policy[a_i].value for a_i in range(len(self.pitcher_actions))])
            return new_policy, problem.value

    def get_value(self, state: GameState = GameState()) -> float:
        """After policy has been calculated, returns the value of a state"""

        return self.raw_values[self.total_states_dict[state]]

    def get_policy(self) -> dict[GameState, list[tuple[A, float]]]:
        """After policy has been calculated, returns the optimal policy as a dictionary, for easier access"""

        optimal_policy = {}
        for i, state in enumerate(self.game_states):
            optimal_policy[state] = []
            for j, prob in enumerate(self.raw_policy[i]):
                if prob > 0.01:
                    optimal_policy[state].append((self.pitcher_actions[j], prob))
        return optimal_policy


def test_era(bd: BaseballData, pitcher_id: int, batter_lineup: list[int]):
    # print(f'Pitcher OBP: {bd.pitchers[pitcher_id].obp}, Batter (first) OBP: {bd.batters[batter_lineup[0]].obp}')

    solver = PolicySolver(bd, pitcher_id, batter_lineup)
    solver.initialize_distributions(save_distributions=True, load_distributions=True)
    solver.calculate_optimal_policy(print_difference=True)

    print(f'ERA {solver.get_value()}')

    solver.save('solved_policy.blosc2')


def main(debug: bool = False):
    if not debug:
        # bd = BaseballData()
        bd = None

        # The resulting ERA is highly dependent on the pitcher and batter chosen
        # For these kinds of tests we only look at players with more appearances than min_obp_cutoff (167)

        # Good pitcher, bad batter
        good_bad_matchup = (666204, 462102)     # obp = 0.146, 0.049 -> ERA 0.005
        bad_good_matchup = (592464, 608061)     # obp = 0.367, 0.316 -> ERA 0.155
        good_good_matchup = (666204, 608061)    # obp = 0.146, 0.316 -> ERA 0.132
        bad_bad_matchup = (592464, 462102)      # obp = 0.367, 0.049 -> ERA 0.003

        # A Cardinals lineup (with Pedro Pages replaced because we don't have enough data for him)
        full_matchup = (666204, [691026, 676475, 575929, 502671, 680977, 571448, 663457, 669357, 608061])

        test_era(bd, *full_matchup)
    else:
        # distributions = load_blosc2('distributions.blosc2')
        transition_distribution = load_blosc2('transition_distribution.blosc2')
        solver = PolicySolver.from_saved('solved_policy.blosc2')
        raw_values, raw_policy = solver.raw_values, solver.raw_policy
        print(f'ERA {solver.get_value()}')
        pass  # Do something with the data or just examine it


if __name__ == '__main__':
    main(debug=False)
