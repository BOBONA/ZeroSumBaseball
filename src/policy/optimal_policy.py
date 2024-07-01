import random
from collections import defaultdict

import cvxpy as cp
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader

try:
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
from src.model.zones import ZONES, Zone, NON_BORDERLINE_ZONES, BORDERLINE_ZONES, get_zones_batched

# To simplify typing, in accordance with the reference paper's notation
type A = tuple[PitchType, Zone]
type O = bool
type S = AtBatState

pitcher_actions: list[A] = [(pitch_type, zone) for zone in NON_BORDERLINE_ZONES for pitch_type in PitchType]
batter_actions: list[O] = [False, True]  # Order is important!

game_states: list[S] = [
    AtBatState(balls=balls, strikes=strikes, outs=outs, first=first, second=second, third=third)
    for balls in range(4) for strikes in range(3) for outs in range(3)
    for first in [False, True] for second in [False, True] for third in [False, True]
]

# Sort by "lateness" of the state, to optimize the value iteration algorithm
game_states.sort(key=lambda st: (st.num_outs + st.num_runs) * 100 + (st.balls + st.strikes) * 10 +
                 int(st.third) * 4 + int(st.second) * 2 + int(st.first), reverse=True)

# Terminal states are stored separately for easier indexing
final_states: list[S] = [
    AtBatState(balls=balls, strikes=strikes, outs=3, first=first, second=second, third=third)
    for balls in range(4) for strikes in range(3)
    for first in [False, True] for second in [False, True] for third in [False, True]
]

total_states = game_states + final_states

total_states_dict = {state: i for i, state in enumerate(total_states)}

# We define some type aliases to improve readability, note how they are indexed into
type Policy = list[list[float]]  # [S_i][A_i] -> float
type TransitionDistribution = list[list[list[dict[int, tuple[float, float]]]]]  # [S_i][A_i][O_i] -> {total_states_i: (next state probability, reward)}

# These distributions can be indexed into according to the lists defined above
type SwingOutcomeDistribution = list[list[list[float]]]   # [S_i][A_i][SwingResult] -> swing result probability
type PitcherControlDistribution = list[list[float]]  # [A_i][ZONE_i] -> pitch outcome zone probability
type BatterPatienceDistribution = list[list[list[float]]]  # [S_i][PitchType][(BORDERLINE_)ZONE_i] -> batter swing probability

assert ZONES[0] == BORDERLINE_ZONES[0]  # BatterPatienceDistribution indexing relies on this

batch_size = 512

# This is a flag that improves the convergence by calculating states with the values of the current iteration
use_ordered_iteration: bool = True


def tuple_help():
    """Utility for default dict"""

    return 0, 0


def calculate_swing_outcome_distribution(matchups: list[tuple[Pitcher, Batter]]) -> dict[tuple[Pitcher, Batter], SwingOutcomeDistribution]:
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
        swing_outcome[(pitcher, batter)] = [[[] for _ in range(len(pitcher_actions))] for _ in range(len(game_states))]

        pitch_states = [(state_i, pitch_i)
                        for state_i in range(len(game_states)) for pitch_i in range(len(pitcher_actions))]
        pitch_data = [Pitch(game_states[state_i], AtBat(None, pitcher, batter, game_states[state_i]),
                            0, pitcher_actions[pitch_i][1], pitcher_actions[pitch_i][0], 0, PitchResult.HIT_SINGLE)
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
                result_distribution[3] = max(result_distribution[3], 0.12)  # Ensure reasonable values
                swing_outcome[(pitcher, batter)][state_i][pitch_i] = result_distribution

    return swing_outcome


def calculate_pitcher_control_distribution(pitchers: list[Pitcher]) -> dict[Pitcher, PitcherControlDistribution]:
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
        pitcher_control[pitcher] = [[0 for _ in range(len(ZONES))] for _ in range(len(pitcher_actions))]

        for pitch_i, pitch in enumerate(pitcher_actions):
            pitch_type, intended_zone = pitch
            zone_center = intended_zone.center()
            gaussian = pitcher_type_control[pitcher][pitch_type]
            gaussian = MultivariateNormal(torch.tensor([zone_center[0], zone_center[1]]), gaussian.covariance_matrix)

            num_samples = 1000
            sample_pitches = gaussian.sample(torch.Size((num_samples,)))
            zones = get_zones_batched(sample_pitches[:, 0], sample_pitches[:, 1])
            for zone_i in zones:
                pitcher_control[pitcher][pitch_i][zone_i] += 1 / num_samples

    return pitcher_control


def calculate_batter_patience_distribution(batters: list[Batter]) -> dict[Batter, BatterPatienceDistribution]:
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
        batter_patience[batter] = [[[0 for _ in range(len(BORDERLINE_ZONES))] for _ in range(len(PitchType))]
                                   for _ in range(len(game_states))]

        pitch_states = [(state_i, type_i, borderline_zone_i) for state_i in range(len(game_states))
                        for type_i in range(len(PitchType)) for borderline_zone_i in range(len(BORDERLINE_ZONES))]
        pitch_data = [Pitch(game_states[state_i], AtBat(None, None, batter, game_states[state_i]),
                            0, BORDERLINE_ZONES[zone_i], PitchType(type_i), 0, PitchResult.HIT_SINGLE)
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


def precalculate_transition_distribution(pitcher: Pitcher, batter: Batter,
                                         batter_patience: BatterPatienceDistribution | None = None,
                                         swing_outcome: SwingOutcomeDistribution | None = None,
                                         pitcher_control: PitcherControlDistribution | None = None) -> TransitionDistribution:
    """
    Precalculates the transition probabilities for a given pitcher and batter pair.
    This method is complicated by the fact that both the pitch outcome and the swing outcome
    are stochastic.
    """

    swing_outcome = calculate_swing_outcome_distribution([(pitcher, batter)])[pitcher, batter] if swing_outcome is None else swing_outcome
    pitcher_control = calculate_pitcher_control_distribution([pitcher])[pitcher] if pitcher_control is None else pitcher_control
    batter_patience = calculate_batter_patience_distribution([batter])[batter] if batter_patience is None else batter_patience

    transition_distribution: TransitionDistribution = [[[defaultdict(tuple_help) for _ in range(len(batter_actions))]
                                                        for _ in range(len(pitcher_actions))] for _ in range(len(game_states))]

    for state_i, state in tqdm(enumerate(game_states), desc='Calculating transition distribution', total=len(game_states)):
        for action_i, action in enumerate(pitcher_actions):
            pitch_type, intended_zone = action
            for batter_swung in batter_actions:

                # Given a pitcher and batter action, calculate the actual outcome distribution
                # The pitch outcome is a distribution over zones given the intended pitch
                for outcome_zone_i, prob in enumerate(pitcher_control[action_i]):

                    # To account for batter patience, we override the outcomes for borderline zones,
                    # to be a measure of the batter's patience. This requires another nested loop
                    swing_probs = {o: float(batter_swung == o) for o in batter_actions}

                    outcome_zone = ZONES[outcome_zone_i]
                    if outcome_zone.is_borderline:
                        patience = batter_patience[state_i][pitch_type][outcome_zone_i]
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
                                next_state.num_runs = 0  # We don't want to distinguish between states with different runs

                                next_state_i = total_states_dict[next_state]
                                current_prob = transition_distribution[state_i][action_i][batter_swung][next_state_i][0]
                                transition_distribution[state_i][action_i][batter_swung][next_state_i] = (current_prob + prob * result_prob * swing_prob, reward)
                        else:
                            next_state = state.transition_from_pitch_result(PitchResult.CALLED_STRIKE if outcome_zone.is_strike
                                                                            else PitchResult.CALLED_BALL)
                            reward = next_state.value() - state.value()
                            next_state.num_runs = 0

                            next_state_i = total_states_dict[next_state]
                            current_prob = transition_distribution[state_i][action_i][batter_swung][next_state_i][0]
                            transition_distribution[state_i][action_i][batter_swung][next_state_i] = (current_prob + prob * swing_prob, reward)

    return transition_distribution


def update_policy(action_quality: list[list[float]], max_pitch_percentage: float = 0.7) -> tuple[list[float], float]:
    """Optimizes a new policy using dynamic programming"""

    policy = cp.Variable(len(pitcher_actions))
    policy_constraints = [policy >= 0, cp.sum(policy) == 1, policy <= max_pitch_percentage]  # Limit the maximum probability of any action

    # We want to minimize the maximum expected value (for the batter) of the next state
    objective = cp.Minimize(cp.maximum(*[sum([policy[a_i] * action_quality[a_i][o]
                                       for a_i in range(len(pitcher_actions))]) for o in batter_actions]))

    problem = cp.Problem(objective, policy_constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL and problem.status != cp.OPTIMAL_INACCURATE:
        raise ValueError(f'Policy optimization failed, status {problem.status}')
    else:
        # if problem.status == cp.OPTIMAL_INACCURATE:
        #     warnings.warn('Inaccurate optimization detected')
        new_policy = [policy[a_i].value for a_i in range(len(pitcher_actions))]
        return new_policy, problem.value


def calculate_optimal_policy(pitcher: Pitcher, batter: Batter, transition_distribution: TransitionDistribution | None = None,
                             beta: float = 1e-5) -> tuple[Policy, list[float]]:
    """
    Uses value iteration algorithm to calculate the optimal policy for our model, given
    the pitcher and batter. https://doi.org/10.1016/B978-1-55860-335-6.50027-1

    A policy (or mixed strategy) defines a probability distribution over actions for each state for
    the pitcher.

    :param transition_distribution: The precalculated transition distribution
    :param pitcher: The pitcher
    :param batter: The batter
    :param beta: The minimum change in value to continue iterating
    :return: The optimal pitcher policy, assigning a probability to each action in each state and the value of each state
    """

    # Stores the "value" of each state, indexed according to total_states
    # Last time I measured, random initialization in this manner converged 1/3 faster
    value = [random.random() for _ in game_states] + [0 for _ in final_states]

    # Stores the policy for each state, indexed according to game_states
    policy: Policy = [[1 / len(pitcher_actions) for _ in pitcher_actions] for _ in game_states]

    transition_distribution = precalculate_transition_distribution(pitcher, batter) if transition_distribution is None else transition_distribution

    difference = float('inf')
    iter_num = 0
    while difference > beta:
        iter_num += 1

        # Independently optimize the policy for each state
        new_policy = []
        new_value = value.copy()

        value_src = new_value if use_ordered_iteration else value  # Faster convergence with "ordered iteration"

        # Note, this can be parallelized
        for state_i, state in tqdm(enumerate(game_states), f'Iterating over values, iter={iter_num}', total=len(game_states)):
            action_quality = [[sum([prob * (reward + value_src[next_state_i])
                                    for next_state_i, (prob, reward) in transition_distribution[state_i][a_i][o].items()])
                               for o in batter_actions] for a_i in range(len(pitcher_actions))]

            new_state_policy, new_value[state_i] = update_policy(action_quality)
            new_policy.append(new_state_policy)

        # Update values
        difference = max([abs(new_value[state_i] - value[state_i]) for state_i in range(len(game_states))])
        policy = new_policy
        value = new_value

        print(difference)

    return policy, value


def main():
    bd = BaseballData.load_with_cache()

    # Note how the resulting ERA is highly dependent on the pitcher and batter chosen
    # Try pitchers[2] (obp_percentile = 0.21) and batters[0] (obp_percentile = 0.95) for a higher ERA
    # Also note that lower percentile batters often have little data and create unstable results from the models

    pitcher = list(bd.pitchers.values())[90]  # obp_percentile = 0.94
    batter = list(bd.batters.values())[513]  # obp_percentile = 0.54

    optimal_policy, value = calculate_optimal_policy(pitcher, batter, beta=1e-3)
    print(f'ERA {value[total_states_dict[AtBatState()]]}')

    torch.save(optimal_policy, 'optimal_policy.pth')
    torch.save(value, 'value.pth')


if __name__ == '__main__':
    main()
    value = torch.load('value.pth')
    print(value[total_states_dict[AtBatState()]])
