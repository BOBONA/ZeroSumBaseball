from collections import defaultdict

import cvxpy as cp
import torch
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from src.data.data_loading import BaseballData
from src.data.datasets import SwingResult, PitchDataset, PitchControlDataset
from src.distributions.batter_patience import BatterSwings, batter_patience_map
from src.distributions.pitcher_control import PitcherControl
from src.distributions.swing_outcome import SwingOutcome, map_swing_outcome
from src.model.at_bat import AtBatState, PitchResult, AtBatOutcome, AtBat
from src.model.pitch import Pitch
from src.model.pitch_type import PitchType
from src.model.players import Pitcher, Batter
from src.model.zones import Zone, NON_BORDERLINE_ZONES, get_zone, BORDERLINE_ZONES, get_zone_batched

# To simplify typing, in accordance with the reference paper's notation
type A = tuple[PitchType, Zone]
type O = bool
type S = AtBatState

pitcher_actions: list[A] = [(pitch_type, zone) for zone in NON_BORDERLINE_ZONES for pitch_type in PitchType]
batter_actions: list[O] = [True, False]
game_states: list[S] = [AtBatState(balls=balls, strikes=strikes) for balls in range(4) for strikes in range(3)]  # Normal states
final_states: list[S] = ([AtBatState(outcome=AtBatOutcome.BASE, balls=balls, strikes=strikes) for balls in range(5) for strikes in range(3)] +  # Base states
                         [AtBatState(outcome=AtBatOutcome.OUT, balls=balls, strikes=strikes) for balls in range(4) for strikes in range(4)])  # Out states

# We define some type aliases to improve readability
type Policy = defaultdict[S, dict[A, float]]
type TransitionDistribution = dict[S, dict[A, dict[O, dict[S, float]]]]

type SwingOutcomeDistribution = dict[S, dict[A, dict[SwingResult, float]]]
type PitcherControlDistribution = dict[A, dict[Zone, float]]
type BatterPatienceDistribution = dict[S, dict[A, float]]


def calculate_swing_outcome_distribution(matchups: list[tuple[Pitcher, Batter]], batch_size: int = 512) -> dict[tuple[Pitcher, Batter], SwingOutcomeDistribution]:
    """
    Calculates the distribution of swing outcomes for a list of pitcher and batter pairs, given the current game state.
    This method takes in a list to allow for batch processing.

    :return: A dictionary mapping a state, pitcher action and batter action to a distribution of swing outcomes
    """

    swing_outcome = defaultdict(lambda: defaultdict(defaultdict))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    swing_outcome_model = SwingOutcome().to(device)
    swing_outcome_model.load_state_dict(torch.load('../../model_weights/swing_outcome, g=2.75.pth'))
    swing_outcome_model.eval()

    for state in tqdm(game_states, desc='Calculating swing outcomes'):
        for pitch in tqdm(pitcher_actions, leave=False):
            # Some finicky code is needed to batch process the swing outcome model, once state gets
            # more complicated, we'll need to batch process a loop out of this
            pitch_type, zone = pitch
            pitches = [Pitch(state, AtBat(None, pitcher, batter, state), zone, pitch_type, 0, PitchResult.HIT) for pitcher, batter in matchups]
            pitch_dataset = PitchDataset(data_source=None, pitches=pitches, map_to=map_swing_outcome)
            pitch_dataloader = DataLoader(pitch_dataset, batch_size=batch_size, shuffle=False)

            for i, (pitch_idx, pitch_data, target) in enumerate(pitch_dataloader):
                pitch_data = [d.to(device) for d in pitch_data]
                outcome_tensor = swing_outcome_model(*pitch_data)

                # Iterate through the batched outcomes
                outcome_list = outcome_tensor.squeeze().tolist()
                if not isinstance(outcome_list[0], list):
                    outcome_list = [outcome_list]

                for j, (pitcher, batter) in enumerate(matchups[i * batch_size:(i + 1) * batch_size]):
                    outcome_distribution = {SwingResult(k): outcome_list[j][k] for k in range(len(outcome_list[j]))}
                    swing_outcome[(pitcher, batter)][state][pitch] = outcome_distribution

    return swing_outcome


def calculate_pitcher_control_distribution(pitchers: list[Pitcher], batch_size: int = 512) -> dict[Pitcher, PitcherControlDistribution]:
    """
    Calculates the distribution of actual pitch outcomes for a given pitcher, given the intended pitch type and zone

    :return: A dictionary mapping a pitcher action to a distribution of actual pitch outcomes over zones
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pitcher_control_model = PitcherControl().to(device)
    pitcher_control_model.load_state_dict(torch.load('../../model_weights/pitcher_control.pth'))
    pitcher_control_model.eval()

    pitcher_type_control = defaultdict(defaultdict)

    # This dataset class automatically iterates over pitch types
    pitch_control_dataset = PitchControlDataset(data_source=None, pitchers=pitchers, empty_data=True)
    dataloader = DataLoader(pitch_control_dataset, batch_size=batch_size, shuffle=False)

    for i, (p_i, p_type, p_data, distribution) in tqdm(enumerate(dataloader), desc='Calculating pitcher control'):
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
    pitcher_control = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for pitcher in tqdm(pitchers, desc='Sampling pitcher control'):
        for pitch in pitcher_actions:
            pitch_type, intended_zone = pitch
            zone_center = intended_zone.center()
            gaussian = pitcher_type_control[pitcher][pitch_type]
            gaussian = MultivariateNormal(torch.tensor([zone_center[0], zone_center[1]]), gaussian.covariance_matrix)

            num_samples = 1000
            sample_pitches = gaussian.sample(torch.Size((num_samples,)))
            zones = get_zone_batched(sample_pitches[:, 0], sample_pitches[:, 1])
            for zone in zones:
                pitcher_control[pitcher][pitch][zone] += 1 / num_samples

    return pitcher_control


def calculate_batter_patience_distribution(batters: list[Batter], batch_size: int = 512) -> dict[Batter, BatterPatienceDistribution]:
    """
    Calculates the distribution of batter patience for a given batter, given the current game state and pitcher action

    :return: A dictionary mapping a state and pitcher action (on a borderline zone) to the probability that the batter will swing
    """

    batter_patience = defaultdict(lambda: defaultdict(dict))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batter_patience_model = BatterSwings().to(device)
    batter_patience_model.load_state_dict(torch.load('../../model_weights/batter_patience.pth'))
    batter_patience_model.eval()

    for state in tqdm(game_states, desc='Calculating batter patience'):
        for pitch_type in PitchType:
            for zone in BORDERLINE_ZONES:
                pitches = [Pitch(state, AtBat(None, None, batter, state), zone, pitch_type, 0, PitchResult.HIT) for batter in batters]
                patience_dataset = PitchDataset(data_source=None, pitches=pitches, map_to=batter_patience_map)
                dataloader = DataLoader(patience_dataset, batch_size=batch_size, shuffle=False)

                for i, (pitch_idx, b_data, swing) in enumerate(dataloader):
                    b_data = [d.to(device) for d in b_data]
                    swing_percent = batter_patience_model(*b_data).squeeze().tolist()

                    if not isinstance(swing_percent, list):
                        swing_percent = [swing_percent]
                    for j, batter in enumerate(batters[i * batch_size:(i + 1) * batch_size]):
                        batter_patience[batter][state][(pitch_type, zone)] = swing_percent[j]

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

    batter_patience = calculate_batter_patience_distribution([batter])[batter] if batter_patience is None else batter_patience
    swing_outcome = calculate_swing_outcome_distribution([(pitcher, batter)])[pitcher, batter] if swing_outcome is None else swing_outcome
    pitcher_control = calculate_pitcher_control_distribution([pitcher])[pitcher] if pitcher_control is None else pitcher_control

    transition_distribution: TransitionDistribution = defaultdict(dict)

    for state in game_states:
        for action in pitcher_actions:
            transition_distribution[state][action] = {}
            pitch_type, intended_zone = action
            for batter_swung in batter_actions:
                transition_distribution[state][action][batter_swung] = defaultdict(float)

                # Given a pitcher and batter action, calculate the actual outcome distribution
                # The pitch outcome is a distribution over zones given the intended pitch
                for outcome_zone, prob in pitcher_control[action].items():
                    # To account for batter patience, we override the outcomes for borderline zones,
                    # to be a measure of the batter's patience. This requires another nested loop
                    swing_probs = {o: float(batter_swung == o) for o in batter_actions}

                    if outcome_zone.is_borderline:
                        patience = batter_patience[state][(pitch_type, outcome_zone)]
                        swing_probs[True] = patience
                        swing_probs[False] = 1 - patience

                    # Loop over the swing "outcomes". When the zone is not borderline, this part isn't
                    # stochastic and the loop is redundant (swing_prob = {True: 1/0, False: 0/1})
                    for outcome_swing, swing_prob in swing_probs.items():
                        if outcome_swing:
                            swing_results = swing_outcome[state][action]
                            for swing_result, result_prob in swing_results.items():
                                next_state = state.transition_from_pitch_result(swing_result.to_pitch_result())
                                transition_distribution[state][action][batter_swung][next_state] += prob * result_prob * swing_prob
                        else:
                            next_state = state.transition_from_pitch_result(PitchResult.CALLED_STRIKE if outcome_zone.is_strike
                                                                            else PitchResult.CALLED_BALL)
                            transition_distribution[state][action][batter_swung][next_state] += prob * swing_prob

    return transition_distribution


def update_policy(action_quality: dict[A, dict[O, float]]) -> tuple[dict[A, float], float]:
    """Optimizes a new policy using dynamic programming"""

    policy = cp.Variable(len(pitcher_actions))
    policy_constraints = [policy >= 0, cp.sum(policy) == 1, policy <= 0.7]  # Limit the maximum probability of any action

    # We want to maximize the minimum expected value of the next state
    objective = cp.Maximize(cp.minimum(*[sum([policy[a_i] * action_quality[a][o]
                                       for a_i, a in enumerate(pitcher_actions)]) for o in batter_actions]))

    problem = cp.Problem(objective, policy_constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL:
        raise ValueError(f'Policy optimization failed, status {problem.status}')
    else:
        new_policy = {a: policy[a_i].value for a_i, a in enumerate(pitcher_actions)}
        return new_policy, problem.value


def calculate_optimal_policy(pitcher: Pitcher, batter: Batter, transition_distribution: TransitionDistribution | None = None,
                             beta: float = 1e-5, discount_factor: float = 0.9) -> tuple[Policy, dict[S, float]]:
    """
    Uses value iteration algorithm to calculate the optimal policy for our model, given
    the pitcher and batter. https://doi.org/10.1016/B978-1-55860-335-6.50027-1

    A policy (or mixed strategy) defines a probability distribution over actions for each state for
    the pitcher.

    :param transition_distribution: The precalculated transition distribution
    :param pitcher: The pitcher
    :param batter: The batter
    :param beta: The minimum change in value to continue iterating
    :param discount_factor: Weights future rewards over closer rewards
    :return: The optimal pitcher policy, assigning a probability to each action in each state and the value of each state
    """

    # Value iteration repeatedly updates the "value" of each state while optimizing the policy
    value: defaultdict[S, float] = defaultdict(float)
    for state in final_states:
        if state.outcome != AtBatOutcome.NONE:
            value[state] = state.value()  # Initialize final states to their true values

    policy: Policy = defaultdict(lambda: {a: 1 / len(pitcher_actions) for a in pitcher_actions})

    transition_distribution = precalculate_transition_distribution(pitcher, batter) if transition_distribution is None else transition_distribution

    # The expected "immediate" reward for each state-action pair
    reward = {state: {a: {o: sum([transition_distribution[state][a][o][next_state] * next_state.value()
                                  for next_state in (game_states + final_states)]) for o in batter_actions} for a in pitcher_actions}  for state in game_states}

    difference = float('inf')
    while difference > beta:
        # Independently optimize the policy for each state
        new_policy = defaultdict(dict)
        new_value = defaultdict(float)
        for state in game_states:
            action_quality = {a: {o: reward[state][a][o] + discount_factor * sum([transition_distribution[state][a][o][next_state] * value[next_state]
                                                                                  for next_state in game_states])
                                  for o in batter_actions} for a in pitcher_actions}
            new_policy[state], new_value[state] = update_policy(action_quality)

        # Update values
        difference = max([abs(new_value[state] - value[state]) for state in game_states])
        policy = new_policy
        value = new_value

    # Remove actions with effectively zero probability
    for state in policy:
        for action in policy[state].copy():
            policy[state][action] = round(policy[state][action], 4)
            if policy[state][action] == 0:
                del policy[state][action]

    return policy, value


def main():
    bd = BaseballData.load_with_cache()

    pitcher = list(bd.pitchers.values())[0]
    batter = list(bd.batters.values())[0]

    optimal_policy, value = calculate_optimal_policy(pitcher, batter, beta=1e-5, discount_factor=0.9)
    print(optimal_policy)


if __name__ == '__main__':
    main()
