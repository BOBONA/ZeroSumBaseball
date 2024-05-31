import random
from collections import defaultdict
from copy import copy

import cvxpy as cp

import torch
from torch.distributions import MultivariateNormal

from src.data.datasets import SwingResult
from src.distributions.batter_patience import BatterSwings
from src.distributions.pitcher_control import PitcherControl
from src.distributions.swing_outcome import SwingOutcome
from src.model.at_bat import AtBatState, PitchResult, AtBatOutcome
from src.model.pitch import Pitch
from src.model.pitch_type import PitchType
from src.model.players import Pitcher, Batter
from src.model.zones import Zone, NON_BORDERLINE_ZONES

# In accordance with the reference paper's notation
type A = tuple[PitchType, Zone]
type O = bool
type S = AtBatState

pitcher_actions: list[A] = [(pitch_type, zone) for zone in NON_BORDERLINE_ZONES for pitch_type in PitchType]
batter_actions: list[O] = [True, False]
game_states: list[S] = [AtBatState(balls=balls, strikes=strikes) for balls in range(4) for strikes in range(3)]


def precalculate_distributions(pitcher: Pitcher, batter: Batter) -> tuple[defaultdict[S, dict[A, dict[SwingResult, float]]], dict[PitchType, MultivariateNormal], defaultdict[S, dict[A, float]]]:
    """Precalculates the available distribution data for a pitcher and batter"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    swing_outcome_model = SwingOutcome().to(device)
    swing_outcome_model.load_state_dict(torch.load('../../model_weights/swing_outcome, g=2.75.pth'))
    swing_outcome_model.eval()
    swing_outcome = defaultdict(dict)
    for state in game_states:
        for pitch in pitcher_actions:
            input_data = pitcher.data, batter.data, Pitch.get_encoding(*pitch), torch.tensor(state.strikes, dtype=torch.float32), torch.tensor(state.balls, dtype=torch.float32)
            input_data = [d.unsqueeze(0) for d in input_data]
            outcome_tensor = swing_outcome_model(*input_data)
            swing_outcome[state][pitch] = {SwingResult(i): val for i, val in enumerate(outcome_tensor.squeeze().tolist())}

    pitcher_control_model = PitcherControl().to(device)
    pitcher_control_model.load_state_dict(torch.load('../../model_weights/pitcher_control.pth'))
    pitcher_control_model.eval()
    pitcher_control = {}
    for pitch_type in PitchType:
        control_tensor = pitcher_control_model(pitcher.data.unsqueeze(0), pitch_type.get_one_hot_encoding().unsqueeze(0))[0]
        pitcher_control[pitch_type] = MultivariateNormal(control_tensor[:2], torch.tensor([[control_tensor[2], control_tensor[4]],
                                                                                      [control_tensor[4], control_tensor[3]]]))

    batter_patience_model = BatterSwings().to(device)
    batter_patience_model.load_state_dict(torch.load('../../model_weights/batter_patience.pth'))
    batter_patience_model.eval()
    batter_patience = defaultdict(dict)
    for state in game_states:
        for pitch in pitcher_actions:
            input_data = batter.data, Pitch.get_encoding(*pitch), torch.tensor(state.strikes, dtype=torch.float32), torch.tensor(state.balls, dtype=torch.float32)
            input_data = [d.unsqueeze(0) for d in input_data]
            batter_patience[state][pitch] = batter_patience_model(*input_data).item()

    return swing_outcome, pitcher_control, batter_patience


def update_policy(state: S, action_quality: defaultdict[tuple[S, A, O], float]) -> dict[A, float]:
    """Optimizes a new policy using dynamic programming"""

    policy = cp.Variable(len(pitcher_actions))
    policy_constraints = [policy >= 0, cp.sum(policy) == 1]

    objective = cp.Maximize(cp.min([cp.sum([policy[a] * action_quality[(state, a, o)]
                                            for a in pitcher_actions]) for o in batter_actions]))

    problem = cp.Problem(objective, policy_constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL:
        raise ValueError(f'Policy optimization failed, status {problem.status}')
    else:
        new_policy = {a: policy[i].value for i, a in enumerate(pitcher_actions)}
        return new_policy


def calculate_optimal_policy(pitcher: Pitcher, batter: Batter, iterations: int, discount_factor: float, decay: float,
                             exploration: float) -> defaultdict[S, dict[A, float]]:
    """
    Uses the minimax-Q algorithm to calculate the optimal policy for our model, given
    the pitcher and batter. https://doi.org/10.1016/B978-1-55860-335-6.50027-1

    :param iterations: The number of iterations to run the algorithm
    :param batter: The batter
    :param pitcher: The pitcher
    :param discount_factor: Weights future rewards over closer rewards
    :param decay: Learning rate decay
    :param exploration: The exploration parameter
    :return: The optimal pitcher policy, assigning a probability to each action in each state
    """

    swing_outcome, pitcher_control, batter_patience = precalculate_distributions(pitcher, batter)

    # Our final policy
    policy: defaultdict[S, dict[A, float]] = defaultdict(lambda: defaultdict(lambda: 1.0))
    alpha = 1.0  # The learning rate

    # The quality of pitcher's action against a batter's action in a given state
    action_quality: defaultdict[tuple[S, A, O], float] = defaultdict(lambda: 1.0)

    # The value of a state
    value: defaultdict[S, float] = defaultdict(lambda: 1.0)

    current_state: AtBatState = AtBatState()
    while iterations > 0:
        # Choose a pitcher action
        explore = random.random() < exploration
        if explore:
            action = random.choice(pitcher_actions)
        else:
            action = random.choices(pitcher_actions, weights=[policy[current_state][action] for action in pitcher_actions], k=1)[0]

        pitch_zone, pitch_type = action
        # TODO Calculate the actual pitch location (sample from pitcher control Gaussian)
        # TODO Decide whether the batter will swing (how does patience factor into this?
        batter_action = True

        # Transition the state
        if not batter_action:  # Batter takes
            result = PitchResult.CALLED_BALL if pitch_zone.is_strike else PitchResult.CALLED_STRIKE
        else:  # Batter swings
            result_distribution = swing_outcome[current_state][action]
            result = random.choices(list(result_distribution.keys()), weights=list(result_distribution.values()), k=1)[0]

        previous_state = copy(current_state)
        current_state.transition_from_pitch_result(result)
        reward = current_state.value()

        # Update values
        action_quality[(previous_state, action, batter_action)] = (
                (1 - alpha) * action_quality[(previous_state, action, batter_action)] +
                alpha * (reward + discount_factor * value[current_state]))

        policy[previous_state] = update_policy(previous_state, action_quality)

        value[previous_state] = min([sum([policy[previous_state][a] * action_quality[(previous_state, a, o)]]
                                         for a in A) for o in O])
        alpha *= decay

        if current_state.outcome != AtBatOutcome.NONE:
            # TODO do we reset the game here?
            current_state = AtBatState()

        iterations -= 1

    return policy
