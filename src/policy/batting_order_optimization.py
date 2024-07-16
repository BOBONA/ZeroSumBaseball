import itertools
import math
import random
from abc import ABC, abstractmethod

import numpy as np
import torch
from tqdm import tqdm

from src.data.data_loading import BaseballData, save_blosc2
from src.model.state import GameState
from src.policy.optimal_policy import PolicySolver


type Permutation = tuple[int, ...]


def swap(perm: Permutation, i: int, j: int) -> Permutation:
    perm = list(perm)
    perm[i], perm[j] = perm[j], perm[i]
    return tuple(perm)


class BattingOrderStrategy(ABC):

    def __init__(self):
        self.tries = dict()
        self.best = None

    @abstractmethod
    def get_next_permutation(self) -> Permutation:
        """Get the next permutation to try. This will be called repeatedly until a new permutation is fetched."""
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        """Returns whether the strategy is complete"""
        pass

    def get_solution(self) -> tuple[Permutation, float]:
        """Returns the strategy's best solution"""

        return self.get_best()

    def step(self, policy_solver: PolicySolver):
        perm = None
        while (perm is None or perm in self.tries) and not self.is_finished():
            perm = self.get_next_permutation()

        if not self.is_finished():
            policy_solver.set_batter_permutation(perm)
            policy_solver.calculate_optimal_policy()

            # We can extract all permutations with the same cycle because of the symmetry of the problem
            for i in range(GameState.num_batters):
                value = policy_solver.get_value(GameState(batter=i))
                cycle_perm = tuple(perm[i:] + perm[:i])
                self.record_try(cycle_perm, value)

    def record_try(self, perm: Permutation, value: float):
        self.tries[perm] = value

        if self.best is None or value > self.best[1]:
            self.best = (perm, value)

    def get_tries(self) -> dict[Permutation, float]:
        return self.tries

    def get_best(self) -> tuple[Permutation, float]:
        return self.best


class BruteForce(BattingOrderStrategy):
    """Run through all permutations of the batting order"""

    def __init__(self):
        super().__init__()
        self.perm = itertools.permutations(range(GameState.num_batters))
        self.next = next(self.perm)

    def get_next_permutation(self) -> Permutation:
        perm = self.next
        self.next = next(self.perm, None)
        return tuple(perm)

    def is_finished(self):
        return self.next is None


class Random(BattingOrderStrategy):
    """Tries random permutations"""

    def __init__(self):
        super().__init__()
        self.size = math.factorial(GameState.num_batters)

    def get_next_permutation(self) -> Permutation:
        perm = list(range(GameState.num_batters))
        while not self.is_finished() or tuple(perm) in self.tries:
            random.shuffle(perm)
        return tuple(perm)

    def is_finished(self):
        return len(self.tries) >= self.size * 0.9


class OneByOne(BattingOrderStrategy):
    """Greedily fills the batting order, from first to last"""

    def __init__(self):
        super().__init__()
        self.perm = tuple(range(GameState.num_batters))
        self.considering_batter = 0
        self.batter_choices = []

    def get_next_permutation(self) -> Permutation:
        perm = swap(self.perm, self.considering_batter, self.considering_batter + len(self.batter_choices))
        self.batter_choices.append(perm)

        if len(self.batter_choices) == GameState.num_batters - self.considering_batter:
            self.perm = max(self.batter_choices, key=lambda x: self.tries[x])
            self.considering_batter += 1
            self.batter_choices = []

        return perm

    def is_finished(self):
        return self.considering_batter >= GameState.num_batters - 1

    def get_solution(self) -> tuple[Permutation, float]:
        return self.perm, self.tries[self.perm]


class GreedyHillClimbing(BattingOrderStrategy):
    """A greedy algorithm that swaps two players as long as the score improves"""

    def __init__(self):
        super().__init__()
        self.possible_swaps = list(itertools.combinations(range(GameState.num_batters), 2))
        self.swap_index = 0
        self.current_perm = tuple(range(GameState.num_batters))

    def get_next_permutation(self) -> Permutation:
        if self.get_best() is not None and self.get_best() != self.current_perm:
            self.current_perm = self.get_best()[0]
            self.swap_index = 0

        perm = swap(self.current_perm, *self.possible_swaps[self.swap_index])
        self.swap_index += 1

        return perm

    def is_finished(self) -> bool:
        return self.swap_index >= len(self.possible_swaps) and self.get_best() != self.current_perm


class StochasticHillClimbing(BattingOrderStrategy):
    """
    TODO A stochastic variation of the hill climbing algorithm, which considers every swap with a probability
    based on the score improvement/loss.
    """

    def __init__(self):
        super().__init__()

    def get_next_permutation(self) -> Permutation:
        pass

    def is_finished(self) -> bool:
        pass


class SamplingHillClimbing(BattingOrderStrategy):
    """TODO A variant of the hill climbing algorithm that samples a number of random swaps and selects the best one"""

    def __init__(self):
        super().__init__()

    def get_next_permutation(self) -> Permutation:
        pass

    def is_finished(self) -> bool:
        pass


def seed():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


def main():
    bd = BaseballData()
    test_match = (666204, [691026, 676475, 575929, 502671, 680977, 571448, 663457, 669357, 608061])  # ERA 1.05

    seed()
    policy_solver = PolicySolver(bd, *test_match)
    policy_solver.initialize_distributions()

    strategy = BruteForce()
    with tqdm(total=math.factorial(GameState.num_batters)) as pbar:
        while not strategy.is_finished():
            strategy.step(policy_solver)
            pbar.update(1)

    save_blosc2(strategy, 'brute_force.blosc2')

    print('Starting permutation:', strategy.get_tries()[tuple(range(GameState.num_batters))])
    print('Best permutation:', strategy.get_best())


if __name__ == '__main__':
    main()
