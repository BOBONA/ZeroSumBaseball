import itertools
import math
import multiprocessing
import random
import threading
import time
from abc import ABC, abstractmethod

from tqdm import tqdm

from src.data.data_loading import BaseballData, save_blosc2, load_blosc2
from src.model.state import DebugRules, GameState
from src.policy.optimal_policy import PolicySolver, seed


type Permutation = tuple[int, ...]

rules = DebugRules


def swap(perm: Permutation, i: int, j: int) -> Permutation:
    perm = list(perm)
    perm[i], perm[j] = perm[j], perm[i]
    return tuple(perm)


class BattingOrderStrategy(ABC):

    def __init__(self):
        self.tries = dict()
        self.steps = []
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
            policy_solver.calculate_optimal_policy(beta=2e-4, use_last_values=True)

            # We can extract all permutations with the same cycle because of the symmetry of the problem
            for i in range(rules.num_batters):
                value = policy_solver.get_value(GameState(batter=i))
                idx = perm.index(i)
                cycle_perm = tuple(perm[idx:] + perm[:idx])
                self.record_try(cycle_perm, value)

            self.steps.append(self.best)

    def record_try(self, perm: Permutation, value: float):
        self.tries[perm] = value

        if self.best is None or value > self.best[1]:
            self.best = (perm, value)

    def get_tries(self) -> dict[Permutation, float]:
        return self.tries

    def get_best(self) -> tuple[Permutation, float]:
        return self.best

    def get_steps(self) -> list[tuple[Permutation, float]]:
        return self.steps


class BruteForce(BattingOrderStrategy):
    """Run through all permutations of the batting order"""

    def __init__(self):
        super().__init__()
        self.perm = itertools.permutations(range(rules.num_batters))
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
        self.size = math.factorial(rules.num_batters)

    def get_next_permutation(self) -> Permutation:
        perm = list(range(rules.num_batters))
        while not self.is_finished() and tuple(perm) in self.tries:
            random.shuffle(perm)
        return tuple(perm)

    def is_finished(self):
        return len(self.tries) >= self.size * 0.9


class OneByOne(BattingOrderStrategy):
    """Greedily fills the batting order, from first to last"""

    def __init__(self):
        super().__init__()
        self.perm = tuple(range(rules.num_batters))
        self.considering_batter = 0
        self.batter_choices = []

    def get_next_permutation(self) -> Permutation:
        if len(self.batter_choices) == rules.num_batters - self.considering_batter:
            self.perm = max(self.batter_choices, key=lambda x: self.tries[x])
            self.considering_batter += 1
            self.batter_choices = []

        perm = swap(self.perm, self.considering_batter, self.considering_batter + len(self.batter_choices))
        self.batter_choices.append(perm)

        return perm

    def is_finished(self):
        return self.considering_batter >= rules.num_batters - 1

    def get_solution(self) -> tuple[Permutation, float]:
        return self.perm, self.tries[self.perm]


class GreedyHillClimbing(BattingOrderStrategy):
    """A greedy algorithm that swaps two players as long as the score improves"""

    def __init__(self):
        super().__init__()
        self.possible_swaps = list(itertools.combinations(range(rules.num_batters), 2))
        self.swap_index = 0
        self.current_perm = tuple(range(rules.num_batters))

    def get_next_permutation(self) -> Permutation:
        if self.get_best() is not None and self.get_best()[0] != self.current_perm:
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


def test_strategy(strategy: BattingOrderStrategy, match: tuple, k: int, i: int = 0):
    policy_solver = PolicySolver(None, *match, rules=DebugRules)
    policy_solver.initialize_distributions(save_distributions=True, load_distributions=True, load_transition=True)

    for step in range(k):
        start = time.time()
        strategy.step(policy_solver)
        end = time.time()

        if step % 10 == 9:
            print(f'{strategy.__class__.__name__}: {step + 1}/{k}, time: {end - start:.2f}s, best: {strategy.get_best()}')

    save_blosc2(strategy, f'{strategy.__class__.__name__.lower()}/{i}.blosc2')
    graph_strategy(strategy)


def test_strategies():
    strategies = [BruteForce, Random, OneByOne, GreedyHillClimbing]
    k = 40
    num_matches = 1
    matches = load_blosc2('matches.blosc2')

    bd = BaseballData()
    for i in range(num_matches):
        match = matches[i]
        PolicySolver(bd, *match, rules=DebugRules).initialize_distributions(save_distributions=True)

        threads = []
        for strategy in strategies:
            strat = strategy()
            thread = multiprocessing.Process(target=test_strategy, args=(strat, match, k, i))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


def test():
    bd = BaseballData()
    test_match = (666204, [691026, 676475, 575929, 502671, 680977, 571448, 663457, 669357, 608061])

    policy_solver = PolicySolver(bd, *test_match, rules=DebugRules)
    policy_solver.initialize_distributions(save_distributions=True)

    strategy = GreedyHillClimbing()
    k = 40
    for _ in tqdm(range(k)):
        strategy.step(policy_solver)

    save_blosc2(strategy, 'greedy.blosc2')

    print('Starting permutation:', list(strategy.get_tries().keys())[0])
    print('Best permutation:', strategy.get_best())

    graph_strategy(strategy)


def graph_strategy(strategy: BattingOrderStrategy):
    import matplotlib.pyplot as plt

    steps = [x[1] for x in strategy.get_steps()]
    x = list(range(1, len(steps)))
    y = [max(steps[0:i]) for i in x]
    plt.plot(x, y)
    plt.show()


def generate_matches():
    """It's nice to have a fixed set of matches instead of generating them every time"""

    bd = BaseballData()
    num_matches = 1000
    matches = []

    for _ in range(num_matches):
        pitcher = random.choice(sorted(bd.pitchers))
        batters = random.sample(sorted(bd.batters), DebugRules.num_batters)
        matches.append((pitcher, batters))

    save_blosc2(matches, 'matches.blosc2')


if __name__ == '__main__':
    seed()
    test_strategies()
