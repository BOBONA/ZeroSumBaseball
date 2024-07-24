import itertools
import math
import multiprocessing
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from src.data.data_loading import BaseballData, save_blosc2, load_blosc2
from src.model.players import Pitcher
from src.model.state import DebugRules, GameState, Rules
from src.policy.optimal_policy import PolicySolver, seed
from src.policy.rosters import rosters, pitchers

type Permutation = tuple[int, ...]

# This can be swapped out for the regular rules, but will require way more computation
rules = DebugRules


def swap(perm: Permutation, i: int, j: int) -> Permutation:
    perm = list(perm)
    perm[i], perm[j] = perm[j], perm[i]
    return tuple(perm)


class BattingOrderStrategy(ABC):
    """
    This is a base class for strategies that try to optimize the batting order.
    It's important to understand how this works to create new strategies and understand the existing ones.
    A strategy can only suggest a new permutation to try. If the permutation has already been tried, nothing will happen.
    The strategy will keep track of all the tries and the best permutation found so far.

    IMPORTANT: Each run of the strategy actually tests rules.num_batters permutations. Because of the symmetry of the problem, for
    [a, b, c] we also test [b, c, a] and [c, a, b]. Keep this in mind when creating a strategy, since it means you
    have more data you get from each run.
    """

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
    """
    Tries to greedily fills the batting order, from first to last.
    Because of the symmetry of the problem, a better permutation is sometimes found
    that doesn't fit the one-by-one approach. In this case it resets with the new permutation
    as a start. If the reset never yields a better permutation, the algorithm is done.
    """

    def __init__(self):
        super().__init__()
        self.perm = tuple(range(rules.num_batters))
        self.considering_batter = 0
        self.restart = False
        self.batter_choices = []
        self.tested = set()

    def get_next_permutation(self) -> Permutation:
        # If a better permutation has been found outside the one-by-one approach, reset the strategy
        if self.get_best() is not None and self.get_best()[0] != self.perm and self.get_best()[0] not in self.tested:
            self.perm = self.get_best()[0]
            self.considering_batter = 0
            self.restart = False
            self.batter_choices = []
            self.tested = set()

        # When all choices for the batter have been tested, select the best one
        if len(self.batter_choices) == rules.num_batters - self.considering_batter:
            self.perm = max(self.batter_choices, key=lambda x: self.tries[x])
            self.considering_batter += 1
            self.batter_choices = []

            # If the algorithm has finished, we try to restart (if we haven't already)
            if self.considering_batter >= rules.num_batters - 1 and not self.restart:
                self.considering_batter = 0
                self.restart = True

        perm = swap(self.perm, self.considering_batter, self.considering_batter + len(self.batter_choices))
        self.batter_choices.append(perm)
        self.tested.add(perm)

        return perm

    def is_finished(self):
        return self.considering_batter >= rules.num_batters - 1


class GreedyHillClimbing(BattingOrderStrategy):
    """
    A greedy algorithm that swaps two players as long as the score improves. Once it has tried
    all possible swaps, it increases the number of swaps to try per iteration.
    """

    def __init__(self):
        super().__init__()
        self.possible_swaps = list(itertools.combinations(range(rules.num_batters), 2))
        random.shuffle(self.possible_swaps)
        self.swap_index = 0
        self.num_swaps = 1
        self.current_perm = tuple(range(rules.num_batters))

    def get_next_permutation(self) -> Permutation:
        if self.get_best() is not None and self.get_best()[0] != self.current_perm:
            self.current_perm = self.get_best()[0]
            self.swap_index = 0

        perm = self.current_perm
        swap_index = self.swap_index
        for _ in range(self.num_swaps):
            idx = swap_index % len(self.possible_swaps)
            perm = swap(perm, *self.possible_swaps[idx])
            swap_index //= len(self.possible_swaps)
        self.swap_index += 1

        if self.swap_index >= len(self.possible_swaps) ** self.num_swaps:
            self.num_swaps += 1
            self.swap_index = 0

        return perm

    def is_finished(self) -> bool:
        # This shouldn't ever happen, since I rewrote the strategy to keep increasing num_swaps
        return self.swap_index >= len(self.possible_swaps) ** self.num_swaps and self.get_best() != self.current_perm


class StochasticHillClimbing(BattingOrderStrategy):
    """
    A stochastic variation of the hill climbing algorithm, which considers every swap with a probability
    based on the score's percentile.
    """

    exp = 4  # A higher value will make the algorithm more greedy

    def __init__(self):
        super().__init__()
        self.possible_swaps = list(itertools.combinations(range(rules.num_batters), 2))
        random.shuffle(self.possible_swaps)
        self.swap_index = 0
        self.current_perm = tuple(range(rules.num_batters))

        self.considered_permutations = set()  # Permutations that have been considered for a swap
        self.all_values = []

    def get_next_permutation(self) -> Permutation:
        # We look at new permutations and consider the worst ones first
        not_considered = set(self.get_tries().keys()) - self.considered_permutations
        not_considered = sorted(not_considered, key=lambda x: self.get_tries()[x])
        for new_perm in not_considered:
            percentile = 1 if len(self.all_values) == 0 else self.calculate_percentile(self.get_tries()[new_perm])
            if random.random() < percentile ** self.exp:
                self.current_perm = new_perm
                self.swap_index = 0

            self.considered_permutations.add(new_perm)
            self.all_values.append(self.get_tries()[new_perm])

        # Then we suggest a new swap
        perm = swap(self.current_perm, *self.possible_swaps[self.swap_index])
        self.swap_index += 1

        return perm

    def calculate_percentile(self, value):
        all_values_array = np.array(self.all_values)
        percentile = np.sum(all_values_array <= value) / len(all_values_array)
        return percentile

    def is_finished(self) -> bool:
        return self.swap_index >= len(self.possible_swaps)


class ShortTermHillClimbing(BattingOrderStrategy):
    """
    A variant of the hill climbing algorithm with short-term memory loss
    It's greedy for the last n permutations that have been tried
    """

    # Basically, a permutation gets 8 tries before it's forgotten
    memory = 8 * rules.num_batters

    def __init__(self):
        super().__init__()
        self.possible_swaps = list(itertools.combinations(range(rules.num_batters), 2))
        random.shuffle(self.possible_swaps)
        self.swap_index = 0
        self.current_perm = tuple(range(rules.num_batters))

    def get_next_permutation(self) -> Permutation:
        # Check the best permutation in our memory
        last_n = list(self.get_tries().keys())[-self.memory:]
        best_perm = None if not last_n else max(last_n, key=lambda x: self.get_tries()[x])
        if self.get_best() is not None and best_perm != self.current_perm:
            self.current_perm = best_perm
            self.swap_index = 0

        # Explore a new swap
        perm = swap(self.current_perm, *self.possible_swaps[self.swap_index])
        self.swap_index += 1

        return perm

    def is_finished(self) -> bool:
        return self.swap_index >= len(self.possible_swaps) and self.get_best() != self.current_perm


class GeneticAlgorithm(BattingOrderStrategy):
    """We attempt to use the edge recombination operator to create new permutations"""

    def get_next_permutation(self) -> Permutation:
        best = sorted(self.get_tries().keys(), key=lambda x: self.get_tries()[x], reverse=True)[:2]
        if len(best) < 2:
            return tuple(range(rules.num_batters))

        perm1, perm2 = random.sample(best, 2)
        child = self.edge_recombination_operator(perm1, perm2)

        # We need to add a bit of randomness to the child
        child = swap(child, random.randint(0, rules.num_batters - 1), random.randint(0, rules.num_batters - 1))
        return child

    @staticmethod
    def edge_recombination_operator(parent1, parent2):
        # Step 1: Create the adjacency matrix
        adjacency = defaultdict(set)
        for p in (parent1, parent2):
            for i, gene in enumerate(p):
                adjacency[gene].add(p[(i - 1) % len(p)])
                adjacency[gene].add(p[(i + 1) % len(p)])

        # Step 2: Create the offspring
        offspring = []
        current = random.choice([parent1[0], parent2[0]])  # Start with a random gene from parent1

        while len(offspring) < len(parent1):
            offspring.append(current)

            # Remove current gene from all adjacency lists
            for adj_list in adjacency.values():
                adj_list.discard(current)

            if adjacency[current]:
                # Find the neighbors with the fewest remaining neighbors
                min_neighbors = min(len(adjacency[x]) for x in adjacency[current])
                candidates = [x for x in adjacency[current] if len(adjacency[x]) == min_neighbors]

                # Choose a random neighbor from the candidates
                current = random.choice(candidates)
            else:
                # If no neighbors, choose a random unvisited gene
                unvisited = set(parent1) - set(offspring)
                current = random.choice(list(unvisited)) if unvisited else None

        return tuple(offspring)

    def is_finished(self) -> bool:
        return False


def test_strategy(strategy: BattingOrderStrategy, match: tuple, k: int, label='strat', print_output=False):
    """Run a strategy for k steps and then save it"""

    policy_solver = PolicySolver(None, *match, rules=rules)
    policy_solver.initialize_distributions(save_distributions=True, load_distributions=True, load_transition=True, path=f'distributions/{label}/')

    for step in range(k):
        start = time.time()
        strategy.step(policy_solver)
        end = time.time()

        if step % 10 == 9 and print_output:
            print(f'{strategy.__class__.__name__}: {step + 1}/{k}, time: {end - start:.2f}s, best: {strategy.get_best()}')

    save_blosc2(strategy, f'{strategy.__class__.__name__.lower()}/{label}.blosc2')
    graph_strategy(strategy, label)


def test_strategies():
    """A routine to test strategies against each other"""

    strategies = [OneByOne, GreedyHillClimbing, GeneticAlgorithm]
    k = 60
    num_matches = 20
    start = 20
    matches = load_blosc2('matches.blosc2')

    bd = BaseballData()
    for i in tqdm(range(start, start + num_matches)):
        match = matches[i]
        PolicySolver(bd, *match, rules=rules).initialize_distributions(save_distributions=True)

        threads = []
        for strategy in strategies:
            strat = strategy()
            thread = multiprocessing.Process(target=test_strategy, args=(strat, match, k, i))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        print('Winning strategy:', max(strategies, key=lambda x: load_blosc2(f'{x.__name__.lower()}/{i}.blosc2').get_best()[1]))


def test_strategy_against_rosters(strategy: type[BattingOrderStrategy], matches: list[tuple], k: int):
    """A routine to test a strategy against specific rosters"""

    for team, match in matches:
        test_strategy(strategy(), match, k, team)
        print(f'{team} done')


def test_against_rosters(load: bool = False):
    """A routine to test a strategies performance against specific rosters"""

    # We just test each lineup against a pitcher from their own team :)
    matches = {team: (pitchers[team], rosters[team]) for team in rosters.keys()}
    strategy = OneByOne  # Performed best in tests

    if not load:
        bd = BaseballData()
        for team, (pitcher, batters) in matches.items():
            PolicySolver(bd, pitcher, batters, rules=rules).initialize_distributions(save_distributions=True, path=f'distributions/{team}/')

    games_per_process = 5
    threads = []
    for i in range(0, len(matches), games_per_process):
        thread = multiprocessing.Process(target=test_strategy_against_rosters, args=(strategy, list(matches.items())[i:i + games_per_process], 60))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()


def test_cardinals():
    """Test the cardinals lineup against an average pitcher"""

    bd = BaseballData()

    average_pitcher = sum(p.data for p in bd.pitchers.values() if p.obp_percentile) / len(bd.pitchers)
    pitcher = Pitcher()
    pitcher.data = average_pitcher
    bd.pitchers['average_pitcher'] = pitcher
    cardinals = rosters['cardinals']
    match = ('average_pitcher', cardinals)
    policy_solver = PolicySolver(bd, *match, rules=Rules)
    policy_solver.initialize_distributions()
    del bd

    # strategy = OneByOne()
    # k = 120
    # for _ in tqdm(range(k)):
    #     strategy.step(policy_solver)
    #     save_blosc2(strategy, f'{strategy.__class__.__name__.lower()}/cardinals_OPTIMAL_ALT.blosc2')
    strategy = load_blosc2(f'onebyone/cardinals_OPTIMAL_ALT.blosc2')
    k = 82
    for _ in tqdm(range(k)):
        strategy.step(policy_solver)
        save_blosc2(strategy, f'{strategy.__class__.__name__.lower()}/cardinals_OPTIMAL_ALT.blosc2')


def graph_strategy(strategy: BattingOrderStrategy, label: str = ""):
    import matplotlib.pyplot as plt

    steps = [x[1] for x in strategy.get_steps()]
    x = list(range(1, len(steps)))
    y = [max(steps[0:i]) for i in x]
    plt.plot(x, y)
    plt.title(f'{label}: {strategy.__class__.__name__} - Best value {strategy.get_best()[1]}')
    plt.show()


def generate_matches():
    """It's nice to have a fixed set of matches instead of generating them every time"""

    bd = BaseballData()
    num_matches = 1000
    matches = []

    for _ in range(num_matches):
        pitcher = random.choice(sorted(bd.pitchers))
        batters = random.sample(sorted(bd.batters), rules.num_batters)
        matches.append((pitcher, batters))

    save_blosc2(matches, 'matches.blosc2')


if __name__ == '__main__':
    seed(i=1)
    test_cardinals()
