from enum import IntEnum
from typing import Self


class PitchResult(IntEnum):
    """An enum representing the possible results of a pitch, used to transition the state."""

    CALLED_BALL = 0
    CALLED_STRIKE = 1

    SWINGING_STRIKE = 2
    SWINGING_FOUL = 3

    HIT_SINGLE = 4
    HIT_DOUBLE = 5
    HIT_TRIPLE = 6
    HIT_HOME_RUN = 7
    HIT_OUT = 8

    def batter_swung(self):
        """Returns whether the batter swung at the pitch (or if he was hit by the pitch)."""

        return (self == PitchResult.SWINGING_STRIKE or self == PitchResult.SWINGING_FOUL or self == PitchResult.HIT_OUT or
                self.batter_hit())

    def batter_hit(self):
        """Returns whether the batter hit the pitch (or was hit by the pitch)."""

        return (self == PitchResult.HIT_SINGLE or self == PitchResult.HIT_DOUBLE or self == PitchResult.HIT_TRIPLE or
                self == PitchResult.HIT_HOME_RUN)


class GameState:
    """Represents the changing state of a game."""

    __slots__ = ['inning', 'balls', 'strikes', 'num_runs', 'outs', 'first', 'second', 'third', 'batter']

    num_innings = 9  # Although in the real world, games can go into extra innings
    num_balls = 4
    num_strikes = 3
    num_outs = 3
    num_batters = 9
    max_runs = 9

    def __init__(self, inning=0, balls=0, strikes=0, runs=0, outs=0, first=False, second=False, third=False, batter=0):
        self.inning = inning
        self.balls = balls
        self.strikes = strikes
        self.num_runs = runs
        self.outs = outs
        self.first = first      # True if runner on first
        self.second = second    # True if runner on second
        self.third = third      # True if runner on third
        self.batter = batter

    def transition_from_pitch_result(self, result: PitchResult) -> tuple[Self, int]:
        next_state = GameState(inning=self.inning, balls=self.balls, strikes=self.strikes, runs=self.num_runs,
                               outs=self.outs, first=self.first, second=self.second, third=self.third, batter=self.batter)

        if next_state.inning >= self.num_innings or next_state.num_runs >= GameState.max_runs:
            return next_state, 0

        if (result == PitchResult.SWINGING_STRIKE or result == PitchResult.CALLED_STRIKE or
                (result == PitchResult.SWINGING_FOUL and next_state.strikes < GameState.num_strikes - 1)):
            next_state.strikes += 1
        elif result == PitchResult.CALLED_BALL:
            next_state.balls += 1
        elif result == PitchResult.HIT_SINGLE:
            next_state.move_batter(1)
        elif result == PitchResult.HIT_DOUBLE:
            next_state.move_batter(2)
        elif result == PitchResult.HIT_TRIPLE:
            next_state.move_batter(3)
        elif result == PitchResult.HIT_HOME_RUN:
            next_state.move_batter(4)
        elif result == PitchResult.HIT_OUT:
            next_state.outs += 1
            next_state.balls = next_state.strikes = 0
            next_state.batter = (next_state.batter + 1) % self.num_batters

        if next_state.balls == self.num_balls:  # Walk
            next_state.move_batter(1)
        if next_state.strikes == self.num_strikes:
            next_state.outs += 1
            next_state.balls = next_state.strikes = 0
            next_state.batter = (next_state.batter + 1) % self.num_batters

        if next_state.num_runs > GameState.max_runs:
            next_state.num_runs = GameState.max_runs

        if next_state.outs >= self.num_outs:
            next_state.inning += 1
            next_state.outs = 0
            next_state.first = next_state.second = next_state.third = False

        return next_state, next_state.num_runs - self.num_runs

    def move_batter(self, num_bases: int):
        """A helper method, advances runners and resets count"""

        if num_bases >= 4:
            self.num_runs += 1 + int(self.first) + int(self.second) + int(self.third)
            self.first = self.second = self.third = False
        elif num_bases == 3:
            self.num_runs += int(self.first) + int(self.second) + int(self.third)
            self.first = self.second = False
            self.third = True
        elif num_bases == 2:
            self.num_runs += int(self.second) + int(self.third)
            self.third = self.first
            self.first = False
            self.second = True
        elif num_bases == 1:
            self.num_runs += int(self.third)
            self.third = self.second
            self.second = self.first
            self.first = True

        self.balls = self.strikes = 0
        self.batter = (self.batter + 1) % self.num_batters

    def value(self) -> int:
        """
        Returns the "value" of the current state. Of course, value is more complicated than a single integer,
        this is a target for the value iteration algorithm
        """

        return self.num_runs

    def __repr__(self):
        return (f"GameState(i{self.inning} b{self.batter}: {self.balls}/{self.strikes}, {self.num_runs}, {self.outs}, "
                f"{'x' if self.first else '-'}{'x' if self.second else '-'}{'x' if self.third else '-'})")

    def __hash__(self):
        return hash((self.inning, self.balls, self.strikes, self.outs, self.first, self.second, self.third, self.batter))

    def __eq__(self, other):
        return hash(self) == hash(other)
