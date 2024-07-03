from enum import IntEnum
from typing import Self

from src.model.game import Game
from src.model.players import Pitcher, Batter


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
        """Returns whether the batter swung at the pitch."""

        return (self == PitchResult.SWINGING_STRIKE or self == PitchResult.SWINGING_FOUL or self == PitchResult.HIT_OUT or
                self.batter_hit())

    def batter_hit(self):
        """Returns whether the batter hit the pitch."""

        return (self == PitchResult.HIT_SINGLE or self == PitchResult.HIT_DOUBLE or self == PitchResult.HIT_TRIPLE or
                self == PitchResult.HIT_HOME_RUN)


class AtBatState:
    """
    Represents the changing state of an at-bat. This is also used to represent the outcome of an at-bat,
    which is made up of the number of outs and the outcome event (if applicable).

    Note that outcome_event is a PitchResult, but that this is only used to represent whether the at-bat ended
    on base, and if so, which base the batter reached. This is not a proper usage of the PitchResult enum.

    Also, note that num_runs is not currently used in computing the optimal policy, although limiting it
    does have a small effect, it also increases the number of states significantly.
    """

    max_runs = 9

    def __init__(self, balls=0, strikes=0, runs=0, outs=0, first=False, second=False, third=False, outcome_event: PitchResult | None = None):
        self.balls = balls
        self.strikes = strikes

        # The state of the first, second, and third bases
        self.num_runs = runs
        self.num_outs = outs
        self.first = first
        self.second = second
        self.third = third

        self.outcome_event = outcome_event  # The outcome pitch result

        self.precomputed_hash = None

    def transition_from_pitch_result(self, result: PitchResult) -> Self:
        next_state = AtBatState(self.balls, self.strikes, self.num_runs, self.num_outs, self.first, self.second, self.third)

        if next_state.num_outs >= 3 or next_state.num_runs >= AtBatState.max_runs:
            return next_state

        if (result == PitchResult.SWINGING_STRIKE or result == PitchResult.CALLED_STRIKE or
                (result == PitchResult.SWINGING_FOUL and next_state.strikes < 3)):
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
            next_state.num_outs += 1

        if next_state.balls == 4:
            next_state.move_batter(1)  # Walk
        if next_state.strikes == 3:
            next_state.num_outs += 1
            next_state.balls = next_state.strikes = 0

        if next_state.num_runs > AtBatState.max_runs:
            next_state.num_runs = AtBatState.max_runs

        return next_state

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

    def value(self) -> int:
        """
        Returns the "value" of the current state. Of course, value is more complicated than a single integer,
        more so this is a target for the value iteration algorithm
        """

        return self.num_runs

    def __repr__(self):
        return (f"AtBatState({self.balls}/{self.strikes}, {self.num_runs}, {self.num_outs}, "
                f"{'x' if self.first else '-'}{'x' if self.second else '-'}{'x' if self.third else '-'})")

    def __hash__(self):
        if self.precomputed_hash is None:
            self.precomputed_hash = hash((self.balls, self.strikes, self.num_outs, self.first, self.second, self.third))
        return self.precomputed_hash

    def __eq__(self, other):
        return hash(self) == hash(other)


class AtBat:
    """Represents a full at-bat event"""

    def __init__(self, game: Game | None, pitcher: Pitcher | None, batter: Batter, outcome_state: AtBatState, ab_id: int | None = None):
        self.game = game
        self.pitcher = pitcher
        self.batter = batter
        self.state = outcome_state
        self.id = ab_id
        self.pitches = []
