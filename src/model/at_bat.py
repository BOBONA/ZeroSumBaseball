from enum import IntEnum
from typing import Self

import torch

from src.model.game import Game
from src.model.players import Pitcher, Batter


class PitchResult(IntEnum):
    CALLED_BALL = 0
    CALLED_STRIKE = 1
    SWINGING_STRIKE = 2
    SWINGING_FOUL = 3
    HIT = 4
    OUT = 5

    def batter_swung(self):
        """Returns whether the batter swung at the pitch."""

        return self in [PitchResult.SWINGING_STRIKE, PitchResult.SWINGING_FOUL, PitchResult.HIT, PitchResult.OUT]


class AtBatOutcome(IntEnum):
    BASE = 0
    OUT = 1
    NONE = 2


class AtBatState:
    """Represents the changing state of an at-bat."""

    def __init__(self, outcome=AtBatOutcome.NONE, balls=0, strikes=0):
        self.outcome = outcome
        self.balls = balls
        self.strikes = strikes

    def transition_from_pitch_result(self, result: PitchResult) -> Self:
        next_state = AtBatState(self.outcome, self.balls, self.strikes)

        if next_state.outcome != AtBatOutcome.NONE:
            return next_state

        if result == PitchResult.SWINGING_STRIKE or result == PitchResult.CALLED_STRIKE:
            next_state.strikes += 1
        elif result == PitchResult.SWINGING_FOUL and next_state.strikes < 2:
            next_state.strikes += 1
        elif result == PitchResult.CALLED_BALL:
            next_state.balls += 1
        elif result == PitchResult.HIT:
            next_state.outcome = AtBatOutcome.BASE
        elif result == PitchResult.OUT:
            next_state.outcome = AtBatOutcome.OUT

        if next_state.balls == 4:
            next_state.outcome = AtBatOutcome.BASE
        if next_state.strikes == 3:
            next_state.outcome = AtBatOutcome.OUT

        return next_state

    def value(self) -> int:
        """Returns the "utility" of the current state."""

        return int(self.outcome == AtBatOutcome.BASE)

    def __repr__(self):
        return f"AtBatState({self.outcome.name}, {self.balls}, {self.strikes})"

    def __hash__(self):
        return hash((self.outcome, self.balls, self.strikes))

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
