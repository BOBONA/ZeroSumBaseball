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
    """
    Represents the changing state of an at-bat.
    """

    def __init__(self, outcome=AtBatOutcome.NONE, balls=0, strikes=0):
        self.outcome = outcome
        self.balls = balls
        self.strikes = strikes

    def transition_from_pitch_result(self, result: PitchResult) -> Self:
        if self.outcome != AtBatOutcome.NONE:
            return self

        if result == PitchResult.SWINGING_STRIKE or result == PitchResult.CALLED_STRIKE:
            self.strikes += 1
        elif result == PitchResult.SWINGING_FOUL and self.strikes < 2:
            self.strikes += 1
        elif result == PitchResult.CALLED_BALL:
            self.balls += 1
        elif result == PitchResult.HIT:
            self.outcome = AtBatOutcome.BASE
        elif result == PitchResult.OUT:
            self.outcome = AtBatOutcome.OUT

        if self.balls == 4:
            self.outcome = AtBatOutcome.BASE
        if self.strikes == 3:
            self.outcome = AtBatOutcome.OUT

        return self


class AtBat:
    """Represents a full at-bat event"""

    def __init__(self, game: Game, pitcher: Pitcher, batter: Batter, outcome_state: AtBatState, ab_id: int | None = None):
        self.game = game
        self.pitcher = pitcher
        self.batter = batter
        self.state = outcome_state
        self.id = ab_id
