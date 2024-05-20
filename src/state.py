from enum import Enum
from typing import Self


class AtBatAction(Enum):
    SWING = "swing"
    TAKE = "take"


class AtBatSwingResult(Enum):
    STRIKE = "strike"
    FOUL = "foul"
    HIT = "hit"
    OUT = "out"


class AtBatOutcome(Enum):
    BASE = "base"
    OUT = "out"
    NONE = "none"


class AtBatState:
    def __init__(self, balls=0, strikes=0, outcome=AtBatOutcome.NONE):
        self.balls = balls
        self.strikes = strikes
        self.outcome = outcome

    def transition_from_swing(self, result: AtBatSwingResult) -> Self:
        if self.outcome != AtBatOutcome.NONE:
            return self

        if result == AtBatSwingResult.STRIKE:
            self.strikes += 1
        elif result == AtBatSwingResult.FOUL and self.strikes < 2:
            self.strikes += 1
        elif result == AtBatSwingResult.HIT:
            self.outcome = AtBatOutcome.BASE
        else:
            self.outcome = AtBatOutcome.OUT

        if self.strikes == 3:
            self.outcome = AtBatOutcome.OUT

        return self

    def __str__(self) -> str:
        return f"{self.balls}-{self.strikes}" if self.outcome == AtBatOutcome.NONE else self.outcome.value
