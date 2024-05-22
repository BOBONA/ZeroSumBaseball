from enum import IntEnum


class PitchType(IntEnum):
    """
    This enum follows a previous implementation, but could be reconsidered.
    It's in its own file to avoid circular imports.
    """

    FOUR_SEAM = 0
    TWO_SEAM = 1
    SLIDER = 2
    CHANGEUP = 3
    CURVE = 4
    CUTTER = 5
