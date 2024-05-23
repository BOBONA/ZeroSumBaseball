from enum import IntEnum

import torch


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

    def get_one_hot_encoding(self):
        """Returns a one-hot encoding of the pitch type."""

        one_hot = torch.zeros(len(PitchType))
        one_hot[self] = 1
        return one_hot
