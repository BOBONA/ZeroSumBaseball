from enum import IntEnum

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.data_loading import BaseballData
from src.model.at_bat import PitchResult


class SwingResult(IntEnum):
    STRIKE = 0
    FOUL = 1
    HIT = 2
    OUT = 3

    def get_one_hot_encoding(self):
        """Returns a one-hot encoding of the pitch type."""

        one_hot = torch.zeros(len(SwingResult))
        one_hot[self] = 1
        return one_hot

    @classmethod
    def from_pitch_result(cls, pitch_result: PitchResult):
        if pitch_result == PitchResult.SWINGING_STRIKE:
            return cls.STRIKE
        elif pitch_result == PitchResult.SWINGING_FOUL:
            return cls.FOUL
        elif pitch_result == PitchResult.HIT:
            return cls.HIT
        elif pitch_result == PitchResult.OUT:
            return cls.OUT
        else:
            raise ValueError(f"Invalid pitch result: {pitch_result}")


class PitchSwingDataset(Dataset):
    """
    Simple wrapper around the Pitch class, returning the pitcher, batter, pitch, and
    strike/ball count for each pitch that the batter swung to.
    """

    def __init__(self, data_source: BaseballData):
        self.pitches = [pitch for pitch in data_source.pitches if pitch.result.batter_swung()]

    def __len__(self):
        return len(self.pitches)

    def __getitem__(self, idx) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]:
        pitch = self.pitches[idx]
        return ((pitch.at_bat.pitcher.data, pitch.at_bat.batter.data,
                pitch.get_one_hot_encoding(),
                torch.tensor(pitch.at_bat_state.strikes, dtype=torch.float32),
                torch.tensor(pitch.at_bat_state.balls, dtype=torch.float32)),
                SwingResult.from_pitch_result(pitch.result).get_one_hot_encoding())
