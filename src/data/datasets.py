import random
from enum import IntEnum
from typing import Self

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

    def __init__(self, data_source: BaseballData, ab_ids: list[int] | None = None):
        ab_ids_set = set(ab_ids)
        self.pitches = [pitch for pitch in data_source.pitches
                        if pitch.result.batter_swung()
                        and pitch.is_valid()
                        and (ab_ids is None or pitch.at_bat.id in ab_ids_set)]

    def __len__(self):
        return len(self.pitches)

    def __getitem__(self, idx) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]:
        pitch = self.pitches[idx]
        return ((pitch.at_bat.pitcher.data, pitch.at_bat.batter.data,
                 pitch.get_one_hot_encoding(),
                 torch.tensor(pitch.at_bat_state.strikes, dtype=torch.float32),
                 torch.tensor(pitch.at_bat_state.balls, dtype=torch.float32)),
                SwingResult.from_pitch_result(pitch.result).get_one_hot_encoding())

    @classmethod
    def get_random_split(cls, data_source: BaseballData, val_split: float = 0.2, seed: int | None = None) -> tuple[Self, Self]:
        """Splits the data into training and validation sets, keeping the at-bats together."""

        ab_ids = list(data_source.at_bats.keys())
        random.seed(seed)
        random.shuffle(ab_ids)

        split_idx = int(len(ab_ids) * (1 - val_split))
        train_ab_ids = ab_ids[:split_idx]
        val_ab_ids = ab_ids[split_idx:]

        return cls(data_source, train_ab_ids), cls(data_source, val_ab_ids)
