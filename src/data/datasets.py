import random
from enum import IntEnum
from typing import Self, Callable

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.data_loading import BaseballData
from src.model.at_bat import PitchResult
from src.model.pitch import Pitch
from src.model.players import Pitcher


class SwingResult(IntEnum):
    STRIKE = 0
    FOUL = 1
    OUT = 2
    HIT = 3

    def get_one_hot_encoding(self):
        """Returns a one-hot encoding of the pitch type."""

        one_hot = torch.zeros(len(SwingResult))
        one_hot[self] = 1
        return one_hot

    @classmethod
    def from_pitch_result(cls, pitch_result: PitchResult):
        assert pitch_result.batter_swung()

        if pitch_result == PitchResult.SWINGING_STRIKE:
            return cls.STRIKE
        elif pitch_result == PitchResult.SWINGING_FOUL:
            return cls.FOUL
        elif pitch_result == PitchResult.HIT:
            return cls.HIT
        elif pitch_result == PitchResult.OUT:
            return cls.OUT


class PitchDataset(Dataset):
    """A versatile pitch dataset, wrapping the BaseballData source"""

    def __init__(self, data_source: BaseballData,
                 filter_on: Callable[[Pitch], bool] | None = None,
                 map_to: Callable[[Pitch], any] | None = None,
                 pitches: list[Pitch] | None = None):
        """
        :param data_source: The data source for the dataset
        :param filter_on: A function that filters the pitches
        :param map_to: A function that maps the pitches to a different type
        :param pitches: A list of pitches to use instead of the data source
        """

        pitch_objects = pitches if pitches is not None else data_source.pitches
        self.data = [map_to(pitch) if map_to is not None
                     else pitch for pitch in pitch_objects
                     if filter_on is None or filter_on(pitch)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> any:
        return self.data[idx]

    @classmethod
    def get_split_on_attribute(cls, data_source: BaseballData, val_split: float = 0.2,
                               attribute: Callable[[Pitch], any] | None = None,
                               filter_on: Callable[[Pitch], bool] | None = None,
                               map_to: Callable[[Pitch], any] | None = None) -> tuple[Self, Self]:
        """
        Splits the data into training and validation sets, with an option to split on a custom attribute.
        Note that the split is not guaranteed to be perfect across the attribute.
        """

        pitches = data_source.pitches
        if attribute is not None:
            pitches = sorted(pitches, key=attribute)

        split_idx = int(len(pitches) * (1 - val_split))
        train_pitches = pitches[:split_idx]
        val_pitches = pitches[split_idx:]

        return (cls(data_source, filter_on, map_to, train_pitches),
                cls(data_source, filter_on, map_to, val_pitches))


class PitchSwingDataset(Dataset):
    """
    Simple wrapper around the Pitch class, returning the pitcher, batter, pitch, and
    strike/ball count for each pitch that the batter swung to.
    """

    def __init__(self, data_source: BaseballData, condition: Callable[[Pitch], bool] | None = None,
                 pitches: list[Pitch] | None = None):
        if pitches is not None:
            self.pitches = pitches
        else:
            self.pitches = [pitch for pitch in data_source.pitches
                            if pitch.result.batter_swung()
                            and pitch.is_valid()
                            and (condition is None or condition(pitch))]

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
        train_ab_ids = set(ab_ids[:split_idx])
        val_ab_ids = set(ab_ids[split_idx:])

        return (cls(data_source, lambda pitch: pitch.at_bat.id in train_ab_ids),
                cls(data_source, lambda pitch: pitch.at_bat.id in val_ab_ids))
