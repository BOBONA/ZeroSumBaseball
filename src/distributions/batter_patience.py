import torch

from src.data.data_loading import BaseballData
from src.data.datasets import PitchDataset


def get_batter_patience_set(data: BaseballData) -> (PitchDataset, PitchDataset):
    return PitchDataset.get_split_on_attribute(
        data, 0.2,
        attribute=lambda p: p.at_bat.batter,
        filter_on=lambda p: p.location.is_borderline,
        map_to=lambda p: (p.at_bat.batter.data, p.get_one_hot_encoding(),
                          torch.tensor(p.at_bat_state.strikes, dtype=torch.float32),
                          torch.tensor(p.at_bat_state.balls, dtype=torch.float32)),
        seed=0
    )
