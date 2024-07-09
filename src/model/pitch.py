import torch

from src.model.at_bat import AtBatState, PitchResult
from src.model.pitch_type import PitchType
from src.model.zones import Zone, Zones, default


class Pitch:
    """
    A Pitch is represented as a pair of a pitch type and a location in the strike zone.
    """

    __slots__ = ['at_bat_state', 'batter_id', 'pitcher_id', 'type', 'zone_idx', 'result', 'speed', 'plate_x', 'plate_z',
                 'game_id', 'at_bat_num', 'pitch_num']

    def __init__(self, at_bat_state: AtBatState, batter_id: int, pitcher_id: int,
                 pitch_type: PitchType | None, location: int | None, pitch_result: PitchResult | None,
                 speed: float | None, plate_x: float | None, plate_z: float | None,
                 game_id: int = -1, at_bat_num: int = -1, pitch_num: int = -1):
        """
        :param at_bat_state: The current state of the at-bat BEFORE the pitch is thrown
        :param batter_id: The ID of the batter
        :param pitcher_id: The ID of the pitcher
        :param pitch_type: The type of pitch thrown
        :param location: The zone index of the pitch, see src.model.zones.Zones
        :param pitch_result: The result of the pitch
        :param speed: The speed of the pitch in MPH
        :param plate_x: The x-coordinate of the pitch in feet
        :param plate_z: The z-coordinate of the pitch in feet
        :param game_id: The ID of the game
        :param at_bat_num: The number of the at-bat
        :param pitch_num: The number of the pitch in the at-bat
        """

        self.at_bat_state = at_bat_state
        self.batter_id = batter_id
        self.pitcher_id = pitcher_id
        self.type = pitch_type
        self.zone_idx = location
        self.result = pitch_result
        self.speed = speed
        self.plate_x = plate_x
        self.plate_z = plate_z
        self.game_id = game_id
        self.at_bat_num = at_bat_num
        self.pitch_num = pitch_num

    def is_valid(self):
        """Returns whether the pitch is valid and can be used for training."""

        return self.type is not None and self.result is not None and self.zone_idx is not None

    def get_one_hot_encoding(self):
        """Returns a one-hot encoding of the pitch."""

        return Pitch.get_encoding(self.type, default.COMBINED_ZONES[self.zone_idx])

    def get_zone(self) -> Zone:
        """Get this pitch's zone."""

        return default.COMBINED_ZONES[self.zone_idx]

    @classmethod
    def get_encoding(cls, pitch_type: PitchType, location: Zone):
        one_hot = torch.zeros(len(PitchType), Zones.DIMENSION, Zones.DIMENSION)
        for x, y in location.coords:
            one_hot[pitch_type.value, x, y] = 1
        return one_hot
