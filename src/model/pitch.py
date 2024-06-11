import torch

from src.model.at_bat import AtBat, AtBatState, PitchResult
from src.model.pitch_type import PitchType
from src.model.zones import ZONES_DIMENSION, Zone


class Pitch:
    """
    A Pitch is represented as a pair of a pitch type and a location in the strike zone.
    """

    def __init__(self, at_bat_state: AtBatState, at_bat: AtBat, pitch_number: int, location: Zone | None,
                 pitch_type: PitchType | None, speed: float, pitch_result: PitchResult | None):
        """
        :param at_bat_state: The current state of the at-bat BEFORE the pitch is thrown
        :param at_bat: The full at-bat event that the pitch is part of
        :param pitch_number: The number of the pitch in the at-bat
        :param location: The location of the pitch
        :param pitch_type: The type of the pitch
        :param speed: The speed of the pitch
        :param pitch_result: The result of the pitch
        """

        self.at_bat_state = at_bat_state
        self.at_bat = at_bat
        self.pitch_number = pitch_number
        self.location = location
        self.type = pitch_type
        self.speed = speed
        self.result = pitch_result

    def is_valid(self):
        """Returns whether the pitch is valid and can be used for training."""

        return self.type is not None and self.result is not None and self.location is not None

    def get_one_hot_encoding(self):
        """Returns a one-hot encoding of the pitch."""

        return Pitch.get_encoding(self.type, self.location)

    @classmethod
    def get_encoding(cls, pitch_type: PitchType, location: Zone):
        one_hot = torch.zeros(len(PitchType), ZONES_DIMENSION, ZONES_DIMENSION)
        for x, y in location.coords:
            one_hot[pitch_type.value, x, y] = 1
        return one_hot
