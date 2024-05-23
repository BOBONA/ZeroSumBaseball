import torch

from src.model.pitch_type import PitchType
from src.model.zones import ZONES_DIMENSION


class Batter:
    """
    A Batter is represented as a 3D tensor of shape (2 * len(PitchType), ZONES_DIMENSION, ZONES_DIMENSION).
    Each pitch type/location combination has a corresponding relative swinging frequency and batting average.

    Attributes:
        data (torch.Tensor): The tensor representing the batter's relative swinging frequency and batting average.
    """

    def __init__(self, data: torch.Tensor = None):
        if data:
            assert (data.size(0) == 2 * len(PitchType) and
                    data.size(1) == ZONES_DIMENSION and
                    data.size(2) == ZONES_DIMENSION)

            self.data = data
        else:
            self.data = torch.zeros(2 * len(PitchType), ZONES_DIMENSION, ZONES_DIMENSION)
            self.data[:len(PitchType), :, :] = 1 / len(PitchType)

    def set_swinging_frequency_data(self, data: torch.Tensor):
        """
        Sets the swinging data for the batter. You can provide relative frequencies or provide total pitches for each type
        since the method normalizes the values.

        :param data: The swinging frequency data, a 3D tensor of shape (len(PitchType), ZONES_DIMENSION, ZONES_DIMENSION) to correspond to each pitch type.
        """

        # Normalize the swinging frequencies
        self.data[len(PitchType):, :, :] = data / data.sum()

    def set_batting_average_data(self, data: torch.Tensor):
        """Sets the batting average data for the batter."""

        self.data[:len(PitchType), :, :] = data


class Pitcher:
    """
    A Pitcher is represented as a 3D tensor of shape (2 * len(PitchType), ZONES_DIMENSION, ZONES_DIMENSION).
    Each pitch type/location combination has a corresponding relative throwing frequency and average velocity.
    Note that for now this tensor is structurally identical to a batter's.

    Attributes:
        data (torch.Tensor): The tensor representing the pitcher's relative throwing frequency and average velocity.
        estimated_control (torch.Tensor): A bivariate normal distribution representing the pitcher's control. Note that
            BaseballData learns this from the data and it is used to train the PitcherControl network.
    """

    def __init__(self, data: torch.Tensor = None):
        if data:
            assert (data.size(0) == 2 * len(PitchType) and
                    data.size(1) == ZONES_DIMENSION and
                    data.size(2) == ZONES_DIMENSION)

            # Normalize the throwing frequencies
            data[:len(PitchType), :, :] /= data[:len(PitchType), :, :].sum(dim=0, keepdim=True)
            self.data = data
        else:
            self.data = torch.zeros(2 * len(PitchType), ZONES_DIMENSION, ZONES_DIMENSION)
            self.data[:len(PitchType), :, :] = 1 / len(PitchType)

        self.estimated_control = torch.zeros(5)

    def set_throwing_frequency_data(self, data: torch.Tensor):
        """
        Sets the throwing data for the pitcher. You can provide relative frequencies or provide total pitches for each type
        since the method normalizes the values.

        :param data: The throwing frequency data, a 3D tensor of shape (len(PitchType), ZONES_DIMENSION, ZONES_DIMENSION) to correspond to each pitch type. The tensor is normalized to sum to 1 along the pitch type dimension.
        """

        # Normalize the throwing frequencies
        self.data[:len(PitchType), :, :] = data / data[:len(PitchType), :, :].sum()

    def set_average_velocity_data(self, data: torch.Tensor):
        """Sets the average velocity data for the pitcher."""

        self.data[len(PitchType):, :, :] = data
