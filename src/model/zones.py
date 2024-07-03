from copy import copy

import torch


class Zone:
    """A Zone is represented with physical coordinates and "virtual" coordinates."""

    def __init__(self, coords: list[tuple[int, int]], left: float, right: float, bottom: float, top: float, is_strike: bool = True, is_borderline: bool = False):
        self.coords = coords
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.is_strike = is_strike
        self.is_borderline = is_borderline

    def center(self) -> tuple[float, float]:
        """Returns the center of the zone."""

        return (self.left + self.right) / 2, (self.bottom + self.top) / 2

    def __repr__(self):
        return f"Zone({self.coords}){' borderline' if self.is_borderline else ''}{' strike' if self.is_strike else ''}"

    def __hash__(self):
        return hash((*self.coords, self.is_strike, self.is_borderline))

    def __eq__(self, other):
        return hash(self) == hash(other)


class Zones:
    """
    We use the following reference for default measurements https://tangotiger.net/strikezone/zone%20chart.png

    Using those measurements, we divide the physical strike zone into 5x5 zones. In our model, (0, 0)
    is the bottom left corner (although how you think of it shouldn't actually matter).

    * ----- *
    | o o o |
    | o o o |
    | o o o |
    * ----- *

    Borderline zones are also included. These are just outside the strike zone and are used to help determine
    batter patience.
    """

    # The width/height of the virtual strike zone
    DIMENSION = 5

    def __init__(self, width=20, sz_top=42, sz_bottom=18):
        """
        :param width: The width of the strike zone
        :param sz_top: The top of the strike zone
        :param sz_bottom: The bottom of the strike zone
        """
        self.ZONES = []

        # Strike zones
        strike_zone_dim = 3
        strike_left = -width / 2
        strike_height = sz_top - sz_bottom
        x_step = width / strike_zone_dim
        y_step = strike_height / strike_zone_dim
        self.ZONES.extend([
            Zone([(i + 1, j + 1)],
                 strike_left + i * x_step, strike_left + (i + 1) * x_step,
                 sz_bottom + j * y_step, sz_bottom + (j + 1) * y_step, is_strike=True)
            for i in range(strike_zone_dim) for j in range(strike_zone_dim)
        ])

        # Ball zones
        inf = float('inf')
        self.ZONES.extend([
            Zone([(0, 0)], -inf, strike_left, -inf, sz_bottom, is_strike=False),
            Zone([(0, 4)], -inf, strike_left, sz_top, inf, is_strike=False),
            Zone([(4, 0)], strike_left + width, inf, -inf, sz_bottom, is_strike=False),
            Zone([(4, 4)], strike_left + width, inf, sz_top, inf, is_strike=False),

            Zone([(0, 1), (0, 2), (0, 3)], -inf, strike_left, sz_bottom, sz_top, is_strike=False),
            Zone([(4, 1), (4, 2), (4, 3)], strike_left + width, inf, sz_bottom, sz_top, is_strike=False),
            Zone([(1, 0), (2, 0), (3, 0)], strike_left, strike_left + width, -inf, sz_bottom, is_strike=False),
            Zone([(1, 4), (2, 4), (3, 4)], strike_left, strike_left + width, sz_top, inf, is_strike=False)
        ])

        self.BORDERLINE_ZONES = [copy(zone) for zone in self.ZONES]
        for zone in self.BORDERLINE_ZONES:
            zone.is_borderline = True

        self.COMBINED_ZONES = self.ZONES + self.BORDERLINE_ZONES

        self.STRIKE_ZONE_WIDTH = width
        self.STRIKE_ZONE_HEIGHT = sz_top - sz_bottom
        self.STRIKE_ZONE_BOTTOM = sz_bottom
        self.STRIKE_ZONE_TOP = sz_top
        self.STRIKE_ZONE_LEFT = strike_left
        self.STRIKE_ZONE_RIGHT = strike_left + width
        self.BALL_SIZE = 3

    def get_zone(self, x_loc: float | None, y_loc: float | None) -> Zone | None:
        """Converts physical coordinates x and y (in inches) to virtual coordinates in the strike zone."""

        if x_loc is None or y_loc is None:
            return None

        for i, zone in enumerate(self.ZONES):
            if zone.left <= x_loc <= zone.right and zone.bottom <= y_loc <= zone.top:
                if self.STRIKE_ZONE_LEFT - self.BALL_SIZE <= x_loc <= self.STRIKE_ZONE_RIGHT + self.BALL_SIZE and \
                    self.STRIKE_ZONE_BOTTOM - self.BALL_SIZE <= y_loc <= self.STRIKE_ZONE_TOP + self.BALL_SIZE and \
                    not (self.STRIKE_ZONE_LEFT + self.BALL_SIZE <= x_loc <= self.STRIKE_ZONE_RIGHT - self.BALL_SIZE and
                         self.STRIKE_ZONE_BOTTOM + self.BALL_SIZE <= y_loc <= self.STRIKE_ZONE_TOP - self.BALL_SIZE):
                    return self.BORDERLINE_ZONES[i]
                else:
                    return zone

    def get_zones_batched(self, x_locs: torch.Tensor, y_locs: torch.Tensor) -> list[int]:
        """
        This batched version of get_zone is significantly faster and necessary for the random
        sampling method used for measuring intended vs actual pitch locations. It returns indices of ZONES.
        """

        borderline_mask = (self.STRIKE_ZONE_LEFT - self.BALL_SIZE <= x_locs) & (x_locs <= self.STRIKE_ZONE_RIGHT + self.BALL_SIZE) & \
                          (self.STRIKE_ZONE_BOTTOM - self.BALL_SIZE <= y_locs) & (y_locs <= self.STRIKE_ZONE_TOP + self.BALL_SIZE) & \
                          ~((self.STRIKE_ZONE_LEFT + self.BALL_SIZE <= x_locs) & (x_locs <= self.STRIKE_ZONE_RIGHT - self.BALL_SIZE) &
                            (self.STRIKE_ZONE_BOTTOM + self.BALL_SIZE <= y_locs) & (y_locs <= self.STRIKE_ZONE_TOP - self.BALL_SIZE))

        result_zones: list = [-1] * len(x_locs)
        for zone_i, zone in enumerate(self.ZONES):
            mask = (zone.left <= x_locs) & (x_locs <= zone.right) & (zone.bottom <= y_locs) & (y_locs <= zone.top)
            indices = torch.nonzero(mask, as_tuple=False).squeeze()
            if indices.numel() == 1:
                result_zones[indices.item()] = zone_i + len(self.ZONES) * borderline_mask[indices.item()]
            elif indices.numel() > 1:
                for idx in indices:
                    result_zones[idx.item()] = zone_i + len(self.ZONES) * borderline_mask[idx.item()]

        return result_zones


# For convenience, here is a default instance of Zones
default = Zones()
