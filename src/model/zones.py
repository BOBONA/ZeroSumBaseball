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
        return f"Zone({self.coords})"

    def __hash__(self):
        return hash((*self.coords, self.is_strike, self.is_borderline))

    def __eq__(self, other):
        return hash(self) == hash(other)


"""
The previous work appears to have some mistakes in their strike zone measurements. I'm fairly sure 
they use https://www.baseballprospectus.com/news/article/40891/prospectus-feature-the-universal-strike-zone/
as a reference, so I'll use that as well.

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

# Note that the borderline zones technically overlap with the ball zones
ZONES = []
BORDERLINE_ZONES = []
NON_BORDERLINE_ZONES = []

STRIKE_ZONE_BOTTOM = 18.29
STRIKE_ZONE_WIDTH = 19.94
STRIKE_ZONE_HEIGHT = 44.08


ZONES_DIMENSION = 5

Inf = float('inf')

# Some more measurements
STRIKE_ZONE_LEFT = -STRIKE_ZONE_WIDTH / 2
STRIKE_ZONE_RIGHT = STRIKE_ZONE_WIDTH / 2
STRIKE_ZONE_TOP = STRIKE_ZONE_BOTTOM + STRIKE_ZONE_HEIGHT
STRIKE_ZONE_X_STEP = STRIKE_ZONE_WIDTH / (ZONES_DIMENSION - 2)
STRIKE_ZONE_Y_STEP = STRIKE_ZONE_HEIGHT / (ZONES_DIMENSION - 2)

# Borderline zone calculations
borderline_fraction = 0.4
borderline_x = STRIKE_ZONE_X_STEP * borderline_fraction
borderline_y = STRIKE_ZONE_Y_STEP * borderline_fraction

BORDERLINE_ZONES.append(Zone([(0, 0)], STRIKE_ZONE_LEFT - borderline_x, STRIKE_ZONE_LEFT, STRIKE_ZONE_BOTTOM - borderline_y, STRIKE_ZONE_BOTTOM, False, True))
BORDERLINE_ZONES.append(Zone([(0, 4)], STRIKE_ZONE_LEFT - borderline_x, STRIKE_ZONE_LEFT, STRIKE_ZONE_TOP, STRIKE_ZONE_TOP + borderline_y, False, True))
BORDERLINE_ZONES.append(Zone([(4, 0)], STRIKE_ZONE_RIGHT, STRIKE_ZONE_RIGHT + borderline_x, STRIKE_ZONE_BOTTOM - borderline_y, STRIKE_ZONE_BOTTOM, False, True))
BORDERLINE_ZONES.append(Zone([(4, 4)], STRIKE_ZONE_RIGHT, STRIKE_ZONE_RIGHT + borderline_x, STRIKE_ZONE_TOP, STRIKE_ZONE_TOP + borderline_y, False, True))

BORDERLINE_ZONES.append(Zone([(0, 1), (0, 2), (0, 3)], STRIKE_ZONE_LEFT - borderline_x, STRIKE_ZONE_LEFT, STRIKE_ZONE_BOTTOM, STRIKE_ZONE_TOP, False, True))
BORDERLINE_ZONES.append(Zone([(4, 1), (4, 2), (4, 3)], STRIKE_ZONE_RIGHT, STRIKE_ZONE_RIGHT + borderline_x, STRIKE_ZONE_BOTTOM, STRIKE_ZONE_TOP, False, True))
BORDERLINE_ZONES.append(Zone([(1, 0), (2, 0), (3, 0)], STRIKE_ZONE_LEFT, STRIKE_ZONE_RIGHT, STRIKE_ZONE_BOTTOM - borderline_y, STRIKE_ZONE_BOTTOM, False, True))
BORDERLINE_ZONES.append(Zone([(1, 4), (2, 4), (3, 4)], STRIKE_ZONE_LEFT, STRIKE_ZONE_RIGHT, STRIKE_ZONE_TOP, STRIKE_ZONE_TOP + borderline_y, False, True))

# Strike zones
x_divisions = [STRIKE_ZONE_LEFT, STRIKE_ZONE_LEFT + STRIKE_ZONE_X_STEP, STRIKE_ZONE_RIGHT - STRIKE_ZONE_X_STEP, STRIKE_ZONE_RIGHT]
y_divisions = [STRIKE_ZONE_BOTTOM, STRIKE_ZONE_BOTTOM + STRIKE_ZONE_Y_STEP, STRIKE_ZONE_TOP - STRIKE_ZONE_Y_STEP, STRIKE_ZONE_TOP]
for i in range(len(x_divisions) - 1):
    for j in range(len(y_divisions) - 1):
        NON_BORDERLINE_ZONES.append(Zone([(i + 1, j + 1)], x_divisions[i], x_divisions[i + 1], y_divisions[j], y_divisions[j + 1], is_strike=True))

# Regular ball zones
NON_BORDERLINE_ZONES.append(Zone([(0, 0)], -Inf, STRIKE_ZONE_LEFT, -Inf, STRIKE_ZONE_BOTTOM, False))
NON_BORDERLINE_ZONES.append(Zone([(0, 4)], -Inf, STRIKE_ZONE_LEFT, STRIKE_ZONE_TOP, Inf, False))
NON_BORDERLINE_ZONES.append(Zone([(4, 0)], STRIKE_ZONE_RIGHT, Inf, -Inf, STRIKE_ZONE_BOTTOM, False))
NON_BORDERLINE_ZONES.append(Zone([(4, 4)], STRIKE_ZONE_RIGHT, Inf, STRIKE_ZONE_TOP, Inf, False))

NON_BORDERLINE_ZONES.append(Zone([(0, 1), (0, 2), (0, 3)], -Inf, STRIKE_ZONE_LEFT, STRIKE_ZONE_BOTTOM, STRIKE_ZONE_TOP, False))
NON_BORDERLINE_ZONES.append(Zone([(4, 1), (4, 2), (4, 3)], STRIKE_ZONE_RIGHT, Inf, STRIKE_ZONE_BOTTOM, STRIKE_ZONE_TOP, False))
NON_BORDERLINE_ZONES.append(Zone([(1, 0), (2, 0), (3, 0)], STRIKE_ZONE_LEFT, STRIKE_ZONE_RIGHT, -Inf, STRIKE_ZONE_BOTTOM, False))
NON_BORDERLINE_ZONES.append(Zone([(1, 4), (2, 4), (3, 4)], STRIKE_ZONE_LEFT, STRIKE_ZONE_RIGHT, STRIKE_ZONE_TOP, Inf, False))

ZONES.extend(BORDERLINE_ZONES)
ZONES.extend(NON_BORDERLINE_ZONES)


def get_zone(x_loc: float | None, y_loc: float | None) -> Zone | None:
    """Converts physical coordinates x and y (in inches) to virtual coordinates in the strike zone."""

    if x_loc is None or y_loc is None:
        return None

    for zone in ZONES:
        if zone.left <= x_loc <= zone.right and zone.bottom <= y_loc <= zone.top:
            return zone


def get_zones_batched(x_locs: torch.Tensor, y_locs: torch.Tensor) -> list[int]:
    """
    This batched version of get_zone is significantly faster and necessary for the random
    sampling method used for measuring intended vs actual pitch locations. It returns indices of ZONES.
    """

    result_zones: list = [-1] * len(x_locs)
    for zone_i, zone in enumerate(ZONES):
        mask = (zone.left <= x_locs) & (x_locs <= zone.right) & (zone.bottom <= y_locs) & (y_locs <= zone.top)
        indices = torch.nonzero(mask, as_tuple=False).squeeze()
        if indices.numel() == 1:
            result_zones[indices.item()] = zone_i
        elif indices.numel() > 1:
            for idx in indices:
                result_zones[idx.item()] = zone_i

    return result_zones
