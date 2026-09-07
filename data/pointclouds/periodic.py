"""The periodic sides of a box: coordinates brought back into it."""

import numpy as np


def wrap(z, box):
    """Coordinates z brought into [0, box) through the periodic side, for
    arrays, with box a scalar or one length per column."""
    z = np.mod(z, box)
    z[z >= box] = 0.0  # np.mod rounds up to the side
    return z
