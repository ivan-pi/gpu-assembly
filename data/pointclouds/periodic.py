"""The periodic box: coordinates into it, displacements through it.

The box is ``[0, side)`` along each axis, with opposite sides
identified, as ``rbf::spatial::PeriodicBox`` has it (docs/spatial.md).
Both functions take arrays and a `box` of one side per column, 0 for
an axis that is not periodic, as scipy's k-d tree takes it.
"""

import numpy as np


def wrap(z, box):
    """Coordinates brought into the box through its periodic sides.

    Parameters
    ----------
    z : (..., 2) array_like
        Coordinates.
    box : (2,) array_like
        The side of the box along each axis, 0 for one that is not
        periodic.

    Returns
    -------
    (..., 2) ndarray
        The coordinates, each periodic one in ``[0, side)``.
    """
    z, box = np.array(z, float), np.asarray(box, float)
    periodic = box > 0
    w = np.mod(z[..., periodic], box[periodic])
    w[w >= box[periodic]] = 0.0  # np.mod rounds up to the side
    z[..., periodic] = w
    return z


def minimum_image(d, box):
    """Displacements shortened through the nearer of the two sides.

    Parameters
    ----------
    d : (..., 2) array_like
        Displacements.
    box : (2,) array_like
        The side of the box along each axis, 0 for one that is not
        periodic.

    Returns
    -------
    (..., 2) ndarray
        The displacements, each periodic one within half a side of 0.
    """
    d, box = np.array(d, float), np.asarray(box, float)
    periodic = box > 0
    d[..., periodic] -= box[periodic] * np.round(d[..., periodic] / box[periodic])
    return d
