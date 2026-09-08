"""The periodic box: coordinates into it, displacements through it.

The box is ``[0, side)`` along each axis, with opposite sides
identified, as ``rbf::spatial::PeriodicBox`` has it (docs/spatial.md).
Both functions take arrays of points in any dimension and a `box` of
one side per axis, 0 for an axis that is not periodic, as scipy's k-d
tree takes it.
"""

import numpy as np


def wrap(z, box):
    """Wraps the point `z` into the periodic extents of `box`.

    Parameters
    ----------
    z : (..., d) array_like
        Coordinates.
    box : (d,) array_like
        The side of the box along each axis, 0 for one that is not
        periodic.

    Returns
    -------
    (..., d) ndarray
        The coordinates, each periodic one in ``[0, side)``.
    """
    z, box = np.array(z, float), np.asarray(box, float)
    periodic = box > 0
    w = np.mod(z[..., periodic], box[periodic])
    w[w >= box[periodic]] = 0.0  # np.mod rounds up to the side
    z[..., periodic] = w
    return z


def minimum_image(d, box):
    """Shortens displacement `d` to fall within the minimum image of `box`.

    Parameters
    ----------
    d : (..., d) array_like
        Displacements.
    box : (d,) array_like
        The side of the box along each axis, 0 for one that is not
        periodic.

    Returns
    -------
    (..., d) ndarray
        The displacements, each periodic one within half a side of 0.
    """
    d, box = np.array(d, float), np.asarray(box, float)
    periodic = box > 0
    d[..., periodic] -= box[periodic] * np.round(d[..., periodic] / box[periodic])
    return d
