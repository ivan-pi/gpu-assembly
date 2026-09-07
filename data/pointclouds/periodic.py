"""The periodic box, as rbf::spatial::PeriodicBox has it (docs/spatial.md):
[0, box) along each axis, with opposite sides identified. Both functions
take arrays, with `box` one length per column and 0 for an axis that is
not periodic, as scipy's k-d tree has it."""

import numpy as np


def wrap(z, box):
    """Coordinates z brought into [0, box) through the periodic sides."""
    z, box = np.array(z, float), np.asarray(box, float)
    periodic = box > 0
    w = np.mod(z[..., periodic], box[periodic])
    w[w >= box[periodic]] = 0.0  # np.mod rounds up to the side
    z[..., periodic] = w
    return z


def minimum_image(d, box):
    """Displacements d shortened through the nearer of the two sides."""
    d, box = np.array(d, float), np.asarray(box, float)
    periodic = box > 0
    d[..., periodic] -= box[periodic] * np.round(d[..., periodic] / box[periodic])
    return d
