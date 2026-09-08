"""Poisson disk sampling of a rectangle, periodic in either axis.

`PoissonDisk` draws points no two of which are closer than a radius,
with no room left for another one, by Bridson's algorithm with the
inner loop compiled by Numba.
"""

from collections import namedtuple
from math import ceil, sqrt

import numpy as np
from numba import njit

__all__ = ["PoissonDisk"]


# ------------------------------------------------------------------- Numba
# Numba compiles functions of plain arguments and cannot cache a jitclass,
# so the state lives in two namedtuples, which it passes in and reads
# without cost, and the loop is split into small functions of those,
# which LLVM inlines.

Grid = namedtuple("Grid", "lx ly perx pery nx ny sx sy")
# lx, ly      the rectangle
# perx, pery  whether each axis is periodic
# nx, ny      the cells across it
# sx, sy      the sides of a cell
Store = namedtuple("Store", "p grid queue count")
# p      the points, in the order drawn
# grid   the index of the point in each cell, -1 for none
# queue  the points that still have candidates to throw
# count  the number of points, the length of the queue

TAKEN, OUTSIDE, TOO_CLOSE = 0, 1, 2  # what became of a point offered


@njit(cache=True)
def wrap_coordinate(z, length):
    """Wraps one coordinate into [0, length)."""
    z %= length
    if z >= length:  # rounded up to the side
        z = 0.0
    return z


@njit(cache=True)
def into_box(grid, x, y):
    """Brings (x, y) into the box, telling whether it is inside.

    In through a periodic side, or out for good through a wall.
    """
    if grid.perx:
        x = wrap_coordinate(x, grid.lx)
    elif x < 0.0 or x >= grid.lx:
        return False, x, y
    if grid.pery:
        y = wrap_coordinate(y, grid.ly)
    elif y < 0.0 or y >= grid.ly:
        return False, x, y
    return True, x, y


@njit(cache=True)
def minimum_image(d, length):
    """Shortens a coordinate difference through the nearer of the sides."""
    if d > 0.5 * length:
        d -= length
    elif d < -0.5 * length:
        d += length
    return d


@njit(cache=True)
def squared_distance(grid, dx, dy):
    """Squares the distance of a difference, through the periodic sides."""
    if grid.perx:
        dx = minimum_image(dx, grid.lx)
    if grid.pery:
        dy = minimum_image(dy, grid.ly)
    return dx * dx + dy * dy


@njit(cache=True)
def cell_of(grid, x, y):
    """Finds the column and row of the cell (x, y) falls in.

    The last cell for a coordinate on the far side.
    """
    return (min(int(x / grid.sx), grid.nx - 1), min(int(y / grid.sy), grid.ny - 1))


@njit(cache=True)
def too_close(store, grid, x, y, r2):
    """Tells whether a point within r of (x, y) sits in the cells around.

    The 5x5 cells around its own, through the periodic sides where there
    are any.
    """
    ci, cj = cell_of(grid, x, y)
    for dj in range(-2, 3):
        jj = cj + dj
        if grid.pery:
            jj %= grid.ny
        elif jj < 0 or jj >= grid.ny:
            continue
        for di in range(-2, 3):
            ii = ci + di
            if grid.perx:
                ii %= grid.nx
            elif ii < 0 or ii >= grid.nx:
                continue
            t = store.grid[jj * grid.nx + ii]
            if (
                t >= 0
                and squared_distance(grid, store.p[t, 0] - x, store.p[t, 1] - y) < r2
            ):
                return True
    return False


@njit(cache=True)
def offer_point(store, grid, x, y, r2):
    """Stores (x, y) in its cell and the queue, if it fits in the box.

    Returns TAKEN, or OUTSIDE a wall, or TOO_CLOSE to a stored point.
    """
    inside, x, y = into_box(grid, x, y)
    if not inside:
        return OUTSIDE
    if too_close(store, grid, x, y, r2):
        return TOO_CLOSE
    n, qn = store.count[0], store.count[1]
    ci, cj = cell_of(grid, x, y)
    store.p[n, 0] = x
    store.p[n, 1] = y
    store.grid[cj * grid.nx + ci] = n
    store.queue[qn] = n
    store.count[0] = n + 1
    store.count[1] = qn + 1
    return TAKEN


@njit(cache=True)
def offer_points(store, grid, pts, r2):
    """Offers the points in turn; stops at the first refused.

    Returns what became of the last one offered.
    """
    for i in range(len(pts)):
        fate = offer_point(store, grid, pts[i, 0], pts[i, 1], r2)
        if fate != TAKEN:
            return fate
    return TAKEN


@njit(cache=True)
def draw_points(store, grid, r, k, nmax, rng):
    """Draws points from the queue until it is empty or there are nmax.

    A queued point throws its k candidates and comes off; every candidate
    that fits is stored and queued in its turn.
    """
    r2 = r * r
    while store.count[1] > 0 and store.count[0] < nmax:
        qi = rng.integers(0, store.count[1])
        s = store.queue[qi]
        for _ in range(k):
            a = 2.0 * np.pi * rng.random()
            b = r * sqrt(1.0 + 3.0 * rng.random())  # uniform over the annulus
            x, y = store.p[s, 0] + b * np.cos(a), store.p[s, 1] + b * np.sin(a)
            if offer_point(store, grid, x, y, r2) == TAKEN and store.count[0] >= nmax:
                break
        else:  # all k thrown: retired
            qn = store.count[1] - 1
            store.queue[qi] = store.queue[qn]
            store.count[1] = qn


class PoissonDisk:
    """A Poisson disk sampler of the rectangle ``[0, Lx) x [0, Ly)``.

    Parameters
    ----------
    radius : float
        The least distance between two points.
    extent : (2,) array_like, default (1.0, 1.0)
        ``(Lx, Ly)``.
    periodic : bool or (2,) tuple of bool, default False
        Whether each axis wraps around: neither for a box with walls, both
        for a periodic box, one for a channel.
    ncandidates : int, default 30
        Candidates a point throws before it is retired.
    seed : optional
        Anything ``numpy.random.default_rng`` accepts; None draws one.

    Attributes
    ----------
    points : (n, 2) ndarray
        Everything so far, in the order added and drawn.
    radius, extent, periodic, ncandidates, seed
        As given.

    Notes
    -----
    Bridson's algorithm [1]_ keeps the points on a grid of square cells
    with sides of at most ``radius / sqrt(2)``, so that no cell can hold
    two points, and a queue of the points that still have candidates to
    throw. Each round takes a point from the queue at random and throws
    `ncandidates` candidates into the annulus between one and two radii
    from it. A candidate is accepted when none of the 5 by 5 cells around
    its own holds a point within the radius; it is then stored, put in
    its cell and queued in its turn. A point whose candidates are all
    refused leaves the queue. The sample is complete when the queue is
    empty.

    Along a periodic axis the search wraps around: a cell index is taken
    modulo the number of cells, and a coordinate difference is measured
    through the nearer of the two sides. The cells then have to tile the
    extent exactly, so that the 5 by 5 cells around a point still reach
    a radius across the seam, which makes them a little smaller than
    ``radius / sqrt(2)``. Wrapping an ordinary sample afterwards would
    not do: the points on either side of the seam have never seen each
    other, and pairs closer than the radius appear across it.

    The interface follows ``scipy.stats.qmc.PoissonDisk``. The loop, after
    Johnson [2]_, is compiled by Numba on the first call of a process and
    cached next to this module.

    References
    ----------
    .. [1] R. Bridson, "Fast Poisson disk sampling in arbitrary
       dimensions," ACM SIGGRAPH 2007 Sketches, 2007.
    .. [2] C. Johnson, "Poisson Disk Sampling," 2015,
       http://connor-johnson.com/2015/04/08/poisson-disk-sampling/
    """

    def __init__(
        self, radius, extent=(1.0, 1.0), *, periodic=False, ncandidates=30, seed=None
    ):
        if radius <= 0:
            raise ValueError("radius must be positive")
        extent = np.asarray(extent, float)
        if extent.shape != (2,) or np.any(extent <= 0):
            raise ValueError("extent must be two positive lengths")

        self.radius = float(radius)
        self.extent = extent
        self.periodic = tuple(bool(p) for p in np.broadcast_to(periodic, 2))
        self.ncandidates = int(ncandidates)
        self.seed = seed
        self.reset()

    def reset(self):
        """Empties the sample and rewinds the random stream."""
        lx, ly = self.extent
        side = self.radius / sqrt(2)  # so that a cell holds one point at most
        nx, ny = ceil(lx / side), ceil(ly / side)
        self._grid = Grid(lx, ly, *self.periodic, nx, ny, lx / nx, ly / ny)
        self._store = Store(
            np.empty((nx * ny, 2)),
            np.full(nx * ny, -1, np.int64),
            np.empty(nx * ny, np.int64),
            np.zeros(2, np.int64),
        )
        self._rng = np.random.default_rng(self.seed)

    def add_points(self, pts):
        """Feeds points to the sample as if drawn.

        Seeds to start from, or the nodes of an inner layer; the points
        drawn later keep the radius from them.

        Parameters
        ----------
        pts : (m, 2) array_like

        Raises
        ------
        ValueError
            For a point outside the rectangle along an axis with walls, or
            closer than the radius to one already in the sample.
        """
        pts = np.asarray(pts, float).reshape(-1, 2)
        fate = offer_points(self._store, self._grid, pts, self.radius**2)
        if fate == OUTSIDE:
            raise ValueError("a point lies outside the rectangle")
        if fate == TOO_CLOSE:
            raise ValueError("a point is closer than the radius to another")

    def random(self, n=1):
        """Draws up to `n` more points, fewer once the space fills."""
        before = int(self._store.count[0])
        if before == 0 and n > 0:  # nothing to throw from: start anywhere
            self.add_points(self._rng.random(2) * self.extent)
        draw_points(
            self._store,
            self._grid,
            self.radius,
            self.ncandidates,
            min(before + n, len(self._store.p)),
            self._rng,
        )
        return self.points[before:]

    def fill_space(self):
        """Draws until nothing fits; returns the points of this call."""
        return self.random(len(self._store.p))

    @property
    def points(self):
        """Returns all points so far, in the order added and drawn."""
        return self._store.p[: self._store.count[0]].copy()
