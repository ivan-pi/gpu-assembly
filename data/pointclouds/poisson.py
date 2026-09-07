"""Poisson disk sampling of a rectangle, optionally periodic in either
axis: no two points closer than a radius r, and no room left for another
one, drawn by Bridson's algorithm with the inner loop compiled by Numba.

The rectangle is [0, Lx) x [0, Ly), and `periodic` says of each axis
whether its two sides are one: neither for a box with walls, both for a
periodic box, one for a channel. A periodic box, filled:

    pts = PoissonDisk(0.02, extent=(1.0, 2.0), periodic=True).fill_space()

A channel, periodic in x, grown from its wall nodes, which then come
first in the result:

    engine = PoissonDisk(0.02, extent=(1.0, 2.0), periodic=(True, False))
    engine.add_points(walls)
    engine.fill_space()
    pts = engine.points

It is the generator's business to scale the result to lattice units.

Bridson's algorithm (2007): a grid of cells of side at most r / sqrt(2),
so that a cell holds one point at most, and a queue of the points that
still have candidates to throw. A queued point throws `ncandidates`
candidates into the annulus between r and 2r around it; a candidate that
finds no point within r in the 5x5 cells around its own is kept and
queued. A filled rectangle holds about 0.62 / r^2 points per unit area
with 30 candidates, 0.56 with 10.

Along a periodic axis the search wraps: the cell index is taken modulo
the cell count, and a difference of coordinates through the nearer of
the two sides. The cells tile the extent exactly along such an axis,
which makes them a little smaller than r / sqrt(2). With a partial last
cell instead, the two cells beyond the seam would span less than r, the
search would stop short of it, and points closer than r could face each
other across the seam: the defect a sample that is wrapped afterwards
shows, and the reason the sampler is periodic itself.

The interface follows scipy.stats.qmc.PoissonDisk: `random` draws up to
n more points, `fill_space` draws until nothing fits, `reset` goes back
to the start, and a seed makes a sample reproducible. The loop, after
Connor Johnson (2015), "Poisson Disk Sampling",
<http://connor-johnson.com/2015/04/08/poisson-disk-sampling/>, is
compiled by Numba on the first call of a process and cached next to this
module.
"""

from collections import namedtuple
from math import ceil, sqrt

import numpy as np
from numba import njit

__all__ = ["PoissonDisk"]


def wrap(z, box):
    """Coordinates z brought into [0, box) through the periodic side, for
    arrays, with box a scalar or one length per column."""
    z = np.mod(z, box)
    z[z >= box] = 0.0  # np.mod rounds up to the side
    return z


# ------------------------------------------------------------------- Numba
# Numba compiles functions of plain arguments and cannot cache a jitclass,
# so the state lives in three namedtuples, which it passes in and reads
# without cost, and the loop is split into small functions of those,
# which LLVM inlines.

Box = namedtuple("Box", "lx ly perx pery")  # the rectangle and its periodic axes
Cells = namedtuple("Cells", "nx ny sx sy")  # the grid: counts and sides of the cells
Store = namedtuple("Store", "px py grid queue count")
# px, py   the points, in the order drawn
# grid     the index of the point in each cell, -1 for none
# queue    the points that still have candidates to throw
# count    the number of points, the length of the queue


@njit(cache=True)
def wrap_coordinate(z, length):
    """One coordinate brought into [0, length) through the periodic side,
    from at most one length out; `wrap` does the same for arrays."""
    if z < 0.0:
        z += length
    elif z >= length:
        z -= length
    if z >= length:  # rounded up to the side
        z = 0.0
    return z


@njit(cache=True)
def into_box(box, x, y):
    """Whether (x, y) is in the box, and where: brought in through a
    periodic side, or out for good through a wall."""
    if box.perx:
        x = wrap_coordinate(x, box.lx)
    elif x < 0.0 or x >= box.lx:
        return False, x, y
    if box.pery:
        y = wrap_coordinate(y, box.ly)
    elif y < 0.0 or y >= box.ly:
        return False, x, y
    return True, x, y


@njit(cache=True)
def minimum_image(d, length):
    """A difference of coordinates through the nearer of the two sides."""
    if d > 0.5 * length:
        d -= length
    elif d < -0.5 * length:
        d += length
    return d


@njit(cache=True)
def squared_distance(box, dx, dy):
    """The squared distance for a difference of two points, through the
    periodic sides where that is shorter."""
    if box.perx:
        dx = minimum_image(dx, box.lx)
    if box.pery:
        dy = minimum_image(dy, box.ly)
    return dx * dx + dy * dy


@njit(cache=True)
def cell_of(cells, x, y):
    """The column and row of the cell (x, y) falls in; the last for a
    coordinate on the far side."""
    return (min(int(x / cells.sx), cells.nx - 1), min(int(y / cells.sy), cells.ny - 1))


@njit(cache=True)
def cell_taken(store, cells, x, y):
    """Whether the cell of (x, y) already holds a point."""
    ci, cj = cell_of(cells, x, y)
    return store.grid[cj * cells.nx + ci] >= 0


@njit(cache=True)
def too_close(store, cells, box, x, y, r2):
    """Whether a point within r of (x, y) sits in the 5x5 cells around
    its own, through the periodic sides where there are any."""
    ci, cj = cell_of(cells, x, y)
    for dj in range(-2, 3):
        jj = cj + dj
        if box.pery:
            jj %= cells.ny
        elif jj < 0 or jj >= cells.ny:
            continue
        for di in range(-2, 3):
            ii = ci + di
            if box.perx:
                ii %= cells.nx
            elif ii < 0 or ii >= cells.nx:
                continue
            t = store.grid[jj * cells.nx + ii]
            if t >= 0 and squared_distance(box, store.px[t] - x, store.py[t] - y) < r2:
                return True
    return False


@njit(cache=True)
def insert_point(store, cells, x, y):
    """(x, y) stored, put in its cell and queued."""
    n, qn = store.count[0], store.count[1]
    ci, cj = cell_of(cells, x, y)
    store.px[n] = x
    store.py[n] = y
    store.grid[cj * cells.nx + ci] = n
    store.queue[qn] = n
    store.count[0] = n + 1
    store.count[1] = qn + 1


@njit(cache=True)
def insert_points(store, cells, xs, ys):
    """Points inserted as they are; False at the first whose cell is
    taken, which means it is closer than r to another."""
    for i in range(len(xs)):
        if cell_taken(store, cells, xs[i], ys[i]):
            return False
        insert_point(store, cells, xs[i], ys[i])
    return True


@njit(cache=True)
def draw_points(store, cells, box, r, k, nmax, rng):
    """Points drawn from the queue until it is empty or there are nmax of
    them: a queued point throws its k candidates and comes off, and every
    candidate that fits is stored and queued in its turn."""
    if store.count[0] == 0 and nmax > 0:  # no seeds: start anywhere
        insert_point(store, cells, rng.random() * box.lx, rng.random() * box.ly)
    r2 = r * r
    while store.count[1] > 0 and store.count[0] < nmax:
        qi = rng.integers(0, store.count[1])
        s = store.queue[qi]
        for _ in range(k):
            a = 2.0 * np.pi * rng.random()
            b = r * sqrt(1.0 + 3.0 * rng.random())  # uniform over the annulus
            inside, x, y = into_box(
                box, store.px[s] + b * np.cos(a), store.py[s] + b * np.sin(a)
            )
            if not inside or cell_taken(store, cells, x, y):
                continue
            if too_close(store, cells, box, x, y, r2):
                continue
            insert_point(store, cells, x, y)
            if store.count[0] >= nmax:
                break
        else:  # all k thrown: retired
            qn = store.count[1] - 1
            store.queue[qi] = store.queue[qn]
            store.count[1] = qn


class PoissonDisk:
    """A Poisson disk sampler of the rectangle [0, Lx) x [0, Ly) with
    minimum distance `radius`.

    extent       (Lx, Ly), the rectangle [0, Lx) x [0, Ly)
    periodic     whether each axis wraps around, one flag or a pair
    ncandidates  candidates a point throws before it is retired
    seed         anything numpy.random.default_rng accepts; None draws one
    seeds        points to start from, kept ahead of the drawn ones in
                 `points`; they are taken as given and must themselves be
                 at least `radius` apart

    `points` holds everything so far, seeds first, in the order drawn.
    """

    def __init__(
        self,
        radius,
        extent=(1.0, 1.0),
        periodic=False,
        ncandidates=30,
        seed=None,
        seeds=None,
    ):
        if radius <= 0:
            raise ValueError("radius must be positive")
        extent = np.asarray(extent, float)
        if extent.shape != (2,) or np.any(extent <= 0):
            raise ValueError("extent must be two positive lengths")
        periodic = np.broadcast_to(np.asarray(periodic, bool), 2).copy()
        if ncandidates < 1:
            raise ValueError("ncandidates must be at least 1")

        self.radius = float(radius)
        self.extent = extent
        self.periodic = periodic
        self.ncandidates = int(ncandidates)
        self.seed = seed
        self._seeds = (
            np.empty((0, 2))
            if seeds is None
            else np.array(seeds, float, ndmin=2).reshape(-1, 2)
        )
        self.reset()

    def reset(self):
        """Back to the start: the seeds alone, the random stream rewound.
        Returns the engine."""
        lx, ly = self.extent
        # cells of side at most r / sqrt(2) tiling the rectangle exactly,
        # so that a cell holds one point at most
        nx, ny = (ceil(L / (self.radius / sqrt(2))) for L in self.extent)
        self._box = Box(lx, ly, bool(self.periodic[0]), bool(self.periodic[1]))
        self._cells = Cells(nx, ny, lx / nx, ly / ny)
        self._store = Store(
            np.empty(nx * ny),
            np.empty(nx * ny),
            np.full(nx * ny, -1, np.int64),
            np.empty(nx * ny, np.int64),
            np.zeros(2, np.int64),
        )
        self._rng = np.random.default_rng(self.seed)
        self.num_generated = 0
        self.add_points(self._seeds)
        return self

    def add_points(self, pts):
        """Feed points to the queue as if drawn: the seeds, or the nodes of
        an inner layer. They are wrapped through the periodic sides, and
        two of them in one cell, which are closer than r, are refused."""
        pts = np.array(pts, float, ndmin=2).reshape(-1, 2)
        if len(pts) == 0:
            return
        if np.any(pts[:, ~self.periodic] < 0) or np.any(
            pts[:, ~self.periodic] >= self.extent[~self.periodic]
        ):
            raise ValueError("a point lies outside the rectangle")
        pts = pts.copy()
        pts[:, self.periodic] = wrap(pts[:, self.periodic], self.extent[self.periodic])
        if not insert_points(self._store, self._cells, pts[:, 0], pts[:, 1]):
            raise ValueError("two of the points are closer than the radius")

    def random(self, n=1):
        """Draw up to n more points and return them; fewer when the space
        fills up first."""
        before = int(self._store.count[0])
        draw_points(
            self._store,
            self._cells,
            self._box,
            self.radius,
            self.ncandidates,
            min(before + n, len(self._store.px)),
            self._rng,
        )
        after = int(self._store.count[0])
        self.num_generated += after - before
        return self.points[before:]

    def fill_space(self):
        """Draw until nothing fits and return the points drawn by this call."""
        return self.random(len(self._store.px))

    @property
    def points(self):
        """All points so far, seeds first, then in the order drawn."""
        n = int(self._store.count[0])
        return np.column_stack((self._store.px[:n], self._store.py[:n]))
