"""Poisson disk sampling of a rectangle, optionally periodic in either
axis: no two points closer than a radius r, and no room left for another
one, drawn by Bridson's algorithm with the inner loop compiled by Numba.

    engine = PoissonDisk(0.02, extent=(1.0, 2.0), periodic=True)   # periodic box
    pts = engine.fill_space()

    engine = PoissonDisk(0.02, extent=(1.0, 2.0), periodic=(True, False),
                         seeds=walls, seed=1234)                  # channel
    pts = engine.fill_space()                       # walls, then the rest

The rectangle is [0, Lx) x [0, Ly), and `periodic` says of each axis
whether its two sides are one: neither for a box with walls, both for a
periodic box, one for a channel. It is the generator's business to scale
the result to lattice units.

The algorithm (Bridson, 2007, "Fast Poisson disk sampling in arbitrary
dimensions", SIGGRAPH sketches): a grid of cells of side r / sqrt(2), so
that a cell holds at most one point, and a queue of active points. A
point is taken off the queue at random and throws `ncandidates`
candidates into the annulus between r and 2r around it, uniformly over
its area; a candidate that lands in the rectangle and finds no point
within r in the 5x5 cells around its own is kept and queued. The first
point is drawn uniformly, unless there are seeds. The result is maximal
up to what the candidates missed: the packing holds about 0.62 / r^2
points per unit area with 30 candidates, 0.56 with 10.

The scalar loop descends from the Python one of Connor Johnson (2015),
"Poisson Disk Sampling", <http://connor-johnson.com/2015/04/08/poisson-
disk-sampling/>, by way of a periodic version of it that once lived in
data/tools. Along a periodic axis the cells tile the box exactly, so the
wrapped search reaches at least r across the seam; with a partial last
cell it reached less, which let points closer than r through.

The interface follows scipy.stats.qmc.PoissonDisk where it can: `random`
draws up to n more points, `fill_space` draws until nothing fits, `reset`
goes back to the start, and the same seed gives the same points. Numba
compiles the loop on the first call of a process, which takes a few
seconds, and caches the result next to this module.
"""

from math import ceil, sqrt

import numpy as np
from numba import njit

from .periodic import wrap


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

    def __init__(self, radius, extent=(1.0, 1.0), periodic=False,
                 ncandidates=30, seed=None, seeds=None):
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
        self._seeds = (np.empty((0, 2)) if seeds is None
                       else np.array(seeds, float, ndmin=2).reshape(-1, 2))
        self.reset()

    def reset(self):
        """Back to the start: the seeds alone, the random stream rewound.
        Returns the engine."""
        r = self.radius
        # cells of side at most r / sqrt(2) tiling the rectangle exactly
        self._cells = np.array([ceil(L / (r / sqrt(2))) for L in self.extent], np.int64)
        self._side = self.extent / self._cells
        ncell = int(self._cells.prod())
        self._px = np.empty(ncell)
        self._py = np.empty(ncell)
        self._grid = np.full(ncell, -1, np.int64)
        self._queue = np.empty(ncell, np.int64)
        self._count = np.zeros(2, np.int64)      # points so far, queue length
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
                pts[:, ~self.periodic] >= self.extent[~self.periodic]):
            raise ValueError("a point lies outside the rectangle")
        pts = pts.copy()
        pts[:, self.periodic] = wrap(pts[:, self.periodic], self.extent[self.periodic])
        if not _add(self._px, self._py, self._grid, self._queue, self._count,
                    pts[:, 0], pts[:, 1], self._cells[0], self._cells[1],
                    self._side[0], self._side[1]):
            raise ValueError("two of the points are closer than the radius")

    def random(self, n=1):
        """Draw up to n more points and return them; fewer when the space
        fills up first."""
        gx, gy = self._cells
        before = int(self._count[0])
        _grow(self._px, self._py, self._grid, self._queue, self._count,
              gx, gy, self._side[0], self._side[1],
              self.extent[0], self.extent[1], bool(self.periodic[0]),
              bool(self.periodic[1]), self.radius, self.ncandidates,
              min(before + n, len(self._px)), self._rng)
        after = int(self._count[0])
        self.num_generated += after - before
        return np.column_stack((self._px[before:after], self._py[before:after]))

    def fill_space(self):
        """Draw until nothing fits and return the points drawn by this call."""
        return self.random(len(self._px))

    @property
    def points(self):
        """All points so far, seeds first, then in the order drawn."""
        n = int(self._count[0])
        return np.column_stack((self._px[:n], self._py[:n]))


# ------------------------------------------------------------------- Numba
# The helpers are separate functions for reading; LLVM inlines them.

@njit(cache=True)
def _cell(z, side, cells):
    """The cell a coordinate falls in, the last one for z == extent."""
    return min(int(z / side), cells - 1)


@njit(cache=True)
def _add(px, py, grid, queue, count, xs, ys, gx, gy, sx, sy):
    """Points put on the grid and the queue as they are; False when a
    cell already holds one."""
    n, qn = count[0], count[1]
    for i in range(len(xs)):
        c = _cell(ys[i], sy, gy) * gx + _cell(xs[i], sx, gx)
        if grid[c] >= 0:
            return False
        px[n] = xs[i]
        py[n] = ys[i]
        grid[c] = n
        queue[qn] = n
        n += 1
        qn += 1
    count[0], count[1] = n, qn
    return True


@njit(cache=True)
def _too_close(x, y, ci, cj, px, py, grid, gx, gy, lx, ly, perx, pery, r2):
    """Whether a point within r of (x, y) sits in the 5x5 cells around
    cell (ci, cj), through the periodic sides where there are any."""
    for dj in range(-2, 3):
        jj = cj + dj
        if pery:
            jj %= gy
        elif jj < 0 or jj >= gy:
            continue
        for di in range(-2, 3):
            ii = ci + di
            if perx:
                ii %= gx
            elif ii < 0 or ii >= gx:
                continue
            t = grid[jj * gx + ii]
            if t < 0:
                continue
            dx = px[t] - x
            dy = py[t] - y
            if perx:
                if dx > 0.5 * lx:
                    dx -= lx
                elif dx < -0.5 * lx:
                    dx += lx
            if pery:
                if dy > 0.5 * ly:
                    dy -= ly
                elif dy < -0.5 * ly:
                    dy += ly
            if dx * dx + dy * dy < r2:
                return True
    return False


@njit(cache=True)
def _grow(px, py, grid, queue, count, gx, gy, sx, sy, lx, ly, perx, pery,
          r, k, nmax, rng):
    """Bridson's loop: points drawn until the queue is empty or there are
    nmax of them. The queue holds the points that still have candidates
    to throw; a point comes off it after its k."""
    n, qn = count[0], count[1]
    r2 = r * r
    if n == 0 and nmax > 0:                      # no seeds: start anywhere
        x = rng.random() * lx
        y = rng.random() * ly
        px[n] = x
        py[n] = y
        grid[_cell(y, sy, gy) * gx + _cell(x, sx, gx)] = n
        queue[qn] = n
        n += 1
        qn += 1
    while qn > 0 and n < nmax:
        qi = rng.integers(0, qn)
        s = queue[qi]
        for _ in range(k):
            a = 2.0 * np.pi * rng.random()
            b = r * sqrt(1.0 + 3.0 * rng.random())   # uniform over the annulus
            x = px[s] + b * np.cos(a)
            y = py[s] + b * np.sin(a)
            if perx:
                if x < 0.0:
                    x += lx
                elif x >= lx:
                    x -= lx
                if x >= lx:                      # rounded up to the side
                    x = 0.0
            elif x < 0.0 or x >= lx:
                continue
            if pery:
                if y < 0.0:
                    y += ly
                elif y >= ly:
                    y -= ly
                if y >= ly:
                    y = 0.0
            elif y < 0.0 or y >= ly:
                continue
            ci = _cell(x, sx, gx)
            cj = _cell(y, sy, gy)
            if grid[cj * gx + ci] >= 0:
                continue
            if _too_close(x, y, ci, cj, px, py, grid, gx, gy, lx, ly, perx, pery, r2):
                continue
            px[n] = x
            py[n] = y
            grid[cj * gx + ci] = n
            queue[qn] = n
            n += 1
            qn += 1
            if n >= nmax:
                break
        else:
            qn -= 1                              # all k thrown: retired
            queue[qi] = queue[qn]
    count[0], count[1] = n, qn
