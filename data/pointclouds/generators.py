"""The clouds the generators in data/gen build, as children of NodeSet:
a perturbed grid, a Poisson disk sample of a periodic box with or
without a hole, and the refined cavity. A child's constructor makes the
points and the markers and hands them to NodeSet with the box and the
title; what a script adds is its command line. Lengths are in lattice
units throughout, with the spacing 1: scaling a case to other units is
left to whoever needs it.
"""

import numpy as np

from .nodeset import MARKERS, NodeSet, wrap


class PerturbedGrid(NodeSet):
    """The node layout of Strzelczyk and Matyka (2022), "How nodes
    layout, refinement and velocity discretization influence convergence
    of the meshless lattice Boltzmann method",
    <https://ssrn.com/abstract=4070398>, Figs. 4 and 8: an N by N
    Cartesian grid with its lowest node at the origin, every coordinate
    displaced by an amount uniform on [-sigma, sigma] spacings. The
    reference uses sigma = 0, 0.02 and 0.2; a node stays in its own cell
    for sigma < 0.5.

    The box is N by N. With `geometry` "periodic" both sides are
    periodic, as in the Taylor-Green test, and a node that leaves the
    box comes back through the opposite side. With "channel" the box is
    periodic in x with walls at y = 0 and y = N, as in the Poiseuille
    test: the grid has N + 1 rows, the first on the bottom wall (marker
    south) and the last on the top (marker north), and a wall node
    slides along its wall but does not leave it. The nodes come row by
    row from y = 0, x fastest, so the wall nodes are the first N and the
    last N."""

    def __init__(self, n, sigma=0.2, geometry="periodic", seed=None):
        periodic = geometry == "periodic"
        x = np.arange(float(n))
        y = np.arange(float(n if periodic else n + 1))
        xv, yv = np.meshgrid(x, y)  # row by row, x fastest
        pts = np.column_stack((xv.ravel(), yv.ravel()))
        m = np.full(len(pts), MARKERS.interior)
        if not periodic:
            m[:n], m[len(pts) - n :] = MARKERS.south, MARKERS.north
        d = np.random.default_rng(seed).uniform(-sigma, sigma, pts.shape)
        d[m != MARKERS.interior, 1] = 0.0  # a wall node stays on its wall
        pts += d
        box = float(n)
        pts[:, 0] = wrap(pts[:, 0], box)
        if periodic:
            pts[:, 1] = wrap(pts[:, 1], box)
        else:
            pts[:, 1] = np.clip(pts[:, 1], 0.0, box)  # only reached by sigma >= 1
        super().__init__(
            pts,
            m,
            (box, box),
            (True, periodic),
            f"perturbed grid, size={n}x{n}, n={n}, sigma={sigma:g}, {geometry}",
        )


class PoissonBox(NodeSet):
    """A Poisson disk sample of the periodic box [0, Lx) x [0, Ly): no
    two nodes closer than `distance`, and no room left for another one,
    from `candidates` throws per node (pointclouds.poisson); about
    0.65 / distance^2 nodes per unit area at the default, more with more
    candidates up to a point.

    With a `hole`, the disk of that radius about the centre of the box is
    cut out: nodes are laid on its circle first, as many as keep them
    `distance` apart, with the marker hole; the sample grows from them,
    so the nearest nodes sit about a spacing off the circle; and what
    lands inside is discarded. That is the unit cell of a square array of
    cylinders. The hole is at least `distance` in radius and leaves that
    much to its image across the periodic sides."""

    def __init__(self, extent, distance=1.0, candidates=100, hole=None, seed=None):
        from .poisson import PoissonDisk  # compiled by numba, only when needed

        extent = np.asarray(extent, float)
        if extent.min() < 2 * distance:
            raise ValueError(
                f"the box is narrower than twice the distance {distance:g}: no room"
            )
        centre = 0.5 * extent
        seeds = np.empty((0, 2))
        if hole is not None:
            if hole < distance:
                raise ValueError(
                    f"a hole of radius {hole:g} is smaller than the distance "
                    f"{distance:g} between nodes"
                )
            if 2 * hole + distance > extent.min():
                raise ValueError(
                    f"a hole of radius {hole:g} leaves less than the distance "
                    f"{distance:g} to its image across the periodic sides of a "
                    f"box {extent.min():g} across"
                )
            n = int(np.pi / np.arcsin(distance / (2 * hole)))  # chords of at least d
            phi = 2 * np.pi * np.arange(n) / n
            seeds = centre + hole * np.column_stack((np.cos(phi), np.sin(phi)))
        sampler = PoissonDisk(
            distance, extent, periodic=True, ncandidates=candidates, seed=seed, seeds=seeds
        )
        sampler.fill_space()
        pts = sampler.points
        inside = np.hypot(*(pts - centre).T) < (hole or 0.0)
        inside[: len(seeds)] = False  # the circle nodes sit on the hole, not in it
        pts = pts[~inside]
        m = np.full(len(pts), MARKERS.interior)
        m[: len(seeds)] = MARKERS.hole
        title = f"periodic poisson, size={extent[0]:g}x{extent[1]:g}, distance={distance:g}"
        area = extent.prod()
        if hole is not None:
            title += f", hole={hole:g}"
            area -= np.pi * hole**2
        super().__init__(pts, m, extent, (True, True), title, area)


# ------------------------------------------------------------ the cavity

SPACINGS = (1.0, 1.5, 2.5)  # at the wall, in the second band, in the middle

TOL = 1e-9


class RefinedCavity(NodeSet):
    """The lid-driven cavity [0, Lx] x [0, Ly], refined towards the walls
    in three bands of `steps` spacings each: the spacing is 1 at the
    wall, 1.5 in the second band and 2.5 in the middle, so the bands from
    both walls fill a cavity of 10 steps, the size a bare `steps` gives,
    and a bigger `size` = (Lx, Ly) gets its middle at the coarsest
    spacing. The proportions are those of the figure in Lin, Wu and
    Zhang (2019), "A mesh-free radial basis function-based
    semi-Lagrangian lattice Boltzmann method for incompressible flows",
    Int. J. Numer. Meth. Fluids 91, 198-211.

    Two distributions realise the spacings. "rings" are concentric
    rectangles inset from the walls, the points h apart along a ring and
    consecutive rings h apart, ending in the centre point of a square,
    or in a segment for a rectangle; each side of a ring holds a whole
    number of spacings, so the spacing along a ring can differ from h by
    a fraction of a percent. "grid" is the tensor product of 1-d
    coordinates graded the same way: rows and columns line up, but a
    point near the middle of a wall sits in a cell of h by 2.5 h. Both
    put the wall nodes first, from the wall inwards for the rings, row
    by row for the grid.

    The markers are south, east, north (the lid) and west for the walls,
    and corner where two walls meet, since a corner node carries the
    boundary data of both walls, which differ in the cavity; whoever
    assembles the boundary conditions decides what a corner gets."""

    def __init__(self, steps=10, size=None, distribution="rings"):
        span = 2 * steps * sum(SPACINGS)  # the bands from both walls
        Lx, Ly = (span, span) if size is None else size
        if min(Lx, Ly) < span - TOL:
            raise ValueError(
                f"the cavity must be at least {span:g} by {span:g}, the width "
                f"of the three bands from both walls at {steps} steps"
            )
        for_rings = distribution == "rings"
        lv = levels(steps, for_rings)
        pts = rings(Lx, Ly, lv) if for_rings else grid(Lx, Ly, lv)
        pts = np.round(pts, 12) + 0.0  # 0.1 + 0.2 style noise, and no -0.0
        super().__init__(
            pts,
            cavity_markers(pts, Lx, Ly),
            (Lx, Ly),
            (False, False),
            f"refined cavity, size={Lx:g}x{Ly:g}, {distribution}, steps={steps}",
        )


def levels(N, for_rings):
    """(spacing, count) per band for N spacings across each band: the number
    of rings, or of steps of the 1-d grid coordinate. Consecutive rings are
    one spacing of the outer ring apart, so a band holds N + 1 rings and
    the first ring of the next band lies one spacing of this band inside
    its last. The last band holds N rings and ends in the centre point,
    which works out because the first two bands together are as wide as
    the third."""
    counts = (N + 1, N + 1, N) if for_rings else (N, N, N)
    return list(zip(SPACINGS, counts))


def side(a, b, h):
    """From a to b in a whole number of steps as close to h as possible;
    the end b is left out."""
    n = max(round(abs(b - a) / h), 1)
    return np.linspace(a, b, n + 1)[:-1]


def ring(x0, x1, y0, y1, h):
    """Points about h apart on the boundary of [x0, x1] x [y0, y1], counter-
    clockwise from (x0, y0); a segment or a point when a side is zero."""
    if x1 - x0 < TOL and y1 - y0 < TOL:
        return np.array([[x0, y0]])
    if x1 - x0 < TOL:
        return np.column_stack(
            (np.full(len(side(y0, y1, h)) + 1, x0), np.append(side(y0, y1, h), y1))
        )
    if y1 - y0 < TOL:
        return np.column_stack(
            (np.append(side(x0, x1, h), x1), np.full(len(side(x0, x1, h)) + 1, y0))
        )
    sx, sy = side(x0, x1, h), side(y0, y1, h)
    south = np.column_stack((sx, np.full(len(sx), y0)))
    east = np.column_stack((np.full(len(sy), x1), sy))
    north = np.column_stack((x1 + x0 - sx, np.full(len(sx), y1)))
    west = np.column_stack((np.full(len(sy), x0), y1 + y0 - sy))
    return np.vstack((south, east, north, west))


def rings(Lx, Ly, levels):
    spacings = [h for h, count in levels for _ in range(count)]
    pts = []
    r = 0.0
    k = 0
    while Lx - 2 * r > -TOL and Ly - 2 * r > -TOL:
        h = spacings[min(k, len(spacings) - 1)]
        pts.append(ring(r, Lx - r, r, Ly - r, h))
        r += h
        k += 1
    return np.vstack(pts)


def graded_coordinate(L, levels):
    """From the wall at 0 to the far wall at L: the levels inwards from
    both walls, and the middle at the coarsest spacing."""
    z = [0.0]
    for h, count in levels:
        z.extend(z[-1] + h * np.arange(1, count + 1))
    z = np.array(z)
    if L - 2 * z[-1] > TOL:
        middle = side(z[-1], L - z[-1], levels[-1][0])
    else:
        middle = np.empty(0)  # the unit side: the bands meet
    return np.concatenate((z[:-1], middle, L - z[::-1]))


def grid(Lx, Ly, levels):
    x, y = np.meshgrid(graded_coordinate(Lx, levels), graded_coordinate(Ly, levels))
    pts = np.column_stack((x.ravel(), y.ravel()))
    on_wall = cavity_markers(pts, Lx, Ly) != MARKERS.interior
    return np.vstack((pts[on_wall], pts[~on_wall]))


def cavity_markers(pts, Lx, Ly):
    x, y = pts[:, 0], pts[:, 1]
    s, e, n, w = y < TOL, x > Lx - TOL, y > Ly - TOL, x < TOL
    m = np.full(len(pts), MARKERS.interior)
    m[s], m[e], m[n], m[w] = MARKERS.south, MARKERS.east, MARKERS.north, MARKERS.west
    m[(s | n) & (e | w)] = MARKERS.corner
    return m
