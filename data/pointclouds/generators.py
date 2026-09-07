"""The clouds the generators in data/tools build, as children of NodeSet:
a perturbed grid, a Poisson disk sample of a periodic box with or
without a hole, and the refined cavity. A child's constructor makes the
points and the markers and hands them to NodeSet with the box and the
title; what a script adds is its command line. Lengths are in lattice
units throughout, with the spacing 1: scaling a case to other units is
left to whoever needs it.
"""

import numpy as np

from .nodeset import MARKERS, NodeSet, boundary_first


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
    last N.

    Parameters
    ----------
    n : int
        Nodes across the box, which is N by N in lattice units.
    sigma : float
        The displacement of a node, uniform on ``[-sigma, sigma]`` spacings.
    geometry : {"periodic", "channel"}
    seed : int, optional
        Of the displacements; random when not given.
    """

    def __init__(self, n, sigma=0.2, *, geometry="periodic", seed=None):
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
        pts += d  # NodeSet wraps the periodic sides
        box = float(n)
        if not periodic:
            pts[:, 1] = np.clip(pts[:, 1], 0.0, box)  # only reached by sigma >= 1
        super().__init__(
            pts,
            m,
            extent=(box, box),
            periodic=(True, periodic),
            title=f"perturbed grid, size={n}x{n}, sigma={sigma:g}, {geometry}",
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
    much to its image across the periodic sides.

    Parameters
    ----------
    extent : (2,) array_like
        The box ``[0, Lx) x [0, Ly)``.
    distance : float
        The least distance between two nodes, 1 in lattice units.
    candidates : int
        The throws a node makes before it is retired; more pack tighter.
    hole : float, optional
        The radius of the disk cut out of the middle of the box.
    seed : int, optional
        Of the sample; random when not given.
    """

    def __init__(self, extent, distance=1.0, *, candidates=100, hole=None, seed=None):
        from .poisson import PoissonDisk  # compiled by numba, only when needed

        extent = np.asarray(extent, float)
        if extent.min() < 2 * distance:
            raise ValueError(
                f"the box is narrower than twice the distance {distance:g}: no room"
            )
        centre = 0.5 * extent
        seeds = ()
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
            distance,
            extent,
            periodic=True,
            ncandidates=candidates,
            seed=seed,
            seeds=seeds,
        )
        sampler.fill_space()
        pts = sampler.points
        title = (
            f"periodic poisson, size={extent[0]:g}x{extent[1]:g}, distance={distance:g}"
        )
        if hole is not None:
            inside = np.hypot(*(pts - centre).T) < hole
            inside[: len(seeds)] = False  # the circle nodes sit on the hole, not in it
            pts = pts[~inside]
            title += f", hole={hole:g}"
        m = np.full(len(pts), MARKERS.interior)
        m[: len(seeds)] = MARKERS.hole
        super().__init__(pts, m, extent=extent, periodic=(True, True), title=title)


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
    assembles the boundary conditions decides what a corner gets.

    Parameters
    ----------
    steps : int
        Spacings across each band.
    size : (2,) tuple of float, optional
        The cavity ``(Lx, Ly)``, at least the 10 steps of the bands.
    distribution : {"rings", "grid"}
    """

    SPACINGS = (1.0, 1.5, 2.5)  # at the wall, in the second band, in the middle

    def __init__(self, steps=10, *, size=None, distribution="rings"):
        span = 2 * steps * sum(self.SPACINGS)  # the bands from both walls
        Lx, Ly = (span, span) if size is None else size
        if min(Lx, Ly) < span:
            raise ValueError(
                f"the cavity must be at least {span:g} by {span:g}, the width "
                f"of the three bands from both walls at {steps} steps"
            )
        rings = distribution == "rings"
        # The spacing of every ring, or of every step of the graded
        # coordinate. A ring lies one spacing of the ring outside it
        # inwards, so a band holds N + 1 rings and the last one N, ending
        # in the centre since the first two bands are as wide as the third.
        counts = (steps + 1, steps + 1, steps) if rings else (steps,) * 3
        h = np.repeat(self.SPACINGS, counts)
        if rings:
            pts, r = [], 0.0
            while 2 * r <= min(Lx, Ly):
                hk = h[min(len(pts), len(h) - 1)]  # the coarsest beyond the bands
                pts.append(self._ring(r, Lx, Ly, hk))
                r += hk
            pts = np.vstack(pts)
        else:
            # from the wall to the end of the bands, the middle at the coarsest
            # spacing if there is room, then the same from the far wall
            z = np.concatenate(([0.0], np.cumsum(h)))
            coordinates = []
            for L in (Lx, Ly):
                middle = np.empty(0)
                if L > 2 * z[-1]:
                    middle = self._side(z[-1], L - z[-1], h[-1])
                coordinates.append(np.concatenate((z[:-1], middle, L - z[::-1])))
            xv, yv = np.meshgrid(*coordinates)
            pts = np.column_stack((xv.ravel(), yv.ravel()))
        pts = np.round(pts, 12) + 0.0  # 0.1 + 0.2 style noise, and no -0.0
        x, y = pts.T
        s, e, n, w = y == 0, x == Lx, y == Ly, x == 0
        m = np.select(
            [(s | n) & (e | w), s, e, n, w],
            [MARKERS.corner, MARKERS.south, MARKERS.east, MARKERS.north, MARKERS.west],
            MARKERS.interior,
        )
        first = boundary_first(m)
        super().__init__(
            pts[first],
            m[first],
            extent=(Lx, Ly),
            title=f"refined cavity, size={Lx:g}x{Ly:g}, {distribution}, steps={steps}",
        )

    # From a to b in a whole number of steps as close to h as possible; the
    # end b is left out.
    @staticmethod
    def _side(a, b, h):
        return np.linspace(a, b, max(round((b - a) / h), 1), endpoint=False)

    # Points about h apart on the boundary of the cavity inset by r, counter-
    # clockwise from (r, r); a segment or a point when a side has shrunk to
    # nothing.
    @classmethod
    def _ring(cls, r, Lx, Ly, h):
        x0, x1, y0, y1 = r, Lx - r, r, Ly - r
        sx, sy = cls._side(x0, x1, h), cls._side(y0, y1, h)
        south = np.column_stack((sx, np.full(len(sx), y0)))
        east = np.column_stack((np.full(len(sy), x1), sy))
        north = np.column_stack((x0 + x1 - sx, np.full(len(sx), y1)))
        west = np.column_stack((np.full(len(sy), x0), y0 + y1 - sy))
        pts = np.vstack((south, east, north, west))
        if x1 == x0 or y1 == y0:  # the four sides lie on each other
            return np.unique(np.round(pts, 12), axis=0)
        return pts
