"""The clouds the generators build, as children of `NodeSet`.

A child's constructor makes the points and the markers and hands them
to `NodeSet` with the box and the title. Lengths are in lattice units
throughout, with the spacing 1: scaling a case to other units is left
to whoever needs it.

.. autosummary::

   Disk
   LShape
   PerturbedGrid
   PoissonBox
   RefinedCavity
"""

import numpy as np

from .nodeset import Marker, NodeSet, boundary_first


class Disk(NodeSet):
    """The disk, or the annulus between two circles, at a fixed spacing.

    The round test cases: the disk ``|z| <= R`` about the origin, on
    which the Poisson equation with a manufactured solution is the usual
    check of an RBF-FD operator, and, with `hole`, the annulus
    ``r0 <= |z| <= R``, the section of the concentric cylinders of
    Taylor-Couette flow. The nodes of the two circles come first, marked
    circle (1) and hole (6), the rest are interior. Dividing the points
    by `radius` gives the unit disk.

    Parameters
    ----------
    radius : float
        The outer radius R.
    spacing : float, default 1.0
        The distance h between neighbouring nodes.
    hole : float, optional
        The radius r0 of the disk cut out of the middle, which makes the
        cloud an annulus; a whole disk without it.
    distribution : {"rings", "spiral"}
        Concentric rings a spacing apart, each turned against the last,
        or a Vogel spiral [1]_ of the same density between the circles.

    Raises
    ------
    ValueError
        For a radius or a spacing that is not positive, and for a hole
        that leaves less than a spacing of either the annulus or its own
        circle.

    Notes
    -----
    The rings hold a whole number of nodes about h apart along the
    circle, so the spacing along a ring differs from h by up to a few
    percent, the more the smaller the ring; the innermost ring of a disk
    is the single node at the centre. Consecutive rings are turned by
    the golden angle against each other, which keeps the nodes off the
    spokes an unturned stack of rings would show.

    The spiral is the one node arrangement here that is quasi-uniform
    rather than structured: node i sits at the golden angle from node
    ``i - 1`` and at the radius that gives every node the same area, as
    many nodes as put them h apart. It is drawn in the annulus a
    hexagonal row spacing ``h sqrt(3) / 2`` inside the two circles, so
    that it does not crowd them; the nodes of a circle are therefore the
    only structured part of it.

    References
    ----------
    .. [1] Vogel, "A better way to construct the sunflower head," Math.
       Biosci. 44, 179-189, 1979,
       doi:10.1016/0025-5564(79)90080-4.
    """

    GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))
    PACKING = 0.96  # the nearest neighbours of a spiral, in sqrt(area per node)

    def __init__(self, radius, spacing=1.0, *, hole=0.0, distribution="rings"):
        if radius <= 0.0 or spacing <= 0.0:
            raise ValueError("the radius and the spacing must be positive")
        if hole:
            if hole < spacing:
                raise ValueError(
                    f"a hole of radius {hole:g} is smaller than the spacing "
                    f"{spacing:g} between nodes"
                )
            if radius - hole < spacing:
                raise ValueError(
                    f"a hole of radius {hole:g} leaves less than the spacing "
                    f"{spacing:g} of the annulus inside a radius of {radius:g}"
                )
        circles = [(radius, Marker.circle)] + ([(hole, Marker.hole)] if hole else [])
        pts = [self._ring(r, spacing) for r, _ in circles]
        m = [np.full(len(p), marker) for p, (_, marker) in zip(pts, circles)]
        fill = self._spiral if distribution == "spiral" else self._rings
        pts.append(fill(hole, radius, spacing))
        m.append(np.full(len(pts[-1]), Marker.interior))
        title = f"{'annulus' if hole else 'disk'}, radius={radius:g}"
        if hole:
            title += f", hole={hole:g}"
        super().__init__(
            np.vstack(pts),
            np.concatenate(m),
            title=title + f", spacing={spacing:g}, {distribution}",
        )

    @classmethod
    def _ring(cls, r, h, turn=0.0):
        """Places a whole number of nodes about h apart on the circle r.

        The circle is turned by `turn`; a ring of no radius is the node
        at the centre.
        """
        n = max(round(2 * np.pi * r / h), 1)
        phi = turn + 2 * np.pi * np.arange(n) / n
        return r * np.column_stack((np.cos(phi), np.sin(phi)))

    @classmethod
    def _rings(cls, r0, r1, h):
        """Fills the annulus between the two circles with turned rings.

        The rings are evenly spaced by about h, leaving that much to
        either circle; a disk gets its innermost ring at the centre
        instead, since it has no inner circle to keep away from.
        """
        if r0:
            k = max(round((r1 - r0) / h), 2)
            radii = r0 + (r1 - r0) * np.arange(1, k) / k
        else:
            k = max(round(r1 / h), 1)
            radii = r1 * np.arange(k) / k
        return np.vstack(
            [cls._ring(r, h, j * cls.GOLDEN_ANGLE) for j, r in enumerate(radii)]
        )

    @classmethod
    def _spiral(cls, r0, r1, h):
        """Draws a Vogel spiral of nodes h apart between the two circles.

        The nodes stay a hexagonal row spacing off either circle, and
        the radii are those of equal areas, so the density is the same
        throughout.
        """
        rin, rout = (r0 + h * np.sqrt(3) / 2 if r0 else 0.0), r1 - h * np.sqrt(3) / 2
        n = max(round(np.pi * (rout**2 - rin**2) * (cls.PACKING / h) ** 2), 1)
        i = np.arange(n)
        r = np.sqrt(rin**2 + (rout**2 - rin**2) * (i + 0.5) / n)
        phi = i * cls.GOLDEN_ANGLE
        return r[:, None] * np.column_stack((np.cos(phi), np.sin(phi)))


class LShape(NodeSet):
    """The L-shaped domain, graded toward its re-entrant corner in rings.

    The canonical corner-singularity test: the square ``[-L, L]^2`` less
    the quadrant ``x > 0, y < 0``, with the re-entrant corner at the
    origin. The Laplace solution ``r^(2/3) sin(2 theta / 3)`` is singular
    there, so uniform refinement loses its rate; the cloud grades the
    spacing toward the corner as ``h(r) ~ h (r / radius)^(1 - 2/3)``, the
    classical a-priori remedy [1]_, used with RBF-FD in [2]_. It is also
    the domain of the MATLAB-logo membrane eigenproblem.

    Parameters
    ----------
    size : float
        The leg length L: each leg of the L is L wide and 2 L long.
    spacing : float, default 1.0
        The distance h between neighbouring nodes away from the corner.
    radius : float, optional
        The radius R of the graded region about the corner, at most a
        spacing short of L so that the graded arcs fit in the legs and
        hand over to the walls; half of `size` when not given.
    exponent : float, default 1.5
        The grading exponent beta: the rings sit at the radii
        ``r_k = R (k/n)**beta``, so the spacing near the corner falls off
        as ``r**(1 - 1/beta)``. The default is ``1/lambda`` for the
        ``lambda = 2/3`` singularity of the corner; 1 spaces the rings
        evenly, an ungraded control case.

    Raises
    ------
    ValueError
        For a size or a spacing that is not positive, an exponent below
        1, and a radius of less than two spacings or reaching within a
        spacing of the legs.

    Notes
    -----
    Inside `radius` the cloud is the arcs of the 270-degree sector at the
    graded radii, each node about a local spacing from its neighbours,
    the arc endpoints on the two walls of the corner; beyond it the rings
    continue at the constant spacing h, clipped to the domain, and the
    walls carry their own evenly spaced nodes. Where a clipped ring meets
    a wall the spacing is ragged by design: a node generator with
    repulsive relaxation can take the cloud as its starting point and
    smooth the seam, and the functionality tests do not mind it.

    The markers name the outward normal, so two walls share one where the
    normals agree: south (1) for ``y = -L`` and the corner wall
    ``y = 0, x > 0``, east (2) for ``x = L`` and the corner wall
    ``x = 0, y < 0``, north (3) for ``y = L``, west (4) for ``x = -L``,
    and corner (5) for the six corners, the re-entrant one included. The
    boundary nodes come first.

    References
    ----------
    .. [1] Mitchell, "A collection of 2D elliptic problems for testing
       adaptive grid refinement algorithms," Appl. Math. Comput. 220,
       350-364, 2013, doi:10.1016/j.amc.2013.05.068.
    .. [2] Oanh, Davydov and Phu, "Adaptive RBF-FD method for elliptic
       problems with point singularities in 2D," Appl. Math. Comput.
       313, 474-497, 2017.
    """

    SECTOR = 1.5 * np.pi  # the 270-degree opening of the re-entrant corner

    def __init__(self, size, spacing=1.0, *, radius=None, exponent=1.5):
        if size <= 0.0 or spacing <= 0.0:
            raise ValueError("the size and the spacing must be positive")
        if exponent < 1.0:
            raise ValueError("an exponent below 1 coarsens toward the corner")
        if radius is None:
            radius = size / 2
        if not 2 * spacing <= radius <= size - spacing:
            raise ValueError(
                f"the graded radius must lie between two spacings "
                f"{2 * spacing:g} and a spacing short of the leg length "
                f"{size - spacing:g}"
            )
        n = max(round(exponent * radius / spacing), 2)
        radii = radius * (np.arange(n + 1) / n) ** exponent
        pts = [np.zeros((1, 2))]
        m = [np.array([Marker.corner])]
        # the graded arcs; their endpoints are the wall nodes of the grading
        for k in range(1, n + 1):
            h = radii[k + 1] - radii[k] if k < n else spacing
            ring = self._arc(radii[k], h)
            mk = np.full(len(ring), Marker.interior)
            mk[0], mk[-1] = Marker.south, Marker.east
            pts.append(ring)
            m.append(mk)
        # uniform arcs beyond the graded region, clipped to the domain
        r = radius + spacing
        while r < size * np.sqrt(2.0):
            ring = self._arc(r, spacing)[1:-1]
            keep = self._inside(ring, size)
            keep &= self._wall_distance(ring, size) >= 0.55 * spacing
            pts.append(ring[keep])
            m.append(np.full(np.count_nonzero(keep), Marker.interior))
            r += spacing
        # the corner walls beyond the grading, then the outer walls
        d = np.arange(radius + spacing, size - 0.5 * spacing, spacing)
        pts.append(np.column_stack((d, np.zeros_like(d))))
        m.append(np.full(len(d), Marker.south))
        pts.append(np.column_stack((np.zeros_like(d), -d)))
        m.append(np.full(len(d), Marker.east))
        verts = self._outline(size)[1:-1]
        walls = (Marker.east, Marker.north, Marker.west, Marker.south)
        for a, b, wall in zip(verts[:-1], verts[1:], walls):
            k = max(round(np.hypot(*(b - a)) / spacing), 1)
            seg = a + (np.arange(k) / k)[:, None] * (b - a)
            mk = np.full(k, wall)
            mk[0] = Marker.corner
            pts.append(seg)
            m.append(mk)
        pts.append(verts[-1:])
        m.append(np.array([Marker.corner]))
        pts = np.round(np.vstack(pts), 12) + 0.0  # rounding noise, and no -0.0
        m = np.concatenate(m)
        first = boundary_first(m)
        super().__init__(
            pts[first],
            m[first],
            title=f"l-shape, size={size:g}, radius={radius:g}, "
            f"exponent={exponent:g}, spacing={spacing:g}",
        )

    @classmethod
    def _arc(cls, r, h, turn=0.0):
        """Places nodes about h apart on the 270-degree arc of radius r.

        With both endpoints, which lie on the walls of the corner.
        """
        n = max(round(cls.SECTOR * r / h), 2)
        phi = turn + cls.SECTOR * np.arange(n + 1) / n
        return r * np.column_stack((np.cos(phi), np.sin(phi)))

    @staticmethod
    def _outline(L):
        """The boundary of the L, counter-clockwise from the corner."""
        return np.array(
            [(0, 0), (L, 0), (L, L), (-L, L), (-L, -L), (0, -L), (0, 0)], float
        )

    @staticmethod
    def _inside(pts, L):
        """Which points lie in the L."""
        x, y = pts.T
        return (np.abs(x) <= L) & (np.abs(y) <= L) & ~((x > 0) & (y < 0))

    @classmethod
    def _wall_distance(cls, pts, L):
        """The distance of every point to the boundary of the L."""
        d = np.full(len(pts), np.inf)
        for a, b in zip(cls._outline(L)[:-1], cls._outline(L)[1:]):
            ab = b - a
            t = np.clip((pts - a) @ ab / (ab @ ab), 0.0, 1.0)
            d = np.minimum(d, np.hypot(*(pts - (a + t[:, None] * ab)).T))
        return d


class PerturbedGrid(NodeSet):
    """A Cartesian grid with every node moved by a small random amount.

    The node layout of Strzelczyk and Matyka [1]_, Figs. 4 and 8: an N by
    N grid with its lowest node at the origin, every coordinate displaced
    by an amount uniform on ``[-sigma, sigma]`` spacings. The box is N by
    N.

    Parameters
    ----------
    n : int
        Nodes across the box.
    sigma : float, default 0.2
        The displacement of a node, in spacings. The reference uses 0,
        0.02 and 0.2; a node stays in its own cell below 0.5.
    geometry : {"periodic", "channel"}
        Both sides periodic, as in the Taylor-Green test, or periodic in
        x with walls at ``y = 0`` and ``y = N``, as in the Poiseuille test.
    seed : int, optional
        Of the displacements; random when not given.

    Notes
    -----
    On the channel the grid has N + 1 rows, the first on the bottom wall
    (marker south) and the last on the top (marker north); a wall node
    slides along its wall but does not leave it. The nodes come row by row
    from ``y = 0``, x fastest, so the wall nodes are the first N and the
    last N.

    References
    ----------
    .. [1] Strzelczyk and Matyka, "How nodes layout, refinement and
       velocity discretization influence convergence of the meshless
       lattice Boltzmann method," 2022, https://ssrn.com/abstract=4070398
    """

    def __init__(self, n, sigma=0.2, *, geometry="periodic", seed=None):
        periodic = geometry == "periodic"
        x = np.arange(float(n))
        y = np.arange(float(n if periodic else n + 1))
        xv, yv = np.meshgrid(x, y)  # row by row, x fastest
        pts = np.column_stack((xv.ravel(), yv.ravel()))
        m = np.full(len(pts), Marker.interior)
        if not periodic:
            m[:n], m[-n:] = Marker.south, Marker.north
        d = np.random.default_rng(seed).uniform(-sigma, sigma, pts.shape)
        d[m != Marker.interior, 1] = 0.0  # a wall node stays on its wall
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
    """A Poisson disk sample of a periodic box, with or without a hole.

    No two nodes are closer than `distance` and no room is left for
    another one, drawn by `pointclouds.poisson.PoissonDisk`: about
    ``0.65 / distance**2`` nodes per unit area with the default number of
    candidates, more with more of them up to a point.

    Parameters
    ----------
    extent : (2,) array_like
        The box ``[0, Lx) x [0, Ly)``.
    distance : float, default 1.0
        The least distance between two nodes.
    candidates : int, default 100
        The number of throws a node makes before it is retired.
    hole : float, optional
        The radius of the disk cut out of the middle of the box.
    seed : int, optional
        Of the sample; random when not given.

    Raises
    ------
    ValueError
        For a box narrower than twice the distance, and for a hole
        smaller than the distance in radius or closer than that to its
        image across the periodic sides.

    Notes
    -----
    With a hole, nodes are laid on its circle first, as many as keep them
    `distance` apart, with the marker hole; the sample grows from them, so
    the nearest nodes sit about a spacing off the circle; and what lands
    inside is discarded. That is the unit cell of a square array of
    cylinders.
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
            distance, extent, periodic=True, ncandidates=candidates, seed=seed
        )
        sampler.add_points(seeds)
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
        m = np.full(len(pts), Marker.interior)
        m[: len(seeds)] = Marker.hole
        super().__init__(pts, m, extent=extent, periodic=(True, True), title=title)


class RefinedCavity(NodeSet):
    """The lid-driven cavity, refined towards the walls in three bands.

    The spacing is 1 at the wall, 1.5 in the second band and 2.5 in the
    middle, each band `steps` spacings wide, so the bands from both walls
    fill a cavity of ``10 * steps``; a bigger `size` gets its middle at
    the coarsest spacing. The proportions are those of the figure in Lin,
    Wu and Zhang [1]_, which shows the point distribution without
    specifying it.

    Parameters
    ----------
    steps : int, default 10
        Spacings across each band.
    size : (2,) tuple of float, optional
        The cavity ``(Lx, Ly)``, at least ``10 * steps`` on either side.
    distribution : {"rings", "grid"}
        Concentric rectangles inset from the walls, or the tensor product
        of 1-d coordinates graded the same way.

    Raises
    ------
    ValueError
        For a size smaller than the bands need.

    Notes
    -----
    The rings have their points h apart along a ring and consecutive
    rings h apart, and end in the centre point of a square or in a
    segment for a rectangle; each side of a ring holds a whole number of
    spacings, so the spacing along a ring can differ from h by a fraction
    of a percent. In the grid, rows and columns line up, but a point near
    the middle of a wall sits in a cell of h by 2.5 h. Both put the wall
    nodes first, from the wall inwards for the rings, row by row for the
    grid.

    The markers are south, east, north (the lid) and west for the walls,
    and corner where two walls meet, since a corner node carries the
    boundary data of both walls, which differ in the cavity; whoever
    assembles the boundary conditions decides what a corner gets.

    References
    ----------
    .. [1] Lin, Wu and Zhang, "A mesh-free radial basis function-based
       semi-Lagrangian lattice Boltzmann method for incompressible
       flows," Int. J. Numer. Meth. Fluids 91, 198-211, 2019.
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
            [Marker.corner, Marker.south, Marker.east, Marker.north, Marker.west],
            Marker.interior,
        )
        first = boundary_first(m)
        super().__init__(
            pts[first],
            m[first],
            extent=(Lx, Ly),
            title=f"refined cavity, size={Lx:g}x{Ly:g}, {distribution}, steps={steps}",
        )

    @staticmethod
    def _side(a, b, h):
        """Divides [a, b) into whole steps as close to h as possible."""
        return np.linspace(a, b, max(round((b - a) / h), 1), endpoint=False)

    @classmethod
    def _ring(cls, r, Lx, Ly, h):
        """Places points about h apart around the cavity inset by r.

        Counter-clockwise from (r, r); a segment or a point when a side has
        shrunk to nothing.
        """
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
