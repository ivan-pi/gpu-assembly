"""The clouds the generators build, as children of `NodeSet`.

A child's constructor makes the points and the markers and hands them
to `NodeSet` with the box and the title. Lengths are in lattice units
throughout, with the spacing 1: scaling a case to other units is left
to whoever needs it.

.. autosummary::

   Disk
   EccentricAnnulus
   LShape
   PerturbedGrid
   PoissonBox
   PolarRegion
   RefinedCavity
"""

import numpy as np
from scipy.spatial import cKDTree

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


class EccentricAnnulus(NodeSet):
    """The annulus between two eccentric circles, Poisson-disk sampled.

    The domain of Wannier flow, the Stokes flow between two rotating
    circular cylinders whose axes do not coincide: it has a closed-form
    solution [1]_ and serves as a benchmark for discretizations of
    curvilinear geometry with strong boundary layers [2]_. The outer
    circle of radius R is centred on the origin, the inner one of
    radius R0 is moved down to ``(0, -e)``, as the reference draws it,
    so the geometry is symmetric about the y axis with the narrow gap
    at the bottom. Nodes are laid on the two circles
    first, as many as keep them a spacing apart -- the outer ones
    first, marked circle (1), then the inner, marked hole (6) -- and
    the interior is filled by `pointclouds.poisson.PoissonDisk` seeded
    with them, with what lands outside the annulus discarded. `Disk`
    makes the concentric annulus deterministically, in rings or in a
    spiral.

    Parameters
    ----------
    radius : float
        R of the outer circle, about the origin.
    spacing : float, default 1.0
        The distance h between neighbouring nodes.
    hole : float
        R0 of the inner circle.
    eccentricity : float, default 0.0
        The distance e of the inner centre below the origin, along -y;
        0 is the concentric annulus, scattered.
    candidates : int, default 100
        The number of throws a node makes before it is retired.
    seed : int, optional
        Of the sample; random when not given.

    Raises
    ------
    ValueError
        For a radius, spacing or hole that is not positive, a negative
        eccentricity, a hole smaller than the spacing, and a gap
        ``R - e - R0`` of less than a spacing at its narrowest.

    Notes
    -----
    Trask, Maxey and Hu [2]_ compute the Wannier flow between cylinders
    of radii ``R0 = pi/10`` and ``R = pi/2`` at the eccentricity
    ``e = pi/5``, rotating at 1 and 1/2; those are the defaults of
    tools/eccentric_annulus.py. The interior sample keeps the spacing
    from the circle nodes, so the nearest interior nodes sit about a
    spacing off the circles; a node generator with repulsive relaxation
    can take the cloud as its starting point.

    References
    ----------
    .. [1] Wannier, "A contribution to the hydrodynamics of
       lubrication," Q. Appl. Math. 8, 1-19, 1950.
    .. [2] Trask, Maxey and Hu, "Compact moving least squares: an
       optimization framework for generating high-order compact
       meshless discretizations," J. Comput. Phys. 326, 596-611, 2016,
       doi:10.1016/j.jcp.2016.08.045.
    """

    def __init__(
        self,
        radius,
        spacing=1.0,
        *,
        hole,
        eccentricity=0.0,
        candidates=100,
        seed=None,
    ):
        from .poisson import PoissonDisk  # compiled by numba, only when needed

        if radius <= 0.0 or spacing <= 0.0 or hole <= 0.0:
            raise ValueError("the radius, the spacing and the hole must be positive")
        if eccentricity < 0.0:
            raise ValueError("the eccentricity cannot be negative: it points along -y")
        if hole < spacing:
            raise ValueError(
                f"a hole of radius {hole:g} is smaller than the spacing "
                f"{spacing:g} between nodes"
            )
        gap = radius - eccentricity - hole
        if gap < spacing:
            raise ValueError(
                f"the hole of radius {hole:g} at eccentricity {eccentricity:g} "
                f"leaves a gap of {gap:g} at its narrowest inside a radius of "
                f"{radius:g}, less than the spacing {spacing:g}"
            )
        centre = np.array([0.0, -eccentricity])
        seeds, m = [], []
        for r, c, marker in ((radius, 0.0, Marker.circle), (hole, centre, Marker.hole)):
            n = int(np.pi / np.arcsin(spacing / (2 * r)))  # chords of at least h
            phi = 2 * np.pi * np.arange(n) / n
            seeds.append(c + r * np.column_stack((np.cos(phi), np.sin(phi))))
            m.append(np.full(n, marker))
        seeds, m = np.vstack(seeds), np.concatenate(m)
        lo = -radius - 0.5 * spacing
        sampler = PoissonDisk(
            spacing,
            np.full(2, 2 * radius + spacing),
            periodic=False,
            ncandidates=candidates,
            seed=seed,
        )
        sampler.add_points(seeds - lo)
        sampler.fill_space()
        drawn = sampler.points[len(seeds) :] + lo
        keep = np.hypot(*drawn.T) < radius
        keep &= np.hypot(*(drawn - centre).T) > hole
        super().__init__(
            np.vstack((seeds, drawn[keep])),
            np.concatenate((m, np.full(np.count_nonzero(keep), Marker.interior))),
            title=f"eccentric annulus, radius={radius:g}, hole={hole:g}, "
            f"eccentricity={eccentricity:g}, spacing={spacing:g}",
        )


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
       313, 474-497, 2017, doi:10.1016/j.amc.2017.06.006.
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
    geometry : {"periodic", "channel", "box"}
        Both sides periodic, as in the Taylor-Green test; periodic in
        x with walls at ``y = 0`` and ``y = N``, as in the Poiseuille
        test; or walls on all four sides, as in the Poisson tests on
        the square.
    seed : int, optional
        Of the displacements; random when not given.

    Notes
    -----
    On the channel the grid has N + 1 rows, the first on the bottom wall
    (marker south) and the last on the top (marker north); a wall node
    slides along its wall but does not leave it. The nodes come row by row
    from ``y = 0``, x fastest, so the wall nodes are the first N and the
    last N.

    The box has N + 1 rows and columns, walled at 0 and N on either
    axis, the markers south (1), east (2), north (3) and west (4) by
    the outward normal and corner (5) where two walls meet; a wall node
    slides along its wall and a corner stays put. The nodes keep the
    row-by-row order, so the east and west nodes lie in their rows.

    References
    ----------
    .. [1] Strzelczyk and Matyka, "How nodes layout, refinement and
       velocity discretization influence convergence of the meshless
       lattice Boltzmann method," 2022, https://ssrn.com/abstract=4070398
    """

    def __init__(self, n, sigma=0.2, *, geometry="periodic", seed=None):
        periodic = (geometry != "box", geometry == "periodic")
        box = float(n)
        x = np.arange(float(n if periodic[0] else n + 1))
        y = np.arange(float(n if periodic[1] else n + 1))
        xv, yv = np.meshgrid(x, y)  # row by row, x fastest
        pts = np.column_stack((xv.ravel(), yv.ravel()))
        never = np.zeros(len(pts), bool)
        s = never if periodic[1] else pts[:, 1] == 0.0
        nn = never if periodic[1] else pts[:, 1] == box
        e = never if periodic[0] else pts[:, 0] == box
        w = never if periodic[0] else pts[:, 0] == 0.0
        m = np.select(
            [(s | nn) & (e | w), s, e, nn, w],
            [Marker.corner, Marker.south, Marker.east, Marker.north, Marker.west],
            Marker.interior,
        )
        d = np.random.default_rng(seed).uniform(-sigma, sigma, pts.shape)
        d[s | nn, 1] = 0.0  # a wall node stays on its wall
        d[e | w, 0] = 0.0
        pts += d  # NodeSet wraps the periodic sides
        for axis in range(2):
            if not periodic[axis]:  # only reached by sigma >= 1
                pts[:, axis] = np.clip(pts[:, axis], 0.0, box)
        super().__init__(
            pts,
            m,
            extent=(box, box),
            periodic=periodic,
            title=f"perturbed grid, size={n}x{n}, sigma={sigma:g}, {geometry}",
        )


class PoissonBox(NodeSet):
    """A Poisson disk sample of a box, periodic or walled, with or
    without a hole.

    No two nodes are closer than `distance` and no room is left for
    another one, drawn by `pointclouds.poisson.PoissonDisk`: about
    ``0.65 / distance**2`` nodes per unit area with the default number of
    candidates, more with more of them up to a point.

    Parameters
    ----------
    extent : (2,) array_like
        The box ``[0, Lx) x [0, Ly)``, closed on the walls.
    distance : float, default 1.0
        The least distance between two nodes.
    boundary : {"periodic", "channel", "walls"}
        Periodic on both sides; periodic in x with walls at ``y = 0``
        and ``y = Ly``, a box walled on two sides; or walls on all
        four, as in the scattered tests on the square.
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
        image across a periodic side, or to a wall.

    Notes
    -----
    With a hole, nodes are laid on its circle first, as many as keep them
    `distance` apart, with the marker hole; the sample grows from them, so
    the nearest nodes sit about a spacing off the circle; and what lands
    inside is discarded. In the periodic box that is the unit cell of a
    square array of cylinders.

    With walls, the four walls are laid first the same way, nodes about
    `distance` apart from the corners inwards, marked south (1), east
    (2), north (3) and west (4) by the outward normal and corner (5) at
    the corners; on the channel only the two walls, south and north,
    their nodes evenly around the periodic x. The wall nodes come
    before the hole's, and the sample grows from them all. The sampler
    works on a box grown by half a distance beyond every wall, so that
    a wall node keeps its 5 by 5 search cells; nothing fits in the
    margin, since everything there is within the distance of a wall
    node.
    """

    def __init__(
        self,
        extent,
        distance=1.0,
        *,
        boundary="periodic",
        candidates=100,
        hole=None,
        seed=None,
    ):
        from .poisson import PoissonDisk  # compiled by numba, only when needed

        extent = np.asarray(extent, float)
        if extent.min() < 2 * distance:
            raise ValueError(
                f"the box is narrower than twice the distance {distance:g}: no room"
            )
        periodic = {
            "periodic": (True, True),
            "channel": (True, False),
            "walls": (False, False),
        }[boundary]
        centre = 0.5 * extent
        seeds, m = [np.empty((0, 2))], [np.empty(0, int)]
        if boundary == "walls":
            corners = np.array([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]) * extent
            walls = (Marker.south, Marker.east, Marker.north, Marker.west)
            for a, b, wall in zip(corners[:-1], corners[1:], walls):
                k = max(int(np.hypot(*(b - a)) / distance), 1)  # steps of >= d
                seeds.append(a + (np.arange(k) / k)[:, None] * (b - a))
                mk = np.full(k, wall)
                mk[0] = Marker.corner
                m.append(mk)
        elif boundary == "channel":
            k = max(int(extent[0] / distance), 1)  # steps of >= d around x
            x = extent[0] * np.arange(k) / k
            for y, wall in ((0.0, Marker.south), (extent[1], Marker.north)):
                seeds.append(np.column_stack((x, np.full(k, y))))
                m.append(np.full(k, wall))
        if hole is not None:
            if hole < distance:
                raise ValueError(
                    f"a hole of radius {hole:g} is smaller than the distance "
                    f"{distance:g} between nodes"
                )
            for axis, wraps in enumerate(periodic):
                if wraps and 2 * hole + distance > extent[axis]:
                    raise ValueError(
                        f"a hole of radius {hole:g} leaves less than the "
                        f"distance {distance:g} to its image across the "
                        f"periodic sides of a box {extent[axis]:g} across"
                    )
                if not wraps and 2 * (hole + distance) > extent[axis]:
                    raise ValueError(
                        f"a hole of radius {hole:g} leaves less than the "
                        f"distance {distance:g} to the walls of a box "
                        f"{extent[axis]:g} across"
                    )
            n = int(np.pi / np.arcsin(distance / (2 * hole)))  # chords of at least d
            phi = 2 * np.pi * np.arange(n) / n
            seeds.append(centre + hole * np.column_stack((np.cos(phi), np.sin(phi))))
            m.append(np.full(n, Marker.hole))
        seeds, m = np.vstack(seeds), np.concatenate(m)
        pad = np.where(periodic, 0.0, 0.5 * distance)
        sampler = PoissonDisk(
            distance,
            extent + 2 * pad,
            periodic=periodic,
            ncandidates=candidates,
            seed=seed,
        )
        sampler.add_points(seeds + pad)
        sampler.fill_space()
        pts = sampler.points - pad
        pts[: len(seeds)] = seeds  # exactly, without the padding round trip
        title = (
            f"{'walled' if boundary == 'walls' else boundary} poisson, "
            f"size={extent[0]:g}x{extent[1]:g}, distance={distance:g}"
        )
        if hole is not None:
            inside = np.hypot(*(pts - centre).T) < hole
            inside[: len(seeds)] = False  # the circle nodes sit on the hole, not in it
            pts = pts[~inside]
            title += f", hole={hole:g}"
        markers = np.full(len(pts), Marker.interior)
        markers[: len(seeds)] = m
        super().__init__(pts, markers, extent=extent, periodic=periodic, title=title)


class PolarRegion(NodeSet):
    """The region between two polar curves, Poisson-disk sampled.

    The star-shaped test domains of the RBF-FD literature: the region
    ``r_in(theta) <= r <= r_out(theta)`` between two closed curves
    about the origin, each a constant or a trigonometric polynomial

    .. math:: r(\\theta) = c_0 + \\sum_k c_k \\cos k \\theta
                               + \\sum_k s_k \\sin k \\theta.

    Bayona, Flyer, Fornberg and Barnett [1]_ solve elliptic equations
    with variable coefficients between the curves
    ``3/10 + sin(t)/10 + 3 sin(5t)/20`` and
    ``1 + cos(t)/5 + 3 sin(4t)/20``, the defaults of
    tools/polar_region.py. Nodes are laid along the curves first,
    evenly in arc length, then the interior is filled by
    `pointclouds.poisson.PoissonDisk` seeded with them over the
    bounding box of the outer curve, and what lands outside the region
    is discarded. The nodes of the outer curve come first, marked
    circle (1), then those of the inner one, marked hole (6).

    Parameters
    ----------
    outer : float or pair of sequences
        The outer curve: the radius of a circle, or the coefficients
        ``((c0, c1, ...), (s1, s2, ...))`` of its cosines and sines,
        either of which may be empty.
    spacing : float, default 1.0
        The distance h between neighbouring nodes.
    inner : float or pair of sequences, optional
        The inner curve, in the same form; the whole region inside
        `outer` without it.
    candidates : int, default 100
        The number of throws a node makes before it is retired.
    seed : int, optional
        Of the sample; random when not given.

    Raises
    ------
    ValueError
        For a curve that is not star-shaped about the origin (a radius
        of 0 or less at some angle), curves within a spacing of each
        other along some ray, or a curve too small for the spacing.

    Notes
    -----
    A curve is sampled densely, its arc length accumulated along the
    chords and divided evenly, which closes the loop exactly; where the
    curve bends sharply a chord falls short of its arc, so the node
    count is lowered until neighbouring nodes are at least a spacing
    apart. A feature narrower than the spacing, such as the hairpin
    lobe of the Bayona inner curve, still forces the nodes of its two
    sides together in the plane; of any two closer than the spacing the
    later one is dropped, so such a feature keeps the nodes the spacing
    has room for. The interior sample keeps the spacing from the curve nodes,
    so the nearest interior nodes sit about a spacing inside the
    curves; cropping the sample to the region is a radial comparison
    at the angle of a point, which is what needs the curves to be
    star-shaped. A node generator with repulsive relaxation can take
    the cloud as its starting point.

    References
    ----------
    .. [1] Bayona, Flyer, Fornberg and Barnett, "On the role of
       polynomials in RBF-FD approximations: II. Numerical solution of
       elliptic PDEs," J. Comput. Phys. 332, 257-273, 2017.
    """

    DENSE = 4096  # samples of a curve for its arc length

    def __init__(self, outer, spacing=1.0, *, inner=None, candidates=100, seed=None):
        from .poisson import PoissonDisk  # compiled by numba, only when needed

        if spacing <= 0.0:
            raise ValueError("the spacing must be positive")
        theta = 2 * np.pi * np.arange(self.DENSE) / self.DENSE
        rout = self._radius(outer, theta)
        curves = [outer] if inner is None else [outer, inner]
        for curve in curves:
            if self._radius(curve, theta).min() <= 0.0:
                raise ValueError(
                    "a curve must be star-shaped about the origin: "
                    "r(theta) positive throughout"
                )
        if inner is not None:
            gap = (rout - self._radius(inner, theta)).min()
            if gap < spacing:
                raise ValueError(
                    f"the curves come within {gap:g} of each other along "
                    f"some ray, less than the spacing {spacing:g}"
                )
        pts = [self._curve(curve, spacing) for curve in curves]
        m = [np.full(len(p), mk) for p, mk in zip(pts, (Marker.circle, Marker.hole))]
        seeds, m = np.vstack(pts), np.concatenate(m)
        keep = self._thin(seeds, spacing)
        seeds, m = seeds[keep], m[keep]
        outline = self._polyline(outer)
        lo = outline.min(axis=0) - 0.5 * spacing
        sampler = PoissonDisk(
            spacing,
            outline.max(axis=0) + 0.5 * spacing - lo,
            periodic=False,
            ncandidates=candidates,
            seed=seed,
        )
        sampler.add_points(seeds - lo)
        sampler.fill_space()
        drawn = sampler.points[len(seeds) :] + lo
        rho, phi = np.hypot(*drawn.T), np.arctan2(drawn[:, 1], drawn[:, 0])
        keep = rho < self._radius(outer, phi)
        if inner is not None:
            keep &= rho > self._radius(inner, phi)
        title = f"polar region, outer={self._name(outer)}"
        if inner is not None:
            title += f", inner={self._name(inner)}"
        super().__init__(
            np.vstack((seeds, drawn[keep])),
            np.concatenate((m, np.full(np.count_nonzero(keep), Marker.interior))),
            title=title + f", spacing={spacing:g}",
        )

    @staticmethod
    def _radius(curve, theta):
        """Evaluates r(theta): a constant, or the trigonometric series."""
        if isinstance(curve, (int, float, np.integer, np.floating)):
            return np.full_like(theta, float(curve))
        cos, sin = (np.atleast_1d(np.asarray(c, float)) for c in curve)
        r = (cos * np.cos(np.multiply.outer(theta, np.arange(len(cos))))).sum(-1)
        return r + (
            sin * np.sin(np.multiply.outer(theta, 1 + np.arange(len(sin))))
        ).sum(-1)

    @staticmethod
    def _name(curve):
        """Writes a curve into a title: its radius, or its coefficients."""
        if isinstance(curve, (int, float, np.integer, np.floating)):
            return f"{curve:g}"
        cos, sin = curve
        return (
            f"cos({','.join(f'{c:g}' for c in np.atleast_1d(cos))})"
            f"+sin({','.join(f'{s:g}' for s in np.atleast_1d(sin))})"
        )

    @staticmethod
    def _thin(pts, h):
        """Keeps the earlier of any two points closer than h.

        A curve feature narrower than the spacing forces the nodes on
        its sides together in the plane however they are spread along
        the arc; what survives is pairwise at least h apart.
        """
        drop = np.zeros(len(pts), bool)
        for i, j in sorted(cKDTree(pts).query_pairs(h * (1.0 - 1e-12))):
            if not drop[i]:
                drop[j] = True
        return ~drop

    @classmethod
    def _polyline(cls, curve):
        """Draws the curve as a dense closed polyline."""
        theta = 2 * np.pi * np.arange(cls.DENSE + 1) / cls.DENSE
        r = cls._radius(curve, theta)
        return r[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))

    @classmethod
    def _curve(cls, curve, h):
        """Places nodes along the curve, evenly in arc length, h apart.

        The arc length accumulates along the chords of the dense
        polyline and is divided evenly; the node count is lowered while
        any two neighbours are closer than h, since a chord cuts a
        bending arc short.
        """
        xy = cls._polyline(curve)
        s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))))
        theta = 2 * np.pi * np.arange(cls.DENSE + 1) / cls.DENSE
        n = int(s[-1] / h)
        while n >= 3:
            t = np.interp(np.arange(n) * s[-1] / n, s, theta)
            r = cls._radius(curve, t)
            pts = r[:, None] * np.column_stack((np.cos(t), np.sin(t)))
            gaps = np.hypot(*np.diff(np.vstack((pts, pts[:1])), axis=0).T)
            if gaps.min() >= h:
                return pts
            n -= 1
        raise ValueError(f"a curve is too small for the spacing {h:g}")


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
