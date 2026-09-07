"""A point cloud with a marker per node and the box it lives in: what the
generators build, the tools read, and both write, search and draw.

    cloud = PoissonBox((32, 32), seed=1)       # a child, pointclouds.generators
    ia, ja = cloud.stencils()                  # the stencil graph, CSR
    cloud.write("case", ".node", (ia, ja))     # case.node and case.graph

The box is [0, Lx) x [0, Ly), and `periodic` says of each axis whether
its two sides are one: the points are brought into the box through the
periodic sides, and the stencil search wraps around them. A cloud read
from a file has no box unless given one. TiledNodeSet lays copies of a
cloud periodic in both axes side by side. The boundary markers of the
node files are fixed here too, as MARKERS.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .io import write_graph, write_node, write_points
from .periodic import minimum_image, wrap
from .stencils import KNN, select_stencils


@dataclass(frozen=True)
class Markers:
    """The boundary markers of the node files: 0 for an interior node,
    the walls numbered counter-clockwise from the bottom, then the
    corners of a cavity, then a hole in the interior. A generator uses
    the ones its geometry has: a periodic side has no wall and so no
    marker, and only a cavity has corners. MARKERS is the one instance,
    and frozen: its fields cannot be reassigned."""

    interior: int = 0
    south: int = 1
    east: int = 2
    north: int = 3
    west: int = 4
    corner: int = 5
    hole: int = 6


MARKERS = Markers()


def boundary_first(markers):
    """Return the permutation that puts the boundary nodes ahead of the interior ones, each group in its order."""
    return np.argsort(markers == MARKERS.interior, kind="stable")


class NodeSet:
    """A point cloud with a marker per node and the box it lives in.

    Parameters
    ----------
    points : (n, 2) array_like
        The coordinates.
    markers : (n,) array_like of int, optional
        The boundary marker of every node, 0 for an interior one; all
        interior when not given.
    extent : (2,) tuple of float, optional
        The box ``[0, Lx) x [0, Ly)``, or None for the open plane.
    periodic : (2,) tuple of bool
        Which sides of the box are one. The points are brought into the
        box through those sides, and the stencil search wraps around them.
    title : str
        What the cloud is, in words: the comment of its node file.

    Attributes
    ----------
    boxsize : list of float
        The side of every periodic axis and 0 for the others, as scipy's
        k-d tree takes it.
    """

    def __init__(
        self, points, markers=None, *, extent=None, periodic=(False, False), title=""
    ):
        self.points = np.array(points, float).reshape(-1, 2)
        n = len(self.points)
        self.markers = np.zeros(n, int) if markers is None else np.asarray(markers, int)
        self.extent = None if extent is None else (float(extent[0]), float(extent[1]))
        self.periodic = (bool(periodic[0]), bool(periodic[1]))
        self.title = title
        if any(self.periodic) and self.extent is None:
            raise ValueError("a periodic cloud needs its box")
        self.boxsize = [
            L if p else 0.0 for L, p in zip(self.extent or (0, 0), self.periodic)
        ]
        self.points = wrap(self.points, self.boxsize)

    def __len__(self):
        return len(self.points)

    def stencils(self, method="knn", value=KNN):
        """Select the stencil of every node.

        Parameters
        ----------
        method : {"knn", "radius", "range"}
            The nearest nodes, the nodes within a distance, or those within a
            square; see `pointclouds.stencils`.
        value : int or float
            The stencil size for knn, the distance for the others.

        Returns
        -------
        ia, ja : ndarray
            The stencils in CSR form, that of node i at ``ja[ia[i]:ia[i + 1]]``,
            the node itself first.
        """
        return select_stencils(self.points, self.boxsize, method, value)

    def nearest(self):
        """Return the index and the distance of the nearest other node of every node."""
        d, j = cKDTree(self.points, boxsize=self.boxsize).query(
            self.points, 2, workers=-1
        )
        twin_first = j[:, 0] != np.arange(len(self))  # a coincident node may lead
        j[twin_first, 1], d[twin_first, 1] = j[twin_first, 0], d[twin_first, 0]
        return j[:, 1], d[:, 1]

    def write(self, stem, ext=".points", graph=None):
        """Write the cloud, and its stencil graph if given.

        Parameters
        ----------
        stem : str
            The files are ``stem + ext`` and ``stem + ".graph"``.
        ext : {".points", ".node"}
            A points file, or a node file with the markers and the title.
        graph : tuple of ndarray, optional
            ``(ia, ja)`` from `stencils`.
        """
        if ext == ".node":
            write_node(stem, self.points, self.markers, self.title)
        else:
            write_points(stem, self.points)
        if graph is not None:
            write_graph(stem, *graph)

    def summary(self, graph=None):
        """Return one line on the cloud: the node counts, the density over the box if there is one, and the graph if given."""
        n, boundary = len(self), int(np.count_nonzero(self.markers))
        text = f"{n} nodes ({n - boundary} interior, {boundary} boundary)"
        if self.extent is not None:
            text += f", {n / np.prod(self.extent):.3g} per unit area of the box"
        if graph is not None:
            ia, _ = graph
            sizes = np.diff(ia)
            text += (
                f", {ia[-1]} edges in the stencil graph, "
                f"stencils of {sizes.min()} to {sizes.max()} nodes"
            )
        return text

    PALETTE = [
        "tab:red",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:olive",
    ]

    def plot(self, ax, labels=False, stencils=()):
        """Draw the nodes on an axes, coloured by marker.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : bool
            Write the index next to every node.
        stencils : list of (int, array_like)
            Nodes with their stencil members, each drawn as a circle about the
            node through its farthest member, the members ringed and the node
            filled; a wrapped member sits at its nearest image.
        """
        xy, marker = self.points, self.markers
        interior = marker == MARKERS.interior
        if interior.any():
            ax.plot(
                *xy[interior].T,
                ".",
                color="0.55",
                ms=3,
                label=f"interior ({interior.sum()})",
            )
        for m in np.unique(marker[~interior]):
            sel = marker == m
            ax.plot(*xy[sel].T, "o", ms=3.5, label=f"marker {m} ({sel.sum()})")
        if labels:
            for i, (x, y) in enumerate(xy):
                ax.annotate(
                    str(i),
                    (x, y),
                    xytext=(2, 2),
                    textcoords="offset points",
                    fontsize=6,
                )
        from matplotlib.patches import Circle

        for s, (i, members) in enumerate(stencils):
            color = self.PALETTE[s % len(self.PALETTE)]
            centre = xy[i]
            at = centre + minimum_image(xy[members] - centre, self.boxsize)
            r = np.linalg.norm(at - centre, axis=1).max()
            ax.add_patch(Circle(centre, r, fill=False, edgecolor=color, lw=1.2))
            ax.plot(*at.T, "o", ms=8, mfc="none", mec=color, mew=1.2)
            ax.plot(
                *centre,
                "o",
                ms=8,
                color=color,
                label=f"stencil of node {i} ({len(members)})",
            )
            ax.annotate(
                str(i),
                centre,
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8, markerscale=1.5)


class TiledNodeSet(NodeSet):
    """Copies of a cloud periodic in both axes, laid side by side.

    The copies go row by row from the origin with x fastest, on the box
    ``MX Lx`` by ``MY Ly``, with the boundary nodes of all the copies ahead
    of the interior ones. They join without a seam, since the cloud is
    periodic, so the stencils are searched over the whole tiling.

    Parameters
    ----------
    cloud : NodeSet
        Periodic in both axes.
    tiles : (2,) tuple of int
        The copies along x and along y, ``(MX, MY)``.
    """

    def __init__(self, cloud, tiles):
        if cloud.periodic != (True, True):
            raise ValueError("only a cloud periodic in both axes tiles without a seam")
        mx, my = tiles
        extent = np.array(cloud.extent)
        shifts = [(ix, iy) for iy in range(my) for ix in range(mx)]
        pts = np.vstack([cloud.points + extent * shift for shift in shifts])
        m = np.tile(cloud.markers, len(shifts))
        first = boundary_first(m)
        super().__init__(
            pts[first],
            m[first],
            extent=extent * tiles,
            periodic=(True, True),
            title=f"{cloud.title}, tile={mx}x{my}",
        )
        self.tiles = (mx, my)

    def plot(self, ax, **kwargs):
        """The cloud, with the outline of the tiles."""
        super().plot(ax, **kwargs)
        (lx, ly), (mx, my) = np.array(self.extent) / self.tiles, self.tiles
        ax.vlines(lx * np.arange(mx + 1), 0, my * ly, color="0.7", lw=0.8)
        ax.hlines(ly * np.arange(my + 1), 0, mx * lx, color="0.7", lw=0.8)
