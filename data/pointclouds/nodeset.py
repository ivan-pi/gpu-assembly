"""Point clouds with a marker per node and the box they live in.

`NodeSet` holds the points, the markers and the box, and knows how to
select its stencils, write its files and draw itself; the generators
in `pointclouds.generators` are its children. `TiledNodeSet` lays
copies of a periodic cloud side by side. `Marker` fixes the marker
convention of the node files.
"""

from enum import IntEnum

import numpy as np
from scipy.spatial import cKDTree

from .io import write_graph, write_node, write_points
from .periodic import minimum_image, wrap
from .stencils import KNN, select_stencils


class Marker(IntEnum):
    """The boundary markers of the node files.

    An interior node has the marker 0, the walls are numbered
    counter-clockwise from the bottom, then come the corners of a cavity
    and a hole in the interior. A generator uses the ones its geometry
    has. A member is an int, so it compares with a marker read from a
    file and fills a numpy array.

    `circle` is another name for the marker 1, the outer boundary of a
    disk or an annulus, which is what the node sets read off the papers
    (data/README.md) carry on it: a round domain has no south wall, and
    the two never meet in one cloud.
    """

    interior = 0
    south = 1
    circle = 1
    east = 2
    north = 3
    west = 4
    corner = 5
    hole = 6


def boundary_first(markers):
    """Returns the permutation that puts the boundary nodes first.

    Parameters
    ----------
    markers : (n,) array_like of int
        The marker of every node, 0 for an interior one.

    Returns
    -------
    (n,) ndarray of int
        Node indices, the boundary first, each group in its order.
    """
    return np.argsort(markers == Marker.interior, kind="stable")


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
    periodic : (2,) tuple of bool, default (False, False)
        Which axes are periodic: the points are wrapped into the box
        along them, and the stencil search wraps around them.
    title : str, optional
        What the cloud is, in words: the comment of its node file.

    Raises
    ------
    ValueError
        For a periodic cloud given without its box.

    Attributes
    ----------
    points : (n, 2) ndarray
    markers : (n,) ndarray of int
    extent : tuple of float or None
    periodic : tuple of bool
    title : str
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
        """Selects the stencil of every node.

        Parameters
        ----------
        method : {"knn", "radius", "range"}
            The nearest nodes, the nodes within a distance, or those within
            a square.
        value : int or float
            The stencil size for knn, the distance for the others.

        Returns
        -------
        Graph
            The stencil of node i is ``graph.stencil(i)``, itself first.

        See Also
        --------
        pointclouds.stencils.select_stencils :
            The search, and what it refuses.
        """
        return select_stencils(self.points, self.boxsize, method, value)

    def nearest(self):
        """Finds the nearest other node of every node.

        Returns
        -------
        index : (n,) ndarray of int
        distance : (n,) ndarray of float
        """
        d, j = cKDTree(self.points, boxsize=self.boxsize).query(
            self.points, 2, workers=-1
        )
        twin_first = j[:, 0] != np.arange(len(self))  # a coincident node may lead
        j[twin_first, 1], d[twin_first, 1] = j[twin_first, 0], d[twin_first, 0]
        return j[:, 1], d[:, 1]

    def write(self, stem, ext=".points", graph=None):
        """Writes the cloud, and its stencil graph if given.

        Parameters
        ----------
        stem : str
            The files are ``stem + ext`` and ``stem + ".graph"``.
        ext : {".points", ".node"}
            A points file, or a node file with the markers and the title.
        graph : Graph, optional
            From `stencils`.
        """
        if ext == ".node":
            write_node(stem, self.points, self.markers, self.title)
        else:
            write_points(stem, self.points)
        if graph is not None:
            write_graph(stem, graph.ia, graph.ja)

    def summary(self, graph=None):
        """Describes the cloud in one line.

        Parameters
        ----------
        graph : Graph, optional
            From `stencils`, to describe it too.

        Returns
        -------
        str
            The node counts, the density over the box if there is one, and
            the edge count and the stencil sizes of the graph if given.
        """
        n, boundary = len(self), int(np.count_nonzero(self.markers))
        text = f"{n} nodes ({n - boundary} interior, {boundary} boundary)"
        if self.extent is not None:
            text += f", {n / np.prod(self.extent):.3g} per unit area of the box"
        if graph is not None:
            sizes = np.diff(graph.ia)
            text += (
                f", {graph.nnz} edges in the stencil graph, "
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

    def plot(self, ax, *, labels=False, stencils=()):
        """Draws the nodes on an axes, coloured by marker.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        labels : bool, default False
            Write the index next to every node.
        stencils : list of tuple, optional
            ``(node, members)`` pairs, each drawn as a circle about the
            node through its farthest member, the members ringed and the
            node filled; a wrapped member sits at its nearest image.
        """
        xy, marker = self.points, self.markers
        interior = marker == Marker.interior
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
    ``MX Lx`` by ``MY Ly``, with the boundary nodes of all the copies
    ahead of the interior ones. They join without a seam, since the cloud
    is periodic, so the stencils are searched over the whole tiling.

    Parameters
    ----------
    cloud : NodeSet
        Periodic in both axes.
    tiles : (2,) tuple of int
        The copies along x and along y, ``(MX, MY)``.

    Raises
    ------
    ValueError
        For a cloud that is not periodic in both axes.

    Attributes
    ----------
    tiles : tuple of int
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
        """Draws the cloud with the outline of the tiles."""
        super().plot(ax, **kwargs)
        (lx, ly), (mx, my) = np.array(self.extent) / self.tiles, self.tiles
        ax.vlines(lx * np.arange(mx + 1), 0, my * ly, color="0.7", lw=0.8)
        ax.hlines(ly * np.arange(my + 1), 0, mx * lx, color="0.7", lw=0.8)
