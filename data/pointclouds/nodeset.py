"""A point cloud with a marker per node and the box it lives in: what the
generators build, the tools read, and both write, search and draw.

    cloud = NodeSet.read("case.node")          # or a child from pointclouds.generators
    ia, ja = cloud.stencils("knn", 18)         # the stencil graph, CSR
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

from .io import read_nodes, write_graph, write_node, write_points
from .periodic import minimum_image, wrap
from .stencils import select_stencils


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


class NodeSet:
    """A point cloud: `points` (n, 2); a marker per node in `markers`, 0
    for an interior node; the box `extent` = (Lx, Ly) it lives in, or
    None for the open plane; `periodic` = (px, py), which sides of the
    box are one; and a `title` in words, the comment of its node file.
    `boxsize` is the side of every periodic axis and 0 for the others,
    as scipy's k-d tree takes it."""

    def __init__(
        self, points, markers=None, extent=None, periodic=(False, False), title=""
    ):
        self.points = np.array(points, float).reshape(-1, 2)
        n = len(self.points)
        self.markers = np.zeros(n, int) if markers is None else np.asarray(markers, int)
        self.extent = None if extent is None else (float(extent[0]), float(extent[1]))
        self.periodic = (bool(periodic[0]), bool(periodic[1]))
        self.title = title
        if any(self.periodic) and self.extent is None:
            raise ValueError("a periodic cloud needs its box")
        wraps = np.array(self.periodic)
        self.boxsize = list(np.where(wraps, self.extent or (0.0, 0.0), 0.0))
        if wraps.any():
            self.points[:, wraps] = wrap(
                self.points[:, wraps], np.array(self.extent)[wraps]
            )

    def __len__(self):
        return len(self.points)

    @classmethod
    def read(cls, fname, extent=None, periodic=(False, False)):
        """The cloud of a points or node file, told apart by extension."""
        return cls(*read_nodes(fname), extent, periodic, title=f"from {fname}")

    def stencils(self, method, value):
        """The stencil graph in CSR form, (ia, ja), with the stencil of
        node i at ja[ia[i]:ia[i + 1]]; see pointclouds.stencils."""
        return select_stencils(self.points, self.boxsize, method, value)

    def neighbours(self, k):
        """Indices and distances of the k nearest other nodes of every
        node, nearest first."""
        d, j = cKDTree(self.points, boxsize=self.boxsize).query(
            self.points, k + 1, workers=-1
        )
        twin_first = j[:, 0] != np.arange(len(self))  # a coincident node may lead
        j[twin_first, 1], d[twin_first, 1] = j[twin_first, 0], d[twin_first, 0]
        return j[:, 1:], d[:, 1:]

    def write(self, stem, ext=".points", graph=None):
        """The points file, or with `ext` ".node" the node file with the
        markers and the title as its comment, and the graph file when a
        graph is given, all under `stem`."""
        if ext == ".node":
            write_node(stem, self.points, self.markers, self.title)
        else:
            write_points(stem, self.points)
        if graph is not None:
            write_graph(stem, *graph)

    def summary(self, graph=None):
        """One line: the node counts, the density over the box if there
        is one, and the graph if given."""
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
        """The nodes coloured by marker, optionally every index and the
        given (node, members) stencils: a circle about the node through
        its farthest member, the members ringed and the node filled, a
        wrapped member at its nearest image."""
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
        for s, (i, members) in enumerate(stencils):
            from matplotlib.patches import Circle

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
    """MX by MY copies of a cloud periodic in both axes, side by side on
    the box MX Lx by MY Ly: the copies row by row from the origin with x
    fastest, and the boundary nodes of all the copies ahead of the
    interior ones. The copies join without a seam, since the cloud is
    periodic, so the stencils are searched over the whole tiling."""

    def __init__(self, cloud, tiles):
        if cloud.periodic != (True, True):
            raise ValueError("only a cloud periodic in both axes tiles without a seam")
        mx, my = tiles
        extent = np.array(cloud.extent)
        shifts = [(ix, iy) for iy in range(my) for ix in range(mx)]
        pts = np.vstack([cloud.points + extent * shift for shift in shifts])
        m = np.tile(cloud.markers, len(shifts))
        first = np.argsort(m == MARKERS.interior, kind="stable")  # the boundary first
        super().__init__(
            pts[first],
            m[first],
            extent * tiles,
            (True, True),
            f"{cloud.title}, tile={mx}x{my}",
        )
        self.tile, self.tiles = cloud.extent, (mx, my)

    def plot(self, ax, labels=False, stencils=()):
        """The cloud, with the outline of the tiles."""
        super().plot(ax, labels, stencils)
        (lx, ly), (mx, my) = self.tile, self.tiles
        ax.vlines(lx * np.arange(mx + 1), 0, my * ly, color="0.7", lw=0.8)
        ax.hlines(ly * np.arange(my + 1), 0, mx * lx, color="0.7", lw=0.8)
