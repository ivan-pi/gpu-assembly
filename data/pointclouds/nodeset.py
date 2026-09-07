"""A point cloud with a marker per node and the box it lives in: what the
generators in data/gen build, the tools in data/tools read, and both
write, search and draw.

    cloud = NodeSet.read("case.node")          # or a child from pointclouds.generators
    ia, ja = cloud.stencils("knn", 18)         # the stencil graph, CSR
    cloud.write("case", ".node", (ia, ja))     # case.node and case.graph
    print(cloud.summary((ia, ja)))

The box is [0, Lx) x [0, Ly), and `periodic` says of each axis whether
its two sides are one: the stencil search and the distances wrap around
the periodic sides. A cloud read from a file has no box unless given
one. `Tiled` lays copies of a cloud periodic in both axes side by side
and is a cloud like any other. The boundary markers of the node files
are fixed here too, as MARKERS.
"""

import functools
from dataclasses import dataclass

import numpy as np

from .io import read_nodes, write_graph, write_node, write_points
from .stencils import select_stencils


@dataclass(frozen=True)
class Markers:
    """The boundary markers of the node files: 0 for an interior node,
    the walls numbered counter-clockwise from the bottom, then the
    corners of a cavity, then a hole in the interior. MARKERS is the one
    instance, and frozen: its fields cannot be reassigned."""

    interior: int = 0
    south: int = 1
    east: int = 2
    north: int = 3
    west: int = 4
    corner: int = 5
    hole: int = 6


MARKERS = Markers()


def wrap(z, box):
    """Coordinates z brought into [0, box) through the periodic side, for
    arrays, with box a scalar or one length per column."""
    z = np.mod(z, box)
    z[z >= box] = 0.0  # np.mod rounds up to the side
    return z


class NodeSet:
    """A point cloud: `points` (n, 2); a marker per node in `markers`, 0
    for an interior node; the box `extent` = (Lx, Ly) it lives in, or
    None for the open plane; `periodic` = (px, py), which sides of the
    box are one; a `title` in words, the comment of its node file; and
    `area`, what the nodes fill, for their density: the box unless a
    child says otherwise."""

    def __init__(
        self,
        points,
        markers=None,
        extent=None,
        periodic=(False, False),
        title="",
        area=None,
    ):
        self.points = np.asarray(points, float).reshape(-1, 2)
        n = len(self.points)
        self.markers = np.zeros(n, int) if markers is None else np.asarray(markers, int)
        self.extent = None if extent is None else (float(extent[0]), float(extent[1]))
        self.periodic = (bool(periodic[0]), bool(periodic[1]))
        if any(self.periodic) and self.extent is None:
            raise ValueError("a periodic cloud needs its box")
        self.title = title
        if area is None:
            if self.extent is not None:
                area = self.extent[0] * self.extent[1]
            else:
                area = float(np.prod(np.ptp(self.points, axis=0))) if n else 0.0
        self.area = area

    def __len__(self):
        return len(self.points)

    @classmethod
    def read(cls, fname, extent=None, periodic=(False, False)):
        """The cloud of a points or node file, told apart by extension."""
        pts, m = read_nodes(fname)
        return cls(pts, m, extent, periodic, title=fname)

    # ---------------------------------------------------------- the box

    @property
    def boxsize(self):
        """The side of every periodic axis and 0 for the others: what
        scipy's k-d tree takes."""
        if self.extent is None:
            return [0.0, 0.0]
        return [L if p else 0.0 for L, p in zip(self.extent, self.periodic)]

    def wrapped(self):
        """The points brought into the box through the periodic sides."""
        pts = self.points.copy()
        for a in (0, 1):
            if self.periodic[a]:
                pts[:, a] = wrap(pts[:, a], self.extent[a])
        return pts

    def minimum_image(self, d):
        """Displacements shortened through the periodic sides."""
        d = np.array(d, float)
        for a in (0, 1):
            if self.periodic[a]:
                d[..., a] -= self.extent[a] * np.round(d[..., a] / self.extent[a])
        return d

    # ------------------------------------------------------- neighbours

    def stencils(self, method, value, order="distance"):
        """The stencil graph in CSR form, (ia, ja), with the stencil of
        node i at ja[ia[i]:ia[i + 1]]; see pointclouds.stencils."""
        return select_stencils(self.wrapped(), self.boxsize, method, value, order)

    def neighbours(self, k, which=None):
        """Indices and distances of the k nearest other nodes of every
        node, or of the nodes in `which`, nearest first."""
        from scipy.spatial import cKDTree

        n = len(self)
        which = np.arange(n) if which is None else np.asarray(which)
        k = min(k, n - 1)
        if k < 1:
            return np.empty((len(which), 0), dtype=int), np.empty((len(which), 0))
        pts = self.wrapped()
        d, j = cKDTree(pts, boxsize=self.boxsize).query(pts[which], k + 1, workers=-1)
        return j[:, 1:], d[:, 1:]

    @functools.cached_property
    def nearest(self):
        """(index, distance) of the nearest other node of every node."""
        j, d = self.neighbours(1)
        return j.reshape(-1), d.reshape(-1)

    def coincident(self):
        """The problem with nodes at distance zero from another, or None."""
        j, d = self.nearest
        same = np.flatnonzero(d == 0)
        if not same.size:
            return None
        pairs = sorted({(min(a, j[a]), max(a, j[a])) for a in same})
        shown = ", ".join(f"({a}, {b})" for a, b in pairs[:5])
        more = f", ... {len(pairs)} pairs" if len(pairs) > 5 else ""
        return f"{self.title}: coincident nodes: {shown}{more}"

    def renumbered(self, order):
        """The same cloud with node order[i] at i."""
        return NodeSet(
            self.points[order],
            self.markers[order],
            self.extent,
            self.periodic,
            self.title,
            self.area,
        )

    # ------------------------------------------------------------ files

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

    # -------------------------------------------------------- reporting

    def summary(self, graph=None):
        """One line: the node counts, the density, and the graph if given."""
        n, boundary = len(self), int(np.count_nonzero(self.markers))
        text = (
            f"{n} nodes ({n - boundary} interior, {boundary} boundary), "
            f"{n / self.area:.3g} per unit area"
        )
        if graph is not None:
            ia, _ = graph
            sizes = np.diff(ia)
            text += (
                f", {ia[-1]} edges in the stencil graph, "
                f"stencils of {sizes.min()} to {sizes.max()} nodes"
            )
        return text

    def describe(self):
        """Print the counts, the markers, the box and the spacing statistics."""
        n, boundary = len(self), np.count_nonzero(self.markers)
        print(f"{self.title}: {n} nodes, {n - boundary} interior, {boundary} boundary")
        values, counts = np.unique(self.markers, return_counts=True)
        print("  markers: " + "  ".join(f"{v}: {c}" for v, c in zip(values, counts)))
        lo, hi = self.points.min(axis=0), self.points.max(axis=0)
        print(
            f"  bounding box: x in [{lo[0]:.6g}, {hi[0]:.6g}], y in [{lo[1]:.6g}, {hi[1]:.6g}]"
        )
        if any(self.periodic):
            outside = np.count_nonzero(
                ((self.points < 0) | (self.points >= self.extent)).any(axis=1)
            )
            sides = " x ".join(
                f"[0, {L:.6g})" if p else f"[0, {L:.6g}]"
                for L, p in zip(self.extent, self.periodic)
            )
            print(
                f"  periodic box {sides}: distances are minimum-image"
                + (f"; {outside} nodes lie outside the box" if outside else "")
            )
        if n > 1:
            j, d = self.nearest
            lo, hi = int(np.argmin(d)), int(np.argmax(d))
            print(
                f"  nearest-neighbour distance: min {d[lo]:.6g} (nodes {lo} and {j[lo]}), "
                f"max {d[hi]:.6g} (node {hi})"
            )
            print(
                f"    mean {d.mean():.6g}, median {np.median(d):.6g}, std {d.std():.6g}"
            )

    # ---------------------------------------------------------- drawing

    PALETTE = [
        "tab:red",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:olive",
        "tab:cyan",
    ]

    def plot(self, ax, labels=False, stencils=()):
        """The nodes coloured by marker, optionally every index and the
        given (node, members) stencils."""
        xy, marker = self.points, self.markers
        interior = marker == MARKERS.interior
        if interior.any():
            ax.plot(
                xy[interior, 0],
                xy[interior, 1],
                ".",
                color="0.55",
                ms=3,
                label=f"interior ({np.count_nonzero(interior)})",
            )
        for m in np.unique(marker[~interior]):
            sel = marker == m
            ax.plot(
                xy[sel, 0],
                xy[sel, 1],
                "o",
                ms=3.5,
                label=f"marker {m} ({np.count_nonzero(sel)})",
            )
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
            self._plot_stencil(ax, i, members, self.PALETTE[s % len(self.PALETTE)])
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8, markerscale=1.5)

    def _plot_stencil(self, ax, i, members, color):
        """A circle about node i through its farthest member, the members
        ringed and the node filled; a wrapped member sits at its nearest image."""
        from matplotlib.patches import Circle

        centre = self.points[i]
        at = centre + self.minimum_image(self.points[members] - centre)
        r = np.linalg.norm(at - centre, axis=1).max()
        ax.add_patch(Circle(centre, r, fill=False, edgecolor=color, lw=1.2))
        ax.plot(at[:, 0], at[:, 1], "o", ms=8, mfc="none", mec=color, mew=1.2)
        ax.plot(
            centre[0],
            centre[1],
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

    def show(self, graph=None, save=None):
        """A figure of the cloud, with the stencil of its middle node when
        a graph is given, shown or written to the file `save`."""
        import matplotlib.pyplot as plt

        stencils = []
        if graph is not None:
            ia, ja = graph
            i = len(self) // 2
            stencils = [(i, ja[ia[i] : ia[i + 1]])]
        fig, ax = plt.subplots(figsize=(6.5, 6))
        self.plot(ax, stencils=stencils)
        ax.set_title(self.title, fontsize=9)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=150)
        else:
            plt.show()


class Tiled(NodeSet):
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
        boundary_first = np.argsort(m == MARKERS.interior, kind="stable")
        super().__init__(
            wrap(pts[boundary_first], extent * tiles),
            m[boundary_first],
            extent * tiles,
            (True, True),
            f"{cloud.title}, tile={mx}x{my}",
            cloud.area * mx * my,
        )
        self.tile, self.tiles = cloud.extent, (mx, my)

    def plot(self, ax, labels=False, stencils=()):
        """The cloud, with the outline of the tiles."""
        super().plot(ax, labels, stencils)
        (lx, ly), (mx, my) = self.tile, self.tiles
        ax.vlines(lx * np.arange(mx + 1), 0, my * ly, color="0.7", lw=0.8)
        ax.hlines(ly * np.arange(my + 1), 0, mx * lx, color="0.7", lw=0.8)
