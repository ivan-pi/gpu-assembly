#!/usr/bin/env python3
"""Check and describe the files of a case, and draw them.

The point, node, graph and ordering files of docs/file_formats.md; the
tool changes none of them, and writes only the figure of --save.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_array

from pointclouds.cli import number
from pointclouds.io import FormatError, plural, read_graph, read_nodes, read_ordering
from pointclouds.nodeset import NodeSet

KINDS = {
    ".points": "cloud",
    ".node": "cloud",
    ".graph": "graph",
    ".iperm": "iperm",
}


def describe_nodes(fname, xy, cloud):
    """Prints the statistics of a cloud and returns the problems seen.

    `fname` is the file it was read from, `xy` the coordinates as read.
    """
    print(f"{fname}: {cloud.summary()}")
    values, counts = np.unique(cloud.markers, return_counts=True)
    print("  markers: " + "  ".join(f"{v}: {c}" for v, c in zip(values, counts)))
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    print(
        f"  bounding box: x in [{lo[0]:.6g}, {hi[0]:.6g}], y in [{lo[1]:.6g}, {hi[1]:.6g}]"
    )
    if cloud.extent is not None:
        outside = np.count_nonzero(((xy < 0) | (xy >= cloud.extent)).any(axis=1))
        print(
            f"  periodic box [0, {cloud.extent[0]:.6g}) x [0, {cloud.extent[1]:.6g}): "
            "distances are minimum-image"
            + (f"; {outside} nodes lie outside the box" if outside else "")
        )
    if len(cloud) < 2:
        return []
    j, d = cloud.nearest()
    lo, hi = int(np.argmin(d)), int(np.argmax(d))
    print(
        f"  nearest-neighbour distance: min {d[lo]:.6g} (nodes {lo} and {j[lo]}), "
        f"max {d[hi]:.6g} (node {hi})"
    )
    print(f"    mean {d.mean():.6g}, median {np.median(d):.6g}, std {d.std():.6g}")
    same = np.flatnonzero(d == 0)
    if not same.size:
        return []
    pairs = sorted({(min(a, j[a]), max(a, j[a])) for a in same})
    shown = ", ".join(f"({a}, {b})" for a, b in pairs[:5])
    more = f", ... {len(pairs)} pairs" if len(pairs) > 5 else ""
    return [f"{fname}: coincident nodes: {shown}{more}"]


class Graph:
    """The stencils of a graph file, as CSR arrays and a sparse pattern."""

    def __init__(self, ia, ja):
        self.ia, self.ja = ia, ja
        self.n, self.nnz = len(ia) - 1, len(ja)
        self.pattern = csr_array(
            (np.ones(self.nnz, np.int8), ja, ia), shape=(self.n, self.n)
        )

    def __len__(self):
        return self.n

    def renumbered(self, iperm):
        """Returns the graph in the numbering `iperm`.

        Moves the rows and relabels the entries.
        """
        order = np.argsort(iperm)
        moved = self.pattern[order][:, order]
        return Graph(moved.indptr, moved.indices)

    def describe(self, fname):
        """Prints the size, row lengths, symmetry and bandwidth."""
        n, ia, ja = self.n, self.ia, self.ja
        print(f"{fname}: {n} nodes, {self.nnz} entries")
        lengths = np.diff(ia)
        if lengths.min() == lengths.max():
            print(f"  rows: {lengths[0]} entries each")
        else:
            print(
                f"  rows: from {lengths.min()} to {lengths.max()} entries, "
                f"mean {lengths.mean():.4g}"
            )
        not_first = np.count_nonzero(ja[ia[:-1]] != np.arange(n))
        if not_first:
            missing = n - np.count_nonzero(self.pattern.diagonal())
            print(
                f"  {not_first} rows do not start with their own node, "
                f"{missing} do not contain it"
            )
        sym = self.pattern.multiply(self.pattern.T).nnz
        print(
            f"  entries with their transpose stored: {sym} of {self.nnz} "
            f"({100 * sym / self.nnz:.1f}%)"
        )
        print(f"  bandwidth: {np.abs(self.pattern.tocoo().row - ja).max()}")

    def spy(self, ax, title):
        """Draws the sparsity pattern, one square per stored entry."""
        n = self.n
        width = ax.figure.get_size_inches()[0] * ax.get_position().width * 72  # points
        ax.scatter(
            self.ja,
            self.pattern.tocoo().row,
            s=max(width / n, 0.8) ** 2,
            marker="s",
            color="black",
            linewidths=0,
        )
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{title}: {n} x {n}, nnz = {self.nnz}", fontsize=9)


EPILOG = """\
examples:
  inspect_points.py case.node --plot --labels
  inspect_points.py case.points case.graph --stencil 0 17
  inspect_points.py case.points case.graph case.iperm --spy
  inspect_points.py case.points --periodic 32 32

One file of each kind, told apart by extension. Each is checked against
its header and the files against each other; the problems found are
listed at the end and make the exit status 1. An ordering file is
applied to the nodes and the graph before they are drawn, so --labels
shows the new indices; only the spy plot also shows the file order."""


@dataclass
class Case:
    """The files of a case as read; any of the three may be missing."""

    cloud: NodeSet = None
    graph: Graph = None
    iperm: np.ndarray = None


def parse_args():
    """Returns the command line; `files` maps a Case field to its file."""
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="one each of .points or .node, .graph and .iperm; "
        "an ordering is applied to the others",
    )
    ap.add_argument(
        "--plot", action="store_true", help="draw the nodes, coloured by marker"
    )
    ap.add_argument(
        "--labels", action="store_true", help="write the index next to every node"
    )
    ap.add_argument(
        "--stencil",
        nargs="+",
        type=int,
        default=[],
        metavar="I",
        help="draw the stencils of these nodes, from the graph or as the K nearest",
    )
    ap.add_argument(
        "-K",
        "--knn",
        type=number(int, least=2),
        metavar="K",
        help="the stencil of a node is its K nearest nodes, without a graph file",
    )
    ap.add_argument(
        "--periodic",
        nargs=2,
        type=number(float, above=0.0),
        metavar=("LX", "LY"),
        help="the periodic box [0, LX) x [0, LY): minimum-image distances",
    )
    ap.add_argument(
        "--spy",
        action="store_true",
        help="draw the sparsity pattern of the graph, before and after an ordering file",
    )
    ap.add_argument(
        "--save", metavar="FILE", help="write the figure to FILE instead of showing it"
    )
    args = ap.parse_args()
    args.plot = args.plot or args.labels or bool(args.stencil)

    files = {}
    for f in args.files:
        kind = KINDS.get(os.path.splitext(f)[1])
        if kind is None:
            sys.exit(
                f"{f}: unknown extension, expected .points, .node, .graph or .iperm"
            )
        if kind in files:
            sys.exit(f"{f}: a {kind} file was already given, {files[kind]}")
        files[kind] = f
    args.files = files
    if args.spy and "graph" not in files:
        sys.exit("--spy needs a graph file")
    if args.plot and "cloud" not in files:
        sys.exit("--plot needs a points or node file")
    if args.stencil and "graph" not in files and args.knn is None:
        sys.exit("--stencil needs a graph file, or -K to build the stencils")
    return args


def draw(args, case):
    """Shows or saves one figure: the nodes, the spy plot, or both.

    Renumbered by the ordering if one was given, with the spy plot in file
    order beside it.
    """
    cloud, graph, file_graph = case.cloud, case.graph, None
    if case.iperm is not None:
        order = np.argsort(case.iperm)
        if cloud is not None:
            cloud = NodeSet(
                cloud.points[order],
                cloud.markers[order],
                extent=cloud.extent,
                periodic=cloud.periodic,
            )
        if graph is not None:
            file_graph, graph = graph, graph.renumbered(case.iperm)
        print("ordering applied: nodes and stencils are in the new numbering below")

    stencils = []
    if args.stencil:
        bad = [i for i in args.stencil if not 0 <= i < len(cloud)]
        if bad:
            sys.exit(f"--stencil: node {bad[0]} is outside [0, {len(cloud)})")
        if graph is not None:
            if args.knn:
                print("-K ignored: the stencils are taken from the graph file")
            ia, ja = graph.ia, graph.ja
        else:
            ia, ja = cloud.stencils("knn", args.knn)
        stencils = [(i, ja[ia[i] : ia[i + 1]]) for i in args.stencil]

    import matplotlib.pyplot as plt

    panels = int(args.plot) + int(args.spy) * (2 if file_graph is not None else 1)
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 5.2), squeeze=False)
    axes = iter(axes[0])
    if args.plot:
        ax = next(axes)
        cloud.plot(ax, labels=args.labels, stencils=stencils)
        ax.set_title(os.path.basename(args.files["cloud"]), fontsize=9)
    if args.spy:
        base = os.path.basename(args.files["graph"])
        if file_graph is not None:
            file_graph.spy(next(axes), f"{base}, file order")
            graph.spy(next(axes), f"{base}, {os.path.basename(args.files['iperm'])}")
        else:
            graph.spy(next(axes), base)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"figure written to {args.save}")
    else:
        plt.show()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="  note: %(message)s", stream=sys.stdout
    )
    box = dict(extent=args.periodic, periodic=(True, True)) if args.periodic else {}
    problems, sizes, case = [], {}, Case()
    for kind, fname in args.files.items():
        try:
            if kind == "cloud":
                xy, m = read_nodes(fname)
                case.cloud = NodeSet(xy, m, **box)
                problems += describe_nodes(fname, xy, case.cloud)
            elif kind == "graph":
                case.graph = Graph(*read_graph(fname))
                case.graph.describe(fname)
            else:
                case.iperm = read_ordering(fname)
                print(f"{fname}: a permutation of {len(case.iperm)} nodes")
        except FormatError as e:
            problems += e.problems
        else:
            sizes[fname] = len(getattr(case, kind))
    if len(set(sizes.values())) > 1:
        problems.append(
            "node counts differ: " + ", ".join(f"{f} has {n}" for f, n in sizes.items())
        )
    if problems:
        print(f"{plural(len(problems), 'problem')}:")
        print(*(f"  {p}" for p in problems), sep="\n")
    if args.plot or args.spy:
        if problems:
            print("no figure: fix the problems above first")
        else:
            draw(args, case)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
