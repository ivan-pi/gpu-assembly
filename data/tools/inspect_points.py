#!/usr/bin/env python3
"""Inspect the point, node, graph and ordering files of docs/file_formats.md.
The tool reads, checks and reports; it never writes a file.

    tools/inspect_points.py case.node --plot --labels
    tools/inspect_points.py case.points case.graph --stencil 0 17
    tools/inspect_points.py case.points case.graph case.iperm --spy
    tools/inspect_points.py case.points --periodic 32 32

One file of each kind, told apart by extension. Each is checked against
its header and the files against each other; the problems found are
listed at the end and make the exit status 1. An ordering file is
applied to the nodes and the graph before they are drawn, so --labels
shows the new indices; only the spy plot also shows the file order.

Needs numpy and scipy; matplotlib draws the figures.
"""

import argparse
import logging
import os
import sys

import numpy as np
from scipy.sparse import csr_array

from pointclouds.cli import number
from pointclouds.io import FormatError, read_graph, read_nodes, read_ordering
from pointclouds.nodeset import NodeSet

KINDS = {".points": "nodes", ".node": "nodes", ".graph": "graph", ".iperm": "ordering"}


def describe_nodes(fname, xy, cloud):
    """Print the counts, the markers, the box and the spacing statistics
    of the cloud read from the file, whose coordinates are `xy` as they
    were read; return the problems seen, the coincident nodes."""
    n, boundary = len(cloud), np.count_nonzero(cloud.markers)
    print(f"{fname}: {n} nodes, {n - boundary} interior, {boundary} boundary")
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
    if n < 2:
        return []
    j, d = (a.ravel() for a in cloud.neighbours(1))
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
    """The stencils of all nodes from a graph file: the CSR arrays, and
    the same as a sparse pattern for the checks."""

    def __init__(self, ia, ja):
        self.ia, self.ja = ia, ja
        self.n, self.nnz = len(ia) - 1, len(ja)
        self.pattern = csr_array(
            (np.ones(self.nnz, np.int8), ja, ia), shape=(self.n, self.n)
        )

    def __len__(self):
        return self.n

    def row(self, i):
        """The stencil of node i."""
        return self.ja[self.ia[i] : self.ia[i + 1]]

    def renumbered(self, iperm):
        """The symmetric renumbering: rows moved and every entry relabelled."""
        order = np.argsort(iperm)
        moved = self.pattern[order][:, order]
        return Graph(moved.indptr, moved.indices)

    def describe(self, fname):
        """Print the size, the row lengths, symmetry and bandwidth, and the
        rows that do not open with their own node when any do not."""
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
        self_first = np.count_nonzero(ja[ia[:-1]] != np.arange(n))
        if self_first:
            missing = n - np.count_nonzero(self.pattern.diagonal())
            print(
                f"  {self_first} rows do not start with their own node, "
                f"{missing} do not contain it"
            )
        sym = self.pattern.multiply(self.pattern.T).nnz
        print(
            f"  entries with their transpose stored: {sym} of {self.nnz} "
            f"({100 * sym / self.nnz:.1f}%)"
        )
        print(f"  bandwidth: {np.abs(self.pattern.tocoo().row - ja).max()}")

    def spy(self, ax, title):
        """The sparsity pattern, one square per stored entry."""
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


def parse_args():
    """The command line, with `files` mapping each kind of file to its name."""
    ap = argparse.ArgumentParser(
        description="Check and describe the files of a case (docs/file_formats.md): "
        "node and marker counts, nearest-neighbour statistics, graph "
        "statistics. Never writes a file."
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
        help="draw the stencils of these nodes, from the graph "
        "or the --k nearest neighbours",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=0,
        metavar="K",
        help="stencil size for --stencil without a graph file",
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
    if args.plot and "nodes" not in files:
        sys.exit("--plot needs a points or node file")
    if args.stencil and "graph" not in files and args.k < 2:
        sys.exit(
            "--stencil needs a graph file, or --k of at least 2 to build the stencils"
        )
    return args


def draw(args, loaded):
    """One figure: the nodes, the spy plot, or both, renumbered by the
    ordering if one was given, and then the spy plot in file order too."""
    cloud, graph, iperm = (
        loaded.get("nodes"),
        loaded.get("graph"),
        loaded.get("ordering"),
    )
    file_graph = None
    if iperm is not None:
        order = np.argsort(iperm)
        if cloud is not None:
            cloud = NodeSet(
                cloud.points[order], cloud.markers[order], cloud.extent, cloud.periodic
            )
        if graph is not None:
            file_graph, graph = graph, graph.renumbered(iperm)
        print("ordering applied: nodes and stencils are in the new numbering below")

    stencils = []
    if args.stencil:
        bad = [i for i in args.stencil if not 0 <= i < len(cloud)]
        if bad:
            sys.exit(f"--stencil: node {bad[0]} is outside [0, {len(cloud)})")
        if graph is not None:
            if args.k:
                print("--k ignored: the stencils are taken from the graph file")
            stencils = [(i, graph.row(i)) for i in args.stencil]
        else:
            ia, ja = cloud.stencils("knn", args.k)
            stencils = [(i, ja[ia[i] : ia[i + 1]]) for i in args.stencil]

    import matplotlib.pyplot as plt

    panels = int(args.plot) + int(args.spy) * (2 if file_graph is not None else 1)
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 5.2), squeeze=False)
    axes = iter(axes[0])
    if args.plot:
        ax = next(axes)
        cloud.plot(ax, args.labels, stencils)
        ax.set_title(os.path.basename(args.files["nodes"]), fontsize=9)
    if args.spy:
        base = os.path.basename(args.files["graph"])
        if file_graph is not None:
            file_graph.spy(next(axes), f"{base}, file order")
            graph.spy(next(axes), f"{base}, {os.path.basename(args.files['ordering'])}")
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
    box = (args.periodic, (True, True)) if args.periodic else (None, (False, False))
    problems, sizes, loaded = [], {}, {}
    for kind, fname in args.files.items():
        try:
            if kind == "nodes":
                xy, m = read_nodes(fname)
                loaded[kind] = cloud = NodeSet(xy, m, *box)
                problems += describe_nodes(fname, xy, cloud)
            elif kind == "graph":
                loaded[kind] = graph = Graph(*read_graph(fname))
                graph.describe(fname)
            else:
                loaded[kind] = iperm = read_ordering(fname)
                print(f"{fname}: a permutation of {len(iperm)} nodes")
        except FormatError as e:
            problems += e.problems
        else:
            sizes[fname] = len(loaded[kind])
    if len(set(sizes.values())) > 1:
        problems.append(
            "node counts differ: " + ", ".join(f"{f} has {n}" for f, n in sizes.items())
        )
    if problems:
        print(f"{len(problems)} problem{'s' * (len(problems) != 1)}:")
        print(*(f"  {p}" for p in problems), sep="\n")
    if args.plot or args.spy:
        if problems:
            print("no figure: fix the problems above first")
        else:
            draw(args, loaded)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
