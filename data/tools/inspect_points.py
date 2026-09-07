#!/usr/bin/env python3
"""Inspect the point, node, graph and ordering files of docs/file_formats.md.
The tool reads, checks and reports; it never writes a file.

    tools/inspect_points.py case.node --plot --labels
    tools/inspect_points.py case.points case.graph --plot --stencil 0 17
    tools/inspect_points.py case.points case.graph case.iperm --spy
    tools/inspect_points.py case.points --periodic 32 32

One file of each kind, told apart by extension. Each is checked against
its header and the files against each other; the problems found are
listed at the end and make the exit status 1. An ordering file is
applied to the nodes and the graph, so they are reported and drawn in
the new numbering; only the spy plot also shows the file order.

Needs numpy. scipy speeds up the neighbour search and matplotlib draws
the figures.
"""

import argparse
import os
import sys

import numpy as np


class Report:
    """Problems are listed at the end and make the exit status 1; notes are
    printed where they arise."""

    def __init__(self):
        self.problems = []

    def problem(self, fname, msg):
        self.problems.append(f"{fname}: {msg}")

    def note(self, msg):
        print(f"  note: {msg}")


def lines_of(fname, comments):
    """(line number, tokens) of every non-blank line; `#` starts a comment
    in the formats that have them."""
    out = []
    with open(fname) as f:
        for no, line in enumerate(f, 1):
            if comments:
                line = line.split("#", 1)[0]
            toks = line.split()
            if toks:
                out.append((no, toks))
    return out


def as_int(tok):
    try:
        return int(tok)
    except ValueError:
        return None


def as_float(tok):
    try:
        return float(tok)
    except ValueError:
        return None


def minimum_image(diff, period):
    """Coordinate differences reduced to the nearest periodic image."""
    if period is None:
        return diff
    return diff - period * np.round(diff / period)


class Nodes:
    """A point cloud with a marker per node, 0 for interior: the contents of
    a points file (every marker 0) or a node file."""

    def __init__(self, xy, marker):
        self.xy = xy
        self.marker = marker

    def __len__(self):
        return len(self.xy)

    @classmethod
    def read(cls, fname, rep):
        """Nodes from a .points or .node file, or None with the problems
        recorded in rep."""
        if fname.endswith(".node"):
            got = cls._read_node_file(fname, rep)
        else:
            got = cls._read_points_file(fname, rep)
        if got is None:
            return None
        xy, marker = got
        bad = np.flatnonzero(~np.isfinite(xy).all(axis=1))
        if bad.size:
            rep.problem(fname, f"{bad.size} node{'s' if bad.size > 1 else ''} with a non-finite "
                        f"coordinate, the first is node {bad[0]}")
            return None
        return cls(xy, marker)

    @staticmethod
    def _read_points_file(fname, rep):
        """`n`, then n lines of `x y`."""
        lines = lines_of(fname, comments=False)
        if not lines:
            rep.problem(fname, "empty file")
            return None
        no, head = lines[0]
        n = as_int(head[0]) if len(head) == 1 else None
        if n is None or n < 0:
            rep.problem(fname, f"line {no}: expected the point count, got '{' '.join(head)}'")
            return None
        rows = lines[1:]
        if len(rows) < n:
            rep.problem(fname, f"header says {n} points, the file has {len(rows)} lines after it")
            return None
        if len(rows) > n:
            rep.note(f"{len(rows) - n} lines after the {n} points, ignored as the format allows")
        xy = np.full((n, 2), np.nan)
        ok = True
        for k, (no, toks) in enumerate(rows[:n]):
            vals = [as_float(t) for t in toks]
            if len(vals) != 2 or None in vals:
                rep.problem(fname, f"line {no}: expected 'x y', got '{' '.join(toks)}'")
                ok = False
                continue
            xy[k] = vals
        return (xy, np.zeros(n, dtype=int)) if ok else None

    @staticmethod
    def _read_node_file(fname, rep):
        """Triangle's format: `n 2 nattr nmark`, then `i x y a... [marker]`
        per node, `#` comments and blank lines allowed."""
        lines = lines_of(fname, comments=True)
        if not lines:
            rep.problem(fname, "empty file")
            return None
        no, head = lines[0]
        hv = [as_int(t) for t in head]
        if len(hv) != 4 or None in hv:
            rep.problem(fname, f"line {no}: expected the header 'n dim nattr nmark', got '{' '.join(head)}'")
            return None
        n, dim, nattr, nmark = hv
        if dim != 2:
            rep.problem(fname, f"dimension {dim} in the header, only 2 is supported")
            return None
        if nattr < 0 or nmark not in (0, 1):
            rep.problem(fname, f"header: nattr must be >= 0 and nmark 0 or 1, got {nattr} and {nmark}")
            return None
        if nattr:
            rep.note(f"{nattr} attribute{'s' if nattr > 1 else ''} per node, not inspected")
        rows = lines[1:]
        if len(rows) != n:
            rep.problem(fname, f"header says {n} nodes, the file has {len(rows)} node lines")
            return None
        ncol = 3 + nattr + nmark
        index = np.empty(n, dtype=int)
        xy = np.full((n, 2), np.nan)
        marker = np.zeros(n, dtype=int)
        ok = True
        for k, (no, toks) in enumerate(rows):
            i = as_int(toks[0])
            x = as_float(toks[1]) if len(toks) > 1 else None
            y = as_float(toks[2]) if len(toks) > 2 else None
            m = as_int(toks[-1]) if nmark and len(toks) == ncol else 0
            if len(toks) != ncol or None in (i, x, y, m):
                rep.problem(fname, f"line {no}: expected {ncol} columns 'i x y"
                            f"{' a' * nattr}{' marker' if nmark else ''}', got '{' '.join(toks)}'")
                ok = False
                continue
            index[k], xy[k], marker[k] = i, (x, y), m
        if not ok:
            return None
        if np.array_equal(index, np.arange(1, n + 1)):
            rep.note("nodes are numbered from 1 in the file; reported from 0 here")
        elif not np.array_equal(index, np.arange(n)):
            bad = np.flatnonzero(index != np.arange(n))[0]
            rep.problem(fname, f"node indices are not consecutive from 0 or 1: "
                        f"line {rows[bad][0]} has index {index[bad]}")
            return None
        if not nmark:
            rep.note("no marker column: every node is interior")
        return xy, marker

    def neighbours(self, k, period=None):
        """Indices and distances of the k nearest other nodes of every node,
        nearest first; minimum-image distances in the periodic box, if any."""
        n = len(self)
        xy = self.xy
        k = min(k, n - 1)
        if k < 1:
            return np.empty((n, 0), dtype=int), np.empty((n, 0))
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            cKDTree = None
        if cKDTree is not None:
            tree = cKDTree(xy) if period is None else cKDTree(np.mod(xy, period), boxsize=period)
            d, j = tree.query(tree.data, k + 1)
            return j[:, 1:], d[:, 1:]
        j = np.empty((n, k), dtype=int)                # brute force, in chunks of rows
        d = np.empty((n, k))
        chunk = max(1, 2_000_000 // n)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            d2 = (minimum_image(xy[a:b, None, :] - xy[None, :, :], period) ** 2).sum(axis=-1)
            d2[np.arange(b - a), np.arange(a, b)] = np.inf
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            dd = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(dd, axis=1)
            j[a:b] = np.take_along_axis(part, order, axis=1)
            d[a:b] = np.sqrt(np.take_along_axis(dd, order, axis=1))
        return j, d

    def describe(self, fname, rep, period=None):
        n = len(self)
        bnd = np.count_nonzero(self.marker)
        print(f"{fname}: {n} nodes, {n - bnd} interior, {bnd} boundary")
        values, counts = np.unique(self.marker, return_counts=True)
        print("  markers: " + "  ".join(f"{v}: {c}" for v, c in zip(values, counts)))
        lo, hi = self.xy.min(axis=0), self.xy.max(axis=0)
        print(f"  bounding box: x in [{lo[0]:.6g}, {hi[0]:.6g}], y in [{lo[1]:.6g}, {hi[1]:.6g}]")
        if period is not None:
            outside = np.count_nonzero((self.xy < 0).any(axis=1) | (self.xy >= period).any(axis=1))
            print(f"  periodic box [0, {period[0]:.6g}) x [0, {period[1]:.6g}): distances are minimum-image"
                  + (f"; {outside} nodes lie outside the box" if outside else ""))
        if n < 2:
            return
        j, d = self.neighbours(1, period)
        d, j = d[:, 0], j[:, 0]
        i = int(np.argmin(d))
        print(f"  nearest-neighbour distance: min {d.min():.6g} (nodes {i} and {j[i]}), "
              f"max {d.max():.6g} (node {np.argmax(d)})")
        print(f"    mean {d.mean():.6g}, median {np.median(d):.6g}, std {d.std():.6g}")
        same = np.flatnonzero(d == 0)
        if same.size:
            pairs = sorted({(min(a, j[a]), max(a, j[a])) for a in same})
            shown = ", ".join(f"({a}, {b})" for a, b in pairs[:5])
            more = f", ... {len(pairs)} pairs" if len(pairs) > 5 else ""
            rep.problem(fname, f"coincident nodes: {shown}{more}")

    def renumbered(self, ordering):
        return Nodes(self.xy[ordering.order], self.marker[ordering.order])

    def stencils(self, which, k, period=None):
        """(node, members) for the requested nodes, as the k nearest
        neighbours with the node itself first."""
        j, _ = self.neighbours(k - 1, period)
        return [(i, np.concatenate(([i], j[i]))) for i in which]

    def plot(self, ax, labels=False, stencils=(), period=None):
        """The nodes coloured by marker, optionally every index and the given
        (node, members) stencils, each as a circle through the farthest
        member with the members ringed and the node filled."""
        from matplotlib.patches import Circle
        xy, marker = self.xy, self.marker
        interior = marker == 0
        if interior.any():
            ax.plot(xy[interior, 0], xy[interior, 1], ".", color="0.55", ms=3,
                    label=f"interior ({np.count_nonzero(interior)})")
        for m in np.unique(marker[~interior]):
            sel = marker == m
            ax.plot(xy[sel, 0], xy[sel, 1], "o", ms=3.5, label=f"marker {m} ({np.count_nonzero(sel)})")
        if labels:
            for i, (x, y) in enumerate(xy):
                ax.annotate(str(i), (x, y), xytext=(2, 2), textcoords="offset points", fontsize=6)
        palette = ["tab:red", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
        for s, (i, members) in enumerate(stencils):
            c = palette[s % len(palette)]
            at = xy[i] + minimum_image(xy[members] - xy[i], period)   # nearest images of the members
            r = np.sqrt(((at - xy[i]) ** 2).sum(axis=1)).max()
            ax.add_patch(Circle(xy[i], r, fill=False, edgecolor=c, lw=1.2))
            ax.plot(at[:, 0], at[:, 1], "o", ms=8, mfc="none", mec=c, mew=1.2)
            ax.plot(xy[i, 0], xy[i, 1], "o", ms=8, color=c, label=f"stencil of node {i} ({len(members)})")
            ax.annotate(str(i), xy[i], xytext=(5, 5), textcoords="offset points", fontsize=8, color=c)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8, markerscale=1.5)


class Graph:
    """The stencils of all nodes in CSR form, from a graph file."""

    def __init__(self, n, ia, ja):
        self.n = n
        self.ia = ia
        self.ja = ja

    def __len__(self):
        return self.n

    @property
    def nnz(self):
        return int(self.ia[-1])

    def row(self, i):
        return self.ja[self.ia[i]:self.ia[i + 1]]

    @classmethod
    def read(cls, fname, rep):
        """`n nnz`, then the stencil of every node as a line of 0-based
        indices. None if the file cannot be used."""
        lines = lines_of(fname, comments=False)
        if not lines:
            rep.problem(fname, "empty file")
            return None
        no, head = lines[0]
        hv = [as_int(t) for t in head]
        if len(hv) != 2 or None in hv or min(hv) < 0:
            rep.problem(fname, f"line {no}: expected the header 'n nnz', got '{' '.join(head)}'")
            return None
        n, nnz = hv
        rows = lines[1:]
        if len(rows) != n:
            rep.problem(fname, f"header says {n} nodes, the file has {len(rows)} stencil lines")
            return None
        ia = np.zeros(n + 1, dtype=int)
        ja = []
        ok = True
        for k, (no, toks) in enumerate(rows):
            vals = [as_int(t) for t in toks]
            if None in vals:
                rep.problem(fname, f"line {no}: a stencil entry is not an integer")
                ok = False
                continue
            ia[k + 1] = ia[k] + len(vals)
            ja.extend(vals)
        if not ok:
            return None
        ja = np.array(ja, dtype=int)
        if ia[-1] != nnz:
            rep.problem(fname, f"header says nnz = {nnz}, the stencils hold {ia[-1]} entries")
        if ja.size and (ja.min() < 0 or ja.max() >= n):
            if ja.min() >= 1 and ja.max() == n:
                rep.problem(fname, f"indices run from {ja.min()} to {n}: the file looks 1-based, "
                            f"the format is 0-based")
            else:
                rep.problem(fname, f"stencil entries outside [0, {n}): from {ja.min()} to {ja.max()}")
            return None
        g = cls(n, ia, ja)
        for i in np.flatnonzero(np.diff(ia) == 0):
            rep.problem(fname, f"line {rows[i][0]}: node {i} has an empty stencil")
        dup = [i for i in range(n) if np.unique(g.row(i)).size != g.row(i).size]
        if dup:
            rep.problem(fname, f"{len(dup)} stencil{'s list' if len(dup) > 1 else ' lists'} a node twice, "
                        f"the first on line {rows[dup[0]][0]}")
        return g

    def describe(self, fname):
        n, ia, ja = self.n, self.ia, self.ja
        lengths = np.diff(ia)
        print(f"{fname}: {n} nodes, {self.nnz} entries")
        if lengths.min() == lengths.max():
            print(f"  rows: {lengths[0]} entries each")
        else:
            print(f"  rows: from {lengths.min()} to {lengths.max()} entries, mean {lengths.mean():.4g}")
        i = np.repeat(np.arange(n), lengths)
        nonempty = lengths > 0
        self_first = ja[ia[:-1][nonempty]] == np.arange(n)[nonempty]
        has_self = np.zeros(n, dtype=bool)
        has_self[i[ja == i]] = True
        if self_first.all():
            print("  every row starts with its own node")
        else:
            print(f"  {np.count_nonzero(~self_first)} rows do not start with their own node, "
                  f"{np.count_nonzero(~has_self)} do not contain it")
        sym = np.count_nonzero(np.isin(ja * n + i, i * n + ja))
        print(f"  entries with their transpose stored: {sym} of {self.nnz} ({100 * sym / max(self.nnz, 1):.1f}%)")
        print(f"  bandwidth: {np.abs(i - ja).max() if self.nnz else 0}")

    def renumbered(self, ordering):
        """The symmetric renumbering: rows moved and every entry relabelled."""
        order, iperm = ordering.order, ordering.iperm
        lengths = np.diff(self.ia)[order]
        ia = np.concatenate(([0], np.cumsum(lengths)))
        ja = np.concatenate([iperm[self.row(o)] for o in order])
        return Graph(self.n, ia, ja)

    def stencils(self, which):
        return [(i, self.row(i)) for i in which]

    def spy(self, ax, title):
        n = self.n
        i = np.repeat(np.arange(n), np.diff(self.ia))
        cell = 0.9 * 72 * 4.5 / max(n, 1)             # about one cell of the axes, in points
        ax.scatter(self.ja, i, s=max(cell, 0.8) ** 2, marker="s", color="black", linewidths=0)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{title}: {n} x {n}, nnz = {self.nnz}", fontsize=9)


class Ordering:
    """A renumbering from an ordering file: iperm[i] is the new index of
    node i, order[i] the old index of the node that lands at i."""

    def __init__(self, iperm):
        self.iperm = iperm
        self.order = np.argsort(iperm)

    def __len__(self):
        return self.iperm.size

    @classmethod
    def read(cls, fname, rep):
        """One integer per line, no header; None unless a permutation."""
        vals = []
        for no, toks in lines_of(fname, comments=False):
            v = as_int(toks[0]) if len(toks) == 1 else None
            if v is None:
                rep.problem(fname, f"line {no}: expected one integer, got '{' '.join(toks)}'")
                return None
            vals.append(v)
        iperm = np.array(vals, dtype=int)
        n = iperm.size
        if not np.array_equal(np.sort(iperm), np.arange(n)):
            if n and iperm.min() == 1 and iperm.max() == n:
                rep.problem(fname, f"values run from 1 to {n}: the file looks 1-based, the format is 0-based")
            else:
                rep.problem(fname, f"the {n} values are not a permutation of 0 .. {n - 1}")
            return None
        return cls(iperm)

    def describe(self, fname):
        print(f"{fname}: a permutation of {len(self)} nodes")


KINDS = {".points": Nodes, ".node": Nodes, ".graph": Graph, ".iperm": Ordering}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Check and describe the files of a case (docs/file_formats.md): "
                    "node and marker counts, nearest-neighbour statistics, graph "
                    "statistics. Never writes a file.")
    ap.add_argument("files", nargs="+", metavar="FILE",
                    help="one each of .points or .node, .graph and .iperm; "
                         "an ordering is applied to the others")
    ap.add_argument("--plot", action="store_true", help="draw the nodes, coloured by marker")
    ap.add_argument("--labels", action="store_true", help="with --plot: the index next to every node")
    ap.add_argument("--stencil", nargs="+", type=int, default=[], metavar="I",
                    help="with --plot: the stencils of these nodes, from the graph "
                         "or the --k nearest neighbours")
    ap.add_argument("--k", type=int, default=0, metavar="K",
                    help="stencil size for --stencil without a graph file")
    ap.add_argument("--periodic", nargs=2, type=float, metavar=("LX", "LY"),
                    help="the periodic box [0, LX) x [0, LY): minimum-image distances")
    ap.add_argument("--spy", action="store_true",
                    help="the sparsity pattern of the graph, before and after an ordering file")
    ap.add_argument("--save", metavar="FILE", help="write the figure to FILE instead of showing it")
    args = ap.parse_args()

    args.given = {}                                    # class -> file name
    for f in args.files:
        kind = KINDS.get(os.path.splitext(f)[1])
        if kind is None:
            sys.exit(f"{f}: unknown extension, expected .points, .node, .graph or .iperm")
        if kind in args.given:
            sys.exit(f"{f}: a {kind.__name__.lower()} file was already given, {args.given[kind]}")
        args.given[kind] = f
    if args.spy and Graph not in args.given:
        sys.exit("--spy needs a graph file")
    if (args.plot or args.stencil) and Nodes not in args.given:
        sys.exit("--plot needs a points or node file")
    if args.stencil and Graph not in args.given and args.k < 2:
        sys.exit("--stencil needs a graph file, or --k of at least 2 to build the stencils")
    if args.periodic and min(args.periodic) <= 0:
        sys.exit("--periodic: the box sides must be positive")
    args.period = np.array(args.periodic) if args.periodic else None
    return args


def draw(args, nodes, graph, ordering, file_graph, rep):
    """One figure: the nodes, the spy plot, or both; two spy panels when an
    ordering was applied."""
    import matplotlib.pyplot as plt
    panels = int(args.plot) + int(args.spy) * (2 if ordering else 1)
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 5.2), squeeze=False)
    axes = list(axes[0])
    if args.plot:
        n = len(nodes)
        bad = [i for i in args.stencil if not 0 <= i < n]
        if bad:
            sys.exit(f"--stencil: node {bad[0]} is outside [0, {n})")
        if not args.stencil:
            stencils = []
        elif graph is not None:
            if args.k:
                rep.note("--k ignored: the stencils are taken from the graph file")
            stencils = graph.stencils(args.stencil)
        else:
            stencils = nodes.stencils(args.stencil, args.k, args.period)
        ax = axes.pop(0)
        nodes.plot(ax, args.labels, stencils, args.period)
        ax.set_title(os.path.basename(args.given[Nodes]), fontsize=9)
    if args.spy:
        base = os.path.basename(args.given[Graph])
        if ordering:
            file_graph.spy(axes.pop(0), f"{base}, file order")
            graph.spy(axes.pop(0), f"{base}, {os.path.basename(args.given[Ordering])}")
        else:
            graph.spy(axes.pop(0), base)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"figure written to {args.save}")
    else:
        plt.show()


def main():
    args = parse_args()
    rep = Report()
    read = {}                                          # class -> object, for the files that read
    for kind, fname in args.given.items():
        obj = kind.read(fname, rep)
        if obj is not None:
            read[kind] = obj
            if kind is Nodes:
                obj.describe(fname, rep, args.period)
            else:
                obj.describe(fname)
    counts = {args.given[kind]: len(obj) for kind, obj in read.items()}
    if len(set(counts.values())) > 1:
        rep.problem("files", "node counts differ: " + ", ".join(f"{f} has {n}" for f, n in counts.items()))

    if rep.problems:
        print(f"{len(rep.problems)} problem{'s' if len(rep.problems) > 1 else ''}:")
        for p in rep.problems:
            print(f"  {p}")
    else:
        print("checks: no problems found")

    nodes, graph, ordering = read.get(Nodes), read.get(Graph), read.get(Ordering)
    consistent = not rep.problems and (nodes is not None or graph is not None)
    file_graph = graph
    if ordering and consistent:
        nodes = nodes.renumbered(ordering) if nodes else None
        graph = graph.renumbered(ordering) if graph else None
        print("ordering applied: nodes and stencils are in the new numbering below")

    if args.plot or args.spy:
        if consistent:
            draw(args, nodes, graph, ordering, file_graph, rep)
        else:
            print("no figure: fix the problems above first")
    sys.exit(1 if rep.problems else 0)


if __name__ == "__main__":
    main()
