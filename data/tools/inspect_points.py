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

Needs numpy. scipy speeds up the neighbour search and matplotlib draws
the figures.
"""

import argparse
import functools
import itertools
import os
import sys

import numpy as np


# ----------------------------------------------------------- text parsing

class Report:
    """What the checks found: problems, listed at the end and making the
    exit status 1, and notes, printed under the description of their file.
    A file with nothing wrong is passed over in silence."""

    def __init__(self):
        self.problems = []
        self.notes = []

    def problem(self, fname, msg):
        self.problems.append(f"{fname}: {msg}")

    def note(self, msg):
        self.notes.append(msg)

    def print_notes(self):
        for msg in self.notes:
            print(f"  note: {msg}")
        self.notes = []

    def print_problems(self):
        """The list of problems, and nothing at all when there are none:
        the exit status already says a file checked out."""
        if not self.problems:
            return
        print(f"{plural(len(self.problems), 'problem')}:")
        for p in self.problems:
            print(f"  {p}")


def plural(k, word):
    """The count and the word, with an s unless the count is 1."""
    return f"{k} {word}{'' if k == 1 else 's'}"


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


def nums(toks, conv):
    """All tokens through int or float, or None if one is not a number."""
    try:
        return [conv(t) for t in toks]
    except ValueError:
        return None


def read_header(fname, rep, form, comments):
    """The non-negative integers of the first line, and the lines after it:
    (values, rows), or None with the problem recorded. `form` names the
    fields, `n nnz` say, and sets how many there are."""
    lines = lines_of(fname, comments)
    if not lines:
        rep.problem(fname, "empty file")
        return None
    no, head = lines[0]
    vals = nums(head, int)
    if vals is None or len(vals) != len(form.split()) or min(vals) < 0:
        rep.problem(fname, f"line {no}: expected the header '{form}', got '{' '.join(head)}'")
        return None
    return vals, lines[1:]


def parse_rows(fname, rep, rows, form, convert):
    """Every row through `convert`, which returns the values of a line or
    None: the list of values, or None with every bad line recorded.
    `form` describes a line in the message."""
    out = []
    ok = True
    for no, toks in rows:
        vals = convert(toks)
        if vals is None:
            rep.problem(fname, f"line {no}: expected '{form}', got '{' '.join(toks)}'")
            ok = False
        else:
            out.append(vals)
    return out if ok else None


def minimum_image(diff, period):
    """Coordinate differences reduced to the nearest periodic image."""
    if period is None:
        return diff
    return diff - period * np.round(diff / period)


# ------------------------------------------------------------------ nodes

class Nodes:
    """A point cloud with a marker per node, 0 for interior, from a points
    file (every marker 0) or a node file, and the periodic box its
    distances are measured in, if any."""

    def __init__(self, xy, marker, fname, period=None):
        self.xy = xy
        self.marker = marker
        self.fname = fname
        self.period = period

    def __len__(self):
        return len(self.xy)

    # reading

    @classmethod
    def read(cls, fname, rep, period=None):
        """Nodes from a .points or .node file, or None with the problems
        recorded."""
        reader = cls._read_node_file if fname.endswith(".node") else cls._read_points_file
        got = reader(fname, rep)
        if got is None:
            return None
        xy, marker = got
        bad = np.flatnonzero(~np.isfinite(xy).all(axis=1))
        if bad.size:
            rep.problem(fname, f"{plural(bad.size, 'node')} with a non-finite coordinate, "
                        f"the first is node {bad[0]}")
            return None
        nodes = cls(xy, marker, fname, period)
        nodes._check_coincident(rep)
        return nodes

    @staticmethod
    def _read_points_file(fname, rep):
        """`n`, then n lines of `x y`: (xy, zero markers), or None."""
        got = read_header(fname, rep, "n", comments=False)
        if got is None:
            return None
        (n,), rows = got
        if len(rows) < n:
            rep.problem(fname, f"header says {n} points, the file has {len(rows)} lines after it")
            return None
        if len(rows) > n:
            rep.note(f"{len(rows) - n} lines after the {n} points, ignored as the format allows")
        vals = parse_rows(fname, rep, rows[:n], "x y",
                          lambda toks: nums(toks, float) if len(toks) == 2 else None)
        if vals is None:
            return None
        return np.array(vals, dtype=float).reshape(n, 2), np.zeros(n, dtype=int)

    @staticmethod
    def _read_node_file(fname, rep):
        """Triangle's format, `n 2 nattr nmark` then `i x y a... [marker]`
        per node: (xy, markers), or None."""
        got = read_header(fname, rep, "n dim nattr nmark", comments=True)
        if got is None:
            return None
        (n, dim, nattr, nmark), rows = got
        if dim != 2:
            rep.problem(fname, f"dimension {dim} in the header, only 2 is supported")
            return None
        if nmark not in (0, 1):
            rep.problem(fname, f"header: nmark must be 0 or 1, got {nmark}")
            return None
        if len(rows) != n:
            rep.problem(fname, f"header says {n} nodes, the file has {len(rows)} node lines")
            return None
        if nattr:
            rep.note(f"{plural(nattr, 'attribute')} per node, not inspected")
        if not nmark:
            rep.note("no marker column: every node is interior")
        form = f"i x y{' a' * nattr}{' marker' if nmark else ''}"
        vals = parse_rows(fname, rep, rows, form, lambda toks: Nodes._node_line(toks, nattr, nmark))
        if vals is None:
            return None
        table = np.array(vals, dtype=float).reshape(n, 4)
        if not Nodes._consecutive(fname, rep, table[:, 0].astype(int), rows):
            return None
        return table[:, 1:3], table[:, 3].astype(int)

    @staticmethod
    def _node_line(toks, nattr, nmark):
        """(index, x, y, marker) of a node line, or None if malformed."""
        if len(toks) != 3 + nattr + nmark:
            return None
        try:
            return int(toks[0]), float(toks[1]), float(toks[2]), int(toks[-1]) if nmark else 0
        except ValueError:
            return None

    @staticmethod
    def _consecutive(fname, rep, index, rows):
        """Whether the index column counts from 0 or from 1."""
        expected = np.arange(index.size)
        if np.array_equal(index, expected + 1):
            rep.note("nodes are numbered from 1 in the file; reported from 0 here")
        elif not np.array_equal(index, expected):
            bad = np.flatnonzero(index != expected)[0]
            rep.problem(fname, f"node indices are not consecutive from 0 or 1: "
                        f"line {rows[bad][0]} has index {index[bad]}")
            return False
        return True

    def _check_coincident(self, rep):
        """Nodes at distance zero from another are a problem."""
        j, d = self.nearest
        same = np.flatnonzero(d == 0)
        if same.size:
            pairs = sorted({(min(a, j[a]), max(a, j[a])) for a in same})
            shown = ", ".join(f"({a}, {b})" for a, b in pairs[:5])
            more = f", ... {len(pairs)} pairs" if len(pairs) > 5 else ""
            rep.problem(self.fname, f"coincident nodes: {shown}{more}")

    # neighbours

    def neighbours(self, k, which=None):
        """Indices and distances of the k nearest other nodes of every node,
        or of the nodes in `which`, nearest first."""
        n = len(self)
        which = np.arange(n) if which is None else np.asarray(which)
        k = min(k, n - 1)
        if k < 1:
            return np.empty((len(which), 0), dtype=int), np.empty((len(which), 0))
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return self._neighbours_brute_force(k, which)
        xy = self.xy if self.period is None else np.mod(self.xy, self.period)
        d, j = cKDTree(xy, boxsize=self.period).query(xy[which], k + 1)
        return j[:, 1:], d[:, 1:]

    def _neighbours_brute_force(self, k, which):
        """The same from the distances of the queried nodes to all others,
        in chunks of rows."""
        xy = self.xy
        j = np.empty((len(which), k), dtype=int)
        d = np.empty((len(which), k))
        chunk = max(1, 2_000_000 // len(self))
        for a in range(0, len(which), chunk):
            q = which[a:a + chunk]
            d2 = (minimum_image(xy[q, None, :] - xy[None, :, :], self.period) ** 2).sum(axis=-1)
            d2[np.arange(len(q)), q] = np.inf
            jj = np.argsort(d2, axis=1, kind="stable")[:, :k]
            j[a:a + chunk] = jj
            d[a:a + chunk] = np.sqrt(np.take_along_axis(d2, jj, axis=1))
        return j, d

    @functools.cached_property
    def nearest(self):
        """(index, distance) of the nearest other node of every node."""
        j, d = self.neighbours(1)
        return j.reshape(-1), d.reshape(-1)

    def renumbered(self, ordering):
        """The same nodes in the new numbering."""
        return Nodes(self.xy[ordering.order], self.marker[ordering.order], self.fname, self.period)

    # reporting

    def describe(self):
        """Print the counts, the bounding box and the spacing statistics."""
        n, bnd = len(self), np.count_nonzero(self.marker)
        print(f"{self.fname}: {n} nodes, {n - bnd} interior, {bnd} boundary")
        values, counts = np.unique(self.marker, return_counts=True)
        print("  markers: " + "  ".join(f"{v}: {c}" for v, c in zip(values, counts)))
        self._print_box()
        if n > 1:
            self._print_spacing()

    def _print_box(self):
        """The bounding box, and the periodic box if one was declared."""
        lo, hi = self.xy.min(axis=0), self.xy.max(axis=0)
        print(f"  bounding box: x in [{lo[0]:.6g}, {hi[0]:.6g}], y in [{lo[1]:.6g}, {hi[1]:.6g}]")
        if self.period is not None:
            outside = np.count_nonzero(((self.xy < 0) | (self.xy >= self.period)).any(axis=1))
            print(f"  periodic box [0, {self.period[0]:.6g}) x [0, {self.period[1]:.6g}): "
                  f"distances are minimum-image" + (f"; {outside} nodes lie outside the box" if outside else ""))

    def _print_spacing(self):
        """The nearest-neighbour distance statistics."""
        j, d = self.nearest
        lo, hi = int(np.argmin(d)), int(np.argmax(d))
        print(f"  nearest-neighbour distance: min {d[lo]:.6g} (nodes {lo} and {j[lo]}), "
              f"max {d[hi]:.6g} (node {hi})")
        print(f"    mean {d.mean():.6g}, median {np.median(d):.6g}, std {d.std():.6g}")

    # drawing

    PALETTE = ["tab:red", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]

    def plot(self, ax, labels=False, stencils=()):
        """The nodes coloured by marker, optionally every index and the given
        (node, members) stencils."""
        self._plot_markers(ax)
        if labels:
            for i, (x, y) in enumerate(self.xy):
                ax.annotate(str(i), (x, y), xytext=(2, 2), textcoords="offset points", fontsize=6)
        for s, (i, members) in enumerate(stencils):
            self._plot_stencil(ax, i, members, self.PALETTE[s % len(self.PALETTE)])
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8, markerscale=1.5)

    def _plot_markers(self, ax):
        """Interior nodes in grey, every marker value in its own colour, the
        counts in the legend."""
        xy, marker = self.xy, self.marker
        interior = marker == 0
        if interior.any():
            ax.plot(xy[interior, 0], xy[interior, 1], ".", color="0.55", ms=3,
                    label=f"interior ({np.count_nonzero(interior)})")
        for m in np.unique(marker[~interior]):
            sel = marker == m
            ax.plot(xy[sel, 0], xy[sel, 1], "o", ms=3.5, label=f"marker {m} ({np.count_nonzero(sel)})")

    def _plot_stencil(self, ax, i, members, color):
        """A circle about node i through its farthest member, the members
        ringed and the node filled; a wrapped member sits at its nearest image."""
        from matplotlib.patches import Circle
        centre = self.xy[i]
        at = centre + minimum_image(self.xy[members] - centre, self.period)
        r = np.linalg.norm(at - centre, axis=1).max()
        ax.add_patch(Circle(centre, r, fill=False, edgecolor=color, lw=1.2))
        ax.plot(at[:, 0], at[:, 1], "o", ms=8, mfc="none", mec=color, mew=1.2)
        ax.plot(centre[0], centre[1], "o", ms=8, color=color, label=f"stencil of node {i} ({len(members)})")
        ax.annotate(str(i), centre, xytext=(5, 5), textcoords="offset points", fontsize=8, color=color)


# ------------------------------------------------------------------ graph

class Graph:
    """The stencils of all nodes in CSR form, from a graph file."""

    def __init__(self, stencils, fname):
        self.fname = fname
        self.ia = np.cumsum([0, *map(len, stencils)])
        self.ja = np.fromiter(itertools.chain.from_iterable(stencils), dtype=int, count=self.nnz)
        self.rows = np.repeat(np.arange(self.n), np.diff(self.ia))   # the row of every entry

    def __len__(self):
        return self.n

    @property
    def n(self):
        return len(self.ia) - 1

    @property
    def nnz(self):
        return int(self.ia[-1])

    def row(self, i):
        """The stencil of node i."""
        return self.ja[self.ia[i]:self.ia[i + 1]]

    def keys(self):
        """i * n + j of every entry: one integer per matrix position."""
        return self.rows * self.n + self.ja

    # reading

    @classmethod
    def read(cls, fname, rep):
        """`n nnz`, then a line of 0-based indices per node: a Graph, or None
        if the file cannot be used."""
        got = read_header(fname, rep, "n nnz", comments=False)
        if got is None:
            return None
        (n, nnz), rows = got
        if len(rows) != n:
            rep.problem(fname, f"header says {n} nodes, the file has {len(rows)} stencil lines")
            return None
        stencils = parse_rows(fname, rep, rows, "integer stencil entries", lambda toks: nums(toks, int))
        if stencils is None:
            return None
        g = cls(stencils, fname)
        if g.nnz != nnz:
            rep.problem(fname, f"header says nnz = {nnz}, the stencils hold {g.nnz} entries")
        lo, hi = g.ja.min(), g.ja.max()
        if lo < 0 or hi >= n:
            if lo >= 1 and hi == n:
                rep.problem(fname, f"indices run from {lo} to {n}: the file looks 1-based, the format is 0-based")
            else:
                rep.problem(fname, f"stencil entries outside [0, {n}): from {lo} to {hi}")
            return None
        dup = g._rows_with_duplicates()
        if dup.size:
            rep.problem(fname, f"{plural(dup.size, 'stencil')} {'lists' if dup.size == 1 else 'list'} "
                        f"a node twice, the first on line {rows[dup[0]][0]}")
        return g

    def _rows_with_duplicates(self):
        """The rows in which some node is listed twice."""
        keys = np.sort(self.keys())
        return np.unique(keys[1:][keys[1:] == keys[:-1]] // self.n)

    # renumbering

    def renumbered(self, ordering):
        """The symmetric renumbering: rows moved and every entry relabelled."""
        return Graph([ordering.iperm[self.row(o)] for o in ordering.order], self.fname)

    # reporting

    def describe(self):
        """Print the size, the row lengths, symmetry and bandwidth, and the
        rows that do not open with their own node when any do not."""
        n, ja, rows = self.n, self.ja, self.rows
        print(f"{self.fname}: {n} nodes, {self.nnz} entries")
        lengths = np.diff(self.ia)
        if lengths.min() == lengths.max():
            print(f"  rows: {lengths[0]} entries each")
        else:
            print(f"  rows: from {lengths.min()} to {lengths.max()} entries, mean {lengths.mean():.4g}")
        self_first = ja[self.ia[:-1]] == np.arange(n)
        if not self_first.all():
            has_self = np.isin(np.arange(n) * (n + 1), self.keys())
            print(f"  {np.count_nonzero(~self_first)} rows do not start with their own node, "
                  f"{np.count_nonzero(~has_self)} do not contain it")
        sym = np.count_nonzero(np.isin(ja * n + rows, self.keys()))
        print(f"  entries with their transpose stored: {sym} of {self.nnz} ({100 * sym / self.nnz:.1f}%)")
        print(f"  bandwidth: {np.abs(rows - ja).max()}")

    # drawing

    def spy(self, ax, title):
        """The sparsity pattern, one square per stored entry."""
        n = self.n
        width = ax.figure.get_size_inches()[0] * ax.get_position().width * 72   # the axes, in points
        ax.scatter(self.ja, self.rows, s=max(width / n, 0.8) ** 2, marker="s", color="black", linewidths=0)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{title}: {n} x {n}, nnz = {self.nnz}", fontsize=9)


# --------------------------------------------------------------- ordering

class Ordering:
    """A renumbering from an ordering file: iperm[i] is the new index of
    node i, order[i] the old index of the node that lands at i."""

    def __init__(self, iperm, fname):
        self.iperm = iperm
        self.order = np.argsort(iperm)
        self.fname = fname

    def __len__(self):
        return self.iperm.size

    @classmethod
    def read(cls, fname, rep):
        """One integer per line, no header: an Ordering, or None unless the
        values are a permutation."""
        vals = parse_rows(fname, rep, lines_of(fname, comments=False), "one integer",
                          lambda toks: nums(toks, int) if len(toks) == 1 else None)
        if vals is None:
            return None
        iperm = np.array(vals, dtype=int).reshape(-1)
        n = iperm.size
        if np.array_equal(np.sort(iperm), np.arange(n)):
            return cls(iperm, fname)
        if n and iperm.min() == 1 and iperm.max() == n:
            rep.problem(fname, f"values run from 1 to {n}: the file looks 1-based, the format is 0-based")
        else:
            rep.problem(fname, f"the {n} values are not a permutation of 0 .. {n - 1}")
        return None

    def describe(self):
        print(f"{self.fname}: a permutation of {len(self)} nodes")


# ------------------------------------------------------------ command line

KINDS = {".points": Nodes, ".node": Nodes, ".graph": Graph, ".iperm": Ordering}


def parse_args():
    """The command line, with `given` mapping each file's class to its name
    and `period` the periodic box as an array or None."""
    ap = argparse.ArgumentParser(
        description="Check and describe the files of a case (docs/file_formats.md): "
                    "node and marker counts, nearest-neighbour statistics, graph "
                    "statistics. Never writes a file.")
    ap.add_argument("files", nargs="+", metavar="FILE",
                    help="one each of .points or .node, .graph and .iperm; "
                         "an ordering is applied to the others")
    ap.add_argument("--plot", action="store_true", help="draw the nodes, coloured by marker")
    ap.add_argument("--labels", action="store_true", help="write the index next to every node")
    ap.add_argument("--stencil", nargs="+", type=int, default=[], metavar="I",
                    help="draw the stencils of these nodes, from the graph "
                         "or the --k nearest neighbours")
    ap.add_argument("--k", type=int, default=0, metavar="K",
                    help="stencil size for --stencil without a graph file")
    ap.add_argument("--periodic", nargs=2, type=float, metavar=("LX", "LY"),
                    help="the periodic box [0, LX) x [0, LY): minimum-image distances")
    ap.add_argument("--spy", action="store_true",
                    help="draw the sparsity pattern of the graph, before and after an ordering file")
    ap.add_argument("--save", metavar="FILE", help="write the figure to FILE instead of showing it")
    args = ap.parse_args()
    args.plot = args.plot or args.labels or bool(args.stencil)
    args.given = classify(args.files)
    check_options(args)
    args.period = np.array(args.periodic) if args.periodic else None
    return args


def classify(files):
    """The class of each file by its extension, {class: file name}; exits on
    an unknown extension or a second file of a kind."""
    given = {}
    for f in files:
        kind = KINDS.get(os.path.splitext(f)[1])
        if kind is None:
            sys.exit(f"{f}: unknown extension, expected .points, .node, .graph or .iperm")
        if kind in given:
            sys.exit(f"{f}: a {kind.__name__.lower()} file was already given, {given[kind]}")
        given[kind] = f
    return given


def check_options(args):
    """Exit on an option that lacks the file it works on."""
    if args.spy and Graph not in args.given:
        sys.exit("--spy needs a graph file")
    if args.plot and Nodes not in args.given:
        sys.exit("--plot needs a points or node file")
    if args.stencil and Graph not in args.given and args.k < 2:
        sys.exit("--stencil needs a graph file, or --k of at least 2 to build the stencils")
    if args.periodic and min(args.periodic) <= 0:
        sys.exit("--periodic: the box sides must be positive")


# ------------------------------------------------------------------- main

def read_files(args, rep):
    """Read and describe every file given, with its notes under it:
    {class: object} for the files that could be read."""
    read = {}
    for kind, fname in args.given.items():
        obj = kind.read(fname, rep, args.period) if kind is Nodes else kind.read(fname, rep)
        if obj is not None:
            read[kind] = obj
            obj.describe()
        rep.print_notes()
    return read


def check_counts(read, rep):
    """A problem if the files disagree on the node count."""
    counts = {obj.fname: len(obj) for obj in read.values()}
    if len(set(counts.values())) > 1:
        rep.problem("files", "node counts differ: " + ", ".join(f"{f} has {n}" for f, n in counts.items()))


def stencils_to_draw(args, nodes, graph, rep):
    """The (node, members) stencils asked for, from the graph or as the k
    nearest neighbours; exits on a node index outside the cloud."""
    if not args.stencil:
        return []
    bad = [i for i in args.stencil if not 0 <= i < len(nodes)]
    if bad:
        sys.exit(f"--stencil: node {bad[0]} is outside [0, {len(nodes)})")
    if graph is not None:
        if args.k:
            rep.note("--k ignored: the stencils are taken from the graph file")
        return [(i, graph.row(i)) for i in args.stencil]
    j, _ = nodes.neighbours(args.k - 1, args.stencil)
    return [(i, np.concatenate(([i], jj))) for i, jj in zip(args.stencil, j)]


def draw(args, read, rep):
    """One figure: the nodes, the spy plot, or both, renumbered by the
    ordering if one was given, and then the spy plot in file order too."""
    nodes, graph, ordering = read.get(Nodes), read.get(Graph), read.get(Ordering)
    file_graph = None
    if ordering is not None:
        nodes = nodes.renumbered(ordering) if nodes else None
        if graph is not None:
            file_graph, graph = graph, graph.renumbered(ordering)
        print("ordering applied: nodes and stencils are in the new numbering below")
    stencils = stencils_to_draw(args, nodes, graph, rep)
    rep.print_notes()

    import matplotlib.pyplot as plt
    panels = int(args.plot) + int(args.spy) * (2 if file_graph is not None else 1)
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 5.2), squeeze=False)
    axes = iter(axes[0])
    if args.plot:
        ax = next(axes)
        nodes.plot(ax, args.labels, stencils)
        ax.set_title(os.path.basename(nodes.fname), fontsize=9)
    if args.spy:
        base = os.path.basename(graph.fname)
        if file_graph is not None:
            file_graph.spy(next(axes), f"{base}, file order")
            graph.spy(next(axes), f"{base}, {os.path.basename(ordering.fname)}")
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
    rep = Report()
    read = read_files(args, rep)
    check_counts(read, rep)
    rep.print_problems()
    if args.plot or args.spy:
        if rep.problems:
            print("no figure: fix the problems above first")
        else:
            draw(args, read, rep)
    sys.exit(1 if rep.problems else 0)


if __name__ == "__main__":
    main()
