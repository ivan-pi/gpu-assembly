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


# ----------------------------------------------------------- text parsing

class Report:
    """The problems found so far. They are listed at the end and make the
    exit status 1; notes are printed where they arise."""

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
    """The integer a token stands for, or None."""
    try:
        return int(tok)
    except ValueError:
        return None


def as_float(tok):
    """The number a token stands for, or None."""
    try:
        return float(tok)
    except ValueError:
        return None


def ints_of(toks):
    """All tokens as integers, or None if one is not an integer."""
    vals = [as_int(t) for t in toks]
    return None if None in vals else vals


def floats_of(toks):
    """All tokens as numbers, or None if one is not a number."""
    vals = [as_float(t) for t in toks]
    return None if None in vals else vals


def read_header(fname, rep, form, comments):
    """The non-negative integers of the first line, and the lines after it:
    (values, rows), or None with the problem recorded. `form` names the
    fields, `n nnz` say, and sets how many there are."""
    lines = lines_of(fname, comments)
    if not lines:
        rep.problem(fname, "empty file")
        return None
    no, head = lines[0]
    vals = ints_of(head)
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
    """A point cloud with a marker per node, 0 for interior: the contents of
    a points file (every marker 0) or a node file."""

    def __init__(self, xy, marker):
        self.xy = xy
        self.marker = marker

    def __len__(self):
        return len(self.xy)

    # reading

    @classmethod
    def read(cls, fname, rep):
        """Nodes from a .points or .node file, or None with the problems
        recorded."""
        reader = cls._read_node_file if fname.endswith(".node") else cls._read_points_file
        got = reader(fname, rep)
        if got is None or not cls._finite(fname, rep, got[0]):
            return None
        return cls(*got)

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
                          lambda toks: floats_of(toks) if len(toks) == 2 else None)
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
        if not Nodes._node_header_ok(fname, rep, n, dim, nattr, nmark, len(rows)):
            return None
        form = f"i x y{' a' * nattr}{' marker' if nmark else ''}"
        vals = parse_rows(fname, rep, rows, form, lambda toks: Nodes._node_line(toks, nattr, nmark))
        if vals is None:
            return None
        table = np.array(vals, dtype=float).reshape(n, 4)
        if not Nodes._consecutive(fname, rep, table[:, 0].astype(int), rows):
            return None
        return table[:, 1:3], table[:, 3].astype(int)

    @staticmethod
    def _node_header_ok(fname, rep, n, dim, nattr, nmark, nrows):
        """Whether the node-file header can be used; notes what it says."""
        if dim != 2:
            rep.problem(fname, f"dimension {dim} in the header, only 2 is supported")
        elif nmark not in (0, 1):
            rep.problem(fname, f"header: nmark must be 0 or 1, got {nmark}")
        elif nrows != n:
            rep.problem(fname, f"header says {n} nodes, the file has {nrows} node lines")
        else:
            if nattr:
                rep.note(f"{nattr} attribute{'s' if nattr > 1 else ''} per node, not inspected")
            if not nmark:
                rep.note("no marker column: every node is interior")
            return True
        return False

    @staticmethod
    def _node_line(toks, nattr, nmark):
        """(index, x, y, marker) of a node line, or None if malformed."""
        if len(toks) != 3 + nattr + nmark:
            return None
        i, x, y = as_int(toks[0]), as_float(toks[1]), as_float(toks[2])
        m = as_int(toks[-1]) if nmark else 0
        if None in (i, x, y, m):
            return None
        return i, x, y, m

    @staticmethod
    def _consecutive(fname, rep, index, rows):
        """Whether the index column counts from 0 or from 1."""
        n = index.size
        if np.array_equal(index, np.arange(1, n + 1)):
            rep.note("nodes are numbered from 1 in the file; reported from 0 here")
        elif not np.array_equal(index, np.arange(n)):
            bad = np.flatnonzero(index != np.arange(n))[0]
            rep.problem(fname, f"node indices are not consecutive from 0 or 1: "
                        f"line {rows[bad][0]} has index {index[bad]}")
            return False
        return True

    @staticmethod
    def _finite(fname, rep, xy):
        """Whether every coordinate is finite; records the first that is not."""
        bad = np.flatnonzero(~np.isfinite(xy).all(axis=1))
        if bad.size:
            rep.problem(fname, f"{bad.size} node{'s' if bad.size > 1 else ''} with a non-finite "
                        f"coordinate, the first is node {bad[0]}")
        return not bad.size

    # neighbours

    def neighbours(self, k, period=None):
        """Indices and distances of the k nearest other nodes of every node,
        nearest first; minimum-image distances in the periodic box, if any."""
        k = min(k, len(self) - 1)
        if k < 1:
            return np.empty((len(self), 0), dtype=int), np.empty((len(self), 0))
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return self._neighbours_brute_force(k, period)
        xy = self.xy if period is None else np.mod(self.xy, period)
        tree = cKDTree(xy, boxsize=period)
        d, j = tree.query(tree.data, k + 1)
        return j[:, 1:], d[:, 1:]

    def _neighbours_brute_force(self, k, period):
        """The same from all pairwise distances, in chunks of rows."""
        n, xy = len(self), self.xy
        j = np.empty((n, k), dtype=int)
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

    def stencils(self, which, k, period=None):
        """(node, members) for the requested nodes, as the k nearest
        neighbours with the node itself first."""
        j, _ = self.neighbours(k - 1, period)
        return [(i, np.concatenate(([i], j[i]))) for i in which]

    def renumbered(self, ordering):
        """The same nodes in the new numbering."""
        return Nodes(self.xy[ordering.order], self.marker[ordering.order])

    # reporting

    def describe(self, fname, rep, period=None):
        """Print the counts, the bounding box and the spacing statistics."""
        self._print_counts(fname)
        self._print_box(period)
        if len(self) > 1:
            self._print_spacing(fname, rep, period)

    def _print_counts(self, fname):
        """The node count, interior against boundary, and the count per marker."""
        n, bnd = len(self), np.count_nonzero(self.marker)
        print(f"{fname}: {n} nodes, {n - bnd} interior, {bnd} boundary")
        values, counts = np.unique(self.marker, return_counts=True)
        print("  markers: " + "  ".join(f"{v}: {c}" for v, c in zip(values, counts)))

    def _print_box(self, period):
        """The bounding box, and the periodic box if one was declared."""
        lo, hi = self.xy.min(axis=0), self.xy.max(axis=0)
        print(f"  bounding box: x in [{lo[0]:.6g}, {hi[0]:.6g}], y in [{lo[1]:.6g}, {hi[1]:.6g}]")
        if period is not None:
            outside = np.count_nonzero((self.xy < 0).any(axis=1) | (self.xy >= period).any(axis=1))
            print(f"  periodic box [0, {period[0]:.6g}) x [0, {period[1]:.6g}): distances are minimum-image"
                  + (f"; {outside} nodes lie outside the box" if outside else ""))

    def _print_spacing(self, fname, rep, period):
        """The nearest-neighbour distance statistics; coincident nodes are a problem."""
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

    # drawing

    PALETTE = ["tab:red", "tab:green", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]

    def plot(self, ax, labels=False, stencils=(), period=None):
        """The nodes coloured by marker, optionally every index and the given
        (node, members) stencils."""
        self._plot_markers(ax)
        if labels:
            self._plot_labels(ax)
        for s, (i, members) in enumerate(stencils):
            self._plot_stencil(ax, i, members, self.PALETTE[s % len(self.PALETTE)], period)
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

    def _plot_labels(self, ax):
        """The index next to every node."""
        for i, (x, y) in enumerate(self.xy):
            ax.annotate(str(i), (x, y), xytext=(2, 2), textcoords="offset points", fontsize=6)

    def _plot_stencil(self, ax, i, members, color, period):
        """A circle about node i through its farthest member, the members
        ringed and the node filled; a wrapped member sits at its nearest image."""
        from matplotlib.patches import Circle
        centre = self.xy[i]
        at = centre + minimum_image(self.xy[members] - centre, period)
        r = np.sqrt(((at - centre) ** 2).sum(axis=1)).max()
        ax.add_patch(Circle(centre, r, fill=False, edgecolor=color, lw=1.2))
        ax.plot(at[:, 0], at[:, 1], "o", ms=8, mfc="none", mec=color, mew=1.2)
        ax.plot(centre[0], centre[1], "o", ms=8, color=color, label=f"stencil of node {i} ({len(members)})")
        ax.annotate(str(i), centre, xytext=(5, 5), textcoords="offset points", fontsize=8, color=color)


# ------------------------------------------------------------------ graph

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
        """The stencil of node i."""
        return self.ja[self.ia[i]:self.ia[i + 1]]

    def row_of_entry(self):
        """The row index of every stored entry, parallel to ja."""
        return np.repeat(np.arange(self.n), np.diff(self.ia))

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
        stencils = parse_rows(fname, rep, rows, "integer stencil entries", ints_of)
        if stencils is None:
            return None
        g = cls.from_rows(n, stencils)
        if g.nnz != nnz:
            rep.problem(fname, f"header says nnz = {nnz}, the stencils hold {g.nnz} entries")
        if not g._in_range(fname, rep):
            return None
        g._check_rows(fname, rep, [no for no, _ in rows])
        return g

    @classmethod
    def from_rows(cls, n, stencils):
        """CSR arrays from one list of indices per node."""
        ia = np.concatenate(([0], np.cumsum([len(s) for s in stencils]))).astype(int)
        ja = np.array([j for s in stencils for j in s], dtype=int)
        return cls(n, ia, ja)

    def _in_range(self, fname, rep):
        """Whether every entry is in [0, n); a 1-based file is told apart."""
        n, ja = self.n, self.ja
        if not ja.size or (ja.min() >= 0 and ja.max() < n):
            return True
        if ja.min() >= 1 and ja.max() == n:
            rep.problem(fname, f"indices run from {ja.min()} to {n}: the file looks 1-based, "
                        f"the format is 0-based")
        else:
            rep.problem(fname, f"stencil entries outside [0, {n}): from {ja.min()} to {ja.max()}")
        return False

    def _check_rows(self, fname, rep, line_numbers):
        """Records the empty stencils and the stencils listing a node twice."""
        for i in np.flatnonzero(np.diff(self.ia) == 0):
            rep.problem(fname, f"line {line_numbers[i]}: node {i} has an empty stencil")
        dup = [i for i in range(self.n) if np.unique(self.row(i)).size != self.row(i).size]
        if dup:
            rep.problem(fname, f"{len(dup)} stencil{'s list' if len(dup) > 1 else ' lists'} a node twice, "
                        f"the first on line {line_numbers[dup[0]]}")

    # renumbering

    def renumbered(self, ordering):
        """The symmetric renumbering: rows moved and every entry relabelled."""
        order, iperm = ordering.order, ordering.iperm
        return Graph.from_rows(self.n, [iperm[self.row(o)] for o in order])

    def stencils(self, which):
        """(node, members) for the requested nodes."""
        return [(i, self.row(i)) for i in which]

    # reporting

    def describe(self, fname):
        """Print the size, the row lengths, the diagonal, symmetry and bandwidth."""
        print(f"{fname}: {self.n} nodes, {self.nnz} entries")
        self._print_row_lengths()
        self._print_diagonal()
        self._print_symmetry()
        i = self.row_of_entry()
        print(f"  bandwidth: {np.abs(i - self.ja).max() if self.nnz else 0}")

    def _print_row_lengths(self):
        """One length if every stencil has the same size, else the range."""
        lengths = np.diff(self.ia)
        if lengths.min() == lengths.max():
            print(f"  rows: {lengths[0]} entries each")
        else:
            print(f"  rows: from {lengths.min()} to {lengths.max()} entries, mean {lengths.mean():.4g}")

    def _print_diagonal(self):
        """Whether every stencil starts with, or at least contains, its own node."""
        n, ia, ja = self.n, self.ia, self.ja
        nonempty = np.diff(ia) > 0
        self_first = ja[ia[:-1][nonempty]] == np.arange(n)[nonempty]
        has_self = np.zeros(n, dtype=bool)
        i = self.row_of_entry()
        has_self[i[ja == i]] = True
        if self_first.all():
            print("  every row starts with its own node")
        else:
            print(f"  {np.count_nonzero(~self_first)} rows do not start with their own node, "
                  f"{np.count_nonzero(~has_self)} do not contain it")

    def _print_symmetry(self):
        """How many entries (i, j) have (j, i) stored as well."""
        i, ja, n = self.row_of_entry(), self.ja, self.n
        sym = np.count_nonzero(np.isin(ja * n + i, i * n + ja))
        print(f"  entries with their transpose stored: {sym} of {self.nnz} ({100 * sym / max(self.nnz, 1):.1f}%)")

    # drawing

    def spy(self, ax, title):
        """The sparsity pattern, one square per stored entry."""
        n = self.n
        cell = 0.9 * 72 * 4.5 / max(n, 1)             # about one cell of the axes, in points
        ax.scatter(self.ja, self.row_of_entry(), s=max(cell, 0.8) ** 2, marker="s",
                   color="black", linewidths=0)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(f"{title}: {n} x {n}, nnz = {self.nnz}", fontsize=9)


# --------------------------------------------------------------- ordering

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
        """One integer per line, no header: an Ordering, or None unless the
        values are a permutation."""
        vals = parse_rows(fname, rep, lines_of(fname, comments=False), "one integer",
                          lambda toks: as_int(toks[0]) if len(toks) == 1 else None)
        if vals is None:
            return None
        iperm = np.array(vals, dtype=int)
        return cls(iperm) if cls._is_permutation(fname, rep, iperm) else None

    @staticmethod
    def _is_permutation(fname, rep, iperm):
        """Whether the values are 0 .. n-1 in some order; a 1-based file is told apart."""
        n = iperm.size
        if np.array_equal(np.sort(iperm), np.arange(n)):
            return True
        if n and iperm.min() == 1 and iperm.max() == n:
            rep.problem(fname, f"values run from 1 to {n}: the file looks 1-based, the format is 0-based")
        else:
            rep.problem(fname, f"the {n} values are not a permutation of 0 .. {n - 1}")
        return False

    def describe(self, fname):
        print(f"{fname}: a permutation of {len(self)} nodes")


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
    if (args.plot or args.stencil) and Nodes not in args.given:
        sys.exit("--plot needs a points or node file")
    if args.stencil and Graph not in args.given and args.k < 2:
        sys.exit("--stencil needs a graph file, or --k of at least 2 to build the stencils")
    if args.periodic and min(args.periodic) <= 0:
        sys.exit("--periodic: the box sides must be positive")


# ------------------------------------------------------------------- main

def read_files(args, rep):
    """Read and describe every file given: {class: object} for those that
    could be read."""
    read = {}
    for kind, fname in args.given.items():
        obj = kind.read(fname, rep)
        if obj is None:
            continue
        read[kind] = obj
        if kind is Nodes:
            obj.describe(fname, rep, args.period)
        else:
            obj.describe(fname)
    return read


def check_counts(args, read, rep):
    """A problem if the files disagree on the node count."""
    counts = {args.given[kind]: len(obj) for kind, obj in read.items()}
    if len(set(counts.values())) > 1:
        rep.problem("files", "node counts differ: " + ", ".join(f"{f} has {n}" for f, n in counts.items()))


def print_problems(rep):
    """The list of problems, or that there were none."""
    if not rep.problems:
        print("checks: no problems found")
        return
    print(f"{len(rep.problems)} problem{'s' if len(rep.problems) > 1 else ''}:")
    for p in rep.problems:
        print(f"  {p}")


def stencils_to_draw(args, nodes, graph, rep):
    """The (node, members) stencils asked for, from the graph or as the k
    nearest neighbours; exits on a node index outside the cloud."""
    bad = [i for i in args.stencil if not 0 <= i < len(nodes)]
    if bad:
        sys.exit(f"--stencil: node {bad[0]} is outside [0, {len(nodes)})")
    if not args.stencil:
        return []
    if graph is None:
        return nodes.stencils(args.stencil, args.k, args.period)
    if args.k:
        rep.note("--k ignored: the stencils are taken from the graph file")
    return graph.stencils(args.stencil)


def draw(args, nodes, graph, file_graph, rep):
    """One figure: the nodes, the spy plot, or both; the spy plot of the
    graph in file order as well when an ordering was applied."""
    import matplotlib.pyplot as plt
    reordered = graph is not file_graph
    panels = int(args.plot) + int(args.spy) * (2 if reordered else 1)
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 5.2), squeeze=False)
    axes = list(axes[0])
    if args.plot:
        ax = axes.pop(0)
        nodes.plot(ax, args.labels, stencils_to_draw(args, nodes, graph, rep), args.period)
        ax.set_title(os.path.basename(args.given[Nodes]), fontsize=9)
    if args.spy:
        base = os.path.basename(args.given[Graph])
        if reordered:
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
    read = read_files(args, rep)
    check_counts(args, read, rep)
    print_problems(rep)

    nodes, graph, ordering = read.get(Nodes), read.get(Graph), read.get(Ordering)
    consistent = not rep.problems and (nodes is not None or graph is not None)
    file_graph = graph
    if ordering and consistent:
        nodes = nodes.renumbered(ordering) if nodes else None
        graph = graph.renumbered(ordering) if graph else None
        print("ordering applied: nodes and stencils are in the new numbering below")

    if args.plot or args.spy:
        if consistent:
            draw(args, nodes, graph, file_graph, rep)
        else:
            print("no figure: fix the problems above first")
    sys.exit(1 if rep.problems else 0)


if __name__ == "__main__":
    main()
