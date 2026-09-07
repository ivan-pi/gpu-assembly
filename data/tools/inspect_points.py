#!/usr/bin/env python3
"""Inspect the point, node, graph and ordering files of docs/file_formats.md.
The tool reads and reports; it never writes a file.

    python3 inspect_points.py case.points
    python3 inspect_points.py case.node --plot
    python3 inspect_points.py case.node --plot --labels
    python3 inspect_points.py case.points case.graph --plot --stencil 0 17
    python3 inspect_points.py case.node --plot --stencil 0 --k 21
    python3 inspect_points.py case.graph --spy
    python3 inspect_points.py case.points case.graph case.iperm --spy
    python3 inspect_points.py case.points case.graph --periodic 32 32 --plot --stencil 0

Give one file of each kind, told apart by extension: .points or .node
for the coordinates, .graph for the stencils, .iperm for an ordering.
Each file is checked as it is read -- the counts against the header, the
indices in range, no entry twice in a stencil, the ordering a
permutation -- and the files against each other, for the same node
count. Every problem found is listed at the end and the exit status is 1
if there was any.

Printed: the node count, the boundary nodes by marker, the bounding
box, the nearest-neighbour distance (min, max, mean, median, std and
the closest pair), and for a graph the row lengths, whether every row
starts with its own node, the share of entries whose transpose is also
stored, and the bandwidth.

--plot draws the nodes coloured by marker. --labels writes the index
next to every node. --stencil draws the stencils of the listed nodes: a
circle through the farthest neighbour, the members ringed, the centre
filled. The stencils come from the graph file, or are the --k nearest
neighbours if there is no graph. --spy draws the sparsity pattern of
the graph, and with an ordering file the pattern before and after the
renumbering. The figures go to the screen, or to a file with --save.

An ordering file is applied to everything it can be: nodes and graph
are reported and drawn in the new numbering, so --labels shows the new
indices, and only the spy plot also shows the old.

--periodic LX LY declares the periodic box [0, LX) x [0, LY): distances
are then minimum-image, so the nearest-neighbour statistics do not see
the box edges, and a stencil that wraps around is drawn about its
centre, with the members at their nearest image.

Needs numpy. scipy speeds up the neighbour search; matplotlib is needed
for the figures.
"""

import argparse
import os
import sys

import numpy as np


class Report:
    """Problems make the exit status 1; notes are just printed."""

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


# ---------------------------------------------------------------- readers

def read_points(fname, rep):
    """The points file: n, then n lines of `x y`. Returns (xy, marker) with
    every marker 0, or None."""
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
    if not ok:
        return None
    return check_finite(fname, rep, xy, np.zeros(n, dtype=int))


def read_nodes(fname, rep):
    """The node file: a Triangle-style header `n 2 nattr nmark`, then
    `i x y a... [marker]` per node. Returns (xy, marker) or None."""
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
    if np.array_equal(index, np.arange(n)):
        pass
    elif np.array_equal(index, np.arange(1, n + 1)):
        rep.note("nodes are numbered from 1 in the file; reported from 0 here")
    else:
        bad = np.flatnonzero((index != np.arange(n)) & (index != np.arange(1, n + 1)))
        rep.problem(fname, f"node indices are not consecutive from 0 or 1: "
                    f"line {rows[bad[0]][0]} has index {index[bad[0]]}")
        return None
    if not nmark:
        rep.note("no marker column: every node is interior")
    return check_finite(fname, rep, xy, marker)


def check_finite(fname, rep, xy, marker):
    bad = np.flatnonzero(~np.isfinite(xy).all(axis=1))
    if bad.size:
        rep.problem(fname, f"{bad.size} node{'s' if bad.size > 1 else ''} with a non-finite "
                    f"coordinate, the first is node {bad[0]}")
        return None
    return xy, marker


def read_graph(fname, rep):
    """The graph file: `n nnz`, then the stencil of every node as a line of
    0-based indices. Returns (n, ia, ja) in CSR form, or None."""
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
    dup = 0
    first_dup = None
    for i in range(n):
        row = ja[ia[i]:ia[i + 1]]
        if row.size == 0:
            rep.problem(fname, f"line {rows[i][0]}: node {i} has an empty stencil")
            continue
        if np.unique(row).size != row.size:
            dup += 1
            first_dup = first_dup or rows[i][0]
    if dup:
        rep.problem(fname, f"{dup} stencil{'s list' if dup > 1 else ' lists'} a node twice, "
                    f"the first on line {first_dup}")
    return n, ia, ja


def read_ordering(fname, rep):
    """The ordering file: one new index per node, no header. Returns the
    old-to-new map, or None."""
    lines = lines_of(fname, comments=False)
    vals = []
    for no, toks in lines:
        v = as_int(toks[0]) if len(toks) == 1 else None
        if v is None:
            rep.problem(fname, f"line {no}: expected one integer, got '{' '.join(toks)}'")
            return None
        vals.append(v)
    iperm = np.array(vals, dtype=int)
    n = iperm.size
    if not (np.sort(iperm) == np.arange(n)).all():
        if n and iperm.min() == 1 and iperm.max() == n:
            rep.problem(fname, f"values run from 1 to {n}: the file looks 1-based, the format is 0-based")
        else:
            rep.problem(fname, f"the {n} values are not a permutation of 0 .. {n - 1}")
        return None
    return iperm


# ------------------------------------------------------------- statistics

def nearest_neighbours(xy, k, period=None):
    """Indices and distances of the k nearest other points of every point,
    nearest first; minimum-image distances in the periodic box, if any."""
    n = len(xy)
    k = min(k, n - 1)
    if k < 1:
        return np.empty((n, 0), dtype=int), np.empty((n, 0))
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        if period is None:
            tree = cKDTree(xy)
        else:
            tree = cKDTree(np.mod(xy, period), boxsize=period)
        d, j = tree.query(tree.data, k + 1)
        return j[:, 1:], d[:, 1:]
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


def minimum_image(diff, period):
    if period is None:
        return diff
    return diff - period * np.round(diff / period)


def report_nodes(fname, rep, xy, marker, period):
    n = len(xy)
    bnd = np.count_nonzero(marker)
    print(f"{fname}: {n} nodes, {n - bnd} interior, {bnd} boundary")
    values, counts = np.unique(marker, return_counts=True)
    print("  markers: " + "  ".join(f"{v}: {c}" for v, c in zip(values, counts)))
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    print(f"  bounding box: x in [{lo[0]:.6g}, {hi[0]:.6g}], y in [{lo[1]:.6g}, {hi[1]:.6g}]")
    if period is not None:
        outside = np.count_nonzero((xy < 0).any(axis=1) | (xy >= period).any(axis=1))
        print(f"  periodic box [0, {period[0]:.6g}) x [0, {period[1]:.6g}): "
              f"distances are minimum-image" + (f"; {outside} nodes lie outside the box" if outside else ""))
    if n < 2:
        return
    j, d = nearest_neighbours(xy, 1, period)
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


def report_graph(fname, rep, n, ia, ja):
    lengths = np.diff(ia)
    nnz = ia[-1]
    print(f"{fname}: {n} nodes, {nnz} entries")
    if lengths.min() == lengths.max():
        print(f"  rows: {lengths[0]} entries each")
    else:
        print(f"  rows: from {lengths.min()} to {lengths.max()} entries, mean {lengths.mean():.4g}")
    i = np.repeat(np.arange(n), lengths)
    self_first = ja[ia[:-1][lengths > 0]] == np.arange(n)[lengths > 0]
    has_self = np.zeros(n, dtype=bool)
    has_self[i[ja == i]] = True
    if self_first.all():
        print("  every row starts with its own node")
    else:
        print(f"  {np.count_nonzero(~self_first)} rows do not start with their own node, "
              f"{np.count_nonzero(~has_self)} do not contain it")
    keys = i * n + ja
    transposed = ja * n + i
    sym = np.count_nonzero(np.isin(transposed, keys))
    print(f"  entries with their transpose stored: {sym} of {nnz} ({100 * sym / max(nnz, 1):.1f}%)")
    print(f"  bandwidth: {np.abs(i - ja).max() if nnz else 0}")


# --------------------------------------------------------------- ordering

def apply_ordering(iperm, nodes, graph):
    """iperm[i] is the new index of node i. Nodes are moved and stencil
    entries relabelled; both return in the new numbering."""
    order = np.argsort(iperm)                     # new index -> old index
    if nodes is not None:
        xy, marker = nodes
        nodes = xy[order], marker[order]
    if graph is not None:
        n, ia, ja = graph
        lengths = np.diff(ia)[order]
        ia_new = np.concatenate(([0], np.cumsum(lengths)))
        ja_new = np.concatenate([iperm[ja[ia[o]:ia[o + 1]]] for o in order])
        graph = n, ia_new, ja_new
    return nodes, graph


# ------------------------------------------------------------------ plots

def stencils_of(graph, xy, which, k, period, rep):
    """The member lists of the requested stencils, from the graph or as
    the k nearest neighbours."""
    n = len(xy)
    out = []
    bad = [i for i in which if not 0 <= i < n]
    if bad:
        sys.exit(f"--stencil: node {bad[0]} is outside [0, {n})")
    if graph is not None:
        _, ia, ja = graph
        if k:
            rep.note("--k ignored: the stencils are taken from the graph file")
        return [(i, ja[ia[i]:ia[i + 1]]) for i in which]
    if not k:
        sys.exit("--stencil needs a graph file, or --k to build the stencils")
    j, _ = nearest_neighbours(xy, k - 1, period)
    for i in which:
        out.append((i, np.concatenate(([i], j[i]))))
    return out


def plot_nodes(ax, xy, marker, labels, stencils, period):
    from matplotlib.patches import Circle
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


def plot_spy(ax, n, ia, ja, title):
    i = np.repeat(np.arange(n), np.diff(ia))
    cell = 0.9 * 72 * 4.5 / max(n, 1)             # about one cell of the axes, in points
    ax.scatter(ja, i, s=max(cell, 0.8) ** 2, marker="s", color="black", linewidths=0)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(f"{title}: {n} x {n}, nnz = {ia[-1]}", fontsize=9)


# ------------------------------------------------------------------- main

KINDS = {".points": "points", ".node": "points", ".graph": "graph", ".iperm": "ordering"}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] + "\n\n" + __doc__.split("\n\n")[2],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", metavar="FILE",
                    help="a .points or .node file, a .graph file, an .iperm file; one of each at most")
    ap.add_argument("--plot", action="store_true", help="draw the nodes, coloured by marker")
    ap.add_argument("--labels", action="store_true", help="with --plot: write the index next to every node")
    ap.add_argument("--stencil", nargs="+", type=int, default=[], metavar="I",
                    help="with --plot: draw the stencils of these nodes")
    ap.add_argument("--k", type=int, default=0, metavar="K",
                    help="stencil size for --stencil when there is no graph file")
    ap.add_argument("--periodic", nargs=2, type=float, metavar=("LX", "LY"),
                    help="the periodic box [0, LX) x [0, LY): distances are minimum-image")
    ap.add_argument("--spy", action="store_true",
                    help="draw the sparsity pattern of the graph; before and after an ordering file")
    ap.add_argument("--save", metavar="FILE", help="write the figure to FILE instead of showing it")
    args = ap.parse_args()
    period = None
    if args.periodic:
        if min(args.periodic) <= 0:
            sys.exit("--periodic: the box sides must be positive")
        period = np.array(args.periodic)

    given = {}
    for f in args.files:
        kind = KINDS.get(os.path.splitext(f)[1])
        if kind is None:
            sys.exit(f"{f}: unknown extension, expected .points, .node, .graph or .iperm")
        if kind in given:
            sys.exit(f"{f}: a {kind} file was already given, {given[kind]}")
        given[kind] = f
    if args.spy and "graph" not in given:
        sys.exit("--spy needs a graph file")
    if (args.plot or args.stencil) and "points" not in given:
        sys.exit("--plot needs a points or node file")

    rep = Report()
    nodes = graph = iperm = None
    if "points" in given:
        f = given["points"]
        nodes = (read_nodes if f.endswith(".node") else read_points)(f, rep)
        if nodes is not None:
            report_nodes(f, rep, *nodes, period)
    if "graph" in given:
        graph = read_graph(given["graph"], rep)
        if graph is not None:
            report_graph(given["graph"], rep, *graph)
    if "ordering" in given:
        iperm = read_ordering(given["ordering"], rep)
        if iperm is not None:
            print(f"{given['ordering']}: a permutation of {iperm.size} nodes")

    counts = {}
    if nodes is not None:
        counts[given["points"]] = len(nodes[0])
    if graph is not None:
        counts[given["graph"]] = graph[0]
    if iperm is not None:
        counts[given["ordering"]] = iperm.size
    if len(set(counts.values())) > 1:
        rep.problem("files", "node counts differ: " + ", ".join(f"{f} has {n}" for f, n in counts.items()))

    if rep.problems:
        print(f"{len(rep.problems)} problem{'s' if len(rep.problems) > 1 else ''}:")
        for p in rep.problems:
            print(f"  {p}")
    else:
        print("checks: no problems found")

    consistent = not rep.problems and (nodes is not None or graph is not None)
    if iperm is not None and consistent:
        old_graph = graph
        nodes, graph = apply_ordering(iperm, nodes, graph)
        if nodes is not None:
            print("ordering applied: nodes and stencils are in the new numbering below")

    if (args.plot or args.spy) and consistent:
        import matplotlib.pyplot as plt
        panels = int(args.plot) + int(args.spy) * (2 if iperm is not None else 1)
        fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 5.2), squeeze=False)
        axes = list(axes[0])
        if args.plot:
            xy, marker = nodes
            stencils = stencils_of(graph, xy, args.stencil, args.k, period, rep) if args.stencil else []
            ax = axes.pop(0)
            plot_nodes(ax, xy, marker, args.labels, stencils, period)
            ax.set_title(os.path.basename(given["points"]), fontsize=9)
        if args.spy:
            base = os.path.basename(given["graph"])
            if iperm is not None:
                plot_spy(axes.pop(0), *old_graph, f"{base}, file order")
                plot_spy(axes.pop(0), *graph, f"{base}, {os.path.basename(given['ordering'])}")
            else:
                plot_spy(axes.pop(0), *graph, base)
        fig.tight_layout()
        if args.save:
            fig.savefig(args.save, dpi=150)
            print(f"figure written to {args.save}")
        else:
            plt.show()
    elif (args.plot or args.spy) and not consistent:
        print("no figure: fix the problems above first")

    sys.exit(1 if rep.problems else 0)


if __name__ == "__main__":
    main()
