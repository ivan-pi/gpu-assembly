"""Stencil selection for the generators in data/gen. The stencil of a
node is the set of nodes it interpolates from, and there are three ways
to select it: its k nearest neighbours (knn), the nodes within a
distance of it (radius), or the nodes within a distance of it along
both axes, a square, which is an orthogonal range search (range).
scipy's k-d tree does the searching, over a box periodic in either
axis, so that a stencil next to a periodic side reaches around it.
Every stencil starts with the node itself, which is what the graph file
expects (docs/file_formats.md), and the stencils come back in the CSR
form the readers use: (ia, ja), with the stencil of node i at
ja[ia[i]:ia[i + 1]].

A generator offers the choice as one option, --stencil METHOD VALUE,
checks it against its box, and selects the stencils of the cloud:

    add_option(parser, default=("knn", 15))
    ...
    method, value = args.stencil
    problem = check(method, value, extent, periodic)
    if problem:
        parser.error(problem)
    ...
    ia, ja = select_stencils(pts, extent, periodic, method, value)

The module also fixes the boundary markers of the node files, the
convention the generators share: MARKERS.interior is 0, the walls are
numbered counter-clockwise from the bottom, then the corners of a
cavity, then a hole in the interior, and MARKERS.style colours them the
same in the --plot of every generator.
"""

import argparse
from collections import namedtuple

import numpy as np

METHODS = ("knn", "radius", "range")

Markers = namedtuple("Markers", "interior south east north west corner hole style")
MARKERS = Markers(0, 1, 2, 3, 4, 5, 6, style=dict(cmap="tab10", vmin=0, vmax=9))


def add_option(ap, default):
    """--stencil METHOD VALUE on the parser, `default` being the
    (method, value) pair used when it is not given."""
    ap.add_argument(
        "--stencil",
        action=StencilOption,
        nargs=2,
        default=default,
        metavar=("METHOD", "VALUE"),
        help="how the stencil of a node is selected: knn K, its K nearest "
        "nodes; radius R, the nodes within a distance R of it; range S, the "
        "nodes within S of it along both axes, a square of side 2 S "
        f"(default: {default[0]} {default[1]})",
    )


class StencilOption(argparse.Action):
    """--stencil METHOD VALUE as parsed: the pair (method, value), with
    the value an int for knn and a float for the other methods, checked
    the way argparse checks a type."""

    def __call__(self, parser, namespace, values, option_string=None):
        method, text = values
        try:
            value = parse_value(method, text)
        except ValueError as e:
            raise argparse.ArgumentError(self, str(e))
        setattr(namespace, self.dest, (method, value))


def parse_value(method, text):
    """The VALUE of --stencil for METHOD, or ValueError saying what is
    wrong with either."""
    if method not in METHODS:
        raise ValueError(f"the method is one of {', '.join(METHODS)}, not '{method}'")
    if method == "knn":
        if not text.isdigit() or int(text) < 1:
            raise ValueError(f"knn takes a number of nodes, at least 1, not '{text}'")
        return int(text)
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{method} takes a distance, not '{text}'") from None
    if value <= 0:
        raise ValueError(f"{method} takes a positive distance, not {text}")
    return value


def check(method, value, extent, periodic):
    """What is wrong with the selection on the box `extent` = (Lx, Ly),
    with `periodic` = (px, py) saying which axes wrap, or None: along a
    periodic axis a stencil cannot reach more than half way around,
    since beyond that a node meets its own image, which its stencil
    cannot hold twice."""
    if method == "knn":
        return None
    half = min((L for L, p in zip(extent, periodic) if p), default=np.inf) / 2
    if value > half:
        return (
            f"--stencil {method} {value:g} reaches more than half way around "
            f"the box, which is {2 * half:g} across: at most {half:g}"
        )
    return None


def select_stencils(pts, extent, periodic, method, value, order="distance"):
    """The stencils of all nodes of `pts` in CSR form, (ia, ja): the node
    itself first in each, then its neighbours nearest first, or by index
    with order="index". `extent` is the box (Lx, Ly) and `periodic` =
    (px, py) says which axes the search wraps around. Raises ValueError
    when two nodes coincide, which no stencil can tell apart."""
    from scipy.spatial import cKDTree

    if order not in ("distance", "index"):
        raise ValueError(f"order is 'distance' or 'index', not '{order}'")
    tree = cKDTree(pts, boxsize=[L if p else 0.0 for L, p in zip(extent, periodic)])
    if method == "knn":
        return nearest(tree, value, order)
    if method == "radius":
        return within(tree, value, 2, order)
    if method == "range":
        return within(tree, value, np.inf, order)  # the max norm bounds a square
    raise ValueError(f"the method is one of {', '.join(METHODS)}, not '{method}'")


def nearest(tree, k, order):
    """The k nearest nodes of every node of the tree, itself first."""
    n = tree.n
    assert k <= n, "a stencil larger than the cloud"
    dist, adj = tree.query(tree.data, max(k, 2), workers=-1)  # two, to see a twin
    refuse_twins(dist, adj)
    adj = adj[:, :k]
    if order == "index":
        adj[:, 1:] = np.sort(adj[:, 1:], axis=1)
    return np.arange(0, n * k + 1, k), adj.ravel()


def within(tree, distance, p, order):
    """The nodes within `distance` of every node of the tree in the
    p-norm, itself first, from the sparse matrix of all pairs of nodes
    that close."""
    n = tree.n
    refuse_twins(*tree.query(tree.data, 2, workers=-1))
    pairs = tree.sparse_distance_matrix(tree, distance, p=p, output_type="coo_matrix")
    keep = pairs.row != pairs.col  # the node itself goes first instead
    row, col, dist = pairs.row[keep], pairs.col[keep], pairs.data[keep]
    if order == "distance":
        o = np.lexsort((col, dist, row))  # by node, then by distance, then by index
    else:
        o = np.lexsort((col, row))  # by node, then by index
    row, col = row[o], col[o]
    first = np.searchsorted(row, np.arange(n + 1))  # where each node's pairs start
    return first + np.arange(n + 1), np.insert(col, first[:-1], np.arange(n))


def refuse_twins(dist, adj):
    """ValueError on a node whose second-nearest is at distance zero; the
    two at that distance are the first two of its row, in either order."""
    same = np.flatnonzero(dist[:, 1] == 0)
    if same.size:
        i, j = sorted(adj[same[0], :2].tolist())
        raise ValueError(f"nodes {i} and {j} coincide")
