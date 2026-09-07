"""Stencil selection. The stencil of a node is the set of nodes it
interpolates from, and there are three ways to select it: its k nearest
neighbours (knn), the nodes within a distance of it (radius), or the
nodes within a distance of it along both axes, a square, which is an
orthogonal range search (range). scipy's k-d tree does the searching,
over a box periodic in either axis, so that a stencil next to a
periodic side reaches around it. Every stencil starts with the node
itself, which is what the graph file expects (docs/file_formats.md),
and the stencils come back in the CSR form the readers use: (ia, ja),
with the stencil of node i at ja[ia[i]:ia[i + 1]].

A generator offers the choice as one option, --graph METHOD=VALUE, and
NodeSet.stencils runs the selection on its cloud and box:

    add_option(parser)
    ...
    ia, ja = cloud.stencils(*args.graph)
"""

import argparse

import numpy as np

METHODS = ("knn", "radius", "range")

# The default stencil: 18 nodes, the 6 terms of a second-order polynomial
# plus 12, for every generator.
DEFAULT_GRAPH = "knn=18"


def add_option(ap, default=DEFAULT_GRAPH):
    """--graph METHOD=VALUE on the parser, `default` being the text used
    when it is not given."""
    ap.add_argument(
        "--graph",
        type=parse_graph,
        default=default,
        metavar="METHOD=VALUE",
        help="how the stencil graph is selected: knn=K, the K nearest nodes "
        "of each; radius=R, the nodes within a distance R of it; range=S, "
        "the nodes within S of it along both axes, a square of side 2 S "
        f"(default: {default})",
    )


def parse_graph(text):
    """--graph METHOD=VALUE as parsed: the pair (method, value), with the
    value an int for knn and a distance for the other methods, or the
    ArgumentTypeError argparse reports as a usage error."""
    method, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError(
            f"expected METHOD=VALUE, like knn=21, not '{text}'"
        )
    if method not in METHODS:
        raise argparse.ArgumentTypeError(
            f"{method} is not a valid method ({', '.join(METHODS)})"
        )
    if method == "knn":
        if not value.isdigit() or int(value) < 1:
            raise argparse.ArgumentTypeError(
                f"knn takes a number of nodes, at least 1, not '{value}'"
            )
        return method, int(value)
    try:
        distance = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{method} takes a distance, not '{value}'")
    if distance <= 0:
        raise argparse.ArgumentTypeError(
            f"{method} takes a positive distance, not {value}"
        )
    return method, distance


def select_stencils(pts, boxsize, method, value, order="distance"):
    """The stencils of all nodes of `pts` in CSR form, (ia, ja): the node
    itself first in each, then its neighbours nearest first, or by index
    with order="index". `boxsize` gives the side of the box along every
    periodic axis, around which the search wraps, and 0 for an axis that
    does not. Raises ValueError when two nodes coincide, which no stencil
    can tell apart, and for a distance that reaches more than half way
    around the box, where a node meets its own image."""
    from scipy.spatial import cKDTree

    if order not in ("distance", "index"):
        raise ValueError(f"order is 'distance' or 'index', not '{order}'")
    n = len(pts)
    tree = cKDTree(pts, boxsize=boxsize)

    if method == "knn":
        assert value <= n, "a stencil larger than the cloud"
        dist, adj = tree.query(pts, max(value, 2), workers=-1)  # two, to see a twin
        refuse_twins(dist, adj)
        adj = adj[:, :value]
        if order == "index":
            adj[:, 1:] = np.sort(adj[:, 1:], axis=1)
        return np.arange(0, n * value + 1, value), adj.ravel()

    # radius and range: all pairs closer than the distance, in the norm
    # whose ball is the disk or the square
    half = min((L for L in boxsize if L > 0), default=np.inf) / 2
    if value > half:
        raise ValueError(
            f"a {method} of {value:g} reaches more than half way around the box, "
            f"which is {2 * half:g} across: at most {half:g}"
        )
    refuse_twins(*tree.query(pts, 2, workers=-1))
    p_norm = {"radius": 2, "range": np.inf}[method]
    pairs = tree.sparse_distance_matrix(tree, value, p=p_norm, output_type="coo_matrix")
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
