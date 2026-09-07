"""The stencil search the generators in data/gen share. The stencil of a
node is the set of nodes it interpolates from: its k nearest neighbours,
the nodes within a distance, or those within a square, found by scipy's
k-d tree over a box periodic in either axis, so that a stencil next to a
periodic side reaches around it. Every stencil starts with the node
itself, which is what the graph file expects (docs/file_formats.md), and
the stencils come back in the CSR form the readers use: (ia, ja), with
the stencil of node i at ja[ia[i]:ia[i + 1]].

A generator adds the three options that pick one with `add_options`,
reads the choice back with `from_args`, checks it against its box with
`check`, and runs `search` on every cloud it makes:

    add_options(parser, knn=15)
    ...
    stencil = from_args(args)
    problem = check(stencil, extent, periodic)
    if problem:
        parser.error(problem)
    ...
    ia, ja = search(pts, extent, periodic, stencil)
"""

from collections import namedtuple

import numpy as np

from .cli import number

__all__ = ["Stencil", "add_options", "from_args", "check", "search"]


# A choice of stencil: kind is "knn", "radius" or "square", and reach the
# k of knn or the distance of the other two.
Stencil = namedtuple("Stencil", "kind reach")


def add_options(ap, knn, why=""):
    """--knn, --radius and --square, mutually exclusive, on the parser;
    `knn` is the stencil size used when none is given, and `why` says
    where it comes from, in the help."""
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--knn",
        type=number(int, least=1),
        default=knn,
        metavar="K",
        help=f"stencil of the K nearest nodes (default: {knn}{', ' + why if why else ''})",
    )
    group.add_argument(
        "--radius",
        type=number(float, above=0.0),
        metavar="R",
        help="stencil of the nodes within a distance R",
    )
    group.add_argument(
        "--square",
        type=number(float, above=0.0),
        metavar="S",
        help="stencil of the nodes within S in both x and y, a square of side 2 S",
    )


def from_args(args):
    """The Stencil the parsed options ask for."""
    if args.radius is not None:
        return Stencil("radius", args.radius)
    if args.square is not None:
        return Stencil("square", args.square)
    return Stencil("knn", args.knn)


def check(stencil, extent, periodic):
    """What is wrong with the stencil on the box `extent`, periodic per
    axis as `periodic` says, or None: along a periodic axis a stencil
    cannot reach more than half way around, since beyond that a node
    meets its own image, which its stencil cannot hold twice."""
    kind, reach = stencil
    periodic = np.broadcast_to(np.asarray(periodic, bool), 2)
    if kind == "knn" or not periodic.any():
        return None
    half = 0.5 * np.asarray(extent, float)[periodic].min()
    if reach > half:
        return (
            f"--{kind} {reach:g} reaches more than half way around the box, "
            f"which is {2 * half:g} across: at most {half:g}"
        )
    return None


def search(pts, extent, periodic, stencil):
    """The stencils of all nodes of `pts` in CSR form, (ia, ja), the node
    itself first in each. `extent` is the box (Lx, Ly) and `periodic`
    says per axis, or for both at once, whether the search wraps around
    it. Raises ValueError for a stencil larger than the cloud, and when
    two nodes coincide, which no stencil can tell apart."""
    from scipy.spatial import cKDTree

    n = len(pts)
    kind, reach = stencil
    periodic = np.broadcast_to(np.asarray(periodic, bool), 2)
    tree = cKDTree(pts, boxsize=np.where(periodic, np.asarray(extent, float), 0.0))
    if kind == "knn":
        if reach > n:
            raise ValueError(
                f"a stencil of {reach} nodes is larger than the cloud of {n}"
            )
        dist, adj = tree.query(pts, max(reach, 2), workers=-1)  # two, to see a twin
        _refuse_twins(dist, adj)
        return np.arange(0, n * reach + 1, reach), adj[:, :reach].ravel()
    _refuse_twins(*tree.query(pts, 2, workers=-1))
    norm = np.inf if kind == "square" else 2  # the max norm bounds a square
    pairs = tree.sparse_distance_matrix(tree, reach, p=norm, output_type="coo_matrix")
    keep = pairs.row != pairs.col  # the node itself goes first instead
    row, col = pairs.row[keep], pairs.col[keep]
    order = np.lexsort((col, row))  # by node, then by neighbour index
    row, col = row[order], col[order]
    first = np.searchsorted(row, np.arange(n + 1))  # where each node's pairs start
    return first + np.arange(n + 1), np.insert(col, first[:-1], np.arange(n))


def _refuse_twins(dist, adj):
    """ValueError on a node whose second-nearest is at distance zero."""
    same = np.flatnonzero(dist[:, 1] == 0)
    if same.size:
        i = int(same[0])
        raise ValueError(f"nodes {i} and {int(adj[i, 1])} coincide")
