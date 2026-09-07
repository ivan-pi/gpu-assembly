"""The stencil search the generators in data/gen share. The stencil of a
node is the set of nodes it interpolates from: its k nearest neighbours,
the nodes within a distance, or those within a square, found by scipy's
k-d tree over a box periodic in either axis, so that a stencil next to a
periodic side reaches around it. Every stencil starts with the node
itself, which is what the graph file expects (docs/file_formats.md).

A generator adds the three options that pick one with `add_options`,
reads the choice back with `from_args`, checks it against its box with
`check`, and runs `search` on every cloud it makes:

    add_options(parser, knn=15)
    ...
    stencil = from_args(args, knn=15)
    problem = check(stencil, extent, periodic, len(pts))
    if problem:
        parser.error(problem)
    rows = search(pts, extent, periodic, stencil)
"""

from collections import namedtuple

import numpy as np

from .cli import number

Stencil = namedtuple("Stencil", "kind reach")
# kind    "knn", "radius" or "square"
# reach   the k of knn, or the distance of the other two

__all__ = ["Stencil", "add_options", "from_args", "check", "search"]


def add_options(ap, knn, why=""):
    """--knn, --radius and --square, mutually exclusive, on the parser;
    `knn` is the stencil size used when none is given, and `why` says
    where it comes from, in the help."""
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--knn",
        type=number(int, least=1),
        metavar="K",
        help=f"stencil of the K nearest nodes (default: {knn}{why})",
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
    return group


def from_args(args, knn):
    """The Stencil the parsed options ask for, k nearest with `knn` of
    them when they ask for nothing."""
    if args.radius is not None:
        return Stencil("radius", args.radius)
    if args.square is not None:
        return Stencil("square", args.square)
    return Stencil("knn", knn if args.knn is None else args.knn)


def check(stencil, extent, periodic, n):
    """What is wrong with the stencil for a cloud of n nodes on the box
    `extent`, periodic per axis as `periodic` says, or None. A stencil
    cannot hold more nodes than there are, and along a periodic axis it
    cannot reach more than half way around: beyond that a node meets its
    own image, which its stencil cannot hold twice."""
    kind, reach = stencil
    if kind == "knn":
        if reach > n:
            return f"--knn {reach} is larger than the {n} nodes of the cloud"
        return None
    periodic = np.broadcast_to(np.asarray(periodic, bool), 2)
    if not periodic.any():
        return None
    half = 0.5 * np.asarray(extent, float)[periodic].min()
    if reach > half:
        return (
            f"--{kind} {reach:g} reaches more than half way around the box, "
            f"which is {2 * half:g} across: at most {half:g}"
        )
    return None


def search(pts, extent, periodic, stencil):
    """The stencil of every node of `pts`, as lists of node indices, the
    node itself first. `extent` is the box (Lx, Ly) and `periodic` says
    per axis, or for both at once, whether the search wraps around it.
    Raises ValueError when two nodes coincide, which leaves a node not
    its own nearest neighbour."""
    from scipy.spatial import cKDTree

    periodic = np.broadcast_to(np.asarray(periodic, bool), 2)
    tree = cKDTree(pts, boxsize=np.where(periodic, np.asarray(extent, float), 0.0))
    kind, reach = stencil
    if kind == "knn":
        adj = tree.query(pts, reach)[1].reshape(len(pts), -1)  # k = 1 comes flat
        if not np.array_equal(adj[:, 0], np.arange(len(pts))):
            raise ValueError(
                "a node is not its own nearest neighbour: two nodes coincide"
            )
        return adj.tolist()  # the node itself opens each row
    norm = np.inf if kind == "square" else 2  # the max norm bounds a square
    adj = tree.query_ball_point(pts, reach, p=norm)
    return [[i] + [j for j in row if j != i] for i, row in enumerate(adj)]
