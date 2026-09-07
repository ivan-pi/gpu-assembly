"""Stencil selection. The stencil of a node is the set of nodes it
interpolates from, and there are three ways to select it: its k nearest
neighbours (knn), the nodes within a distance of it (radius), or the
nodes within a distance of it along both axes, a square, which is an
orthogonal range search (range). scipy's k-d tree does the searching,
over a box periodic in either axis, so that a stencil next to a
periodic side reaches around it. Every stencil starts with the node
itself, which is what the graph file expects (docs/file_formats.md),
then its neighbours nearest first, and the stencils come back in the
CSR form the readers use: (ia, ja), with the stencil of node i at
ja[ia[i]:ia[i + 1]].

    ia, ja = select_stencils(pts, boxsize, "knn", 18)

`boxsize` gives the side of the box along every periodic axis, around
which the search wraps, and 0 for an axis that does not; NodeSet passes
its own. Two nodes that coincide are refused, since no stencil can tell
them apart, and so is a distance that reaches more than half way around
the box, where a node meets its own image.
"""

import numpy as np
from scipy.spatial import cKDTree


def select_stencils(pts, boxsize, method, value):
    n = len(pts)
    tree = cKDTree(pts, boxsize=boxsize)

    if method == "knn":
        assert (
            2 <= value <= n
        ), "a stencil of two nodes at least, and no more than the cloud"
        dist, adj = tree.query(pts, value, workers=-1)
        twins = np.flatnonzero(dist[:, 1] == 0)
        if twins.size:
            i, j = sorted(adj[twins[0], :2])
            raise ValueError(f"nodes {i} and {j} coincide")
        return np.arange(0, n * value + 1, value), adj.ravel()

    # radius and range: all pairs closer than the distance, in the norm
    # whose ball is the disk or the square
    half = min((L for L in boxsize if L > 0), default=np.inf) / 2
    if value > half:
        raise ValueError(
            f"a {method} of {value:g} reaches more than half way around the box, "
            f"which is {2 * half:g} across: at most {half:g}"
        )
    p_norm = {"radius": 2, "range": np.inf}[method]
    pairs = tree.sparse_distance_matrix(tree, value, p=p_norm, output_type="coo_matrix")
    keep = pairs.row != pairs.col  # the node itself goes first instead
    row, col, dist = pairs.row[keep], pairs.col[keep], pairs.data[keep]
    twins = np.flatnonzero(dist == 0)
    if twins.size:
        i, j = sorted((row[twins[0]], col[twins[0]]))
        raise ValueError(f"nodes {i} and {j} coincide")
    o = np.lexsort((col, dist, row))  # by node, then by distance, then by index
    row, col = row[o], col[o]
    first = np.searchsorted(row, np.arange(n + 1))  # where each node's pairs start
    return first + np.arange(n + 1), np.insert(col, first[:-1], np.arange(n))
