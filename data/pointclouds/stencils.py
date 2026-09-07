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

    ia, ja = select_stencils(pts, boxsize)          # the KNN nearest
    ia, ja = select_stencils(pts, boxsize, "radius", 2.5)

`boxsize` gives the side of the box along every periodic axis, around
which the search wraps, and 0 for an axis that does not; NodeSet passes
its own. Two nodes that coincide are refused, since no stencil can tell
them apart, and so is a distance that reaches more than half way around
the box, where a node meets its own image.
"""

import numpy as np
from scipy.spatial import cKDTree

KNN = 18  # the default stencil: the 6 terms of a second-order polynomial plus 12


def select_stencils(pts, boxsize, method="knn", value=KNN):
    """Select the stencil of every node of a cloud.

    Parameters
    ----------
    pts : (n, 2) ndarray
        The nodes.
    boxsize : (2,) array_like
        The side of the box along every periodic axis, around which the
        search wraps, and 0 for an axis that does not.
    method : {"knn", "radius", "range"}
        The k nearest neighbours, the nodes within a distance, or the nodes
        within a distance along both axes, a square.
    value : int or float
        The k of knn, the distance of the others.

    Returns
    -------
    ia, ja : ndarray
        The stencils in CSR form, that of node i at ``ja[ia[i]:ia[i + 1]]``:
        the node itself first, then its neighbours nearest first.

    Raises
    ------
    ValueError
        When two nodes coincide, which no stencil can tell apart; for a
        stencil larger than the cloud; and for a distance that reaches
        more than half way around the box, where a node meets its own
        image.
    """
    n = len(pts)
    tree = cKDTree(pts, boxsize=boxsize)

    if method == "knn":
        if value > n:
            raise ValueError(f"a stencil of {value} nodes from a cloud of {n}")
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
