"""Stencil selection: the nodes a node interpolates from.

Three ways to select them: the k nearest neighbours, the nodes within
a distance, or the nodes within a distance along both axes, a square,
which is an orthogonal range search. scipy's k-d tree does the
searching, over a box periodic in either axis. `KNN` is the default
stencil size. The stencils come back as a `Graph`.
"""

import numpy as np
from scipy.sparse import csr_array
from scipy.spatial import cKDTree

KNN = 18  # the default stencil: the 6 terms of a second-order polynomial plus 12


class Graph:
    """The stencils of a cloud, in CSR form.

    Parameters
    ----------
    ia : (n + 1,) array_like of int
        The row pointer: the stencil of node i is ``ja[ia[i]:ia[i + 1]]``.
    ja : (nnz,) array_like of int
        The column indices, the nodes of the stencils.

    Attributes
    ----------
    ia, ja : ndarray of int
        As given. ``graph[i]`` is the stencil of node i, ``len(graph)``
        the number of nodes.
    nnz : int
        The number of entries, the edges of the graph.
    pattern : scipy.sparse.csr_array
        The sparsity pattern, a 1 for every entry.
    """

    def __init__(self, ia, ja):
        self.ia, self.ja = np.asarray(ia), np.asarray(ja)
        self.nnz = len(self.ja)
        n = len(self)
        self.pattern = csr_array((np.ones(self.nnz, np.int8), self.ja, self.ia), (n, n))

    def __len__(self):
        return len(self.ia) - 1

    def __getitem__(self, i):
        return self.ja[self.ia[i] : self.ia[i + 1]]

    def bandwidth(self):
        """Returns the largest ``|i - j|`` over the entries."""
        rows = np.repeat(np.arange(len(self)), np.diff(self.ia))
        return int(np.abs(rows - self.ja).max())

    def renumbered(self, iperm):
        """Returns the graph in the numbering `iperm`.

        Moves the rows and relabels the entries; the order within a
        stencil is not kept.
        """
        order = np.argsort(iperm)
        moved = self.pattern[order][:, order]
        return type(self)(moved.indptr, moved.indices)


def select_stencils(pts, boxsize, method="knn", value=KNN):
    """Selects the stencil of every node of a cloud.

    Every stencil starts with the node itself, which is what the graph
    file expects, then holds its neighbours nearest first.

    Parameters
    ----------
    pts : (n, 2) ndarray
        The nodes.
    boxsize : (2,) array_like
        The side of the box along every periodic axis, around which the
        search wraps, and 0 for an axis that does not.
    method : {"knn", "radius", "range"}
        The k nearest neighbours, the nodes within a distance, or the
        nodes within a distance along both axes, a square.
    value : int or float
        The k of knn, the distance of the others; `KNN` by default.

    Returns
    -------
    Graph

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
        return Graph(np.arange(0, n * value + 1, value), adj.ravel())

    # radius and range: every pair of nodes closer than the distance, in
    # the norm whose ball is the disk or the square
    half = min((L for L in boxsize if L > 0), default=np.inf) / 2
    if value > half:
        raise ValueError(
            f"a {method} of {value:g} reaches more than half way around the box, "
            f"which is {2 * half:g} across: at most {half:g}"
        )
    p_norm = {"radius": 2, "range": np.inf}[method]
    pairs = tree.sparse_distance_matrix(tree, value, p=p_norm, output_type="coo_matrix")

    # The pairs of a node with itself are dropped, and the node is put at
    # the front of its stencil instead below.
    others = pairs.row != pairs.col
    row, col, dist = pairs.row[others], pairs.col[others], pairs.data[others]
    twins = np.flatnonzero(dist == 0)
    if twins.size:
        i, j = sorted((row[twins[0]], col[twins[0]]))
        raise ValueError(f"nodes {i} and {j} coincide")

    # The pairs sorted by node, then by distance, then by index, so that
    # the pairs of node i are consecutive and start at first[i].
    order = np.lexsort((col, dist, row))
    row, col = row[order], col[order]
    first = np.searchsorted(row, np.arange(n + 1))

    # Stencil i is node i, then its pairs: one entry more per node.
    ia = first + np.arange(n + 1)
    ja = np.insert(col, first[:-1], np.arange(n))
    return Graph(ia, ja)
