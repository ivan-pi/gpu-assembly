#!/usr/bin/env python3
"""Compute a renumbering of the nodes of a graph file and write it as an
ordering file, both of docs/file_formats.md.

    tools/reorder_graph.py case.graph                # case.iperm, by rcm
    tools/reorder_graph.py case.graph --method nd    # nested dissection
    tools/reorder_graph.py case.graph --method amd   # minimum degree
    tools/reorder_graph.py case.graph -o other.iperm
    tools/reorder_graph.py case.graph -o -           # to standard output

The methods:

  rcm   reverse Cuthill-McKee, from scipy.sparse.csgraph: a bandwidth-
        reducing ordering, which numbers neighbouring nodes close to
        each other and so keeps the entries of a row near the diagonal.
  nd    multilevel nested dissection, from METIS through pymetis: an
        ordering that reduces the fill-in of a sparse direct
        factorisation, the one to use before such a factorisation.
  amd   approximate minimum degree, SuiteSparse's AMD through
        scikit-sparse: reduces the fill-in too, by greedy elimination
        rather than recursive bisection, cheaper and often as good.

All three order the undirected graph of the stencils, the pattern of
A + A^T with the diagonal dropped: none of them cares which way an edge
points, and METIS requires it so. The ordering file holds the new index
of every node; `inspect_points.py case.graph case.iperm --spy` shows the
effect.

The report, on standard error so that `-o -` leaves the file alone on
standard output, gives the bandwidth before and after, the measure rcm
works on, and the nonzeros of the Cholesky factor before and after, the
measure nd and amd work on, when scikit-sparse is there to count them.

Needs numpy and scipy; nd needs pymetis, amd and the factor count need
scikit-sparse.
"""

import argparse
import os
import sys

import numpy as np
from scipy.sparse import csr_array, triu

from pointclouds.cli import output_stem
from pointclouds.io import FormatError, read_graph, write_ordering


def adjacency_of(ia, ja):
    """The stencils as an undirected graph without self-loops, the pattern
    of A + A^T less the diagonal, as a CSR array with sorted indices. The
    values count the directions of an edge, 2 when both stencils have it
    and 1 when one does; the orderings here take the pattern alone."""
    n = len(ia) - 1
    a = csr_array((np.ones(len(ja)), ja, ia), shape=(n, n))
    upper = triu(a + a.T, k=1)                  # every edge once, without the self-loops
    adjacency = (upper + upper.T).tocsr()
    adjacency.sort_indices()
    return adjacency


def rcm(adjacency):
    """Return the inverse ordering, the new index of every node, by
    reverse Cuthill-McKee."""
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    perm = reverse_cuthill_mckee(adjacency, symmetric_mode=True)   # perm[new] = old
    return np.argsort(perm)


def nd(adjacency, seed=None):
    """Return the inverse ordering by METIS nested dissection. The seed
    drives the random matching of its coarsening; METIS has a fixed
    default, so the ordering is the same from run to run unless one is
    given."""
    try:
        import pymetis
    except ImportError:
        sys.exit("--method nd needs pymetis (pip install pymetis)")
    options = pymetis.Options() if seed is None else pymetis.Options(seed=seed)   # METIS takes integers only
    perm, iperm = pymetis.nested_dissection(pymetis.CSRAdjacency(adjacency.indptr, adjacency.indices),
                                            options=options)
    return np.asarray(iperm, dtype=int)         # perm is the other direction, argsort(iperm)


def amd(adjacency):
    """Return the inverse ordering by approximate minimum degree."""
    try:
        from sksparse.amd import amd as suitesparse_amd
    except ImportError:
        sys.exit("--method amd needs scikit-sparse (pip install scikit-sparse, built against SuiteSparse)")
    perm = suitesparse_amd(adjacency.tocsc())   # perm[new] = old
    return np.argsort(perm)


METHODS = ("rcm", "nd", "amd")


def order(method, adjacency, seed):
    """The inverse ordering by the named method; the seed is for nd."""
    if method == "rcm":
        return rcm(adjacency)
    if method == "nd":
        return nd(adjacency, seed)
    return amd(adjacency)


def bandwidth(ia, ja, iperm):
    """The largest |i - j| over the stencil entries, in the new numbering."""
    rows = np.repeat(np.arange(len(ia) - 1), np.diff(ia))
    return int(np.abs(iperm[rows] - iperm[ja]).max())


def factor_nonzeros(adjacency, iperm):
    """The nonzeros of the Cholesky factor of the undirected graph in the
    new numbering, diagonal included, by cholmod's symbolic factorisation;
    None without scikit-sparse."""
    try:
        from sksparse.cholmod import symbfact
    except ImportError:
        return None
    perm = np.argsort(iperm)
    return int(np.sum(symbfact(adjacency[perm][:, perm].tocsc()).count))


def parse_args():
    ap = argparse.ArgumentParser(
        description="Renumber the nodes of a graph file to reduce bandwidth (rcm) or "
                    "fill-in (nd, amd) and write the ordering file (docs/file_formats.md).")
    ap.add_argument("graph", metavar="GRAPH", help="the .graph file to order")
    ap.add_argument("-m", "--method", choices=METHODS, default="rcm",
                    help="rcm: reverse Cuthill-McKee (default); nd: METIS nested dissection; "
                         "amd: SuiteSparse approximate minimum degree")
    ap.add_argument("-o", "--output", metavar="FILE",
                    help="the ordering file, default GRAPH with the extension .iperm; "
                         "- writes to standard output")
    ap.add_argument("--seed", type=int, help="the random seed of METIS, for nd")
    args = ap.parse_args()
    if args.output is None:
        args.output = os.path.splitext(args.graph)[0] + ".iperm"
    return args


def main():
    args = parse_args()
    try:
        ia, ja = read_graph(args.graph)
    except FormatError as e:
        sys.exit(str(e))
    adjacency = adjacency_of(ia, ja)
    n = adjacency.shape[0]
    print(f"{args.graph}: {n} nodes, {len(ja)} entries, {adjacency.nnz // 2} undirected edges",
          file=sys.stderr)

    iperm = order(args.method, adjacency, args.seed)
    assert np.array_equal(np.sort(iperm), np.arange(n))

    identity = np.arange(n)
    report = f"{args.method}: bandwidth {bandwidth(ia, ja, identity)} -> {bandwidth(ia, ja, iperm)}"
    before = factor_nonzeros(adjacency, identity)
    if before is not None:
        report += f", factor nonzeros {before} -> {factor_nonzeros(adjacency, iperm)}"
    print(report, file=sys.stderr)

    stem, _ = output_stem(args.output, ".iperm", known=(".iperm",))
    write_ordering(stem, iperm)
    if stem != "-":
        print(f"ordering written to {stem}.iperm", file=sys.stderr)


if __name__ == "__main__":
    main()
