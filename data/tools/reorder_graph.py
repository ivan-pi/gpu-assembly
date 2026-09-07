#!/usr/bin/env python3
"""Renumber the nodes of a graph file and write the ordering file.

Both files are of docs/file_formats.md; the orderings are reverse
Cuthill-McKee, nested dissection and approximate minimum degree.
"""

import argparse
import os
import sys

import numpy as np
from scipy.sparse import csr_array, triu

from pointclouds.io import FormatError, read_graph, write_ordering


def adjacency_of(ia, ja):
    """Return the stencils as an undirected graph without self-loops, the pattern of ``A + A^T`` less the diagonal, as a CSR array with sorted indices."""
    n = len(ia) - 1
    a = csr_array((np.ones(len(ja)), ja, ia), shape=(n, n))
    upper = triu(a + a.T, k=1)  # every edge once, without the self-loops
    adjacency = (upper + upper.T).tocsr()
    adjacency.sort_indices()
    return adjacency


def rcm(adjacency, **kwargs):
    """Return the new index of every node by reverse Cuthill-McKee."""
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    perm = reverse_cuthill_mckee(adjacency, symmetric_mode=True)  # perm[new] = old
    return np.argsort(perm)


def nd(adjacency, seed=None, **kwargs):
    """Return the new index of every node by METIS nested dissection; the seed drives the random matching of the coarsening."""
    try:
        import pymetis
    except ImportError:
        sys.exit("--method nd needs pymetis (pip install pymetis)")
    # Options(seed=None) is refused, so the default is a separate call
    options = pymetis.Options() if seed is None else pymetis.Options(seed=seed)
    perm, iperm = pymetis.nested_dissection(
        pymetis.CSRAdjacency(adjacency.indptr, adjacency.indices), options=options
    )
    return np.asarray(iperm, dtype=int)  # perm is the other direction, argsort(iperm)


def amd(adjacency, **kwargs):
    """Return the new index of every node by approximate minimum degree."""
    try:
        from sksparse.amd import amd as suitesparse_amd
    except ImportError:
        sys.exit(
            "--method amd needs scikit-sparse (pip install scikit-sparse, built against SuiteSparse)"
        )
    perm = suitesparse_amd(adjacency.tocsc())  # perm[new] = old
    return np.argsort(perm)


METHODS = {"rcm": rcm, "nd": nd, "amd": amd}  # all take the options they ignore


def bandwidth(ia, ja, iperm):
    """Return the largest |i - j| over the stencil entries, in the new numbering."""
    rows = np.repeat(np.arange(len(ia) - 1), np.diff(ia))
    return int(np.abs(iperm[rows] - iperm[ja]).max())


def factor_nonzeros(adjacency, iperm):
    """Return the nonzeros of the Cholesky factor in the new numbering by cholmod's symbolic factorisation, or None without scikit-sparse."""
    try:
        from sksparse.cholmod import symbfact
    except ImportError:
        return None
    perm = np.argsort(iperm)
    return int(np.sum(symbfact(adjacency[perm][:, perm].tocsc()).count))


EPILOG = """\
examples:
  reorder_graph.py case.graph                # case.iperm, by rcm
  reorder_graph.py case.graph --method nd    # nested dissection
  reorder_graph.py case.graph -o -           # to standard output

rcm, from scipy, reduces the bandwidth by numbering neighbouring nodes
close together. nd, METIS through pymetis, and amd, SuiteSparse through
scikit-sparse, reduce the fill-in of a sparse direct factorisation, by
recursive bisection and by greedy elimination. All three order the
undirected graph of the stencils, the pattern of A + A^T without the
diagonal. The report, on standard error so that `-o -` leaves the file
alone, gives the bandwidth before and after and, with scikit-sparse, the
nonzeros of the Cholesky factor; `inspect_points.py case.graph
case.iperm --spy` shows the effect."""


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("graph", metavar="GRAPH", help="the .graph file to order")
    ap.add_argument(
        "-m",
        "--method",
        choices=METHODS.keys(),
        default="rcm",
        help="rcm: reverse Cuthill-McKee (default); nd: METIS nested dissection; "
        "amd: SuiteSparse approximate minimum degree",
    )
    ap.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="the ordering file, default GRAPH with the extension .iperm; "
        "- writes to standard output",
    )
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
    print(
        f"{args.graph}: {n} nodes, {len(ja)} entries, {adjacency.nnz // 2} undirected edges",
        file=sys.stderr,
    )

    iperm = METHODS[args.method](adjacency, seed=args.seed)
    assert np.array_equal(np.sort(iperm), np.arange(n))

    identity = np.arange(n)
    report = f"{args.method}: bandwidth {bandwidth(ia, ja, identity)} -> {bandwidth(ia, ja, iperm)}"
    before = factor_nonzeros(adjacency, identity)
    if before is not None:
        report += f", factor nonzeros {before} -> {factor_nonzeros(adjacency, iperm)}"
    print(report, file=sys.stderr)

    stem = args.output.removesuffix(".iperm")
    write_ordering(stem, iperm)
    if stem != "-":
        print(f"ordering written to {stem}.iperm", file=sys.stderr)


if __name__ == "__main__":
    main()
