#!/usr/bin/env python3
"""Compute a renumbering of the nodes of a graph file and write it as an
ordering file, both of docs/file_formats.md.

    tools/reorder_graph.py case.graph                # case.iperm, by rcm
    tools/reorder_graph.py case.graph --method nd    # nested dissection
    tools/reorder_graph.py case.graph -o other.iperm
    tools/reorder_graph.py case.graph -o -           # to standard output

The methods:

  rcm   reverse Cuthill-McKee, from scipy.sparse.csgraph: a bandwidth-
        reducing ordering, which numbers neighbouring nodes close to
        each other and so keeps the entries of a row near the diagonal.
  nd    multilevel nested dissection, from METIS through pymetis: a
        fill-reducing ordering, the one for a sparse direct factorisation.

Both order the undirected graph of the stencils, the pattern of A + A^T
with the diagonal dropped: neither cares which way an edge points, and
METIS requires it so. The ordering file holds the new index of every
node; `inspect_points.py case.graph case.iperm --spy` shows the effect.
The report goes to standard error, so `-o -` leaves the file alone on
standard output.

Needs numpy and scipy; nd needs pymetis too.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # data/, for pointclouds
from pointclouds import Report, output_stem, read_graph, write_ordering


def undirected(ia, ja):
    """The stencils as an undirected graph without self-loops, the pattern
    of A + A^T less the diagonal, as a CSR array with sorted indices."""
    from scipy.sparse import csr_array
    n = len(ia) - 1
    rows = np.repeat(np.arange(n), np.diff(ia))
    r, c = np.concatenate([rows, ja]), np.concatenate([ja, rows])
    off = r != c
    s = csr_array((np.ones(np.count_nonzero(off)), (r[off], c[off])), shape=(n, n))
    s.sum_duplicates()
    s.sort_indices()
    return s


def rcm(s, seed):
    """iperm by reverse Cuthill-McKee; the seed is not used."""
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    order = reverse_cuthill_mckee(s, symmetric_mode=True)   # the old index at every new position
    return np.argsort(order)


def nd(s, seed):
    """iperm by METIS nested dissection, which returns both directions."""
    try:
        import pymetis
    except ImportError:
        sys.exit("--method nd needs pymetis (pip install pymetis)")
    options = pymetis.Options() if seed is None else pymetis.Options(seed=seed)
    _, iperm = pymetis.nested_dissection(pymetis.CSRAdjacency(s.indptr, s.indices), options=options)
    return np.asarray(iperm, dtype=int)


METHODS = {"rcm": rcm, "nd": nd}


def bandwidth(ia, ja, iperm):
    """The largest |i - j| over the stencil entries, in the new numbering."""
    rows = np.repeat(np.arange(len(ia) - 1), np.diff(ia))
    return int(np.abs(iperm[rows] - iperm[ja]).max())


def factor_nonzeros(s, iperm):
    """The nonzeros of the Cholesky factor of the undirected graph in the
    new numbering, diagonal included: what a direct solver would store.
    Row k of the factor holds the nodes on the elimination-tree paths from
    the earlier neighbours of k up to k, and the walk that counts them
    builds the tree as it goes, since a node without a parent yet that
    row k reaches has k as its parent. Costs one Python step per nonzero."""
    n = s.shape[0]
    order = np.argsort(iperm)
    parent = np.full(n, -1)
    mark = np.full(n, -1)                                        # the last row that reached each node
    count = n
    for k in range(n):
        mark[k] = k
        o = order[k]
        for j in iperm[s.indices[s.indptr[o]:s.indptr[o + 1]]].tolist():
            while j < k and mark[j] != k:
                mark[j] = k
                count += 1
                if parent[j] < 0:
                    parent[j] = k
                j = parent[j]
    return count


def parse_args():
    ap = argparse.ArgumentParser(
        description="Renumber the nodes of a graph file to reduce bandwidth (rcm) or "
                    "fill (nd) and write the ordering file (docs/file_formats.md).")
    ap.add_argument("graph", metavar="GRAPH", help="the .graph file to order")
    ap.add_argument("-m", "--method", choices=METHODS, default="rcm",
                    help="rcm: reverse Cuthill-McKee (default); nd: METIS nested dissection")
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
    rep = Report(stream=sys.stderr)
    got = read_graph(args.graph, rep)
    rep.print_notes()
    rep.print_problems()
    if got is None or rep.problems:
        sys.exit(1)
    ia, ja = got
    s = undirected(ia, ja)
    print(f"{args.graph}: {s.shape[0]} nodes, {len(ja)} entries, {s.nnz // 2} undirected edges",
          file=sys.stderr)

    iperm = METHODS[args.method](s, args.seed)
    assert np.array_equal(np.sort(iperm), np.arange(len(iperm)))
    same = np.arange(len(iperm))
    print(f"{args.method}: bandwidth {bandwidth(ia, ja, same)} -> {bandwidth(ia, ja, iperm)}, "
          f"factor nonzeros {factor_nonzeros(s, same)} -> {factor_nonzeros(s, iperm)}", file=sys.stderr)

    stem, _ = output_stem(args.output, ".iperm", known=(".iperm",))
    write_ordering(stem, iperm)
    if stem != "-":
        print(f"ordering written to {stem}.iperm", file=sys.stderr)


if __name__ == "__main__":
    main()
