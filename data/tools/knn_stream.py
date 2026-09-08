"""Write the kNN graph of a random point cloud as the binary stream
examples/bench_spmv.f90 reads, to time the sparse products on the
pattern the library meets.

    python knn_stream.py 1000000 21 morton knn.bin

The stream is int32 n, int32 k, then ja as a C-order (n, k) int32
array, 0-based, then the values as a C-order (n, k) float64 array. In
Fortran that is ja(k, n), the row layout of rbf_csr; the benchmark
transposes it for rbf_ellpack. The values are a smooth function of the
neighbour distance, since their magnitude does not matter to the
timing.

The ordering decides the locality of the gather and so most of the
run time: `random` leaves the points as drawn, `morton` sorts them
along a Z-curve, as rbf::morton_order does. Within a row the
neighbours come sorted by distance, the node itself first, as
NodeSet::stencils hands them out; `--sort-rows` sorts them by index
instead, which makes the j-th entries of consecutive rows more
coherent, the access pattern ELLPACK gathers along.
"""

import argparse

import numpy as np
from scipy.spatial import cKDTree


def morton_keys(pts, bits=16):
    """Z-curve keys of points in the unit square, `bits` per axis."""
    q = np.minimum((pts * 2**bits).astype(np.uint64), 2**bits - 1)

    def spread(v):
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        v = (v | (v << 1)) & 0x55555555
        return v

    return spread(q[:, 0]) | (spread(q[:, 1]) << 1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("n", type=int, help="number of points")
    p.add_argument("k", type=int, help="neighbours per point, the node itself included")
    p.add_argument("order", choices=["random", "morton"])
    p.add_argument("out", help="the stream to write")
    p.add_argument(
        "--sort-rows", action="store_true", help="sort each row by column index"
    )
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    pts = rng.random((args.n, 2))
    if args.order == "morton":
        pts = pts[np.argsort(morton_keys(pts), kind="stable")]

    d, ja = cKDTree(pts).query(pts, k=args.k, workers=-1)
    a = 1.0 / (1.0 + d / d[:, 1:].mean())
    if args.sort_rows:
        o = np.argsort(ja, axis=1, kind="stable")
        ja = np.take_along_axis(ja, o, axis=1)
        a = np.take_along_axis(a, o, axis=1)

    with open(args.out, "wb") as f:
        np.array([args.n, args.k], dtype=np.int32).tofile(f)
        np.ascontiguousarray(ja, dtype=np.int32).tofile(f)
        np.ascontiguousarray(a, dtype=np.float64).tofile(f)

    i = np.arange(args.n)[:, None]
    print(
        f"{args.out}: n={args.n} k={args.k} {args.order}"
        f"{' rows sorted' if args.sort_rows else ''}: "
        f"mean |ja - i| = {np.abs(ja - i).mean():.0f}, "
        f"mean |ja(i+1, j) - ja(i, j)| = {np.abs(np.diff(ja, axis=0)).mean():.0f}"
    )


if __name__ == "__main__":
    main()
