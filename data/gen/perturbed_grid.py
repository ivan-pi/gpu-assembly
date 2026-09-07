#!/usr/bin/env python3
"""Point clouds for the unit square whose nodes are those of a Cartesian
grid moved by a small random amount, with the periodic stencil graph that
goes with them.

    python3 perturbed_grid.py -n 40 tg_40
    python3 perturbed_grid.py -n 40 --sigma 0.02 --knn 21 tg_40
    python3 perturbed_grid.py -n 40 --geometry channel poiseuille_40.node
    python3 perturbed_grid.py -n 40 --realizations 50 --seed 1234 tg_40

The node layout of Strzelczyk and Matyka (2022), "How nodes layout,
refinement and velocity discretization influence convergence of the
meshless lattice Boltzmann method", <https://ssrn.com/abstract=4070398>,
Figs. 4 and 8. An N by N grid of spacing h = 1/N is laid on the unit
square with its lowest node at the origin, and every coordinate is then
displaced by an amount drawn uniformly from [-sigma h, sigma h]. The
reference uses sigma = 0, 0.02 and 0.2; a node stays in its own cell for
sigma < 0.5. Two geometries differ in what happens at the sides:

  periodic  Both sides are periodic, as in the Taylor-Green test. A node
            that leaves the square is wrapped back in through the
            opposite side, so the cloud stays N by N.
  channel   Periodic in x, walls at y = 0 and y = 1, as in the Poiseuille
            test. The grid has N + 1 rows, the first on the bottom wall
            and the last on the top; the wall nodes slide along their
            wall but do not leave it.

The nodes are written row by row from y = 0 upwards, x fastest, so the
wall nodes of a channel are the first N and the last N lines of the file.

The stencil of a node is the set of nodes it interpolates from: its k
nearest neighbours by default, the nodes within a given distance with
--radius, or those within a square with --square. The search knows which
sides are periodic, so a stencil next to a periodic side reaches around
it. Distances are given in spacings h, and every stencil starts with the
node itself.

Random grids are meant to be averaged over, so a run generates one
realization per --realizations and numbers the files. Realization i
depends on the seed and on i alone: asking for more of them extends the
series rather than replacing it, and a run without --seed reports the
seed it drew so that it can be repeated.

The output is a points file and a graph file, in the numbering the two
share. A name ending in `.node` writes a node file instead of the points
file, with a comment line naming the command and a marker per node
(docs/file_formats.md):

    0  interior
    1  bottom wall, y = 0
    3  top wall, y = 1

The two walls are numbered as in cavity_refined.py, counter-clockwise
from the bottom, which leaves 2 and 4 for the east and west walls: those
sides are periodic here and carry no nodes of their own.
"""

import argparse
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

INTERIOR, SOUTH, NORTH = 0, 1, 3

# A periodic box more than twice as wide as the data never wraps, which is
# how the channel gets a periodic x and an open y from one k-d tree.
OPEN = 3.0


def grid(n, geometry):
    """The Cartesian nodes and their markers: N by N on the periodic
    square, N by N + 1 on the channel, whose last row is the top wall."""
    h = 1.0 / n
    x = np.arange(n) * h
    y = np.arange(n if geometry == "periodic" else n + 1) * h
    xv, yv = np.meshgrid(x, y)                   # row by row, x fastest
    pts = np.column_stack((xv.ravel(), yv.ravel()))
    m = np.full(len(pts), INTERIOR)
    if geometry == "channel":
        m[: n], m[len(pts) - n :] = SOUTH, NORTH
    return pts, m


def perturb(pts, m, n, sigma, geometry, rng):
    """Every coordinate displaced by up to sigma spacings, except the one
    across the wall, which would take a wall node off its wall."""
    d = rng.uniform(-sigma / n, sigma / n, pts.shape)
    d[m != INTERIOR, 1] = 0.0
    pts = pts + d
    pts[:, 0] = wrap(pts[:, 0])
    if geometry == "periodic":
        pts[:, 1] = wrap(pts[:, 1])
    else:
        pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)     # only reached by sigma >= 1
    return pts


def wrap(z):
    """Into [0, 1) through the periodic side."""
    z = np.mod(z, 1.0)
    z[z >= 1.0] = 0.0                            # np.mod rounds up to 1.0
    return z


def stencils(pts, n, geometry, knn, radius, square):
    """The stencil of every node, as lists of node indices."""
    tree = cKDTree(pts, boxsize=[1.0, 1.0 if geometry == "periodic" else OPEN])
    if radius is not None:
        adj = tree.query_ball_point(pts, radius / n)
    elif square is not None:
        adj = tree.query_ball_point(pts, square / n, p=np.inf)   # the max norm
    else:
        adj = tree.query(pts, knn)[1].tolist()
    rows = [[i] + [j for j in row if j != i] for i, row in enumerate(adj)]
    if knn is not None and any(len(row) != knn for row in rows):
        sys.exit("a node is not its own nearest neighbour: two nodes coincide, "
                 "which takes a --sigma of about 0.5 or more")
    return rows


def write_points(fname, pts):
    with open(fname, "w") as f:
        f.write(f"{len(pts)}\n")
        for x, y in pts.tolist():
            f.write(f"{x!r} {y!r}\n")            # repr: shortest round-trip text


def write_node(fname, pts, m, provenance):
    with open(fname, "w") as f:
        f.write(f"# {provenance}\n")
        f.write(f"{len(pts)} 2 0 1\n")
        for i, ((x, y), mi) in enumerate(zip(pts.tolist(), m.tolist())):
            f.write(f"{i} {x!r} {y!r} {mi}\n")


def write_graph(fname, rows):
    with open(fname, "w") as f:
        f.write(f"{len(rows)} {sum(len(row) for row in rows)}\n")
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")


def plot(pts, m, rows):
    import matplotlib.pyplot as plt
    plt.scatter(pts[:, 0], pts[:, 1], c=m, s=8, cmap="tab10", vmin=0, vmax=9)
    if rows:                                     # one stencil, to see it wrap
        middle = rows[len(pts) // 2]
        plt.scatter(pts[middle, 0], pts[middle, 1], marker="s", s=24,
                    facecolors="none", edgecolors="tab:orange")
    plt.axis("equal")
    plt.show()


HELP = __doc__.split("\n\n")[0] + """

Distances are in spacings h = 1/N; markers are 0 interior, 1 bottom wall,
3 top wall. The docstring at the top of the script describes the two
geometries and the series of realizations."""


def main():
    ap = argparse.ArgumentParser(description=HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", help="output file: the points file and the graph "
                    "file next to it, or a node file with the markers and the "
                    "graph if the name ends in .node")
    ap.add_argument("-n", "--nodes", type=int, required=True, metavar="N",
                    help="nodes across the unit square; the spacing is h = 1/N")
    ap.add_argument("-s", "--sigma", type=float, default=0.2, metavar="SIGMA",
                    help="displacement of a node, in spacings: uniform on "
                         "[-SIGMA h, SIGMA h] (default: 0.2)")
    ap.add_argument("-g", "--geometry", choices=("periodic", "channel"),
                    default="periodic",
                    help="periodic on both sides, or walls at y = 0 and y = 1 "
                         "and periodic in x (default: periodic)")

    stencil = ap.add_mutually_exclusive_group()
    stencil.add_argument("--knn", type=int, metavar="K",
                         help="stencil of the K nearest nodes (default: 15, the "
                              "stencil size of the reference)")
    stencil.add_argument("--radius", type=float, metavar="R",
                         help="stencil of the nodes within R spacings")
    stencil.add_argument("--square", type=float, metavar="S",
                         help="stencil of the nodes within S spacings in both x "
                              "and y, a square of side 2 S h")

    ap.add_argument("--seed", type=int,
                    help="seed of the displacements (default: drawn and reported)")
    ap.add_argument("--realizations", type=int, default=1, metavar="R",
                    help="independent grids to write, numbered from 0 (default: 1)")
    ap.add_argument("--no-graph", action="store_true",
                    help="write the coordinates only, without the stencil graph")
    ap.add_argument("--plot", action="store_true",
                    help="show the first grid, coloured by marker, with one stencil")
    args = ap.parse_args()

    n, sigma = args.nodes, args.sigma
    if n < 2:
        sys.exit("-n must be at least 2")
    if sigma < 0.0:
        sys.exit("--sigma must not be negative")
    if sigma >= 0.5:
        print(f"warning: --sigma {sigma:g} moves a node out of its own cell, "
              f"and nodes may end up on top of each other", file=sys.stderr)
    if args.knn is None and args.radius is None and args.square is None:
        args.knn = 15
    for name, reach in (("--radius", args.radius), ("--square", args.square)):
        if reach is not None and not 0.0 < reach <= 0.5 * n:
            # Beyond half the box a node is its own neighbour through the
            # periodic side, which the stencil of a node cannot hold twice.
            sys.exit(f"{name} must be positive and at most {0.5 * n:g} spacings, "
                     f"half the width of the box")
    if args.realizations < 1:
        sys.exit("--realizations must be at least 1")

    seed = np.random.SeedSequence().entropy if args.seed is None else args.seed
    if args.seed is None:
        print(f"seed {seed}")
    streams = np.random.SeedSequence(seed).spawn(args.realizations)

    name = args.output
    node_file = name.endswith(".node")
    base = os.path.splitext(name)[0] if node_file or name.endswith(".points") else name
    width = len(str(args.realizations - 1))

    stencil_flag = (f"--knn {args.knn}" if args.knn is not None else
                    f"--radius {args.radius:g}" if args.radius is not None else
                    f"--square {args.square:g}")

    pts0, m = grid(n, args.geometry)
    if args.knn is not None and not 1 <= args.knn <= len(pts0):
        sys.exit(f"--knn must be at least 1 and at most {len(pts0)}, the number of nodes")

    for i, stream in enumerate(streams):
        stem = base if args.realizations == 1 else f"{base}_{i:0{width}d}"
        pts = perturb(pts0, m, n, sigma, args.geometry, np.random.default_rng(stream))

        if node_file:
            command = (f"perturbed_grid.py -n {n} --sigma {sigma:g} "
                       f"--geometry {args.geometry} {stencil_flag} --seed {seed}")
            if args.realizations > 1:
                command += f" --realizations {args.realizations}, number {i}"
            write_node(stem + ".node", pts, m, "produced by " + command)
        else:
            write_points(stem + ".points", pts)

        rows = None
        if not args.no_graph:
            rows = stencils(pts, n, args.geometry, args.knn, args.radius, args.square)
            write_graph(stem + ".graph", rows)

        walls = f", {np.count_nonzero(m)} of them on a wall" if args.geometry == "channel" else ""
        sizes = [len(row) for row in rows] if rows else []
        graph = (f", {sum(sizes)} stencil entries, {min(sizes)} to {max(sizes)} per node"
                 if sizes else "")
        print(f"{stem}: {len(pts)} nodes{walls}{graph}")

        if args.plot and i == 0:
            plot(pts, m, rows)


if __name__ == "__main__":
    main()
