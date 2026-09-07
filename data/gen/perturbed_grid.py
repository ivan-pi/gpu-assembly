#!/usr/bin/env python3
"""Point clouds whose nodes are those of a Cartesian grid moved by a small
random amount, with the periodic stencil graph that goes with them.

    python3 perturbed_grid.py -n 40 tg_40
    python3 perturbed_grid.py -n 40 --sigma 0.02 --knn 21 tg_40
    python3 perturbed_grid.py -n 40 --geometry channel poiseuille_40.node
    python3 perturbed_grid.py -n 40 --realizations 50 --seed 1234 tg_40

The node layout of Strzelczyk and Matyka (2022), "How nodes layout,
refinement and velocity discretization influence convergence of the
meshless lattice Boltzmann method", <https://ssrn.com/abstract=4070398>,
Figs. 4 and 8. An N by N grid is laid on the box with its lowest node at
the origin, and every coordinate is then displaced by an amount drawn
uniformly from [-sigma h, sigma h], with h the spacing. The reference
uses sigma = 0, 0.02 and 0.2; a node stays in its own cell for
sigma < 0.5.

Everything is in lattice units: the box is N by N and the spacing is
h = 1, so the coordinates are the lattice sites the LBM works on and a
distance is a number of spacings. `--size L` scales the box to L by L,
which gives h = L/N; `--size 1` is the unit square of the reference. The
nodes sit on the lattice rather than at cell centres, so a Cartesian
sigma = 0 grid runs from 0 to N - h rather than from h/2 to N - h/2; on a
periodic box the two differ by a shift.

Two geometries differ in what happens at the sides:

  periodic  Both sides are periodic, as in the Taylor-Green test. A node
            that leaves the box is wrapped back in through the opposite
            side, so the cloud stays N by N.
  channel   Periodic in x, walls at y = 0 and y = N, as in the Poiseuille
            test. The grid has N + 1 rows, the first on the bottom wall
            and the last on the top; the wall nodes slide along their
            wall but do not leave it.

The nodes are written row by row from y = 0 upwards, x fastest, so the
wall nodes of a channel are the first N and the last N lines of the file.

The stencil of a node is the set of nodes it interpolates from: its k
nearest neighbours by default, the nodes within a given distance with
--radius, or those within a square with --square. The search knows which
sides are periodic, so a stencil next to a periodic side reaches around
it. Distances are given in spacings, and every stencil starts with the
node itself.

Random grids are meant to be averaged over, so a run generates one
realization per --realizations and numbers the files. Realization i
depends on the seed and on i alone: asking for more of them extends the
series rather than replacing it, and a run without --seed reports the
seed it drew so that it can be repeated.

The output is a points file and a graph file, in the numbering the two
share. A name ending in `.node` writes a node file instead of the points
file, with a comment line naming the command and a marker per node
(docs/file_formats.md). The name `-` writes to standard output, and
`-- -.node` a node file there, after the option separator since the name
starts with a dash. Only one file fits down a pipe, so both take
--no-graph and a single realization. The report goes to standard error,
so the stream carries the file alone. The markers are:

    0  interior
    1  bottom wall, y = 0
    3  top wall, y = N

The two walls are numbered as in cavity_refined.py, counter-clockwise
from the bottom, which leaves 2 and 4 for the east and west walls: those
sides are periodic here and carry no nodes of their own.
"""

import argparse
import contextlib
import sys

import numpy as np
from scipy.spatial import cKDTree

INTERIOR, SOUTH, NORTH = 0, 1, 3


def number(kind, least=None, above=None):
    """An argparse type that also carries a bound: argparse turns the
    ArgumentTypeError into its own usage message, naming the option."""
    def parse(text):
        value = kind(text)                       # a ValueError here: "invalid int"
        if least is not None and value < least:
            raise argparse.ArgumentTypeError(f"must be at least {least:g}")
        if above is not None and value <= above:
            raise argparse.ArgumentTypeError(f"must be greater than {above:g}")
        return value
    parse.__name__ = kind.__name__               # the name argparse reports
    return parse


def grid(n, h, periodic):
    """The Cartesian nodes and their markers: N by N on the periodic box,
    N by N + 1 on the channel, whose last row is the top wall."""
    x = np.arange(n) * h
    y = np.arange(n if periodic else n + 1) * h
    xv, yv = np.meshgrid(x, y)                   # row by row, x fastest
    pts = np.column_stack((xv.ravel(), yv.ravel()))
    m = np.full(len(pts), INTERIOR)
    if not periodic:
        m[: n], m[len(pts) - n :] = SOUTH, NORTH
    return pts, m


def perturb(pts, m, jitter, size, periodic, rng):
    """Every coordinate displaced by up to jitter, except the one across
    the wall, which would take a wall node off its wall."""
    d = rng.uniform(-jitter, jitter, pts.shape)
    d[m != INTERIOR, 1] = 0.0
    pts = pts + d
    pts[:, 0] = wrap(pts[:, 0], size)
    if periodic:
        pts[:, 1] = wrap(pts[:, 1], size)
    else:
        pts[:, 1] = np.clip(pts[:, 1], 0.0, size)    # only reached by sigma >= 1
    return pts


def wrap(z, size):
    """Into [0, size) through the periodic side."""
    z = np.mod(z, size)
    z[z >= size] = 0.0                           # np.mod rounds up to size
    return z


def stencils(pts, h, size, periodic, kind, reach):
    """The stencil of every node, as lists of node indices. A zero side
    leaves that axis aperiodic, which is what the channel wants in y."""
    tree = cKDTree(pts, boxsize=[size, size if periodic else 0.0])
    if kind == "knn":
        adj = tree.query(pts, reach)[1]
        if not np.array_equal(adj[:, 0], np.arange(len(pts))):
            sys.exit("a node is not its own nearest neighbour: two nodes coincide, "
                     "which takes a --sigma of about 0.5 or more")
        return adj.tolist()                      # the node itself opens each row
    norm = np.inf if kind == "square" else 2     # the max norm bounds a square
    adj = tree.query_ball_point(pts, reach * h, p=norm)
    return [[i] + [j for j in row if j != i] for i, row in enumerate(adj)]


def open_out(stem, ext):
    """The file this stem and extension name, or standard output for `-`,
    which stays open."""
    return (contextlib.nullcontext(sys.stdout) if stem == "-"
            else open(stem + ext, "w"))


def write_points(stem, pts):
    with open_out(stem, ".points") as f:
        f.write(f"{len(pts)}\n")
        for x, y in pts.tolist():
            f.write(f"{x!r} {y!r}\n")            # repr: shortest round-trip text


def write_node(stem, pts, m, provenance):
    with open_out(stem, ".node") as f:
        f.write(f"# {provenance}\n")
        f.write(f"{len(pts)} 2 0 1\n")
        for i, ((x, y), mi) in enumerate(zip(pts.tolist(), m.tolist())):
            f.write(f"{i} {x!r} {y!r} {mi}\n")


def write_graph(stem, rows):
    with open_out(stem, ".graph") as f:
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

Lengths are in lattice units, in which the box is N by N and the spacing
is h = 1; distances are in spacings. Markers are 0 interior, 1 bottom
wall, 3 top wall. The docstring at the top of the script describes the
two geometries and the series of realizations."""


def main():
    ap = argparse.ArgumentParser(description=HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", help="output file: the points file and the graph "
                    "file next to it, or a node file with the markers and the "
                    "graph if the name ends in .node; `-` is standard output")
    ap.add_argument("-n", "--nodes", type=number(int, least=2), required=True, metavar="N",
                    help="nodes across the box")
    ap.add_argument("--size", type=number(float, above=0.0), metavar="L",
                    help="side of the box, in the same units as the coordinates "
                         "(default: N, which makes the spacing h = 1)")
    ap.add_argument("-s", "--sigma", type=number(float, least=0.0), default=0.2,
                    metavar="SIGMA",
                    help="displacement of a node, in spacings: uniform on "
                         "[-SIGMA h, SIGMA h] (default: 0.2)")
    ap.add_argument("-g", "--geometry", choices=("periodic", "channel"),
                    default="periodic",
                    help="periodic on both sides, or walls at y = 0 and y = N "
                         "and periodic in x (default: periodic)")

    stencil = ap.add_mutually_exclusive_group()
    stencil.add_argument("--knn", type=number(int, least=1), metavar="K",
                         help="stencil of the K nearest nodes (default: 15, the "
                              "stencil size of the reference)")
    stencil.add_argument("--radius", type=number(float, above=0.0), metavar="R",
                         help="stencil of the nodes within R spacings")
    stencil.add_argument("--square", type=number(float, above=0.0), metavar="S",
                         help="stencil of the nodes within S spacings in both x "
                              "and y, a square of side 2 S h")

    ap.add_argument("--seed", type=int,
                    help="seed of the displacements (default: drawn and reported)")
    ap.add_argument("--realizations", type=number(int, least=1), default=1, metavar="R",
                    help="independent grids to write, numbered from 0 (default: 1)")
    ap.add_argument("--no-graph", action="store_true",
                    help="write the coordinates only, without the stencil graph")
    ap.add_argument("--plot", action="store_true",
                    help="show the first grid, coloured by marker, with one stencil")
    args = ap.parse_args()

    n, sigma, periodic = args.nodes, args.sigma, args.geometry == "periodic"
    size = float(n) if args.size is None else args.size    # lattice units by default
    h = size / n
    if sigma >= 0.5:
        print(f"warning: --sigma {sigma:g} moves a node out of its own cell, "
              f"and nodes may end up on top of each other", file=sys.stderr)

    if args.radius is not None:
        kind, reach = "radius", args.radius
    elif args.square is not None:
        kind, reach = "square", args.square
    else:
        kind, reach = "knn", 15 if args.knn is None else args.knn
    if kind != "knn" and reach > 0.5 * n:
        # Beyond half the box a node is its own neighbour through the
        # periodic side, which the stencil of a node cannot hold twice.
        ap.error(f"--{kind} reaches more than half way around the box: at most "
                 f"{0.5 * n:g} spacings for -n {n}")

    seed = np.random.SeedSequence().entropy if args.seed is None else args.seed
    streams = np.random.SeedSequence(seed).spawn(args.realizations)

    node_file = args.output.endswith(".node")
    base = args.output.removesuffix(".node").removesuffix(".points")
    width = len(str(args.realizations - 1))

    piped = base == "-"                          # `-`, `-.points` or `-.node`
    if piped and not args.no_graph:
        ap.error("only one file fits down a pipe: add --no-graph to write the "
                 "coordinates to standard output, or name a file for the pair")
    if piped and args.realizations > 1:
        ap.error("a series needs file names to go in: --realizations cannot "
                 "write to standard output")

    pts0, m = grid(n, h, periodic)
    if kind == "knn" and reach > len(pts0) and not args.no_graph:
        ap.error(f"--knn is larger than the {len(pts0)} nodes of the grid")
    if args.seed is None:
        print(f"seed {seed}", file=sys.stderr)

    # The comment of a node file is the command that reproduces it.
    command = (f"produced by perturbed_grid.py -n {n} --size {size:g} "
               f"--sigma {sigma:g} --geometry {args.geometry} "
               f"{'--no-graph' if args.no_graph else f'--{kind} {reach:g}'} "
               f"--seed {seed}")
    if args.realizations > 1:
        command += f" --realizations {args.realizations}, number"
    walls = f", {np.count_nonzero(m)} of them on a wall" if not periodic else ""

    for i, stream in enumerate(streams):
        stem = base if args.realizations == 1 else f"{base}_{i:0{width}d}"
        pts = perturb(pts0, m, sigma * h, size, periodic,
                      np.random.default_rng(stream))

        if node_file:
            write_node(stem, pts, m,
                       command if args.realizations == 1 else f"{command} {i}")
        else:
            write_points(stem, pts)

        rows, graph = None, ""
        if not args.no_graph:
            rows = stencils(pts, h, size, periodic, kind, reach)
            write_graph(stem, rows)
            sizes = [len(row) for row in rows]
            graph = (f", {sum(sizes)} stencil entries, "
                     f"{min(sizes)} to {max(sizes)} per node")

        print(f"{'standard output' if piped else stem}: {len(pts)} nodes{walls}{graph}",
              file=sys.stderr)

        if args.plot and i == 0:
            plot(pts, m, rows)


if __name__ == "__main__":
    main()
