#!/usr/bin/env python3
"""Point clouds whose nodes are those of a Cartesian grid moved by a small
random amount, with the periodic stencil graph that goes with them.

    python3 perturbed_grid.py -n 40 tg_40
    python3 perturbed_grid.py -n 40 --sigma 0.02 --graph knn=21 tg_40
    python3 perturbed_grid.py -n 40 --geometry channel poiseuille_40.node
    python3 perturbed_grid.py -n 40 --seed 1234 tg_40

The node layout of Strzelczyk and Matyka (2022), "How nodes layout,
refinement and velocity discretization influence convergence of the
meshless lattice Boltzmann method", <https://ssrn.com/abstract=4070398>,
Figs. 4 and 8. An N by N grid is laid on the box with its lowest node at
the origin, and every coordinate is then displaced by an amount drawn
uniformly from [-sigma, sigma] spacings. The reference uses sigma = 0,
0.02 and 0.2; a node stays in its own cell for sigma < 0.5.

Everything is in lattice units: the box is N by N and the spacing is 1,
so the coordinates are the lattice sites the LBM works on and every
length below -- a displacement, the reach of a stencil -- is a number of
spacings. Scaling a case to other units is left to whoever needs it. The
nodes sit on the lattice rather than at cell centres, so a Cartesian
sigma = 0 grid runs from 0 to N - 1 rather than from 0.5 to N - 0.5; on a
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

The stencil of a node is the set of nodes it interpolates from,
selected by --graph: its k nearest neighbours (`knn=K`, the default
with the 15 of the reference), the nodes within a distance
(`radius=R`), or the nodes within a square (`range=S`). The search knows which
sides are periodic, so a stencil next to a periodic side reaches around
it, and every stencil starts with the node itself.

A run without --seed draws one and reports it, so that the grid can be
repeated; a set of independent grids to average over is a set of runs
with different seeds.

The output is a points file and a graph file, in the numbering the two
share. A name ending in `.node` writes a node file instead of the points
file, with a comment line naming the command and a marker per node
(docs/file_formats.md). The markers are:

    0  interior
    1  bottom wall, y = 0
    3  top wall, y = N

The numbering is the one the generators share, counter-clockwise from
the bottom wall (pointclouds/stencils.py), which leaves 2 and 4 for the
east and west walls: those sides are periodic here and carry no nodes of
their own.
"""

import argparse

import numpy as np

from pointclouds import stencils
from pointclouds.cli import number, output_stem
from pointclouds.io import write_graph, write_node, write_points
from pointclouds.poisson import wrap
from pointclouds.stencils import MARKERS


def grid(n, periodic):
    """The Cartesian nodes and their markers: N by N on the periodic box,
    N by N + 1 on the channel, whose last row is the top wall."""
    x = np.arange(float(n))
    y = np.arange(float(n if periodic else n + 1))
    xv, yv = np.meshgrid(x, y)  # row by row, x fastest
    pts = np.column_stack((xv.ravel(), yv.ravel()))
    m = np.full(len(pts), MARKERS.interior)
    if not periodic:
        m[:n], m[len(pts) - n :] = MARKERS.south, MARKERS.north
    return pts, m


def perturb(pts, m, sigma, box, periodic, rng):
    """Every coordinate displaced by up to sigma spacings, except the one
    across the wall, which would take a wall node off its wall."""
    d = rng.uniform(-sigma, sigma, pts.shape)
    d[m != MARKERS.interior, 1] = 0.0
    pts = pts + d
    pts[:, 0] = wrap(pts[:, 0], box)
    if periodic:
        pts[:, 1] = wrap(pts[:, 1], box)
    else:
        pts[:, 1] = np.clip(pts[:, 1], 0.0, box)  # only reached by sigma >= 1
    return pts


def plot(pts, m, graph):
    import matplotlib.pyplot as plt

    plt.scatter(pts[:, 0], pts[:, 1], c=m, s=8, **MARKERS.style)
    if graph:  # one stencil, to see it wrap
        ia, ja = graph
        middle = ja[ia[len(pts) // 2] : ia[len(pts) // 2 + 1]]
        plt.scatter(
            pts[middle, 0],
            pts[middle, 1],
            marker="s",
            s=24,
            facecolors="none",
            edgecolors="tab:orange",
        )
    plt.axis("equal")
    plt.show()


HELP = __doc__.split("\n\n")[0] + """

Lengths are in lattice units: the box is N by N and the spacing is 1.
Markers are 0 interior, 1 bottom wall, 3 top wall. The docstring at the
top of the script describes the two geometries."""


def main():
    ap = argparse.ArgumentParser(
        description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "output",
        help="output file: the points file and the graph "
        "file next to it, or a node file with the markers and the "
        "graph if the name ends in .node",
    )
    ap.add_argument(
        "-n",
        "--nodes",
        type=number(int, least=2),
        required=True,
        metavar="N",
        help="nodes across the box",
    )
    ap.add_argument(
        "-s",
        "--sigma",
        type=number(float, least=0.0),
        default=0.2,
        metavar="SIGMA",
        help="displacement of a node, in spacings: uniform on "
        "[-SIGMA, SIGMA] (default: 0.2)",
    )
    ap.add_argument(
        "-g",
        "--geometry",
        choices=("periodic", "channel"),
        default="periodic",
        help="periodic on both sides, or walls at y = 0 and y = N "
        "and periodic in x (default: periodic)",
    )

    stencils.add_option(ap, default="knn=15")
    ap.add_argument(
        "--seed", type=int, help="seed of the grid (default: drawn and reported)"
    )
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="write the coordinates only, without the stencil graph",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="show the grid, coloured by marker, with one stencil",
    )
    args = ap.parse_args()

    n, sigma, periodic = args.nodes, args.sigma, args.geometry == "periodic"
    box = float(n)  # N by N at a spacing of one
    extent, wraps = (box, box), (True, periodic)
    if sigma >= 0.5:
        print(
            f"warning: --sigma {sigma:g} moves a node out of its own cell, "
            f"and nodes may end up on top of each other"
        )

    method, value = args.graph
    if not args.no_graph:
        problem = stencils.check(method, value, extent, wraps)
        if problem:
            ap.error(problem)
    stem, ext = output_stem(args.output, default=".points")
    seed = args.seed if args.seed is not None else np.random.SeedSequence().entropy
    if args.seed is None:
        print(f"seed {seed}")

    pts0, m = grid(n, periodic)
    pts = perturb(pts0, m, sigma, box, periodic, np.random.default_rng(seed))
    graph = None
    if not args.no_graph:
        graph = stencils.select_stencils(pts, extent, wraps, method, value)

    if ext == ".node":
        write_node(
            stem,
            pts,
            m,
            f"perturbed grid, size={n}x{n}, n={n}, sigma={sigma:g}, {args.geometry}",
        )
    else:
        write_points(stem, pts)
    boundary = np.count_nonzero(m)
    report = f"{stem}: {len(pts)} nodes ({len(pts) - boundary} interior, {boundary} boundary)"
    if graph:
        ia, ja = graph
        write_graph(stem, ia, ja)
        sizes = np.diff(ia)
        report += (
            f", {ia[-1]} edges in the stencil graph, "
            f"stencils of {sizes.min()} to {sizes.max()} nodes"
        )
    print(report)

    if args.plot:
        plot(pts, m, graph)


if __name__ == "__main__":
    main()
