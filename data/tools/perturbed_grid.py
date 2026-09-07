#!/usr/bin/env python3
"""Point clouds whose nodes are those of a Cartesian grid moved by a small
random amount, with the periodic stencil graph that goes with them.

    python3 perturbed_grid.py -n 40 tg_40
    python3 perturbed_grid.py -n 40 --sigma 0.02 -K 21 tg_40
    python3 perturbed_grid.py -n 40 --geometry channel poiseuille_40.node
    python3 perturbed_grid.py -n 40 --seed 1234 tg_40

The node layout of Strzelczyk and Matyka (2022), as PerturbedGrid in
pointclouds.generators describes it: an N by N grid in lattice units,
every coordinate displaced by up to sigma spacings, on a box periodic on
both sides (the Taylor-Green test) or a channel periodic in x with walls
at y = 0 and y = N (the Poiseuille test), whose wall nodes carry the
markers 1 (bottom) and 3 (top).

The stencil of a node is its K nearest nodes (--knn-graph K, or -K),
18 by default where the reference uses 15, the nodes within a distance
(--radius-graph R, or -R), or those within a square (--range-graph S);
the search wraps around the periodic sides. A run without --seed draws one and reports
it, so that the grid can be repeated.

The output is a points file and a graph file in the same numbering, or a
node file with the markers if the name ends in .node
(docs/file_formats.md).
"""

import argparse

import numpy as np

from pointclouds import cli
from pointclouds.generators import PerturbedGrid


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Lengths are in lattice units: the box is N by N and the spacing "
        "is 1. Markers are 0 interior, 1 bottom wall, 3 top wall.",
    )
    ap.add_argument(
        "output",
        help="output file: the points file and the graph file next to it, "
        "or a node file with the markers and the graph if the name ends "
        "in .node",
    )
    ap.add_argument(
        "-n",
        "--nodes",
        type=cli.number(int, least=2),
        required=True,
        metavar="N",
        help="nodes across the box",
    )
    ap.add_argument(
        "-s",
        "--sigma",
        type=cli.number(float, least=0.0),
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
    cli.add_graph_options(ap)
    ap.add_argument(
        "--seed", type=int, help="seed of the grid (default: drawn and reported)"
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="show the grid, coloured by marker, with one stencil",
    )
    args = ap.parse_args()

    if args.sigma >= 0.5:
        print(
            f"warning: --sigma {args.sigma:g} moves a node out of its own cell, "
            f"and nodes may end up on top of each other"
        )
    if args.seed is None:
        args.seed = np.random.SeedSequence().entropy
        print(f"seed {args.seed}")
    stem, ext = cli.output_stem(args.output, default=".points")

    cloud = PerturbedGrid(args.nodes, args.sigma, args.geometry, args.seed)
    graph = cloud.stencils(*args.graph) if args.graph else None
    cloud.write(stem, ext, graph)
    print(f"{stem}: {cloud.summary(graph)}")
    if args.plot:
        cli.show(cloud, graph)


if __name__ == "__main__":
    main()
