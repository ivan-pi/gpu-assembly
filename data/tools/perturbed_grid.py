#!/usr/bin/env python3
"""Generate a randomly perturbed Cartesian grid with its stencil graph.

The cloud is PerturbedGrid in pointclouds.generators, the node layout of
Strzelczyk and Matyka (2022).
"""

import argparse

from pointclouds import cli
from pointclouds.generators import PerturbedGrid

EPILOG = """\
examples:
  perturbed_grid.py -n 40 tg_40
  perturbed_grid.py -n 40 --sigma 0.02 -K 21 tg_40
  perturbed_grid.py -n 40 --geometry channel poiseuille_40.node
  perturbed_grid.py -n 40 --seed 1234 tg_40

Lengths are in lattice units: the box is N by N and the spacing 1. The
periodic box is the Taylor-Green test; the channel, periodic in x with
walls at y = 0 and y = N, the Poiseuille test, its wall nodes with the
markers 1 (bottom) and 3 (top). The stencil search wraps around the
periodic sides. The output is a points file and a graph file in the
same numbering, or a node file with the markers if named so
(docs/file_formats.md)."""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("output", help="the points file, or a node file if named so")
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
        help="displacement of a node, uniform on [-SIGMA, SIGMA] spacings; "
        "a node stays in its cell below 0.5 (default: 0.2)",
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
        "--seed",
        type=int,
        help="seed of the displacements (default: random)",
    )
    ap.add_argument(
        "--plot", action="store_true", help="show the grid, with one stencil"
    )
    args = ap.parse_args()

    if args.sigma >= 0.5:
        print(f"warning: --sigma {args.sigma:g} may put nodes on top of each other")
    stem, ext = cli.output_stem(args.output, default=".points")

    cloud = PerturbedGrid(
        args.nodes, args.sigma, geometry=args.geometry, seed=args.seed
    )
    graph = cloud.stencils(*args.graph) if args.graph else None
    cloud.write(stem, ext, graph)
    print(f"{stem}: {cloud.summary(graph)}")
    if args.plot:
        cli.show(cloud, graph)


if __name__ == "__main__":
    main()
