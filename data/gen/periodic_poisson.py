#!/usr/bin/env python3
"""A Poisson disk sample of a periodic box, on its own or around a
circular hole, with the stencil graph that goes with it.

    python3 periodic_poisson.py --size 32 32 tg_32
    python3 periodic_poisson.py --size 64 64 --hole 20 cylinder_64.node
    python3 periodic_poisson.py --size 32 32 --tile 4 4 tg_128

Lengths are in lattice units: no two nodes are closer than the distance
d, 1 by default, and how many fit is set by --candidates, as PoissonBox
in pointclouds.generators describes. The earlier poisson_32_21 case has
its nodes 0.9 apart, which --distance 0.9 reproduces.

--hole R, or --solid-fraction PHI for the disk covering the fraction PHI
of the box, cuts a disk out of the middle, with nodes on its circle
(marker 6): the unit cell of a square array of cylinders. --tile MX MY
lays MX by MY copies of the sample side by side into a box of MX Lx by
MY Ly (Tiled in pointclouds.nodeset): the copies join without a seam,
and the stencils are searched over the whole tiling.

The output is a points file and a graph file in the same numbering, or a
node file with the markers if the name ends in .node
(docs/file_formats.md).
"""

import argparse

import numpy as np

from pointclouds import stencils
from pointclouds.cli import number, output_stem
from pointclouds.generators import PoissonBox
from pointclouds.nodeset import Tiled

HELP = __doc__.split("\n\n")[0] + """

Lengths are in lattice units. Markers are 0 interior, 6 the wall of the
cylinder."""


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
        "--size",
        nargs=2,
        type=number(float, above=0.0),
        required=True,
        metavar=("LX", "LY"),
        help="sides of the box, in lattice units",
    )
    ap.add_argument(
        "-d",
        "--distance",
        type=number(float, above=0.0),
        default=1.0,
        metavar="D",
        help="least distance between two nodes (default: 1)",
    )
    ap.add_argument(
        "--candidates",
        type=number(int, least=1),
        default=100,
        metavar="K",
        help="candidates a node throws before it is retired: more "
        "of them pack the nodes tighter, up to a point (default: 100)",
    )
    hole = ap.add_mutually_exclusive_group()
    hole.add_argument(
        "--hole",
        type=number(float, above=0.0),
        metavar="R",
        help="cut the disk of radius R out of the middle of the box, "
        "with nodes on its circle",
    )
    hole.add_argument(
        "--solid-fraction",
        type=number(float, above=0.0),
        metavar="PHI",
        help="the same hole, sized to cover the fraction PHI of the box",
    )
    ap.add_argument(
        "--tile",
        nargs=2,
        type=number(int, least=1),
        default=(1, 1),
        metavar=("MX", "MY"),
        help="lay MX by MY copies of the sample side by side "
        "(default: 1 1, the sample alone)",
    )
    stencils.add_option(ap)
    ap.add_argument(
        "--seed", type=int, help="seed of the sample (default: drawn and reported)"
    )
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="write the coordinates only, without the stencil graph",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="show the cloud, coloured by marker, with one stencil and the "
        "outline of the tiles",
    )
    args = ap.parse_args()

    hole = args.hole
    if args.solid_fraction is not None:
        if args.solid_fraction >= 1.0:
            ap.error("--solid-fraction must be less than 1")
        hole = float(np.sqrt(args.solid_fraction * np.prod(args.size) / np.pi))
    stem, ext = output_stem(args.output, default=".points")
    seed = args.seed if args.seed is not None else np.random.SeedSequence().entropy
    if args.seed is None:
        print(f"seed {seed}")

    cloud = PoissonBox(args.size, args.distance, args.candidates, hole, seed)
    if tuple(args.tile) != (1, 1):
        cloud = Tiled(cloud, args.tile)
    graph = None if args.no_graph else cloud.stencils(*args.graph)
    cloud.write(stem, ext, graph)
    print(f"{stem}: {cloud.summary(graph)}")
    if args.plot:
        cloud.show(graph)


if __name__ == "__main__":
    main()
