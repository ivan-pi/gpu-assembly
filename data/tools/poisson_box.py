#!/usr/bin/env python3
"""Generate a Poisson disk sample of a periodic box with its stencil graph.

The cloud is PoissonBox in pointclouds.generators, tiled by TiledNodeSet
in pointclouds.nodeset when asked.
"""

import argparse

import numpy as np

from pointclouds import cli
from pointclouds.generators import PoissonBox
from pointclouds.nodeset import TiledNodeSet

EPILOG = """\
examples:
  poisson_box.py --size 32 32 tg_32
  poisson_box.py --size 64 64 --hole 20 cylinder_64.node
  poisson_box.py --size 32 32 --tile 4 4 tg_128

Lengths are in lattice units. The nodes on the circle of a hole come
first, marked 6. The output is a points file and a graph file in the
same numbering, or a node file with the markers if named so
(docs/file_formats.md)."""


def hole_radius(solid_fraction, size):
    """Return the radius of the disk covering the fraction of the box."""
    lx, ly = size
    area = solid_fraction * lx * ly
    return np.sqrt(area / np.pi)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("output", help="the points file, or a node file if named so")
    ap.add_argument(
        "--size",
        nargs=2,
        type=cli.number(float, above=0.0),
        required=True,
        metavar=("LX", "LY"),
        help="sides of the box",
    )
    ap.add_argument(
        "-d",
        "--distance",
        type=cli.number(float, above=0.0),
        default=1.0,
        metavar="D",
        help="least distance between two nodes (default: 1)",
    )
    ap.add_argument(
        "--candidates",
        type=cli.number(int, least=1),
        default=100,
        metavar="K",
        help="candidates a node throws before it is retired: more of them "
        "pack the nodes tighter, up to a point (default: 100)",
    )
    hole = ap.add_mutually_exclusive_group()
    hole.add_argument(
        "--hole",
        type=cli.number(float, above=0.0),
        metavar="R",
        help="cut the disk of radius R out of the middle of the box",
    )
    hole.add_argument(
        "--solid-fraction",
        type=cli.number(float, above=0.0),
        metavar="PHI",
        help="the same hole, sized to cover the fraction PHI of the box",
    )
    ap.add_argument(
        "--tile",
        nargs=2,
        type=cli.number(int, least=1),
        default=(1, 1),
        metavar=("MX", "MY"),
        help="lay MX by MY copies of the sample side by side (default: 1 1)",
    )
    cli.add_graph_options(ap)
    ap.add_argument("--seed", type=int, help="seed of the sample (default: random)")
    ap.add_argument(
        "--plot", action="store_true", help="show the cloud, with one stencil"
    )
    args = ap.parse_args()

    hole = args.hole
    if args.solid_fraction is not None:
        if args.solid_fraction >= 1.0:
            ap.error("--solid-fraction must be less than 1")
        hole = hole_radius(args.solid_fraction, args.size)
    stem, ext = cli.output_stem(args.output, default=".points")

    cloud = PoissonBox(
        args.size, args.distance, candidates=args.candidates, hole=hole, seed=args.seed
    )
    if tuple(args.tile) != (1, 1):
        cloud = TiledNodeSet(cloud, args.tile)
    graph = cloud.stencils(*args.graph) if args.graph else None
    cloud.write(stem, ext, graph)
    print(f"{stem}: {cloud.summary(graph)}")
    if args.plot:
        cli.show(cloud, graph)


if __name__ == "__main__":
    main()
