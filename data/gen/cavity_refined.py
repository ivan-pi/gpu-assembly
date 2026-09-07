#!/usr/bin/env python3
"""Point clouds for the lid-driven cavity [0, Lx] x [0, Ly], refined
towards the walls in three levels, written as a node file with boundary
markers.

    python3 cavity_refined.py cavity.node
    python3 cavity_refined.py --steps 20 cavity_fine.node
    python3 cavity_refined.py --distribution grid --size 100 130 cavity_tall.node

Everything is in lattice units: the spacing is 1 at the wall, 1.5 in the
second band and 2.5 in the middle, so the three bands from both walls
fill a cavity of 10 N for --steps N, and --size LX LY makes it bigger
with its middle at the coarsest spacing. RefinedCavity in
pointclouds.generators describes the two distributions and the markers:
0 interior, 1 south wall, 2 east, 3 north (the lid), 4 west, 5 corner.

The output is a node file, or a points file without the markers if the
name ends in .points (docs/file_formats.md).
"""

import argparse

from pointclouds.cli import number, output_stem
from pointclouds.generators import RefinedCavity

HELP = __doc__.split("\n\n")[0] + """

Lengths are in lattice units, in which the spacing is 1 at the wall and
the cavity is 10 N by 10 N unless --size makes it bigger. Boundary
markers: 0 interior, 1 south wall, 2 east, 3 north (the lid), 4 west,
5 corner."""


def main():
    ap = argparse.ArgumentParser(
        description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "output",
        help="output file: a node file with the markers, or a "
        "points file without them if the name ends in .points",
    )
    ap.add_argument(
        "--distribution",
        choices=("rings", "grid"),
        default="rings",
        help="concentric rectangles, or a tensor-product grid (default: rings)",
    )
    ap.add_argument(
        "--size",
        nargs=2,
        type=number(float, above=0.0),
        metavar=("LX", "LY"),
        help="sides of the cavity in lattice units, at least the "
        "10 N the bands need (default: 10 N by 10 N, the bands "
        "alone)",
    )
    ap.add_argument(
        "-n",
        "--steps",
        type=number(int, least=1),
        default=10,
        metavar="N",
        help="spacings across each band (default: 10)",
    )
    ap.add_argument(
        "--plot", action="store_true", help="show the cloud, coloured by marker"
    )
    args = ap.parse_args()

    stem, ext = output_stem(args.output, default=".node")
    cloud = RefinedCavity(args.steps, args.size, args.distribution)
    cloud.write(stem, ext)
    print(f"{stem}{ext}: {cloud.summary()}")
    if args.plot:
        cloud.show()


if __name__ == "__main__":
    main()
