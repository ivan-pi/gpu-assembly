#!/usr/bin/env python3
"""Generate the lid-driven cavity refined towards its walls, as a node file.

The cloud is RefinedCavity in pointclouds.generators.
"""

import argparse

from pointclouds import cli
from pointclouds.generators import RefinedCavity

EPILOG = """\
examples:
  refined_cavity.py cavity.node
  refined_cavity.py --steps 20 cavity_fine.node
  refined_cavity.py --distribution grid --size 100 130 cavity_tall.node

Lengths are in lattice units: the spacing is 1 at the wall, 1.5 in the
second band and 2.5 in the middle, so the three bands from both walls
fill a cavity of 10 N for N steps. Markers: 0 interior, 1 south wall,
2 east, 3 north (the lid), 4 west, 5 corner. A name ending in .points
gives a points file, without the markers (docs/file_formats.md)."""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("output", help="the node file, or a points file if named so")
    ap.add_argument(
        "--distribution",
        choices=("rings", "grid"),
        default="rings",
        help="concentric rectangles, or a tensor-product grid (default: rings)",
    )
    ap.add_argument(
        "--size",
        nargs=2,
        type=cli.number(float, above=0.0),
        metavar=("LX", "LY"),
        help="sides of the cavity, at least the 10 N of the bands, whose "
        "middle is then at the coarsest spacing (default: 10 N by 10 N)",
    )
    ap.add_argument(
        "-n",
        "--steps",
        type=cli.number(int, least=1),
        default=10,
        metavar="N",
        help="spacings across each band (default: 10)",
    )
    ap.add_argument("--plot", action="store_true", help="show the cloud")
    args = ap.parse_args()

    stem, ext = cli.output_stem(args.output, default=".node")
    cloud = RefinedCavity(args.steps, args.size, args.distribution)
    cloud.write(stem, ext)
    print(f"{stem}{ext}: {cloud.summary()}")
    if args.plot:
        cli.show(cloud)


if __name__ == "__main__":
    main()
