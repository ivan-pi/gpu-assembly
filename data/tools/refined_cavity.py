#!/usr/bin/env python3
"""Generate the lid-driven cavity, refined towards its walls.

The cloud is RefinedCavity in pointclouds.generators.
"""

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
    ap = cli.parser(__doc__, EPILOG)
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
    cli.add_output_options(ap, ".node")
    args = ap.parse_args()

    cloud = RefinedCavity(args.steps, size=args.size, distribution=args.distribution)
    cli.finish(cloud, args)


if __name__ == "__main__":
    main()
