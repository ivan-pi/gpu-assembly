#!/usr/bin/env python3
"""Generate the disk, or an annulus, with its stencil graph.

The cloud is Disk in pointclouds.generators, in concentric rings or in
a Vogel spiral.
"""

from pointclouds import cli
from pointclouds.generators import Disk

EPILOG = """\
examples:
  disk.py --radius 20 disk_20
  disk.py --radius 20 --distribution spiral disk_20_spiral.node
  disk.py --radius 20 --hole 6 -K 21 couette_20_6.node
  disk.py --radius 1 --spacing 0.05 unit_disk.node

Lengths are in lattice units, so a radius of R at the spacing 1 is the
disk of radius R; dividing the coordinates by R gives the unit disk.
The nodes of the outer circle come first, marked 1, then those of the
hole, marked 6. The output is a points file and a graph file in the
same numbering, or a node file with the markers if named so
(docs/file_formats.md)."""


def main():
    ap = cli.parser(__doc__, EPILOG)
    ap.add_argument(
        "--radius",
        type=cli.number(float, above=0.0),
        required=True,
        metavar="R",
        help="radius of the disk, about the origin",
    )
    ap.add_argument(
        "-d",
        "--spacing",
        type=cli.number(float, above=0.0),
        default=1.0,
        metavar="H",
        help="distance between neighbouring nodes (default: 1)",
    )
    ap.add_argument(
        "--hole",
        type=cli.number(float, above=0.0),
        default=0.0,
        metavar="R0",
        help="cut the disk of radius R0 out of the middle, which makes the "
        "cloud the annulus R0 <= r <= R between two concentric cylinders",
    )
    ap.add_argument(
        "--distribution",
        choices=("rings", "spiral"),
        default="rings",
        help="concentric rings a spacing apart, turned against each other, "
        "or a quasi-uniform Vogel spiral (default: rings)",
    )
    cli.add_graph_options(ap)
    cli.add_output_options(ap, ".points")
    args = ap.parse_args()

    cloud = Disk(
        args.radius,
        args.spacing,
        hole=args.hole,
        distribution=args.distribution,
    )
    cli.finish(cloud, args)


if __name__ == "__main__":
    main()
