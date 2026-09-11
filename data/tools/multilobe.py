#!/usr/bin/env python3
"""Generate a multilobe catalyst pellet with its stencil graph.

The cloud is Multilobe in pointclouds.generators: the union of
overlapping circular lobes, its boundary laid first and the interior a
Poisson disk sample, with the junctions sharp or filleted.
"""

from pointclouds import cli
from pointclouds.generators import Multilobe

EPILOG = """\
examples:
  multilobe.py --radius 12 trilobe_12
  multilobe.py --radius 12 --fillet 3 trilobe_12_fillet.node
  multilobe.py --radius 12 --lobes 4 quadrilobe_12.node

Lengths are in lattice units. The pellet is the union of the lobes,
circles of radius R about centres OFFSET from the origin, the first
lobe pointing up. Where two lobes meet the boundary turns through a
reentrant corner (300 degrees for the default trilobe), where the
boundary flux of a reaction-diffusion solution is singular; a fillet
rounds each junction with a tangent arc and switches the singularity
off. The junction nodes are marked 5 and every other boundary node 1;
with a fillet the whole boundary is smooth and marked 1. The output is
a points file and a graph file in the same numbering, or a node file
with the markers if named so (docs/file_formats.md)."""


def main():
    ap = cli.parser(__doc__, EPILOG)
    ap.add_argument(
        "--radius",
        type=cli.number(float, above=0.0),
        required=True,
        metavar="R",
        help="radius of a lobe",
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
        "--lobes",
        type=cli.number(int, least=2),
        default=3,
        metavar="K",
        help="circles about the origin: 3 is the trilobe, 4 the "
        "quadrilobe (default: 3)",
    )
    ap.add_argument(
        "--offset",
        type=cli.number(float, above=0.0),
        default=None,
        metavar="E",
        help="distance of the lobe centres from the origin "
        "(default: the radius, the usual pellet)",
    )
    ap.add_argument(
        "--fillet",
        type=cli.number(float, least=0.0),
        default=0.0,
        metavar="RHO",
        help="round each junction with an arc of radius RHO tangent to "
        "both lobes (default: 0, sharp corners)",
    )
    ap.add_argument(
        "--candidates",
        type=cli.number(int, least=1),
        default=100,
        metavar="K",
        help="candidates a node throws before it is retired: more of them "
        "pack the nodes tighter, up to a point (default: 100)",
    )
    cli.add_graph_options(ap)
    ap.add_argument("--seed", type=int, help="seed of the sample (default: random)")
    cli.add_output_options(ap, ".points")
    args = ap.parse_args()

    cloud = Multilobe(
        args.radius,
        args.spacing,
        lobes=args.lobes,
        offset=args.offset,
        fillet=args.fillet,
        ncandidates=args.candidates,
        seed=args.seed,
    )
    cli.finish(cloud, args)


if __name__ == "__main__":
    main()
