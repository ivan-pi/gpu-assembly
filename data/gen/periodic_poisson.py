#!/usr/bin/env python3
"""A Poisson disk sample of a periodic box, on its own or around a
circular hole, with the stencil graph that goes with it.

    python3 periodic_poisson.py --size 32 32 tg_32
    python3 periodic_poisson.py --size 64 64 --hole 20 cylinder_64.node
    python3 periodic_poisson.py --size 32 32 --tile 4 4 tg_128

Lengths are in lattice units: no two nodes are closer than the distance
d, 1 by default, and how many fit is set by --candidates, about 0.65 / d^2
per unit area at the default, more with more candidates up to a point.
The earlier poisson_32_21 case has its nodes 0.9 apart, which
--distance 0.9 reproduces.

--hole R, or --solid-fraction PHI for the disk covering the fraction PHI
of the box, cuts a disk out of the middle: the unit cell of a square
array of cylinders. Nodes are laid on its circle first, marker 6, and
the sample grows from them, so the nearest nodes sit a spacing off the
cylinder. --tile MX MY lays MX by MY copies of the sample side by side
into a box of MX Lx by MY Ly, numbered copy by copy with the circle
nodes of all the copies first; the sample is periodic, so the copies
join without a seam, and the stencils are searched over the whole
tiling.

The output is a points file and a graph file in the same numbering, or a
node file with the markers if the name ends in .node
(docs/file_formats.md).
"""

import argparse

import numpy as np

from pointclouds import stencils
from pointclouds.cli import number, output_stem
from pointclouds.io import write_graph, write_node, write_points
from pointclouds.poisson import PoissonDisk, wrap
from pointclouds.stencils import Markers


def circle(radius, centre, d):
    """Nodes on the circle, counter-clockwise from the x axis, as many as
    keep consecutive ones at least d apart along the chord; the radius
    is at least d."""
    n = int(np.pi / np.arcsin(d / (2.0 * radius)))
    phi = 2.0 * np.pi * np.arange(n) / n
    return centre + radius * np.column_stack((np.cos(phi), np.sin(phi)))


def sample(extent, d, candidates, hole, seed):
    """The nodes of one sample and their markers: the circle nodes
    first, if there is a hole, then the sample grown from them."""
    centre = 0.5 * extent
    seeds = circle(hole, centre, d) if hole else np.empty((0, 2))
    sampler = PoissonDisk(
        d, extent, periodic=True, ncandidates=candidates, seed=seed, seeds=seeds
    )
    sampler.fill_space()
    pts = sampler.points
    inside = np.hypot(*(pts - centre).T) < (hole or 0.0)
    inside[: len(seeds)] = False  # the circle nodes sit on the hole, not in it
    pts = pts[~inside]
    m = np.full(len(pts), Markers.interior)
    m[: len(seeds)] = Markers.hole
    return pts, m


def tiled(pts, m, extent, tiles):
    """MX by MY copies of the cloud side by side, the copies row by row
    from the origin with x fastest, on the box `extent * tiles`, and the
    boundary nodes of all the copies ahead of the interior ones."""
    mx, my = tiles
    shifts = [(ix, iy) for iy in range(my) for ix in range(mx)]
    pts = np.vstack([pts + extent * shift for shift in shifts])
    m = np.tile(m, len(shifts))
    order = np.concatenate(
        (np.flatnonzero(m != Markers.interior), np.flatnonzero(m == Markers.interior))
    )
    return wrap(pts[order], extent * tiles), m[order]


def plot(pts, m, graph, extent, tiles):
    import matplotlib.pyplot as plt

    plt.scatter(pts[:, 0], pts[:, 1], c=m, s=8, **Markers.style)
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
    lx, ly = extent
    mx, my = tiles
    plt.vlines(lx * np.arange(mx + 1), 0, my * ly, color="0.7", lw=0.8)
    plt.hlines(ly * np.arange(my + 1), 0, mx * lx, color="0.7", lw=0.8)
    plt.axis("equal")
    plt.show()


HELP = __doc__.split("\n\n")[0] + """

Lengths are in lattice units. Markers are 0 interior, 6 the wall of the
cylinder. The docstring at the top of the script has the details."""


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

    stencils.add_option(ap, default="knn=21")
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

    lx, ly = args.size
    extent, tiles = np.array(args.size), np.array(args.tile)
    d = args.distance
    hole = args.hole
    if args.solid_fraction is not None:
        if args.solid_fraction >= 1.0:
            ap.error("--solid-fraction must be less than 1")
        hole = float(np.sqrt(args.solid_fraction * extent.prod() / np.pi))
    if hole is not None:
        if hole < d:
            ap.error(
                f"the hole must be at least the distance {d:g} between nodes "
                f"in radius, and {hole:g} is not"
            )
        if 2.0 * hole + d > extent.min():
            ap.error(
                f"the hole must leave a spacing between itself and its image "
                f"across the periodic sides: a radius of {hole:g} in a box "
                f"{extent.min():g} across does not"
            )
    if extent.min() < 2.0 * d:
        ap.error(
            f"the box is narrower than twice the distance {d:g}: no room for nodes"
        )

    method, value = args.graph
    box = extent * tiles
    if not args.no_graph:
        problem = stencils.check(method, value, box, (True, True))
        if problem:
            ap.error(problem)
    stem, ext = output_stem(args.output, default=".points")
    seed = args.seed if args.seed is not None else np.random.SeedSequence().entropy
    if args.seed is None:
        print(f"seed {seed}")

    pts, m = tiled(*sample(extent, d, args.candidates, hole, seed), extent, tiles)
    graph = None
    if not args.no_graph:
        graph = stencils.select_stencils(pts, box, (True, True), method, value)

    if ext == ".node":
        geometry = f"size={lx:g}x{ly:g}, distance={d:g}"
        if hole is not None:
            geometry += f", hole={hole:g}"
        if tiles.prod() > 1:
            geometry += f", tile={tiles[0]}x{tiles[1]}"
        write_node(stem, pts, m, f"periodic poisson, {geometry}")
    else:
        write_points(stem, pts)
    area = box.prod() - (0.0 if hole is None else tiles.prod() * np.pi * hole**2)
    boundary = np.count_nonzero(m)
    report = (
        f"{stem}: {len(pts)} nodes ({len(pts) - boundary} interior, {boundary} boundary), "
        f"{len(pts) / area:.3g} per unit area"
    )
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
        plot(pts, m, graph, extent, tiles)


if __name__ == "__main__":
    main()
