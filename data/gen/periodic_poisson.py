#!/usr/bin/env python3
"""Scattered point clouds of a periodic box, drawn by Poisson disk
sampling, on their own or around a circular hole, with the periodic
stencil graph that goes with them.

    python3 periodic_poisson.py --size 32 32 tg_32
    python3 periodic_poisson.py --size 32 32 --knn 21 --seed 1234 tg_32
    python3 periodic_poisson.py --size 64 64 --hole 20 cylinder_64.node
    python3 periodic_poisson.py --size 32 32 --tile 4 4 tg_128
    python3 periodic_poisson.py --size 32 32 --realizations 50 --seed 1234 tg_32

The box is [0, Lx) x [0, Ly) with both sides periodic, and the nodes are
a Poisson disk sample of it (pointclouds.poisson): no two closer than a
distance d, and no room left for another one, which makes them scattered
but evenly spread. The sampler is periodic itself, so the spacing holds
across the sides as it does inside.

How tightly they pack is set by the candidates a node throws before it
is retired, --candidates: about 0.65 / d^2 nodes per unit area with the
default 100, 0.62 with 30, 0.68 with 300, where it levels off, at a
sampling time that grows with the count. The distance is the stronger
lever: the earlier cases (poisson_32_21) have their nodes 0.9 apart and
0.81 of them per unit area, which --distance 0.9 reproduces.

Everything is in lattice units: the box is Lx by Ly and d is 1 unless
--distance says otherwise, so every length below -- the distance, the
radius of the hole, the reach of a stencil -- is a number of spacings.
Scaling a case to other units is left to whoever needs it.

Two geometries differ in what is cut out of the box:

  periodic  The box alone, as in the Taylor-Green test. Every node is
            interior and the sample fills the box.
  cylinder  The box less the disk of radius R about its centre, given
            as --hole R or as --solid-fraction PHI, the fraction of the
            box the disk covers: the unit cell of the periodic flow
            through a square array of cylinders. Nodes are laid on the
            circle first, as many as keep them d apart, and the sample
            grows from them into the rest of the box, so the nodes
            nearest the cylinder sit about a spacing off it. The nodes
            inside the circle are discarded. The circle nodes come first
            in the file.

--tile MX MY lays MX by MY copies of the sample side by side: a box of
MX Lx by MY Ly holding MX MY times the nodes, sampled once. The sample
is periodic, so the copies join without a seam, and the stencil search
runs over the whole tiling, so no stencil knows where they meet. The
tiling has the spacing of the sample and the size of the tiling: in
lattice units the cloud is bigger, in the units of a fixed physical box
it is finer, and either way it is cheaper than sampling the whole of it.
A tiled cylinder case is an MX by MY array of cylinders. The nodes are
numbered copy by copy, the copies row by row from the origin with x
fastest, and the circle nodes of all the copies come first. A pattern
repeats in it, of course, with the period of the sample.

The stencil of a node is the set of nodes it interpolates from: its k
nearest neighbours by default, the nodes within a given distance with
--radius, or those within a square with --square. The search wraps
around both sides, so a stencil next to a side reaches around it, and
every stencil starts with the node itself.

Random clouds are meant to be averaged over, so a run generates one
realization per --realizations and numbers the files. Realization i
depends on the seed and on i alone: asking for more of them extends the
series rather than replacing it, and a run without --seed reports the
seed it drew so that it can be repeated.

The output is a points file and a graph file, in the numbering the two
share. A name ending in `.node` writes a node file instead of the points
file, with a comment line naming the command and a marker per node
(docs/file_formats.md). The name `-` writes to standard output, and
`-- -.node` a node file there, after the option separator since the name
starts with a dash. Only one file fits down a pipe, so both take
--no-graph and a single realization. The report goes to standard error,
so the stream carries the file alone. The markers are:

    0  interior
    6  the circle, the wall of the cylinder

The numbering is the one the generators share (pointclouds/markers.py):
the sides are periodic and carry no nodes of their own, so the markers
of the four walls and the corners are not used.
"""

import argparse
import sys

import numpy as np

from pointclouds import stencils
from pointclouds.cli import number, output_stem
from pointclouds.io import write_graph, write_node, write_points
from pointclouds.markers import HOLE, INTERIOR, MARKER_STYLE
from pointclouds.poisson import PoissonDisk, wrap


def circle(radius, centre, d):
    """Nodes on the circle, counter-clockwise from the x axis, as many as
    keep consecutive ones at least d apart along the chord."""
    n = int(np.pi / np.arcsin(min(1.0, d / (2.0 * radius))))
    phi = 2.0 * np.pi * np.arange(n) / n
    return centre + radius * np.column_stack((np.cos(phi), np.sin(phi)))


def sample(extent, d, candidates, hole, seed):
    """The nodes of one realization and their markers: the circle nodes
    first, if there is a hole, then the sample grown from them."""
    centre = 0.5 * extent
    seeds = circle(hole, centre, d) if hole else None
    engine = PoissonDisk(
        d, extent, periodic=True, ncandidates=candidates, seed=seed, seeds=seeds
    )
    engine.fill_space()
    pts = engine.points
    nb = 0 if seeds is None else len(seeds)
    if hole:
        inside = np.hypot(*(pts - centre).T) < hole
        inside[:nb] = False  # the circle nodes sit on the hole, not in it
        pts = pts[~inside]
    m = np.full(len(pts), INTERIOR)
    m[:nb] = HOLE
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
        (np.flatnonzero(m != INTERIOR), np.flatnonzero(m == INTERIOR))
    )
    return wrap(pts[order], extent * tiles), m[order]


def plot(pts, m, rows, extent, tiles):
    import matplotlib.pyplot as plt

    plt.scatter(pts[:, 0], pts[:, 1], c=m, s=8, **MARKER_STYLE)
    if rows:  # one stencil, to see it wrap
        middle = rows[len(pts) // 2]
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

Lengths are in lattice units: the box is LX by LY and no two nodes are
closer than the distance D, 1 by default. Markers are 0 interior, 6 the
wall of the cylinder. The docstring at the top of the script describes
the two geometries, the tiling and the series of realizations."""


def main():
    ap = argparse.ArgumentParser(
        description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "output",
        help="output file: the points file and the graph "
        "file next to it, or a node file with the markers and the "
        "graph if the name ends in .node; `-` is standard output",
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

    stencils.add_options(ap, knn=21, why=", as in the poisson_32_21 case")

    ap.add_argument(
        "--seed",
        type=int,
        help="seed of the sample (default: drawn and reported)",
    )
    ap.add_argument(
        "--realizations",
        type=number(int, least=1),
        default=1,
        metavar="R",
        help="independent clouds to write, numbered from 0 (default: 1)",
    )
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="write the coordinates only, without the stencil graph",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="show the first cloud, coloured by marker, with one stencil "
        "and the outline of the tiles",
    )
    args = ap.parse_args()

    extent, tiles = np.array(args.size), np.array(args.tile)
    d = args.distance
    hole = args.hole
    if args.solid_fraction is not None:
        if args.solid_fraction >= 1.0:
            ap.error("--solid-fraction must be less than 1")
        hole = np.sqrt(args.solid_fraction * extent.prod() / np.pi)
    if hole is not None:
        if hole < d:
            ap.error(
                f"a hole of radius {hole:g} is smaller than the distance "
                f"{d:g} between nodes and would hold none on its circle"
            )
        if 2.0 * hole >= extent.min():
            ap.error(
                f"the hole must fit in the box: a radius of {hole:g} "
                f"does not, the box is {extent.min():g} across"
            )
    if extent.min() < 2.0 * d:
        ap.error(
            f"the box is narrower than twice the distance {d:g}: no room for nodes"
        )

    stencil = stencils.from_args(args, knn=21)
    box = extent * tiles
    if not args.no_graph:
        problem = stencils.check(stencil, box, True, np.inf)  # the count later
        if problem:
            ap.error(problem)

    seed = np.random.SeedSequence().entropy if args.seed is None else args.seed
    streams = np.random.SeedSequence(seed).spawn(args.realizations)

    base, ext = output_stem(args.output, default=".points")
    width = len(str(args.realizations - 1))

    piped = base == "-"  # `-`, `-.points` or `-.node`
    if piped and not args.no_graph:
        ap.error(
            "only one file fits down a pipe: add --no-graph to write the "
            "coordinates to standard output, or name a file for the pair"
        )
    if piped and args.realizations > 1:
        ap.error(
            "a series needs file names to go in: --realizations cannot "
            "write to standard output"
        )
    if args.seed is None:
        print(f"seed {seed}", file=sys.stderr)

    # The comment of a node file is the command that reproduces it.
    command = (
        f"produced by periodic_poisson.py --size {extent[0]:g} {extent[1]:g} "
        f"--distance {d:g} --candidates {args.candidates} "
        f"{f'--hole {hole:g} ' if hole is not None else ''}"
        f"--tile {tiles[0]} {tiles[1]} "
        f"{'--no-graph' if args.no_graph else f'--{stencil.kind} {stencil.reach:g}'} "
        f"--seed {seed}"
    )
    if args.realizations > 1:
        command += f" --realizations {args.realizations}, number"
    area = box.prod() - (0.0 if hole is None else tiles.prod() * np.pi * hole**2)

    for i, stream in enumerate(streams):
        stem = base if args.realizations == 1 else f"{base}_{i:0{width}d}"
        pts, m = sample(extent, d, args.candidates, hole, stream)
        pts, m = tiled(pts, m, extent, tiles)
        if not args.no_graph and stencils.check(stencil, box, True, len(pts)):
            ap.error(stencils.check(stencil, box, True, len(pts)))

        if ext == ".node":
            write_node(
                stem, pts, m, command if args.realizations == 1 else f"{command} {i}"
            )
        else:
            write_points(stem, pts)

        rows, graph = None, ""
        if not args.no_graph:
            rows = stencils.search(pts, box, True, stencil)
            write_graph(stem, rows)
            sizes = [len(row) for row in rows]
            graph = (
                f", {sum(sizes)} stencil entries, "
                f"{min(sizes)} to {max(sizes)} per node"
            )

        nb = np.count_nonzero(m)
        walls = f", {nb} of them on the cylinder{'s' if tiles.prod() > 1 else ''}"
        print(
            f"{'standard output' if piped else stem}: {len(pts)} nodes"
            f"{walls if nb else ''}, {len(pts) / area:.3g} per unit area{graph}",
            file=sys.stderr,
        )

        if args.plot and i == 0:
            plot(pts, m, rows, extent, tiles)


if __name__ == "__main__":
    main()
