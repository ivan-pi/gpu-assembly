#!/usr/bin/env python3
"""Show a Poisson disk sample of the unit square.

Three panels: the points with their Delaunay triangulation, the contours
of a function interpolated over it, and the surface of that function.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation

from pointclouds.cli import number
from pointclouds.poisson import PoissonDisk


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="examples:\n  poisson_demo.py\n  poisson_demo.py --radius 0.05 --save sample.png",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--radius",
        type=number(float, above=0.0),
        default=0.02,
        help="minimum distance between points (default: 0.02)",
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="seed of the sample (default: 0)"
    )
    ap.add_argument("--save", metavar="FILE", help="write the figure to FILE")
    args = ap.parse_args()

    pts = PoissonDisk(args.radius, ncandidates=10, seed=args.seed).fill_space()
    x, y = pts[:, 0], pts[:, 1]
    z = np.sin(6 * x) + np.cos(2 * y)
    tri = Triangulation(x, y)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.triplot(tri, color="0.8")
    ax.plot(x, y, "o", ms=2)
    ax.set_aspect("equal")
    ax.set_title(f"{len(pts)} points, r = {args.radius:g}")
    ax = fig.add_subplot(1, 3, 2)
    ax.tricontourf(tri, z, 40, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_title("sin(6x) + cos(2y) on the triangulation")
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    ax.plot_trisurf(x, y, z, cmap="viridis", linewidth=0.2)
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=100)
    else:
        plt.show()


if __name__ == "__main__":
    main()
