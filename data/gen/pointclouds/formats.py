"""The files the generators in data/gen write, in the formats specified by
docs/file_formats.md, which is where the details live: this module only
puts them on disk.

Every number is written as the shortest text that reads back to the same
double, which is what `repr` gives, so a file reproduces the values it was
given. A writer takes a stem rather than a file name and appends its own
extension, so one stem names the whole case: `case.points` and
`case.graph` are the pair the readers expect, in the same node numbering.

The boundary markers are the convention the generators share, numbered
counter-clockwise from the bottom wall. A node file carries one per node,
zero for an interior node and any nonzero value for a boundary one; which
nonzero value means what is ours to fix, and this is where it is fixed. A
generator uses the ones its geometry has: a periodic side has no wall and
so no marker, and only a cavity has corners.
"""

import contextlib
import sys

INTERIOR, SOUTH, EAST, NORTH, WEST, CORNER = 0, 1, 2, 3, 4, 5

# Marker colours for --plot: fixed here so that a marker means the same
# colour whichever generator drew the cloud.
MARKER_STYLE = dict(cmap="tab10", vmin=0, vmax=9)


def open_out(stem, ext):
    """The file this stem and extension name, or standard output for `-`,
    which stays open."""
    return (contextlib.nullcontext(sys.stdout) if stem == "-"
            else open(stem + ext, "w"))


def write_points(stem, pts):
    """A points file: the count, then a coordinate pair per line."""
    with open_out(stem, ".points") as f:
        f.write(f"{len(pts)}\n")
        for x, y in pts.tolist():
            f.write(f"{x!r} {y!r}\n")            # repr: shortest round-trip text


def write_node(stem, pts, m, provenance):
    """A node file: a comment naming the command that produced it, the
    header, then a numbered node with its marker per line."""
    with open_out(stem, ".node") as f:
        f.write(f"# {provenance}\n")
        f.write(f"{len(pts)} 2 0 1\n")
        for i, ((x, y), mi) in enumerate(zip(pts.tolist(), m.tolist())):
            f.write(f"{i} {x!r} {y!r} {mi}\n")


def write_graph(stem, rows):
    """A graph file: the node count and the entry count, then the stencil
    of each node, which starts with the node itself."""
    with open_out(stem, ".graph") as f:
        f.write(f"{len(rows)} {sum(len(row) for row in rows)}\n")
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")
