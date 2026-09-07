"""Writers for the files of docs/file_formats.md, which is where the
details live: this module only puts them on disk. The generators in
data/gen write the point, node and graph files, the tools in data/tools
the ordering file.

Every number is written as the shortest text that reads back to the same
double, which is what `repr` gives, so a file reproduces the values it was
given. A writer takes a stem rather than a file name and appends its own
extension, so one stem names the whole case: `case.points` and
`case.graph` are the pair the readers expect, in the same node numbering.
"""

import contextlib
import sys


def open_out(stem, ext):
    """The file this stem and extension name, or standard output for `-`,
    which stays open."""
    return contextlib.nullcontext(sys.stdout) if stem == "-" else open(stem + ext, "w")


def write_points(stem, pts):
    """A points file: the count, then a coordinate pair per line."""
    with open_out(stem, ".points") as f:
        f.write(f"{len(pts)}\n")
        for x, y in pts.tolist():
            f.write(f"{x!r} {y!r}\n")  # repr: shortest round-trip text


def write_node(stem, pts, m, provenance):
    """A node file: a comment naming the command that produced it, the
    header, then a numbered node with its marker per line."""
    with open_out(stem, ".node") as f:
        f.write(f"# {provenance}\n")
        f.write(f"{len(pts)} 2 0 1\n")
        for i, ((x, y), mi) in enumerate(zip(pts.tolist(), m.tolist())):
            f.write(f"{i} {x!r} {y!r} {mi}\n")


def write_graph(stem, ia, ja):
    """A graph file from the stencils in CSR form, the stencil of node i
    at ja[ia[i]:ia[i + 1]]: the node count and the edge count, then one
    stencil per line."""
    starts = ia.tolist()
    with open_out(stem, ".graph") as f:
        f.write(f"{len(starts) - 1} {starts[-1]}\n")
        for a, b in zip(starts[:-1], starts[1:]):
            f.write(" ".join(map(str, ja[a:b].tolist())) + "\n")


def write_ordering(stem, iperm):
    """An ordering file: the new index of every node, one per line."""
    with open_out(stem, ".iperm") as f:
        for i in iperm.tolist():
            f.write(f"{i}\n")
