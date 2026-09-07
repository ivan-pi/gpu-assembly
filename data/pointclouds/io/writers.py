"""Writers for the files of docs/file_formats.md.

Every number is written as the shortest text that reads back to the
same double, which is what ``repr`` gives, so a file reproduces the
values it was given. A writer takes a stem rather than a file name and
appends its own extension, so one stem names the whole case:
``case.points`` and ``case.graph`` are the pair the readers expect, in
the same node numbering. The stem ``-`` writes to standard output.
"""

import contextlib
import sys


def open_out(stem, ext):
    """The file this stem and extension name, or standard output for `-`,
    which stays open.
    """
    return contextlib.nullcontext(sys.stdout) if stem == "-" else open(stem + ext, "w")


def write_points(stem, pts):
    """Write a points file: the count, then a coordinate pair per line.

    Parameters
    ----------
    stem : str
    pts : (n, 2) array_like
    """
    with open_out(stem, ".points") as f:
        f.write(f"{len(pts)}\n")
        for x, y in pts.tolist():
            f.write(f"{x!r} {y!r}\n")  # repr: shortest round-trip text


def write_node(stem, pts, m, title):
    """Write a node file: a comment, the header, then a node per line.

    Parameters
    ----------
    stem : str
    pts : (n, 2) array_like
    m : (n,) array_like of int
        The marker of every node.
    title : str
        The comment on the first line.
    """
    with open_out(stem, ".node") as f:
        f.write(f"# {title}\n")
        f.write(f"{len(pts)} 2 0 1\n")
        for i, ((x, y), mi) in enumerate(zip(pts.tolist(), m.tolist())):
            f.write(f"{i} {x!r} {y!r} {mi}\n")


def write_graph(stem, ia, ja):
    """Write a graph file: the node and edge counts, then a stencil per line.

    Parameters
    ----------
    stem : str
    ia, ja : array_like of int
        The stencils in CSR form, that of node i at
        ``ja[ia[i]:ia[i + 1]]``.
    """
    starts = ia.tolist()
    with open_out(stem, ".graph") as f:
        f.write(f"{len(starts) - 1} {starts[-1]}\n")
        for a, b in zip(starts[:-1], starts[1:]):
            f.write(" ".join(map(str, ja[a:b].tolist())) + "\n")


def write_ordering(stem, iperm):
    """Write an ordering file: the new index of every node, one per line.

    Parameters
    ----------
    stem : str
    iperm : (n,) array_like of int
    """
    with open_out(stem, ".iperm") as f:
        for i in iperm.tolist():
            f.write(f"{i}\n")
