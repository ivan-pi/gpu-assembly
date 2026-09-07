"""The command-line conventions the scripts in data/tools share.

argparse has no bounds of its own, but a `type` is any callable and an
ArgumentTypeError raised in one comes back as argparse's own message,
naming the option and exiting 2 like any other usage error; `number` is
that callable with a bound attached.

The output argument of a generator is a stem to which the writers append
their extensions, so that one name covers the several files of a case. An
extension the generator knows picks the format and is not doubled. The
options that select the stencil graph, and the figure --plot shows, are
the same in every generator, so they are here too.
"""

import argparse

from .stencils import KNN


def number(kind, *, least=None, above=None):
    """Return an argparse type with a bound: `least` inclusive, `above` exclusive."""

    def parse(text):
        value = kind(text)  # a ValueError here: "invalid int"
        if least is not None and value < least:
            raise argparse.ArgumentTypeError(f"must be at least {least:g}")
        if above is not None and value <= above:
            raise argparse.ArgumentTypeError(f"must be greater than {above:g}")
        return value

    parse.__name__ = kind.__name__  # the name argparse reports
    return parse


def output_stem(name, default):
    """Return the stem of an output name and the extension that picks its format: the one it ends in, or `default` if none."""
    for ext in (".node", ".points"):
        if name.endswith(ext):
            return name.removesuffix(ext), ext
    return name, default


class Pair(argparse.Action):
    """An option stored with the method it stands for: -K 18 gives
    args.graph = ("knn", 18).
    """

    def __call__(self, ap, namespace, value, option):
        setattr(namespace, self.dest, (self.const, value))


def add_graph_options(ap):
    """How the stencil graph is selected, one of --knn-graph K,
    --radius-graph R and --range-graph S, as args.graph = (method, value)
    for NodeSet.stencils, or None for --no-graph; the KNN nearest nodes of
    pointclouds.stencils unless given."""
    group = ap.add_mutually_exclusive_group()
    pair = dict(dest="graph", action=Pair)
    group.add_argument(
        "-K",
        "--knn-graph",
        **pair,
        const="knn",
        type=number(int, least=2),
        metavar="K",
        help=f"the stencil of a node is its K nearest nodes (default: {KNN})",
    )
    group.add_argument(
        "-R",
        "--radius-graph",
        **pair,
        const="radius",
        type=number(float, above=0.0),
        metavar="R",
        help="the stencil of a node is the nodes within a distance R of it",
    )
    group.add_argument(
        "--range-graph",
        **pair,
        const="range",
        type=number(float, above=0.0),
        metavar="S",
        help="the stencil of a node is the nodes within S of it along both "
        "axes, a square of side 2 S",
    )
    group.add_argument(
        "--no-graph",
        dest="graph",
        action="store_const",
        const=None,
        help="write the coordinates only, without the stencil graph",
    )
    ap.set_defaults(graph=("knn", KNN))


def show(cloud, graph=None):
    """Show the figure of --plot: the cloud, with the stencil of its middle node when a graph is given."""
    import matplotlib.pyplot as plt

    stencils = []
    if graph is not None:
        ia, ja = graph
        i = len(cloud) // 2
        stencils = [(i, ja[ia[i] : ia[i + 1]])]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    cloud.plot(ax, stencils=stencils)
    ax.set_title(cloud.title, fontsize=9)
    fig.tight_layout()
    plt.show()
