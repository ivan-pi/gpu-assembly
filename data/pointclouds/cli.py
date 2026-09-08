"""The command-line conventions the scripts in data/tools share.

argparse has no bounds of its own, but a `type` is any callable and an
ArgumentTypeError raised in one comes back as argparse's own message,
naming the option and exiting 2 like any other usage error; `number` is
that callable with a bound attached.

A generator takes its output as a stem to which the writers append their
extensions, so that one name covers the several files of a case. An
extension the generator knows picks the format and is not doubled. The
output argument, the options that select the stencil graph, and what a
generator does with the cloud once it has it, are the same in every
generator, so they are here too.
"""

import argparse

from .stencils import KNN


def parser(doc, epilog):
    """Builds the parser of a script from its docstring and an epilog.

    The description is the first line of `doc`; `epilog` is printed as
    it is, after the options.
    """
    return argparse.ArgumentParser(
        description=doc.split("\n")[0],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def number(kind, *, least=None, above=None, below=None):
    """Builds an argparse type that rejects values beyond a bound.

    Parameters
    ----------
    kind : type
        `int` or `float`.
    least : number, optional
        The value must be at least this.
    above : number, optional
        The value must be greater than this.
    below : number, optional
        The value must be less than this.

    Returns
    -------
    callable
        Parses a string to `kind` and raises ``argparse.ArgumentTypeError``
        outside the bound, which argparse reports naming the option.
    """

    def parse(text):
        value = kind(text)  # a ValueError here: "invalid int"
        if least is not None and value < least:
            raise argparse.ArgumentTypeError(f"must be at least {least:g}")
        if above is not None and value <= above:
            raise argparse.ArgumentTypeError(f"must be greater than {above:g}")
        if below is not None and value >= below:
            raise argparse.ArgumentTypeError(f"must be less than {below:g}")
        return value

    parse.__name__ = kind.__name__  # the name argparse reports
    return parse


def add_output_options(ap, default):
    """Adds the output argument and ``--plot`` of a generator to a parser.

    Parameters
    ----------
    ap : argparse.ArgumentParser
    default : {".points", ".node"}
        The format of the output, unless its name ends in the other
        extension.
    """
    other = ".node" if default == ".points" else ".points"
    ap.add_argument(
        "output", help=f"the {default[1:]} file, or a {other[1:]} file if named so"
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="show the cloud, and a stencil if there is a graph",
    )
    ap.set_defaults(default_ext=default)


class Pair(argparse.Action):
    """Stores an option's value paired with the method it stands for.

    ``-K 18`` gives ``args.graph = ("knn", 18)``.
    """

    def __call__(self, ap, namespace, value, option):
        setattr(namespace, self.dest, (self.const, value))


def add_graph_options(ap):
    """Adds the options that select the stencil graph to a parser.

    One of ``--knn-graph K`` (``-K``), ``--radius-graph R`` (``-R``),
    ``--range-graph S`` and ``--no-graph``, mutually exclusive, parsed to
    ``args.graph = (method, value)`` for `NodeSet.stencils`, or None for
    ``--no-graph``; the `KNN` nearest nodes unless given.

    Parameters
    ----------
    ap : argparse.ArgumentParser
    """
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


def output_stem(name, default):
    """Splits an output name into its stem and the extension of its format.

    Parameters
    ----------
    name : str
        The output argument of a generator.
    default : {".points", ".node"}
        The extension when `name` ends in neither.

    Returns
    -------
    stem : str
        `name` without the extension.
    ext : str
        The extension `name` ends in, or `default`.
    """
    for ext in (".node", ".points"):
        if name.endswith(ext):
            return name.removesuffix(ext), ext
    return name, default


def finish(cloud, args):
    """Writes the cloud and its graph, reports one line and shows the plot.

    Parameters
    ----------
    cloud : NodeSet
    args : argparse.Namespace
        Parsed by a parser with `add_output_options` and, for the graph,
        `add_graph_options`.
    """
    stem, ext = output_stem(args.output, args.default_ext)
    selection = getattr(args, "graph", None)  # None without add_graph_options
    graph = cloud.stencils(*selection) if selection else None
    cloud.write(stem, ext, graph)
    print(f"{stem}{ext}: {cloud.summary(graph)}")
    if args.plot:
        show(cloud, graph)


def show(cloud, graph=None):
    """Shows the figure of ``--plot``.

    Parameters
    ----------
    cloud : NodeSet
    graph : Graph, optional
        To draw the stencil of the middle node too.
    """
    import matplotlib.pyplot as plt

    stencils = []
    if graph is not None:
        i = len(cloud) // 2
        stencils = [(i, graph.stencil(i))]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    cloud.plot(ax, stencils=stencils)
    ax.set_title(cloud.title, fontsize=9)
    fig.tight_layout()
    plt.show()
