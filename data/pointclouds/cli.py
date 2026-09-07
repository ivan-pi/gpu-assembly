"""The command-line conventions the generators in data/gen share.

argparse has no bounds of its own, but a `type` is any callable and an
ArgumentTypeError raised in one comes back as argparse's own message,
naming the option and exiting 2 like any other usage error; `number` is
that callable with a bound attached. Anything that takes two arguments to
decide belongs in the script, through `parser.error`.

The output argument of a generator is a stem to which the writers append
their extensions, so that one name covers the several files of a case. An
extension the generator knows picks the format and is not doubled; the
stem `-` writes to standard output, which is why a generator's report
belongs on standard error.

A generator of random clouds takes a seed, and draws and reports one
when none is given, so that any run can be repeated; `add_run_options`
adds it with the options every such generator has, `draw_seed` reads it
back, and `output` the name of the files with the check a pipe needs.
"""

import argparse
import sys

import numpy as np


def number(kind, least=None, above=None):
    """An argparse type that also carries a bound."""

    def parse(text):
        value = kind(text)  # a ValueError here: "invalid int"
        if least is not None and value < least:
            raise argparse.ArgumentTypeError(f"must be at least {least:g}")
        if above is not None and value <= above:
            raise argparse.ArgumentTypeError(f"must be greater than {above:g}")
        return value

    parse.__name__ = kind.__name__  # the name argparse reports
    return parse


def output_stem(name, default, known=(".node", ".points")):
    """The stem of an output name and the extension that picks its format:
    the one it ends in, or `default` if it ends in none of them."""
    for ext in known:
        if name.endswith(ext):
            return name.removesuffix(ext), ext
    return name, default


def add_run_options(ap, what, plot):
    """--seed, --no-graph and --plot on the parser, for a generator that
    draws a random `what` (say "grid"), with `plot` the help of --plot."""
    ap.add_argument(
        "--seed",
        type=int,
        help=f"seed of the {what} (default: drawn and reported)",
    )
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="write the coordinates only, without the stencil graph",
    )
    ap.add_argument("--plot", action="store_true", help=plot)


def draw_seed(args):
    """The seed of the run: the one given, or one drawn and reported on
    standard error, so that the run can be repeated."""
    if args.seed is not None:
        return args.seed
    drawn = np.random.SeedSequence().entropy
    print(f"seed {drawn}", file=sys.stderr)
    return drawn


def output(ap, args, default):
    """The stem and the extension the output names, `default` being the
    extension when it names none, after the check a pipe needs: only one
    file fits down it, so standard output takes --no-graph."""
    stem, ext = output_stem(args.output, default)
    if stem == "-" and not args.no_graph:
        ap.error(
            "only one file fits down a pipe: add --no-graph to write the "
            "coordinates to standard output, or name a file for the pair"
        )
    return stem, ext


def stencil_report(ia):
    """The graph for the report line: the entry count and the range of
    stencil sizes, from the row pointer."""
    sizes = np.diff(ia)
    return f", {ia[-1]} stencil entries, {sizes.min()} to {sizes.max()} per node"
