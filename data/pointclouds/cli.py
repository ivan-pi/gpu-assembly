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

A generator of random clouds draws a series of them, one per realization,
so that they can be averaged over: `add_series_options` adds the options
of such a series and `series` reads them back, with the checks a pipe
needs, as a `Series` that names the files and carries the random streams.
"""

import argparse
import sys
from collections import namedtuple

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


def add_series_options(ap, what, plot):
    """--seed, --realizations, --no-graph and --plot on the parser, for a
    generator that draws a series of `what` (say "grids"), with `plot` the
    help of --plot."""
    ap.add_argument(
        "--seed",
        type=int,
        help=f"seed of the {what} (default: drawn and reported)",
    )
    ap.add_argument(
        "--realizations",
        type=number(int, least=1),
        default=1,
        metavar="R",
        help=f"independent {what} to write, numbered from 0 (default: 1)",
    )
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="write the coordinates only, without the stencil graph",
    )
    ap.add_argument("--plot", action="store_true", help=plot)


class Series(namedtuple("Series", "seed streams stems ext piped")):
    """What a run of a generator writes: the seed, one random stream per
    realization, the stem of each realization's files and the extension
    that picks their format, and whether they go to standard output.
    Realization i depends on the seed and on i alone, so asking for more
    of them extends the series rather than replacing it."""

    __slots__ = ()

    def provenance(self, options, i):
        """The comment of a node file, the command that reproduces
        realization i: `options` as the generator spells its own, then
        the seed and the place in the series."""
        tail = (
            f" --realizations {len(self.stems)}, number {i}"
            if len(self.stems) > 1
            else ""
        )
        return f"{options} --seed {self.seed}{tail}"

    def written(self, i):
        """Where realization i went, for the report."""
        return "standard output" if self.piped else self.stems[i]


def series(ap, args, default):
    """The Series a parsed command line asks for, `default` being the
    extension when the output names none. Only one file fits down a pipe,
    so standard output takes --no-graph and a single realization; the
    other cases are usage errors. A seed that was not given is drawn and
    reported on standard error, so that the run can be repeated."""
    base, ext = output_stem(args.output, default)
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
    seed = np.random.SeedSequence().entropy if args.seed is None else args.seed
    if args.seed is None:
        print(f"seed {seed}", file=sys.stderr)
    streams = np.random.SeedSequence(seed).spawn(args.realizations)
    width = len(str(args.realizations - 1))
    stems = (
        [base]
        if args.realizations == 1
        else [f"{base}_{i:0{width}d}" for i in range(args.realizations)]
    )
    return Series(seed, streams, stems, ext, piped)


def stencil_report(ia):
    """The graph for the report line: the entry count and the range of
    stencil sizes, from the row pointer."""
    sizes = np.diff(ia)
    return f", {ia[-1]} stencil entries, {sizes.min()} to {sizes.max()} per node"
