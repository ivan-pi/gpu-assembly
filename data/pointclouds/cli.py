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
"""

import argparse


def number(kind, least=None, above=None):
    """An argparse type that also carries a bound."""
    def parse(text):
        value = kind(text)                       # a ValueError here: "invalid int"
        if least is not None and value < least:
            raise argparse.ArgumentTypeError(f"must be at least {least:g}")
        if above is not None and value <= above:
            raise argparse.ArgumentTypeError(f"must be greater than {above:g}")
        return value
    parse.__name__ = kind.__name__               # the name argparse reports
    return parse


def output_stem(name, default, known=(".node", ".points")):
    """The stem of an output name and the extension that picks its format:
    the one it ends in, or `default` if it ends in none of them."""
    for ext in known:
        if name.endswith(ext):
            return name.removesuffix(ext), ext
    return name, default
