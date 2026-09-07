"""Readers for the points, node, graph and ordering files of
docs/file_formats.md, which check what they read.

A reader takes a file name and returns the contents as numpy arrays, or
raises FormatError when the file does not follow its format, with every
problem found, so that a tool can list them all. What is worth knowing
but is no problem -- a node file that counts from 1, lines after the end
that the format allows -- goes to the logger of this module at level
INFO, for a tool to show or ignore.

Indices come back 0-based, as the formats have them; a node file that
counts from 1 is accepted and reported from 0.
"""

import itertools
import logging

import numpy as np

log = logging.getLogger(__name__)


class FormatError(ValueError):
    """A file that does not follow its format: `problems` lists what is wrong, each entry naming the file, and ``str()`` joins them by line."""

    def __init__(self, fname, *messages):
        self.problems = [f"{fname}: {m}" for m in messages]
        super().__init__("\n".join(self.problems))


def plural(k, word):
    """Return the count and the word, with an s unless the count is 1."""
    return f"{k} {word}{'' if k == 1 else 's'}"


# ----------------------------------------------------------- text parsing


def _lines_of(fname, comments):
    """(line number, tokens) of every non-blank line; `#` starts a comment
    in the formats that have them.
    """
    out = []
    with open(fname) as f:
        for no, line in enumerate(f, 1):
            if comments:
                line = line.split("#", 1)[0]
            toks = line.split()
            if toks:
                out.append((no, toks))
    return out


def _nums(toks, conv):
    """All tokens through int or float, or None if one is not a number."""
    try:
        return [conv(t) for t in toks]
    except ValueError:
        return None


def _read_header(fname, form, comments):
    """The non-negative integers of the first line, and the lines after it:
    (values, rows). `form` names the fields, `n nnz` say, and sets how
    many there are.
    """
    lines = _lines_of(fname, comments)
    if not lines:
        raise FormatError(fname, "empty file")
    no, head = lines[0]
    vals = _nums(head, int)
    if vals is None or len(vals) != len(form.split()) or min(vals) < 0:
        raise FormatError(
            fname, f"line {no}: expected the header '{form}', got '{' '.join(head)}'"
        )
    return vals, lines[1:]


def _parse_rows(fname, rows, form, convert):
    """Every row through `convert`, which returns the values of a line or
    None: the list of values, or FormatError naming every bad line.
    `form` describes a line in the message.
    """
    out, bad = [], []
    for no, toks in rows:
        vals = convert(toks)
        if vals is None:
            bad.append(f"line {no}: expected '{form}', got '{' '.join(toks)}'")
        else:
            out.append(vals)
    if bad:
        raise FormatError(fname, *bad)
    return out


# ------------------------------------------------------------------ nodes


def read_nodes(fname):
    """Read a points or node file, told apart by extension: ``(xy, markers)`` with a marker per node, 0 for interior."""
    reader = read_node if fname.endswith(".node") else read_points
    xy, marker = reader(fname)
    bad = np.flatnonzero(~np.isfinite(xy).all(axis=1))
    if bad.size:
        raise FormatError(
            fname,
            f"{plural(bad.size, 'node')} with a non-finite coordinate, "
            f"the first is node {bad[0]}",
        )
    return xy, marker


def read_points(fname):
    """Read a points file, ``n`` then n lines of ``x y``: ``(xy, markers)`` with every marker 0."""
    (n,), rows = _read_header(fname, "n", comments=False)
    if len(rows) < n:
        raise FormatError(
            fname, f"header says {n} points, the file has {len(rows)} lines after it"
        )
    if len(rows) > n:
        log.info(
            "%s: %d lines after the %d points, ignored as the format allows",
            fname,
            len(rows) - n,
            n,
        )
    vals = _parse_rows(
        fname,
        rows[:n],
        "x y",
        lambda toks: _nums(toks, float) if len(toks) == 2 else None,
    )
    return np.array(vals, dtype=float).reshape(n, 2), np.zeros(n, dtype=int)


def read_node(fname):
    """Read a node file in Triangle's format, ``n 2 nattr nmark`` then ``i x y a... [marker]`` per node: ``(xy, markers)``."""
    (n, dim, nattr, nmark), rows = _read_header(
        fname, "n dim nattr nmark", comments=True
    )
    if dim != 2:
        raise FormatError(fname, f"dimension {dim} in the header, only 2 is supported")
    if nmark not in (0, 1):
        raise FormatError(fname, f"header: nmark must be 0 or 1, got {nmark}")
    if len(rows) != n:
        raise FormatError(
            fname, f"header says {n} nodes, the file has {len(rows)} node lines"
        )
    if nattr:
        log.info("%s: %s per node, not inspected", fname, plural(nattr, "attribute"))
    if not nmark:
        log.info("%s: no marker column: every node is interior", fname)
    form = f"i x y{' a' * nattr}{' marker' if nmark else ''}"
    vals = _parse_rows(fname, rows, form, lambda toks: _node_line(toks, nattr, nmark))
    table = np.array(vals, dtype=float).reshape(n, 4)
    _check_consecutive(fname, table[:, 0].astype(int), rows)
    return table[:, 1:3], table[:, 3].astype(int)


def _node_line(toks, nattr, nmark):
    """(index, x, y, marker) of a node line, or None if malformed."""
    if len(toks) != 3 + nattr + nmark:
        return None
    try:
        return (
            int(toks[0]),
            float(toks[1]),
            float(toks[2]),
            int(toks[-1]) if nmark else 0,
        )
    except ValueError:
        return None


def _check_consecutive(fname, index, rows):
    """The index column has to count from 0 or from 1."""
    expected = np.arange(index.size)
    if np.array_equal(index, expected + 1):
        log.info(
            "%s: nodes are numbered from 1 in the file; reported from 0 here", fname
        )
    elif not np.array_equal(index, expected):
        bad = np.flatnonzero(index != expected)[0]
        raise FormatError(
            fname,
            f"node indices are not consecutive from 0 or 1: "
            f"line {rows[bad][0]} has index {index[bad]}",
        )


# ------------------------------------------------------------------ graph


def stencils_to_csr(stencils):
    """Return the row pointer and the column indices of a list of stencils, ``(ia, ja)``."""
    ia = np.cumsum([0, *map(len, stencils)])
    ja = np.fromiter(
        itertools.chain.from_iterable(stencils), dtype=int, count=int(ia[-1])
    )
    return ia, ja


def read_graph(fname):
    """Read a graph file, ``n nnz`` then a line of 0-based indices per node: the stencils in CSR form as ``(ia, ja)``."""
    (n, nnz), rows = _read_header(fname, "n nnz", comments=False)
    if len(rows) != n:
        raise FormatError(
            fname, f"header says {n} nodes, the file has {len(rows)} stencil lines"
        )
    stencils = _parse_rows(
        fname, rows, "integer stencil entries", lambda toks: _nums(toks, int)
    )
    ia, ja = stencils_to_csr(stencils)
    problems = []
    if ia[-1] != nnz:
        problems.append(f"header says nnz = {nnz}, the stencils hold {ia[-1]} entries")
    lo, hi = ja.min(), ja.max()
    if lo >= 1 and hi == n:
        problems.append(
            f"indices run from {lo} to {n}: the file looks 1-based, the format is 0-based"
        )
    elif lo < 0 or hi >= n:
        problems.append(f"stencil entries outside [0, {n}): from {lo} to {hi}")
    else:
        dup = _rows_with_duplicates(ia, ja)
        if dup.size:
            problems.append(
                f"a node listed twice in {plural(dup.size, 'stencil')}, "
                f"the first on line {rows[dup[0]][0]}"
            )
    if problems:
        raise FormatError(fname, *problems)
    return ia, ja


def _rows_with_duplicates(ia, ja):
    """The rows in which some node is listed twice."""
    n = len(ia) - 1
    # i * n + j of every entry
    keys = np.sort(np.repeat(np.arange(n), np.diff(ia)) * n + ja)
    return np.unique(keys[1:][keys[1:] == keys[:-1]] // n)


# --------------------------------------------------------------- ordering


def read_ordering(fname):
    """Read an ordering file, one integer per line and no header: `iperm`, the new index of every node, checked to be a permutation."""
    vals = _parse_rows(
        fname,
        _lines_of(fname, comments=False),
        "one integer",
        lambda toks: _nums(toks, int) if len(toks) == 1 else None,
    )
    iperm = np.array(vals, dtype=int).reshape(-1)
    n = iperm.size
    if np.array_equal(np.sort(iperm), np.arange(n)):
        return iperm
    if n and iperm.min() == 1 and iperm.max() == n:
        raise FormatError(
            fname,
            f"values run from 1 to {n}: the file looks 1-based, the format is 0-based",
        )
    raise FormatError(fname, f"the {n} values are not a permutation of 0 .. {n - 1}")
