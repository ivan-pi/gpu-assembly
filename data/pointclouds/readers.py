"""Readers for the points, node, graph and ordering files of
docs/file_formats.md, which check what they read.

A reader takes a file name and a Report and returns the contents as numpy
arrays, or None when the file cannot be used, with every problem found
recorded in the report. A reader never prints and never exits: the tool
decides what to say, where, and whether to go on, since a problem that
leaves the contents usable, such as a header count that disagrees with
the lines, is recorded and the contents returned anyway.

Indices come back 0-based, as the formats have them; a node file that
counts from 1 is accepted and reported from 0.
"""

import itertools

import numpy as np


# ----------------------------------------------------------------- report

class Report:
    """What the checks found: problems, listed at the end and making the
    exit status 1, and notes, printed under the description of their file.
    A file with nothing wrong is passed over in silence. Everything goes
    to `stream`, standard output unless told otherwise."""

    def __init__(self, stream=None):
        self.problems = []
        self.notes = []
        self.stream = stream

    def problem(self, fname, msg):
        self.problems.append(f"{fname}: {msg}")

    def note(self, msg):
        self.notes.append(msg)

    def print_notes(self):
        for msg in self.notes:
            print(f"  note: {msg}", file=self.stream)
        self.notes = []

    def print_problems(self):
        """The list of problems, and nothing at all when there are none:
        the exit status already says a file checked out."""
        if not self.problems:
            return
        print(f"{plural(len(self.problems), 'problem')}:", file=self.stream)
        for p in self.problems:
            print(f"  {p}", file=self.stream)


def plural(k, word):
    """The count and the word, with an s unless the count is 1."""
    return f"{k} {word}{'' if k == 1 else 's'}"


# ----------------------------------------------------------- text parsing

def lines_of(fname, comments):
    """(line number, tokens) of every non-blank line; `#` starts a comment
    in the formats that have them."""
    out = []
    with open(fname) as f:
        for no, line in enumerate(f, 1):
            if comments:
                line = line.split("#", 1)[0]
            toks = line.split()
            if toks:
                out.append((no, toks))
    return out


def nums(toks, conv):
    """All tokens through int or float, or None if one is not a number."""
    try:
        return [conv(t) for t in toks]
    except ValueError:
        return None


def read_header(fname, rep, form, comments):
    """The non-negative integers of the first line, and the lines after it:
    (values, rows), or None with the problem recorded. `form` names the
    fields, `n nnz` say, and sets how many there are."""
    lines = lines_of(fname, comments)
    if not lines:
        rep.problem(fname, "empty file")
        return None
    no, head = lines[0]
    vals = nums(head, int)
    if vals is None or len(vals) != len(form.split()) or min(vals) < 0:
        rep.problem(fname, f"line {no}: expected the header '{form}', got '{' '.join(head)}'")
        return None
    return vals, lines[1:]


def parse_rows(fname, rep, rows, form, convert):
    """Every row through `convert`, which returns the values of a line or
    None: the list of values, or None with every bad line recorded.
    `form` describes a line in the message."""
    out = []
    ok = True
    for no, toks in rows:
        vals = convert(toks)
        if vals is None:
            rep.problem(fname, f"line {no}: expected '{form}', got '{' '.join(toks)}'")
            ok = False
        else:
            out.append(vals)
    return out if ok else None


# ------------------------------------------------------------------ nodes

def read_nodes(fname, rep):
    """The nodes of a .points or .node file, told apart by extension:
    (xy, marker) with a marker per node, 0 for interior, or None with the
    problems recorded."""
    reader = read_node if fname.endswith(".node") else read_points
    got = reader(fname, rep)
    if got is None:
        return None
    xy, marker = got
    bad = np.flatnonzero(~np.isfinite(xy).all(axis=1))
    if bad.size:
        rep.problem(fname, f"{plural(bad.size, 'node')} with a non-finite coordinate, "
                    f"the first is node {bad[0]}")
        return None
    return xy, marker


def read_points(fname, rep):
    """`n`, then n lines of `x y`: (xy, zero markers), or None."""
    got = read_header(fname, rep, "n", comments=False)
    if got is None:
        return None
    (n,), rows = got
    if len(rows) < n:
        rep.problem(fname, f"header says {n} points, the file has {len(rows)} lines after it")
        return None
    if len(rows) > n:
        rep.note(f"{len(rows) - n} lines after the {n} points, ignored as the format allows")
    vals = parse_rows(fname, rep, rows[:n], "x y",
                      lambda toks: nums(toks, float) if len(toks) == 2 else None)
    if vals is None:
        return None
    return np.array(vals, dtype=float).reshape(n, 2), np.zeros(n, dtype=int)


def read_node(fname, rep):
    """Triangle's format, `n 2 nattr nmark` then `i x y a... [marker]`
    per node: (xy, markers), or None."""
    got = read_header(fname, rep, "n dim nattr nmark", comments=True)
    if got is None:
        return None
    (n, dim, nattr, nmark), rows = got
    if dim != 2:
        rep.problem(fname, f"dimension {dim} in the header, only 2 is supported")
        return None
    if nmark not in (0, 1):
        rep.problem(fname, f"header: nmark must be 0 or 1, got {nmark}")
        return None
    if len(rows) != n:
        rep.problem(fname, f"header says {n} nodes, the file has {len(rows)} node lines")
        return None
    if nattr:
        rep.note(f"{plural(nattr, 'attribute')} per node, not inspected")
    if not nmark:
        rep.note("no marker column: every node is interior")
    form = f"i x y{' a' * nattr}{' marker' if nmark else ''}"
    vals = parse_rows(fname, rep, rows, form, lambda toks: _node_line(toks, nattr, nmark))
    if vals is None:
        return None
    table = np.array(vals, dtype=float).reshape(n, 4)
    if not _consecutive(fname, rep, table[:, 0].astype(int), rows):
        return None
    return table[:, 1:3], table[:, 3].astype(int)


def _node_line(toks, nattr, nmark):
    """(index, x, y, marker) of a node line, or None if malformed."""
    if len(toks) != 3 + nattr + nmark:
        return None
    try:
        return int(toks[0]), float(toks[1]), float(toks[2]), int(toks[-1]) if nmark else 0
    except ValueError:
        return None


def _consecutive(fname, rep, index, rows):
    """Whether the index column counts from 0 or from 1."""
    expected = np.arange(index.size)
    if np.array_equal(index, expected + 1):
        rep.note("nodes are numbered from 1 in the file; reported from 0 here")
    elif not np.array_equal(index, expected):
        bad = np.flatnonzero(index != expected)[0]
        rep.problem(fname, f"node indices are not consecutive from 0 or 1: "
                    f"line {rows[bad][0]} has index {index[bad]}")
        return False
    return True


# ------------------------------------------------------------------ graph

def stencils_to_csr(stencils):
    """The row pointer and the column indices of a list of stencils:
    (ia, ja), with the stencil of node i at ja[ia[i]:ia[i + 1]]."""
    ia = np.cumsum([0, *map(len, stencils)])
    ja = np.fromiter(itertools.chain.from_iterable(stencils), dtype=int, count=int(ia[-1]))
    return ia, ja


def read_graph(fname, rep):
    """`n nnz`, then a line of 0-based indices per node: the stencils in
    CSR form as (ia, ja), or None if the file cannot be used. An entry
    count that disagrees with the header, or a node listed twice in a
    row, is recorded and the graph returned all the same."""
    got = read_header(fname, rep, "n nnz", comments=False)
    if got is None:
        return None
    (n, nnz), rows = got
    if len(rows) != n:
        rep.problem(fname, f"header says {n} nodes, the file has {len(rows)} stencil lines")
        return None
    stencils = parse_rows(fname, rep, rows, "integer stencil entries", lambda toks: nums(toks, int))
    if stencils is None:
        return None
    ia, ja = stencils_to_csr(stencils)
    if ia[-1] != nnz:
        rep.problem(fname, f"header says nnz = {nnz}, the stencils hold {ia[-1]} entries")
    lo, hi = ja.min(), ja.max()
    if lo < 0 or hi >= n:
        if lo >= 1 and hi == n:
            rep.problem(fname, f"indices run from {lo} to {n}: the file looks 1-based, the format is 0-based")
        else:
            rep.problem(fname, f"stencil entries outside [0, {n}): from {lo} to {hi}")
        return None
    dup = _rows_with_duplicates(ia, ja)
    if dup.size:
        rep.problem(fname, f"{plural(dup.size, 'stencil')} {'lists' if dup.size == 1 else 'list'} "
                    f"a node twice, the first on line {rows[dup[0]][0]}")
    return ia, ja


def _rows_with_duplicates(ia, ja):
    """The rows in which some node is listed twice."""
    n = len(ia) - 1
    keys = np.sort(np.repeat(np.arange(n), np.diff(ia)) * n + ja)   # i * n + j of every entry
    return np.unique(keys[1:][keys[1:] == keys[:-1]] // n)


# --------------------------------------------------------------- ordering

def read_ordering(fname, rep):
    """One integer per line, no header: iperm, the new index of every node,
    or None unless the values are a permutation."""
    vals = parse_rows(fname, rep, lines_of(fname, comments=False), "one integer",
                      lambda toks: nums(toks, int) if len(toks) == 1 else None)
    if vals is None:
        return None
    iperm = np.array(vals, dtype=int).reshape(-1)
    n = iperm.size
    if np.array_equal(np.sort(iperm), np.arange(n)):
        return iperm
    if n and iperm.min() == 1 and iperm.max() == n:
        rep.problem(fname, f"values run from 1 to {n}: the file looks 1-based, the format is 0-based")
    else:
        rep.problem(fname, f"the {n} values are not a permutation of 0 .. {n - 1}")
    return None
