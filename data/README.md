# Test cases

Small point clouds and their stencil graphs, in the formats of
[docs/file_formats.md](../docs/file_formats.md), for the functionality
tests. They are read by the tests and the examples, so they live here
rather than under `test/`.

Keep the cases small: they exist to exercise the code, not to measure it.
Anything larger than a few hundred kilobytes should be generated when
needed instead of committed.

A case is a pair `<name>.points` and `<name>.graph` in the same
numbering. The name is free; what it stands for is recorded in the table
below.

## Cases

| Case | Points | k | Origin |
|---|---|---|---|

## Generating

Scripts that produced a case go in `gen/`.

- `gen/cavity_refined.py`: the lid-driven cavity `[0, Lx] x [0, Ly]`,
  refined towards the walls in three levels, as a `.node` file with
  boundary markers or a `.points` file without. Needs numpy;
  `--plot` needs matplotlib.

## Boundary markers

Node files written here mark the walls of a rectangle counter-clockwise
from the bottom, and give the corners a marker of their own, since a
corner belongs to two walls whose boundary data may differ:

| Marker | Nodes |
|---|---|
| 0 | interior |
| 1 | south wall, `y = 0` |
| 2 | east wall, `x = Lx` |
| 3 | north wall, `y = Ly` |
| 4 | west wall, `x = 0` |
| 5 | corners |
