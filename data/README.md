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
  markers for the four walls and the corners. Needs numpy; `--plot`
  needs matplotlib.
- `gen/perturbed_grid.py`: the unit square with the nodes of a Cartesian
  grid of spacing `h = 1/N` displaced by up to `sigma h`, periodic on
  both sides or a channel with walls at `y = 0` and `y = 1`, as a
  `.points` file or a `.node` file with wall markers, together with the
  `.graph` of the stencils, whose search wraps around the periodic
  sides. `--realizations` writes a numbered series to average over and
  `--seed` makes it repeatable. Needs numpy and scipy; `--plot` needs
  matplotlib.
