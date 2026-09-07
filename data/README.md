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
  markers for the four walls and the corners. In lattice units the
  spacing is 1 at the wall, 1.5 and 2.5 further in, and `--steps N`
  gives a cavity of `10 N`; `--size` scales it to other units. Needs
  numpy; `--plot` needs matplotlib.
- `gen/perturbed_grid.py`: an `N` by `N` box in lattice units, spacing
  `h = 1`, with every node of the Cartesian grid displaced by up to
  `sigma` spacings; periodic on both sides, or a channel with walls at
  `y = 0` and `y = N`. Writes a `.points` file or a `.node` file with
  wall markers, together with the `.graph` of the stencils, whose search
  wraps around the periodic sides. `--size` rescales the box to other
  units, `--realizations` writes a numbered series to average over and
  `--seed` makes it repeatable. Needs numpy and scipy; `--plot` needs
  matplotlib.
