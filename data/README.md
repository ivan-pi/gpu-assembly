# Test cases

Small point clouds and their stencil graphs, in the formats of
[docs/file_formats.md](../docs/file_formats.md), for the functionality
tests. They are read by the tests and the examples, so they live here
rather than under `test/`.

Keep the cases small: they exist to exercise the code, not to measure it.
Anything larger than a few hundred kilobytes should be generated when
needed instead of committed.

## Naming

A case is a pair `<name>.points` and `<name>.graph` in the same numbering,
with

```
<distribution>_<resolution>_<k>
```

- `distribution`: how the points were placed, such as `cartesian` or
  `poisson` (Poisson disk sampling)
- `resolution`: the resolution parameter of the generator, the number of
  points per side for a Cartesian grid. It is not the point count: the
  first line of the points file holds that.
- `k`: the stencil size, the number of entries in every row of the graph

Further variants get a suffix rather than a change of the fields above,
so existing names stay valid: `poisson_64_21_periodic`,
`poisson_64_21_3d`.

Both files of a case are committed, the graph too, although it derives
from the points. The tests must not depend on Python or on the neighbour
ordering of a particular library version; the generator scripts document
how a case was made and let it be regenerated for comparison.

## Cases

| Case | Points | k | Generator | Seed |
|---|---|---|---|---|

## Generating

Generator scripts go in `gen/`, one per distribution, writing a points
file, plus a script that computes the k-nearest-neighbour graph of a
points file. Random generators take the seed on the command line, and
the table above records it, so that a case can be reproduced.
