ckdtree (SciPy 1.18.1)
https://github.com/scipy/scipy/releases/tag/v1.18.1  (commit e4e854e
, retrieved 2026-09-07)

Modified: `src/ckdtree_decl.h` drops the numpy include, so `src/` builds
with neither numpy nor Python. Built as the `ckdtree` CMake target and
wrapped by `rbf::spatial::KdTree`.
