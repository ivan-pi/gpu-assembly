# gpu-assembly

Batched RBF-FD operator assembly on the GPU, with the node handling,
renumbering and file I/O it needs on the host.

The project is mixed-language, C++ and Fortran.

## Requirements

The host library and its tests:

- CMake 3.23 or newer
- A C++20 compiler (gcc, clang, nvc++)
- A Fortran 2008 compiler (gfortran, ifx, nvfortran)
- (optional) OpenMP, for host parallelism and, in a future version,
  target offload
- [nanoflann](https://github.com/jlblancoc/nanoflann), a header-only k-d
  tree, vendored under `third_party/`
- [ckdtree](https://github.com/scipy/scipy/tree/main/scipy/spatial/ckdtree),
  SciPy's k-d tree, vendored under `third_party/` and built from source

The GPU parts additionally:

- The NVHPC toolchain, `nvc++` and `nvfortran`
- [MathDx](https://docs.nvidia.com/cuda/cusolverdx/), for cuSolverDx

## Building

The host library (node set, renumbering, curve keys) and its tests build
with any C++20/Fortran toolchain:

```
cmake -B build
cmake --build build
ctest --test-dir build
```

The flow benchmark examples (`examples/`, the Fortran module
`rbf_benchmarks` and the header `rbf_flow_benchmarks.h`) build by
default as the `rbf_benchmarks` target; pass
`-DGPU_ASSEMBLY_BUILD_EXAMPLES=OFF` to leave them out.

CI exercises GCC (`g++`/`gfortran`) and the LLVM toolchain
(`clang++`/`flang`, versions 20 and 22). To build with LLVM flang:

```
CXX=clang++-20 FC=flang-20 cmake -B build
```

On Ubuntu 24.04 the packages are `clang-20 flang-20 libomp-20-dev`;
`flang-22` ships with Ubuntu 26.04.

The CUDA parts (cuSolverDx assembly kernels, LBM D2Q9 kernels,
CUDA Fortran demo) are opt-in and need the NVHPC toolchain plus MathDx:

```
cmake -B build -DGPU_ASSEMBLY_ENABLE_CUDA=ON \
      -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_Fortran_COMPILER=nvfortran
```

Export the MathDx path:

```
export MATHDX_ROOT=$HOME/nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06/
```

## Formatting and linting

The C++ and CUDA are formatted with [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
in the style of `.clang-format` (Google's, with a 4-space indent and
100 columns) and linted with [clang-tidy](https://clang.llvm.org/extra/clang-tidy/)
per `.clang-tidy`; the Python under `data/` with
[Black](https://black.readthedocs.io/) and [Ruff](https://docs.astral.sh/ruff/),
with the settings in `data/pyproject.toml`. All of it runs through
[pre-commit](https://pre-commit.com/), whose `.pre-commit-config.yaml`
pins the tool versions that CI checks with (`.github/workflows/style.yml`):

```
pip install pre-commit
pre-commit install                 # format on every commit from now on
pre-commit run --all-files         # or by hand, over the whole tree
```

clang-tidy wants the compile database of a configured build tree and
is a manual stage:

```
cmake -B build
pre-commit run --hook-stage manual clang-tidy --all-files
```

A block the formatter must leave alone, such as the tables in
`src/cuda_arch.h`, sits between `// clang-format off` and
`// clang-format on`.

Claude Code runs the same hooks after each of its edits
(`.claude/hooks/format.sh`, a `PostToolUse` hook in `.claude/settings.json`);
its session-start hook installs the tools in a fresh remote session. The
reformatting commit is listed in `.git-blame-ignore-revs`, which
`git config blame.ignoreRevsFile .git-blame-ignore-revs` makes blame
skip.

Documentation:
- [docs/nodeset.md](docs/nodeset.md): the `NodeSet` class, its
  renumbering and the stencil search
- [docs/spatial.md](docs/spatial.md): `rbf::spatial`, the periodic box
  and the k-d tree the stencil search runs on
- [docs/renumbering.md](docs/renumbering.md): permutations,
  space-filling-curve orderings and graph renumbering
- [docs/file_formats.md](docs/file_formats.md): the file formats the
  library reads and writes
- [docs/periodic_benchmarks.md](docs/periodic_benchmarks.md): the
  analytic periodic flow fields in `examples/`, for verifying a lattice
  Boltzmann implementation

Test cases, small point clouds with their stencil graphs, live in
[data/](data/README.md).

References:
- [RBF-FD Stencil Sizer](https://ivan-pi.github.io/tools/rbf_fd_stencil_sizer.html)
- [cuSOLVERDx](https://docs.nvidia.com/cuda/cusolverdx/)
- [How to call “Cuda C” device routine from “Cuda Fortran Kernel/device routine”…?](https://forums.developer.nvidia.com/t/how-to-call-cuda-c-device-routine-from-cuda-fortran-kernel-device-routine/200755)
- [Calling Thrust from CUDA Fortran](https://cudamusing.blogspot.com/2011/06/calling-thrust-from-cuda-fortran.html)
