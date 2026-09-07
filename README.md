# gpu-assembly

Batched RBF-FD operator assembly on the GPU, with the node handling,
renumbering and file I/O it needs on the host.

The project is mixed-language. The host library is C++ headers over a few
Fortran modules -- the space-filling-curve keys (`rbf_ordering.F90`), the
working precision (`rbf_precision.f90`) and the periodic box
(`rbf_periodic_box.f90`) -- which meet through the ISO C binding: the
Fortran entry points are `bind(c)` and the C++ side declares them
`extern "C"`, so nothing depends on a compiler's name mangling. The GPU
parts come in both languages too, as CUDA C++ (`rbf_operators.h`,
`lbm/d2q9_kernels.cu`) and CUDA Fortran (`main.f90`, `rbf_cuda.f90`,
`lbm/d2q9_kernels.cuf`).

## Requirements

The host library and its tests:

- CMake 3.23 or newer
- A C++20 compiler (gcc, clang, nvc++). The headers use `std::span`
  throughout and constrain a few templates with `requires`, so C++17 is
  not enough.
- A Fortran compiler with the 2008 bit intrinsics; gfortran 13, ifx and
  nvfortran 23 and newer are known to work, and older nvfortran is
  covered by a fallback in `rbf_ordering.F90`.
- OpenMP, optional. The stencil search and the key kernels use it when
  the toolchain has it and run serially when it does not.

[nanoflann](https://github.com/jlblancoc/nanoflann) is vendored under
`third_party/`, so there is nothing to install for the k-d tree. The GPU
parts additionally need the NVHPC toolchain (`nvc++`, `nvfortran`) and
MathDx for cuSolverDx; the CUDA Fortran sources build with nvfortran only
and are skipped with a warning under any other Fortran compiler.

## Building

The host library (node set, renumbering, curve keys) and its tests build
with any C++20/Fortran toolchain:

```
cmake -B build
cmake --build build
ctest --test-dir build
```

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

Documentation:
- [docs/nodeset.md](docs/nodeset.md): the `NodeSet` class, its
  renumbering and the stencil search
- [docs/renumbering.md](docs/renumbering.md): permutations,
  space-filling-curve orderings and graph renumbering
- [docs/file_formats.md](docs/file_formats.md): the file formats the
  library reads and writes
- [docs/periodic_benchmarks.md](docs/periodic_benchmarks.md): the
  analytic periodic flow fields in `examples/`, for verifying a lattice
  Boltzmann implementation

References:
- [RBF-FD Stencil Sizer](https://ivan-pi.github.io/tools/rbf_fd_stencil_sizer.html)
- [cuSOLVERDx](https://docs.nvidia.com/cuda/cusolverdx/)
- [How to call “Cuda C” device routine from “Cuda Fortran Kernel/device routine”…?](https://forums.developer.nvidia.com/t/how-to-call-cuda-c-device-routine-from-cuda-fortran-kernel-device-routine/200755)
- [Calling Thrust from CUDA Fortran](https://cudamusing.blogspot.com/2011/06/calling-thrust-from-cuda-fortran.html)