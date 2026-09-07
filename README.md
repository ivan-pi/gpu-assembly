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
- [nanoflann](https://github.com/jlblancoc/nanoflann), vendored under
  `third_party/`

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