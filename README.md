
## Building

The host library (node set, renumbering, curve keys) and its tests build
with any C++20/Fortran toolchain:

```
cmake -B build
cmake --build build
ctest --test-dir build
```

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