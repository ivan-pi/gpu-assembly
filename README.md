
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

## Lattice Boltzmann layer

`src/lbm/` holds a header-only LBM layer on top of the host library:
lattice, collision models, streaming operators assembled by RBF-FD
(semi-Lagrangian or Lax-Wendroff, arrival- or departure-centred
stencils), native and Eigen backends, and fused or split time steppers.
Its design, and how to extend it along each axis, is described in
[docs/lbm_architecture.md](docs/lbm_architecture.md). Eigen is optional
and picked up automatically when `find_package(Eigen3)` succeeds.

The periodic flow benchmark driver runs a Taylor-Green vortex (or a
shear layer) on a jittered point cloud and reports the decay error,
MLUPS and bandwidth:

```
./build/lbm_bench --nodes 128 --steps 1000 --io 100
./build/lbm_bench --scheme lw --collision trt --stepper split --backend eigen
./build/lbm_bench --stencils departure --cfl 1.5
./build/lbm_bench --help
```

The file formats the library reads and writes are described in
[docs/file_formats.md](docs/file_formats.md).

References:
- [RBF-FD Stencil Sizer](https://ivan-pi.github.io/tools/rbf_fd_stencil_sizer.html)
- [cuSOLVERDx](https://docs.nvidia.com/cuda/cusolverdx/)
- [How to call “Cuda C” device routine from “Cuda Fortran Kernel/device routine”…?](https://forums.developer.nvidia.com/t/how-to-call-cuda-c-device-routine-from-cuda-fortran-kernel-device-routine/200755)
- [Calling Thrust from CUDA Fortran](https://cudamusing.blogspot.com/2011/06/calling-thrust-from-cuda-fortran.html)