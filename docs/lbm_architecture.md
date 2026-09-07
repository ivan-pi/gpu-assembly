# LBM on scattered nodes: architecture

This document describes how the lattice Boltzmann layer (`src/lbm/`) is
put together, why it is cut where it is, and how to extend it along each
of its axes. It replaces the earlier sketches (`eigen_example.cpp`,
`gpulbm.cpp`, `D2Q9SL`, `Stream_cuSPARSE`, the three `plbm_mod` Fortran
modules); section 7 maps those onto the new components.

## 1. What varies

Seven things must be interchangeable without slowing the time loop down.
The first design decision is *when* each of them is decided.

| Axis | Choices | Decided at | Component |
|---|---|---|---|
| Lattice | D2Q9 now; D3Q19/D3Q27 later | compile time (type) | `lbm::D2Q9<T>` (`lbm_lattice.h`) |
| Collision model | BGK, TRT; RR, cumulant, regularised to come | compile time (functor type) | `lbm::collision::*` (`lbm_collision.h`) |
| Streaming scheme | semi-Lagrangian, Lax-Wendroff | assembly time | `lbm::build_streaming_weights` (`lbm_schemes.h`) |
| Stencil centring | arrival point (one pattern), departure point (one pattern per direction) | assembly time; one flag at run time | `StreamingWeights::shared_pattern`, `EllView::shared` |
| Execution space | host/OpenMP, CUDA; OpenMP target, OpenACC to come | compile time (tag type) | `lbm::space::*`, `Array<T,Space>`, `Exec<Space>` (`lbm_space.h`) |
| Streaming backend | native ELL kernels, Eigen; MKL, cuSPARSE to come | compile time (streamer class) | `NativeStreamer`, `EigenStreamer`, ... |
| Assembly implementation | host API (run-time sizes and operators); device API (static) | callable, adapted once | `rbf::HostAssembler` (`rbf_assembly.h`) as stand-in; `assemble_interp_weights` (`rbf_operators.h`) |

"Compile time" means a template parameter; the driver resolves the
command-line choice once, in a factory, and the time loop then makes one
virtual call per step (section 2.3). Nothing in a kernel is dispatched
dynamically.

## 2. Three decisions that untangle the hierarchy

### 2.1 A streaming scheme is an assembly recipe, not a run-time object

Semi-Lagrangian streaming interpolates the post-collision population at
the departure point:

    f_q(x_i, t+dt) = sum_j A_q(i,j) f_q(x_j, t),   A_q = RBF-FD interpolation weights at x_i - c_q dt

Lax-Wendroff streaming is the second-order Taylor expansion of the same
advection:

    f_q(t+dt) = [ I - dt (c_q . D) + dt^2/2 (c_q . D)^2 ] f_q(t)

with `D` the RBF-FD derivative operators. Written out, that bracket is a
sparse matrix per direction with the *same* pattern as the derivative
operators, and it can be formed once, before the time loop.

So at run time both schemes are identical: Q-1 sparse matrix-vector
products per step (the rest population is the identity). The scheme only
decides which weights to compute. This is why there is one operator type
(`EllOperator`), one set of kernels, and one exchange format
(`StreamingWeights`), and why `lbm_schemes.h` contains no kernels: it
contains recipes that call an assembler and return weights.

The only trace a scheme leaves at run time is the pattern count: arrival
point stencils share one pattern between all directions, departure point
stencils (semi-Lagrangian only; Lax-Wendroff expands about the arrival
point) have one per direction. A kernel specialised on `Shared` loads
each column index once and reuses it for all Q-1 directions.

The distinction is not academic. On a jittered 96x96 cloud with k = 21,
P = 4, arrival-centred semi-Lagrangian streaming blows up at CFL 1.5
(departure point about one spacing from the node, where the interpolant
starts to extrapolate), while departure-centred stencils run stably at
CFL 1.5 and 2.5 with the same stencil size. Arrival-centred stencils
survive CFL 1.5 only when enlarged to k = 37.

### 2.2 Backends are streamers built from one host exchange type

The old `D2Q9SL` sketch tried to be one class for every library, with a
`using MatDescr_t = ...` per vendor and an `#ifdef` ladder in
`stream()`. That couples every backend to every other and forces a
lowest-common-denominator interface (per-direction SpMV).

Instead there is one host-side exchange format, produced by any
assembler and consumed by any backend:

```cpp
template<typename T, typename I>
struct StreamingWeights {
    I n; int k;                              // rows, fixed row length
    std::vector<std::vector<I>> patterns;    // 1 (shared) or Q-1, each n*k row-major
    std::vector<int> pattern_of;             // per direction; -1 = identity
    std::vector<std::vector<T>> a;           // per direction, n*k, laid out like the pattern
};
```

This is CSR with an implicit row pointer (`ia[s] = s*k`), which is what
`NodeSet::stencils`, `rbf::periodic_knn`, the cuSolverDx kernel and the
host assembler all produce naturally, so no backend needs a conversion
the others do not.

A *streamer* is a class that owns whatever its library needs, is
constructed once from `StreamingWeights`, and exposes

```cpp
void stream(const T* f_in, T* f_out, I ldf) const;   // all Q directions
std::size_t bytes_per_step() const;                  // for bandwidth reporting
static const char* name();
```

`EigenStreamer` (`lbm_streamer_eigen.h`) is the worked example: a row
pointer and Q-1 `Eigen::Map<const SparseMatrix>` over the weight arrays.
An MKL streamer holds `sparse_matrix_t` handles created by
`mkl_sparse_d_create_csr`; a cuSPARSE streamer holds device copies of
`ia/ja/a`, Q-1 `cusparseSpMatDescr_t` that share `ia` and `ja`, one
reused `cusparseDnVecDescr_t` pair and a buffer sized once, exactly as in
the `Stream_cuSPARSE` sketch. The Q-1 `cusparseSpMV` calls are a natural
CUDA-graph capture. Each of these is one file that includes one library.

Vendor libraries only offer SpMV, so they plug into the `SplitStepper`.
The `NativeStreamer` and the fused path (below) use `EllOperator`, the
native re-packing of the same weights.

### 2.3 Static inside kernels, one virtual call per step outside

This is the traits-versus-virtual question. Both are used, at different
levels, and the boundary between them is the kernel launch.

**Inside a kernel: templates and functors, never virtual calls.** A
collision model is a functor over the populations of one node,

```cpp
template<class L> struct BGK {
    T omega;
    LBM_HD void operator()(T (&f)[L::Q], typename L::Macros& m) const;
};
```

copied by value into the kernel functor, so its `operator()` inlines and
the whole per-node body becomes straight-line code over registers. This
is what lets `StreamCollideKernel` fuse gathering, collision and store
without a trip through memory. A virtual call per node would cost the
inlining, the register allocation across the call, and on a GPU the
indirect branch; that is the "compromise the performance" the design
must avoid.

Managed memory does not change this. It makes a host-constructed object
*addressable* from the device, but the object's vtable pointer refers to
host code: CUDA only allows virtual calls on objects constructed on the
device (and forbids passing objects with virtual functions to a
`__global__` function by value). So managed memory would buy convenience
for data placement, not host/device polymorphism. It remains available
as an `Array` specialisation for anyone who wants it; it is not the
default because explicit `upload`/`download` keeps the traffic visible.

**Outside the kernel: one abstract interface, one call per time step.**

```cpp
template<class L, typename I, class Space>
struct Stepper {
    virtual void step(const T* f_in, T* f_out, I ldf, MacroPtrs<T> out) = 0;
    virtual std::string name() const = 0;
    virtual std::size_t bytes_per_step() const = 0;
};
```

`FusedStepper<Op, Collision>` and `SplitStepper<L,I,Space,Collision,Streamer>`
implement it; the driver's `make_stepper()` turns the command line
(`--collision trt --stepper split --backend eigen`) into the right
instantiation once, and the time loop holds a `std::unique_ptr<Stepper>`.
A virtual call every `dt` is free, and it is what gives the flexibility
the user of the driver wants without the traits machinery leaking into
`main`.

The rule of thumb: anything that is evaluated per node is a template
parameter; anything evaluated per step or less may be a virtual call or a
runtime flag. `EllView::shared` is a runtime flag read once per step to
pick between two instantiations of the same kernel, not a per-node branch.

## 3. Layers and data flow

```
 nodes x, y ─────► periodic_knn ────► ja (n x k) ──────┐
                                                        │
 scheme, dt ─────► centres + functionals ───────────────┼──► assemble(centres, ja, functionals)
                   (lbm_schemes.h)                      │        host API (run-time)  |  device API (static)
                                                        │             │
                                                        ▼             ▼
                                              StreamingWeights  (host, row-major fixed-k CSR)
                                                        │
                        ┌───────────────────────────────┼──────────────────────────┐
                        ▼                               ▼                          ▼
        EllOperator<L,I,Space,Layout>            EigenStreamer              MklStreamer / CuSparseStreamer
        (native, row- or col-major ELL)          (host CSR maps)            (to be written; same shape)
                        │                               │                          │
          ┌─────────────┴────────────┐                  │                          │
          ▼                          ▼                  ▼                          ▼
 FusedStepper<Op,Collision>   SplitStepper<..,NativeStreamer<Op>>   SplitStepper<..,EigenStreamer>  ...
          └──────────────────────────┴──────────────────┴──────────────────────────┘
                                                        ▼
                                   Stepper<L,I,Space>::step()  ◄──  driver: time loop, diagnostics, IO
```

Everything above the dashed line is host code and runs once. Everything
below runs every step in the chosen `Space`.

## 4. Components

### 4.1 Lattice (`lbm_lattice.h`)

`D2Q9<T>` is a static class: `Q`, `D`, `rest` (index of the zero-velocity
direction, or -1), `cs2`, `cx[]`, `cy[]`, `opp[]`, `w[]`, and two
`LBM_HD` per-node maps, `macros(f)` and `equilibrium(rho, ux, uy, feq)`,
with the same arithmetic as `d2q9_kernels.cu`. Every other component is
templated on the lattice and reads only this interface. A D3Q19 is a
second such class plus a `cz[]`; `StreamingWeights` and the kernels
already index by direction, not by a fixed 9.

### 4.2 Memory and execution spaces (`lbm_space.h`)

A space is a tag type. `Array<T, Space>` owns a buffer in it (aligned
heap memory on the host, `cudaMalloc` under nvcc) with `upload` and
`download` to and from host pointers. `Exec<Space>::parallel_for(n, f)`
runs a functor over `[0, n)`: an OpenMP loop on the host, one thread per
index on a GPU. `LBM_HD` expands to `__host__ __device__` under nvcc and
to nothing otherwise, so every kernel functor in `lbm_streaming.h`
compiles unchanged for both.

The three Fortran modules (`acc_lbm.F90`, `omp_lbm.f90`, `plbm.F90`) were
the same code three times because Fortran has no way to hand a loop body
to a driver; here the kernel bodies exist once and the three variants are
three `Exec` specialisations. An OpenMP-target space is `Array` with
`omp_target_alloc` and an `Exec` whose `parallel_for` is
`#pragma omp target teams distribute parallel for`; OpenACC likewise with
`acc parallel loop`. Both need the functor types to be device-callable,
which with nvc++ means compiling the translation unit with `-mp=gpu` or
`-acc` and letting the compiler see the whole functor; that is the case
here because all kernels are header templates.

### 4.3 Assembly: two APIs, one contract (`rbf_assembly.h`)

There are two assembly APIs by design, and they are not meant to share
code:

- **Device** (`rbf_operators.h`): static. Stencil size, degree, PHS
  exponent and the operator set are template parameters of
  `interp_config`, and only conventional operator sets are offered. A GPU
  pays for the specialisation once and gains throughput on every stencil.
- **Host**: dynamic. Stencil size, degree, exponent and the list of
  functionals are run-time values. A CPU has the latency to branch once
  per column, and the flexibility is worth more than the specialisation.
  The existing host API (not yet in the repository) is this API; nothing
  here depends on its argument list.

What the LBM layer needs from either is one callable:

```cpp
std::vector<std::vector<T>>
assemble(std::span<const T> xc, std::span<const T> yc,      // stencil centres, length n
         std::span<const I> ja,                              // n*k column indices, row-major
         std::span<const rbf::Functional<T>> L);             // what to evaluate, local frame
// returns one n*k weight array per functional, laid out like ja
```

with `Functional = {value | dx | dy | dxx | dxy | dyy | laplace, x, y}`,
`(x, y)` relative to the stencil centre. Centres are passed separately
from the cloud because departure-point stencils are centred on points
that are not nodes. The existing host API is adapted to this with a
lambda; the device API with a small host class around
`assemble_interp_weights` plus a download.

`rbf::HostAssembler` is a self-contained stand-in for the host API so
that the layer can be built and tested without it: per stencil, gather
with minimum-image displacements, fill the collocation matrix, one
right-hand side per functional (the functional is chosen once per column,
each case being the plain formula for that derivative), dense LU, scatter
the first k rows. It is written for clarity, not speed.

### 4.4 Scheme recipes (`lbm_schemes.h`)

`build_streaming_weights<L>(scheme, stencils, dt, x, y, k, assemble, knn, box)`
is generic over the assembler and the neighbour search (`knn(qx, qy, k)`,
so that queries need not be nodes) and produces `StreamingWeights`:

| scheme / stencils | patterns | assemble calls | functionals |
|---|---|---|---|
| semi-Lagrangian / arrival | 1 | 1 | value at `-c_q dt`, q = 1..Q-1 |
| semi-Lagrangian / departure | Q-1 | Q-1 | value at (0,0) about the departure point |
| Lax-Wendroff / arrival | 1 | 1 | dx, dy, dxx, dxy, dyy at (0,0), combined per direction |

The Lax-Wendroff combination needs the identity, so it locates the node
in its own stencil row rather than assuming it is first.

### 4.5 Native operator (`EllOperator`, `lbm_streaming.h`)

`EllOperator<L, I, Space, Layout>` re-packs the weights into a fixed-k
ELL block per pattern and per direction and moves them to `Space`. The
storage order is a per-space policy:

- `row_major` (host default): entry `(s, j)` at `s*k + j`. A thread walks
  its row contiguously, which a CPU core and its prefetcher want. It is
  also the assembly layout, so packing is a copy.
- `col_major` (CUDA default): entry `(s, j)` at `j*ld + s` with `ld`
  padded to a multiple of 64. Consecutive rows are consecutive addresses,
  so a warp handling consecutive rows reads coalesced.

Kernels address entries through `EllView::at(s, j)`, so they are written
once. The choice matters: on the 4-core host used for development,
switching the fused kernel from column- to row-major took it from 14 to
58 MLUPS on a 96x96 cloud with k = 21.

### 4.6 Kernels

All are functors over one row `s`, run through `Exec<Space>`:

| kernel | reads | writes |
|---|---|---|
| `ell_gather_row<Shared>` | `A_q` rows, `f_in` | register array `g[Q]` |
| `StreamCollideKernel<View, Collision, Shared>` | as above | `f_out` row, macros (optional) |
| `StreamKernel<View, Shared>` | as above | `f_out` row |
| `CollideKernel<L, I, Collision>` | `f` row | `f` row in place, macros (optional) |
| `MacrosKernel`, `EquilibriumKernel` | | initialisation and diagnostics |

`MacroPtrs` is a nullable triple `rho, ux, uy`; a step writes them only
when asked (the driver asks on output steps), so the common step does not
pay for three extra stores per node. This subsumes the `indp`/`rho`/`vel`
side outputs of `bgk_kernel_split`.

### 4.7 Steppers and the state convention

The stored populations are **post-collision**, and a step is
**stream, then collide**. With that convention both steppers have the
same state semantics and the same free by-product (the moments at the new
time), and they agree bit-for-bit up to summation order: the test suite
checks fused, split-native and split-Eigen against each other to 1e-12
after 400 steps. Because collision conserves `rho` and `rho u`,
`compute_macros` on the stored state gives the moments of the
pre-collision populations at the same time, so diagnostics need no
special casing.

- `FusedStepper<Op, Collision>`: one pass; gathers the Q-1 streamed
  populations into registers, collides, stores. Populations cross memory
  once per step.
- `SplitStepper<L,I,Space,Collision,Streamer>`: `Streamer::stream`, then
  `CollideKernel` in place on the output. Works with any streamer, native
  or vendor.

### 4.8 Driver (`examples/lbm_bench.cpp`)

Generates a jittered periodic cloud (or reads a `.points` file),
optionally renumbers it along a Morton or Hilbert curve, assembles,
builds the stepper from the options, initialises from
`flow_benchmarks::taylor_green` or `shear_layer`, runs, and reports
`rho` bounds, `umax/u0`, the Taylor-Green decay factor and relative L2
velocity error, MLUPS and an effective bandwidth from
`Stepper::bytes_per_step()`. `--out prefix` writes VTK through
`rbf::io::write_vtk_polydata`.

## 5. Performance notes

Bytes per node per step, D2Q9, double populations, k = 21, 32-bit
indices, in the bandwidth-bound regime where every array crosses memory
once:

| path | indices | weights | populations | total |
|---|---|---|---|---|
| fused, shared pattern | 84 | 1344 | 72 + 72 | 1572 |
| fused, per-direction patterns | 672 | 1344 | 144 | 2160 |
| split native, shared | 84 | 1344 | 144 + 144 | 1716 |
| split vendor CSR (Eigen/MKL/cuSPARSE) | 672 (+ row ptr) | 1344 | 288 | 2304 |

Measured on the 4-core development host (96x96 jittered nodes, 300
steps): fused 58 MLUPS, split native 52, split Eigen 26, fused with
departure patterns 37. The ordering follows the table; the Eigen gap is
larger than the bytes alone predict because its Q-1 separate SpMV calls
each re-stream the populations.

Three consequences:

1. **The weights dominate.** 1344 of about 1600 bytes are `A_q`. Storing
   the weights in `float` while keeping `double` populations would cut
   the traffic by 40%; `EllOperator` and `EllView` only need a second
   scalar type parameter for that. This is the single largest
   optimisation left on the table, and it is orthogonal to everything
   else here.
2. **Fusion is worth more than index sharing** on this lattice
   (144 bytes versus 588), but index sharing is what makes departure
   patterns cost 37% more rather than 100%.
3. **Vendor SpMV is a convenience path.** Its value is validation and a
   quick GPU port through cuSPARSE, not peak throughput; the native
   kernels are where the effort should go.

Other points:

- The kernels take functors by value and raw `__restrict__` pointers; no
  virtual or indirect calls, no bounds checks in the loop.
- `ldf` (population leading dimension) and `ld` (ELL, column-major) are
  padded to 64 elements so each direction plane starts aligned.
- Node ordering (`--order morton|hilbert`, from `rbf_reorder.h`) is a
  locality knob for the gathers and costs nothing at run time.
- `Stepper::bytes_per_step()` reproduces the bandwidth accounting of the
  Fortran `*_bw` functions.

## 6. Extension recipes

**A collision model.** Write a functor in `lbm_collision.h` with the
signature in 2.3 and a static `name()`, then add one line to
`make_stepper()` in the driver. Recursive-regularised and cumulant
collisions are local, so they fit as they are. The two-step regularised
scheme in `acc_lbm.F90` (`ts_macros` + `tslbm_step`) also fits: it is a
functor that computes the stress from the gathered populations and
reconstructs; its "stream the reconstructed populations" order is the
convention of 4.7.

**A streaming backend.** One header, one class satisfying the streamer
contract in 2.2, constructed from `StreamingWeights`. Use it through
`SplitStepper`. For a device library the constructor uploads and the
streamer's `stream()` takes device pointers; the driver's `Space` selects
the matching `Array`.

**An execution space.** One `Array<T, NewSpace>` specialisation and one
`Exec<NewSpace>` in `lbm_space.h`, plus a `default_ell_layout` if it is
not row-major. No kernel changes.

**A lattice.** A static class like `D2Q9<T>` with the same members.

**A scheme.** A branch in `build_streaming_weights`. The assembler
contract already allows per-stencil centres; a scheme with per-node
evaluation points (velocity-dependent departure points) needs the
functional list to become per-stencil, a small extension of the same
call.

**Mixed precision.** A second scalar type on `EllOperator`/`EllView`
for the weights (section 5).

**Boundary conditions** are outside the periodic benchmarks. The natural
hook is a post-step kernel over `NodeSet::bnd` (or rows of `A_q`
replaced at assembly time); the stepper interface does not need to
change for either.

## 7. Where the old pieces went

| Old | New |
|---|---|
| `geom::StencilSet` (eigen_example.cpp) | `rbf::NodeSet` + `rbf::periodic_knn` |
| `StreamingOperator` (Eigen maps over `a(:,q)`) | `StreamingWeights` + `EigenStreamer` |
| `ell_cache`, `pack_ellpack_col_data` | `EllOperator` (column-major layout) |
| `ellpack_mv_col`, `ell_mv`, `sell_stream` | `StreamKernel` / `NativeStreamer` |
| `collide_bgk`, `d2q9_collide`, `bgk_kernel_split` | `collision::BGK` + `CollideKernel` + `MacroPtrs` |
| `D2Q9SL` with its `#ifdef` ladder | one streamer class per backend |
| `Stream_cuSPARSE` | the cuSPARSE streamer, to be written to the contract in 2.2 |
| `rbfx_assembly_periodic_xy` (Fortran) | the host assembly API, adapted to the contract in 4.3; `rbf::HostAssembler` stands in until it lands |
| `plbm_mod` x3 (OpenACC / OpenMP target / CUDA Fortran) | one kernel set + `Exec<Space>` specialisations |
| `collision_bw`, `streaming_bw`, `total_bw` | `Stepper::bytes_per_step()` |
| `flow_benchmarks.hpp` | `examples/rbf_flow_benchmarks.h`, unchanged apart from a `ky/kx` typo |
| `d2q9_kernels.cu` / `.cuf` | kept; `EquilibriumKernel`, `MacrosKernel`, `CollideKernel<BGK>` through `Exec<Cuda>` compute the same things and can be cross-checked against them |

## 8. Status

Built and tested on the host (g++ 13, OpenMP, Eigen 3.4; `ctest`):
`rbf_periodic.h`, `rbf_assembly.h`, all of `src/lbm/` except the CUDA
branches, `examples/lbm_bench.cpp`, `test/test_lbm.cpp`.
The tests cover polynomial exactness of the assembly for all six
functionals, row sums of all three weight recipes, collision invariants,
agreement of the three stepper/backend combinations, and a Taylor-Green
run against the analytic decay (semi-Lagrangian/BGK and
Lax-Wendroff/TRT, both within 1%).

Compiles under nvcc by construction but not exercised here (no CUDA
toolchain in the development container): `Array<T, space::Cuda>`,
`Exec<space::Cuda>`, the `LBM_HD` kernels as device code.
`rbf_operators.h` is untouched apart from its include of
`cuda_arch.hpp`, corrected to `cuda_arch.h`, the file that exists.

Not written: the MKL and cuSPARSE streamers, the device assembler wrapper,
the OpenMP-target space, mixed-precision weights, boundary conditions.
