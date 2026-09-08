<!--
AI disclaimer: this project was written with the assistance of an AI model.
The code, tests and benchmarks were reviewed and validated by running them,
but AI-generated material can contain subtle errors; verify before relying
on it for production use.

Assisted-by: Claude:claude-fable-5
-->
# Fast small LU (dgetrf/dgetrs) and LDL^T (dsytf2/dsytrs) for RBF-FD stencil systems

Column-major, LAPACK conventions, n ~ 10..100, single core. The library is six
source files plus one header; everything else is tests, benchmarks and the
research history that led here.

## Library (`src/`)

| file | role |
|---|---|
| `reclu.h`             | the API (5 functions) and the storage requirements of the fast paths |
| `reclu_lu_naive.cpp`  | **reference** LU: dgetf2 algorithm in plain loops |
| `reclu_lu_opt.cpp`    | optimized LU (AVX2/FMA): zero-row padding, 8x4 register tile, fused pivot search, early reciprocal. Bit-identical to the reference |
| `reclu_solve_naive.cpp` | **reference** solve: swap, unit-L forward, U backward, column oriented |
| `reclu_solve_opt.cpp` | optimized solve: reciprocal diagonal, 4-pivot blocks, 4/8 right-hand sides per vector register; `reclu_solve_asm` uses inline-asm bodies for the two streaming loops (+2..5%) |
| `reclu_sy_naive.cpp`  | **reference** symmetric indefinite LDL^T with Bunch-Kaufman pivoting (dsytf2) and solve (dsytrs): one implementation over a fill-mode trait (`LowerFill`/`UpperFill`: walk direction, 2x2 partner, triangle ranges), `MatrixView` indexing, plain operators, `omp simd` on the column loops and an `omp simd reduction` dot, LAPACK's operation order |
| `reclu_sy_opt.cpp`    | optimized LDL^T/solve (AVX2/FMA), both `uplo` equally fast: group-aligned 4-column tiles with constant lane masks, fused column-max search, right-hand sides in lanes. Bit-identical to the reference |
| `reclu_simd.h`        | two shared helpers (register pin, lane mask) |

    reclu_lu(m, n, A, lda, ipiv, &info);              reclu_lu_opt(...)          // same signature
    reclu_solve_naive(m, nrhs, LU, lda, ipiv, B, ldb); reclu_solve_opt(...)  reclu_solve_asm(...)

    reclu_sytf2(uplo, n, A, lda, ipiv, &info);           reclu_sytf2_opt(...)       // uplo 'U' or 'L'
    reclu_sytrs(uplo, n, nrhs, A, lda, ipiv, B, ldb);    reclu_sytrs_opt(...)

Fast paths need `lda % 4 == 0`, `lda >= roundup4(m)`, 32-byte-aligned buffers; `reclu_lu_opt` and
`reclu_sytf2_opt('L')` zero the pad rows and the solves rely on them (`'U'` only reads them).
Otherwise the optimized routines copy through an aligned scratch buffer. Non-AVX2 targets get the
naive code. `m <= RECLU_MAXM (128)`; `m < RECLU_SMALL (20)` uses the naive kernels.

## Tests (`tests/`, run with `ctest`)

| test | cases | what |
|---|---|---|
| `test_lu`                | 13380 | `reclu_lu_opt` bit-identical to reference: ties, zeros, denormals, NaN, singular, saddle-point, rectangular, every `lda` variant with garbage pad rows |
| `test_solve`             |  1470 | opt/asm bit-identical, <=1e-12 from naive, residual <=1e-14 |
| `test_solve_adversarial` | 10546 | pivot patterns, singular LU (Inf/NaN pattern), Hilbert / tiny diagonal (backward error), overflow, denormal, NaN/Inf/-0 in B, pad rows preserved, fallback paths |
| `test_sy_naive_vs_lapack` |  760 | reference LDL^T/solve vs netlib `dsytf2_`/`dsytrs_` (dlopen, skipped if absent): identical pivot sequences (one near tie), factors within 6e-14 norm-wise, solutions within 1e-11, other triangle untouched |
| `test_sy`                | 5040 | `reclu_sytf2_opt` bit-identical to the reference for both `uplo` (SPD, indefinite, forced 2x2 pivots, integer ties, singular, 1e150/1e-160 scaling, NaN), other triangle and pad rows untouched; solve within 16 n eps of the reference norm-wise (forward substitution bit-identical, the transposed dot product is contracted differently by the compiler in the reference), Inf/NaN pattern identical on singular input, backward error <= 1e-13, every `lda` variant |

## Benchmarks (`bench/`)

| target | what |
|---|---|
| `bench_solve`  | Google Benchmark: naive / opt / asm / MKL / OpenBLAS `dgetrs`, n = 20..64, nrhs = 1,4,8 |
| `bench_sy`     | Google Benchmark: `dsytf2` and `dsytrs`, `uplo` L/U, naive / opt / netlib / MKL (+ `reclu_lu_opt` for scale), n = 20..64, nrhs = 1,4 |
| `sweep`        | LU, n = 4..100: naive, opt (`lda = n` and padded), OpenBLAS, netlib, optional MKL -> CSV |
| `sweep_solve`  | solve, n = 4..100, `nrhs` argument -> CSV |
| `bench_align`  | buffer alignment x lda micro-benchmark |
| `plot_sweep.py`, `plot_sweep_solve.py` | figures from the CSVs (`figures/`) |

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build
    OPENBLAS_NUM_THREADS=1 OPENBLAS_SO=/path/to/libopenblas.so ./build/sweep > results_sweep.csv
    ./build/sweep_solve 5 > results_sweep_solve_nrhs5.csv
    python3 bench/plot_sweep.py results_sweep.csv -o figures ; python3 bench/plot_sweep_solve.py results_sweep_solve_nrhs5.csv -o figures

MKL (sequential, LP64): `-DWITH_MKL=ON -DMKL_ROOT=/usr` (apt: `libmkl-dev libmkl-sequential`).
Once MKL is linked, a dlopen'd netlib LAPACK resolves its BLAS to MKL, so take the netlib column
from a non-MKL run (`plot_sweep.py` does this when given both CSVs).

## Experiments (`experiments/`, `-DBUILD_EXPERIMENTS=ON`)

The compile-time-size kernels (`reclu_static.hpp`, `reclu_opt.hpp`), the blocked rank-8 LU
(`reclu_blk.hpp`, with `-DRECLU_BLK_PROFILE` phase timing), their tests and the multi-variant
`bench_lu`. Kept because they answer the original question: with the kernel written properly,
compile-time sizes are worth only 3-4% over runtime sizes above n ~ 30, and the blocked update
gave nothing beyond the tuned rank-1 kernel at these sizes.

## What bounds the kernels (see `tools/`)

No PMU was available, so the analysis uses `llvm-mca` on the hot loops (`tools/mca_loops.sh`), a
clock probe (`tools/clock_probe.cpp`, ~2.65 GHz effective here) and a fit of the sweep data
T = a n^3 + b n^2 + c n. For the LU at n = 64 the tile loop is 80% of the time and runs at
~11.7 cycles per 8x4 iteration: it sits on Skylake's single store port (8 stores) *and* on the
4-wide rename limit (~30 uops) at once. Two experiments confirm this: a rank-2 update that
halves the stores but not the uops gains only ~1.1x (`experiments/reclu_lu_r2.cpp`); AVX-512
zmm tiles, which halve the uops per element, gain 1.2-1.35x at n = 48..80
(`experiments/reclu_lu_avx512.cpp`, `bench_variants`). The remaining 20% is the per-step serial
chain (~70 cycles/step) which no vector width touches. The solve loops are load-bound (10 loads
per 8 rows); an AVX-512 lane kernel would double the RHS per register at the same load count.

## Hardware counters with Likwid (`bench/likwid_probe.cpp`)

This VM exposes no PMU (CPUID.0xA version 0), so counters could not be read here; the probe is
built and linked against Likwid 5.5 (`-DWITH_LIKWID=ON -DLIKWID_ROOT=...`) but must be run on a
machine with PMU access. It marks four regions: `lu_naive`, `lu_opt`, `solve_naive`, `solve_opt`.

    likwid-perfctr -C 0 -m -g FLOPS_DP ./build/likwid_probe 64 4 20000   # GFLOP/s + scalar/128/256/512-bit split
    likwid-perfctr -C 0 -m -g L2       ./build/likwid_probe 64 4 20000   # L1<->L2 traffic (in-cache intensity)
    likwid-perfctr -C 0 -m -g MEM_DP   ./build/likwid_probe 96 4 20000   # DRAM intensity once out of cache
    likwid-perfctr -C 0 -m -g TMA      ./build/likwid_probe 64 4 20000   # top-down: front-end / back-end / bad spec / retiring
    likwid-perfctr -C 0 -m -g DATA     ./build/likwid_probe 64 4 20000   # load/store ratio

Vectorization ratio = (256-bit + 512-bit packed FLOPs) / all FLOPs from FLOPS_DP. Intensity in the
L1 regime is FLOPs per byte of L1 traffic, i.e. use the DATA group's load/store counts x 32 B, or the
L2 group when the working set exceeds L1 (n > ~80). Note `-m` (marker mode) and pin with `-C 0`.
Likwid does not support Apple Silicon; on the M2 use `xctrace`/kperf counters instead.

## Results (Xeon @ 2.1 GHz, AVX2, GCC 13, 1 core)

LU at n = 30 / 48 / 64: naive 3.0 / 9.6 / 21 us; optimized 2.1 / 5.5 / 12 us; MKL 5.3 / 12.9 / 23 us;
OpenBLAS 5.5 / 14 / 25 us. Solve with 4 RHS at n = 64: naive 5.4 us, optimized 2.0 us, MKL 2.4 us.
Full curves in `figures/` and `results_*.csv`. Machine noise across sessions is +-10%; compare
numbers from one run only.

LDL^T (Bunch-Kaufman) on a random symmetric indefinite matrix (17 2x2 pivots and 45 interchanges
at n = 64), ns per factorization, `L / U`:

| n | **opt** | naive | MKL | netlib |
|---|---|---|---|---|
| 20 |   **878 /   879** |   950 /   868 |  1206 /  1303 |  1581 /  1766 |
| 30 |  **2073 /  1783** |  2279 /  1915 |  3527 /  3323 |  4638 /  4073 |
| 48 |  **5389 /  4976** |  5950 /  5686 |  9836 /  9920 | 11938 / 11488 |
| 64 | **10502 /  9335** | 12525 / 11695 | 19642 / 18638 | 23168 / 23418 |

The reference's simd loops are why it is faster than MKL here and the optimized kernel's margin over
it is 1.1-1.25x; against MKL the kernel is 1.4-2.0x, against netlib 1.8-2.5x. `U` is a few percent
faster than `L` because its interchange and 2x2 bookkeeping touch contiguous memory.

`dsytrs`, ns, `L / U`:

| n | nrhs | **opt** | naive | MKL | netlib |
|---|---|---|---|---|---|
| 20 | 1 |  **520 /  391** |  510 /  383 | 1120 / 1202 | 1543 / 1560 |
| 20 | 4 |  **310 /  286** |  824 /  697 | 1467 / 1559 | 1836 / 1997 |
| 30 | 1 |  **814 /  594** |  804 /  580 | 1721 / 1795 | 2577 / 2523 |
| 30 | 4 |  **543 /  502** | 1395 / 1279 | 2382 / 2311 | 3115 / 2997 |
| 48 | 1 | **1433 / 1015** | 1402 / 1007 | 2866 / 2919 | 4045 / 4066 |
| 48 | 4 | **1145 / 1187** | 2888 / 2608 | 4371 / 3968 | 5086 / 4996 |
| 64 | 1 | **2193 / 1505** | 2213 / 1467 | 3974 / 4115 | 5706 / 5393 |
| 64 | 4 | **1928 / 1972** | 4500 / 4379 | 5767 / 5392 | 7801 / 6863 |

With 4 right-hand sides in vector lanes the kernel is 2.3-2.7x the reference and 2.5-5x MKL. With a
single right-hand side the substitution is a latency chain and the reference's simd loops are as
fast as the vector kernel up to n ~ 100, so `reclu_sytrs_opt` uses them for a 1..3 RHS remainder
below `RECLU_SY_SOLVE1_MIN = 112`.

## Attribution

This code base was developed with AI assistance, following the attribution convention of the
Linux kernel documentation. Every source, test, benchmark, build and documentation file carries
the tag below in its header. No specialized analysis tools beyond standard build tooling were
used (assembly was inspected with binutils; timings with Google Benchmark).

    Assisted-by: Claude:claude-fable-5
