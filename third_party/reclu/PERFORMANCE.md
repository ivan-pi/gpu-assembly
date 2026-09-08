<!--
AI disclaimer: this document was written with the assistance of an AI model.
The techniques described were implemented, measured and tested in this
repository, but AI-generated material can contain subtle errors; verify
before relying on it.

Assisted-by: Claude:claude-fable-5
-->

# Performance techniques in `reclu_lu_opt`, `reclu_solve_opt`, `reclu_sytf2_opt` and `reclu_sytrs_opt` (x86-64)

This document explains *why* the optimized LU factorization (`dgetrf`-style) and
solve (`dgetrs`-style) kernels in this repository are fast for small dense
systems (n ≈ 20..100), and what did **not** work. All numbers are single-core
measurements on a Skylake-X Xeon (AVX2/AVX-512, ~2.65 GHz effective, GCC 13,
`-O3 -march=native`). The naive reference implementations (`reclu_lu`,
`reclu_solve_naive`) are plain loops of the same algorithms and are the
correctness oracle for everything below.

The short version: at these sizes the problem is not flops. It is stores,
front-end µops, store-to-load forwarding, and the serial dependency chain of
partial pivoting. Every technique here targets one of those.

---

## 1. Storage contract: padded rows, 32-byte alignment

Both kernels require column-major storage with `lda % 4 == 0` and
`lda >= roundup4(m)`, and treat rows `m .. roundup4(m)-1` as their own
(they are zeroed by `reclu_lu_opt` and relied upon by `reclu_solve_opt`).

**Why.** A 256-bit vector holds 4 doubles. If the row count is not a multiple
of 4, the last group of rows needs either a masked or a scalar tail. Both are
poison here: a masked store, or four scalar stores, followed one pivot step
later by a full-width vector load of the same bytes, cannot store-forward and
stalls for ~15 cycles. At small n almost every column write *is* a tail. With
zero padding every access is a full, unmasked load/store that forwards
normally. Zero rows are harmless to the algorithm: they never win the pivot
search (`0 > x` is false, also for NaN `x`), they scale to ±0, and they only
ever update themselves.

**Alignment.** These kernels are store-bound; a 256-bit store that straddles a
64-byte line is split into two. A 16-byte-misaligned buffer cost ~35%. Use
`aligned_alloc(32, …)` (or 64) for the matrices. Note that `std::vector` and
Julia arrays below ~2 KB only guarantee 16 bytes.

**If copying into a padded scratch buffer is not allowed** (`experiments/reclu_lu_masked.cpp`):
the in-place kernel with masked tail groups is bit-identical but 7-20% slower than the copy path
and 25-45% slower than padded storage at `lda = n`, `n % 4 != 0` (the masked tail stores do not
forward, and odd `lda` breaks column alignment). A scalar tail loop
(`experiments/reclu_lu_scalartail.cpp`) is 8-16% better than masking for 1-row tails but equal
or worse for 3-row tails. Making the remainder a template parameter
(`experiments/reclu_lu_tail.cpp`: 1 row scalar, 2 rows one xmm, 3 rows xmm + scalar, with the
same access widths used for the next step's loads so every tail store forwards) is the best
in-place option: parity with the copy path up to n ~ 65, 5-17% behind above, and 1.0-1.35x
behind padded storage -- the remainder being the column misalignment of an odd `lda`, which
cannot be fixed in place. A caller-supplied workspace (LAPACK-style
`work` argument) recovers the copy path without stack use; padded allocation removes the copy
altogether.

**Effect.** Removes a 4-periodic sawtooth of 15–25% between multiples and
non-multiples of 4 (the copy path that handles unpadded input costs that
much), see `figures/sweep_throughput.png`.

---

## 2. LU: register-tiled rank-1 update with a blended first group

The trailing update `A(i,j) -= L(i,k) * U(k,j)` is done as an **8-row × 4-column
tile**: two vector loads of the pivot column feed 8 FMAs (4 columns × 2 row
groups) before being discarded. Per 8 rows and 4 columns that is 10 loads,
8 FMAs, 8 stores.

The 4-row group containing row `k+1` is special: rows ≤ k inside it are
finished U entries. Instead of a masked store, the group is loaded whole, the
update is computed for all four lanes, and the result is **blended** back with
the old values for the dead lanes (`_mm256_blendv_pd` with a per-`k%4` lane
mask), then stored at full width. One blend per column per step, and every
store forwards.

**Instruction selection had to be pinned.** GCC rewrote the load+FMA pair into
`vfnmadd213pd` with a memory operand plus a `vmovapd` register copy of the pivot
vector (three copies per four FMAs), and with the runtime row start `k+1`
inside a fixed bound it fully unrolled the loop into a jump-in chain with
spills. Two fixes: `#pragma GCC unroll 1` on the row loop, and an empty
`asm("" : "+x"(v))` on each freshly loaded accumulator (`RECLU_PIN`), which
forces the FMA into the in-place `231` form. Clang did not need either, but is
not harmed by them.

---

## 3. LU: fused pivot search, exact LAPACK tie-breaking

The reference does three passes per step that all touch column `k+1`: the pivot
scan (scalar, branchy), the row swap, and the update. The scan is folded into
the update of column `k+1` (the first tile of every step): a per-lane running
maximum with a strict `>` (first hit within each lane) plus a cross-lane
reduction with "larger wins, ties → smaller index". Because lane *l* only ever
sees rows `≡ first_row + l (mod 4)` in increasing order, this reproduces the
scalar scan exactly, including LAPACK's first-index tie-breaking. The tracker
is seeded with 0 at the diagonal row, so an all-zero column yields the diagonal
and NaN never wins (the reference keeps the diagonal for a NaN diagonal too).

The cross-lane reduce is deliberately **branchy** (a 4-element loop on a stack
copy): the branch predictor usually guesses the next pivot correctly and lets
the next step start speculatively. A branchless permute/min version added a
~20-cycle dependent chain per step and was slower at n ≤ 48.

**Effect.** Together with §2 this is the difference between the generic
compiler-vectorized code and the tuned kernel: ~1.3× at n = 30..48.

---

## 4. LU: shortening the per-step serial chain

Each pivot step has an unavoidable serial chain: resolve pivot → row swap →
`1/a_kk` → scale column → first tile → next pivot. A curve fit of the sweep
(`T ≈ 0.097 n³ + 0.48 n² + 71 n` cycles) shows it costs ~70 cycles per step,
14% of the time at n = 64 and most of it at n = 20.

The one cheap trick: because column `k+1` is final after the *first* tile of
step `k`, the next pivot is resolved and its reciprocal (`vdivsd`, ~14 cycles
latency) is issued **before** the remaining tiles, so the out-of-order core
overlaps the next step's chain with this step's bulk work. Worth a few percent
at n ≥ 48; nothing changes the chain's length itself at a given n. (What does:
batching independent stencils and interleaving their steps, which is outside
the scope of a single-matrix kernel.)

---

## 5. LU: size dispatch and scratch-buffer placement

- `m < 20`: the naive loop is as fast or faster (per-step fixed costs of the
  tiled kernel are not amortized), so it is used directly.
- Unpadded input goes through a scratch copy. That buffer (128 KB) lives in a
  separate `noinline` function: placed in the entry function's frame, Ubuntu's
  default `-fstack-clash-protection` probes every 4 KB page on *every* call,
  a flat ~100 ns penalty that also hit the small-size path.

---

## 6. Solve: reciprocal diagonal

The substitution `x_k = (b_k − Σ) / u_kk` puts a 14-cycle divide on the
critical chain of every pivot step. `reclu_solve_opt` computes all `1/u_kk` up
front with four-wide vector divides (independent, pipelined) and multiplies on
the chain instead (4 cycles). This is the only arithmetic difference from the
naive kernel: ≤ 1 ulp per pivot step. For well-conditioned systems the results
agree to ~1e-15; for numerically singular ones the two forward solutions
legitimately differ by κ·ε while both have small backward error, which is what
the adversarial tests check. (LAPACK's `dtrsm` divides precisely to stay
bit-reproducible against reference implementations; that is a trade-off, not a
requirement.)

---

## 7. Solve: 4-pivot blocks

Instead of an axpy per pivot (one store per FMA, and a store→load forward at
every step), pivots are handled in blocks of 4: the ≤ 4×4 triangle in scalar
code, then one **rank-4 streaming update** of the remaining rows — 1 load,
4 FMAs with memory operands, 1 store per 4 rows. This divides the traffic on
`x` by 4 and turns the serial chain into one segment per block. Blocks of 8
were slower (the 8×8 scalar triangle costs more than the passes it saves).

---

## 8. Solve: right-hand sides in vector lanes

The single-RHS solve is latency-bound; no throughput trick fixes that. But
RBF-FD applies several operators per stencil, i.e. several RHS per LU. With
`nrhs ≥ 4` the four systems ride in the four lanes of a `ymm`: X is kept
row-major (transposed in and out), each row update is one load, four broadcast
loads of L entries, four FMAs, one store — **16 useful flops per row store**, and
the dependency chain is shared by four solves. Eight RHS reuse each broadcast
for two accumulators. Leftover columns use the single-RHS kernel.

**Effect.** 2.8–4.2× over the naive per-column loop at `nrhs = 4`, and the
solve then costs less than the factorization by ~4× at n = 64. MKL's `dgetrs`
does the same internally, which is why its 4-RHS time equals its 1-RHS time.

---

## 9. Inline assembly (`reclu_solve_asm`)

The two streaming loops of the solve exist as inline-asm bodies. They gain
2–5% for one RHS at n ≥ 48 and nothing elsewhere: on an out-of-order core the
static order of ~30 µops in a loop body is irrelevant, and GCC's instruction
selection was already right once pinned (§2). Their value is that they cannot
drift between compilers. Superoptimizers (GSO, STOKE) are not applicable here
at all: they handle short branch-free integer sequences, not SIMD loops.

---

## 10. What bounds the kernels, and what did not help

Without a PMU the analysis uses `llvm-mca` port pressure (`tools/mca_loops.sh`),
a clock probe from fixed-latency chains (`tools/clock_probe.cpp`) and the
curve fit above.

**LU tile (steady state).** Per 8×4 iteration: 8 store-data µops on Skylake's
single store port (8 cycles), and ~30 µops through a 4-wide rename stage
(7.5 cycles). The loop sits on **both** limits at once, measured 11.7
cycles/iteration including short-loop overheads. Two experiments confirm it:

| experiment | what it changes | result |
|---|---|---|
| `experiments/reclu_lu_r2.cpp` — rank-2 update, two pivot steps per pass, step-(k+1) swap fused in, bit-identical | halves stores, not µops | 1.05–1.15× in cache; 1.4× only once L1 is exceeded |
| `experiments/reclu_lu_avx512.cpp` — zmm tiles, 8-row groups, native masks | halves both stores and µops per element | 1.25–1.5× at n = 45..80, ~1.0× outside; no frequency penalty observed |
| rank-8 blocked LU (panel / trsm / gemm) | GEMM-like reuse | no gain at these sizes: panel latency and row swaps dominate |
| 4K-aliasing check (`lda` padded to avoid 512 B strides) | — | no effect |

**Row swaps.** ~20% of LU time and store-port bound at ~1.7 cycles/swap (two
stores each). Leaving rows unpermuted (index vector) costs more than it saves,
because dead rows then have to be masked through every later update.

**Solve loops.** Load-bound: 10 loads per 8 rows against 4 cycles of FMA. An
AVX-512 lane kernel would run at the same 5 cycles per 2 rows while carrying
8 RHS, i.e. ~2× for `nrhs ≥ 8`; the `{1to8}` embedded broadcast still costs a
load µop, so the gain is the lane count, not the instruction count.

**AVX2 vs AVX-512.** For a stencil distribution centred on n = 30..50 the zmm
tile is worth 15–25%; for n = 60..80 about 40%. It is kept as an experiment:
a second code path and 8-row padding for a size-dependent gain.

---

## 11. Compile-time sizes

The original question. With the naive loops, template sizes were worth 12–30%
on GCC — entirely because GCC emits alias-versioning checks, scalar fallbacks
and tail handling for the runtime version — and 0% on Clang. With the kernel
written as above (explicit tiles, explicit padding), compile-time sizes are
worth 3–4% above n ≈ 30 and ~1.3× only at n = 10 where per-step overhead
dominates. `reclu_lu_opt` therefore takes `m, n, lda` as arguments; one
compiled kernel serves all stencil sizes.

---

## 12. LDL^T with Bunch-Kaufman pivoting (`reclu_sytf2_opt`, `reclu_sytrs_opt`)

The symmetric indefinite factorization reuses the LU's storage contract and
column-streaming design, with three things specific to it.

**Staircase tiles with constant lane masks.** The trailing update touches a
triangle: column `j` lives in rows `[j, n)` for `'L'` and `[0, j]` for `'U'`.
A 4-column tile that starts at an arbitrary column has one or two boundary
row groups whose lane masks depend on `j`, and the first implementation
computed those masks with data-dependent branches (`continue` on dead
column-groups, clamped range arithmetic). GCC turned that into a 1777-line
decision tree for `'L'` (685 lines for `'U'`, whose boundary comes after
the streaming loop), and the result was an `'L'` kernel no faster than the
naive loops while `'U'` gained 1.3x. Aligning the tiles to row groups fixes
it: columns `4t..4t+3` all start in row group `4t`, so every tile has
exactly one blended group and the masks are compile-time constants
(`lanes >= w` for `'L'`, `lanes <= w` for `'U'`); the leading columns up to
the next group boundary form a partial tile that uses the same masks
indexed by lane. That took `'L'` at n = 48 from 6.9 to 4.4 us and `'U'`
from 5.0 to 4.1 us on SPD input (no pivoting), and made the two `uplo`
paths the same code with mirrored row ranges. The general lesson from the
LU holds again: at these sizes the per-tile fixed cost decides, and what
the compiler does with boundary logic is part of that cost.

**Order of the walk is not free.** The rank-2 update writes `w_k`, `w_k+1`
back into the pivot columns after each tile, and later tiles read the pivot
columns from their own diagonal onward, so `'L'` must walk columns
ascending and `'U'` descending (LAPACK's loop order). Reversing the `'L'`
walk to mirror `'U'` breaks bit-identity for exactly this reason, and does
not help speed anyway.

**Fused column max.** The next pivot's `idamax` over the next column is
accumulated in the first tile of the update (first-index semantics via a
strict compare and lane order), as in the LU. One detail matters for
matching LAPACK on NaN input: `idamax` starts from the first element, so a
leading NaN is never beaten and is returned; the tracker has to special-case
that after the reduction. The remaining pivot logic (row max for the
Bunch-Kaufman test, symmetric interchange of a column segment with a strided
row segment) is scalar and identical to the reference.

**Solve.** Forward substitution streams one pivot at a time (a 2x2 pivot is
one pass with two FMAs) and is bit-identical to the reference; the
transposed substitution is a dot product per pivot with four chains (row
`i` in chain `i & 3`, combined as `(t0+t1)+(t2+t3)`), the order the
reference spells out. Four right-hand sides ride in ymm lanes with
row-major `X`. A single right-hand side is a latency chain through `x[k]`
(~10 cycles per pivot whichever way it is written), and the reference's
`omp simd` loops with an `omp simd reduction` dot are as fast as the vector
kernel up to n ~ 100 (the kernel pulls ahead by 6-13% at n = 112-128), so
`reclu_sytrs_opt` hands a 1..3 RHS remainder to the reference below
`RECLU_SY_SOLVE1_MIN = 112`. A 4-pivot blocking like the LU solve's would
need blocks that never split a 2x2 pivot; it was not attempted.

**What a plain reference can and cannot pin down.** The reference is a
one-to-one transcription of the LAPACK operations: `A(i, j)` indexing
through a `MatrixView`, plain operators, no `std::fma`, no unrolling. The
streaming loops carry restrict-qualified column pointers and `omp simd`
(worth about 10% over letting GCC version the loops with runtime alias
checks), and the dot product of the transposed solve is an
`omp simd reduction(+: t)`. For the axpy-shaped updates (rank-1, rank-2,
forward solve) GCC contracts consistently and the optimized kernels are
bit-identical to the reference. The dot product is a different matter: a
reduction pragma permits reassociation, so the compiler picks the
summation tree (without the pragma GCC still vectorizes it, but in order,
with unfused products and a contracted scalar epilogue - the decision even
changes with the inlining context, and three variants of the optimized dot
each matched one build and not another). The transposed solve is therefore
*specified* at a few ulp (`test_sy` checks 16 n eps norm-wise; the observed
worst case is 0.3 n eps) and the optimized dot is written the fast way.
Bit-identity of a reduction against a plain-operator reference is a
property of the compiler, not of the code.

**Results** (Skylake-X, random symmetric indefinite, 17 2x2 pivots and 45
interchanges at n = 64): factorization 0.88 / 2.1 / 5.4 / 10.5 us for
`'L'` and 0.88 / 1.8 / 5.0 / 9.3 us for `'U'` at n = 20 / 30 / 48 / 64,
against 0.95 / 2.3 / 6.0 / 12.5 us (L) for the simd reference,
1.2 / 3.5 / 9.8 / 19.6 us MKL and 1.6 / 4.6 / 11.9 / 23.2 us netlib. The LDL^T has half the flops of the LU
but runs only ~1.3x faster than `reclu_lu_opt`: its columns are shorter on
average, the pivot search is longer, and the symmetric interchange is a
strided row swap.

## 13. Correctness discipline

Every technique above was gated on tests: `reclu_lu_opt` is **bit-identical**
to the reference over 13k adversarial cases (ties, exact zeros, denormals,
NaN, singular, saddle-point, rectangular, every `lda` variant with garbage in
the pad rows); the solve is bit-identical between its intrinsic and asm
variants, within a few ulp of naive, and checked by backward error on
ill-conditioned input over 12k cases. The LDL^T reference was first validated
against netlib `dsytf2`/`dsytrs` (760 cases: identical pivot sequences, factors
within 6e-14 norm-wise), and `reclu_sytf2_opt` is bit-identical to it over 5k
cases for both `uplo`; the solve agrees to a fraction of n eps (see §12). Three comparison lessons: NaN sign/payload differs
between compilers and must be ignored; denormal results need an absolute
tolerance floor of a few × 2⁻¹⁰⁷⁴; and factors of near-singular integer
matrices differ at the 1e-5 relative level between fused and unfused
arithmetic where the true value is an exact cancellation (a 1e-17 residual vs
0), so factor comparisons across compilers must be norm-wise, not
element-wise.

Numbers to remember (n = 64, one core, this machine): naive LU 21 µs, optimized
12 µs (AVX-512: 9.5), MKL 23, OpenBLAS 25; solve with 4 RHS naive 5.4 µs,
optimized 2.0, MKL 2.4. LDL^T at n = 64: naive 11.7-12.5 µs, optimized 9.3-10.5 µs (both
`uplo`), MKL 18-19, netlib 22-23; `dsytrs` with 4 RHS naive 4.4-4.5 µs,
optimized 1.9-2.0, MKL 5.4-5.8.
