/*
 * AI disclaimer: this file was written with the assistance of an AI model.
 * The code, tests and benchmarks were reviewed and validated by running
 * them, but AI-generated material can contain subtle errors; verify before
 * relying on it for production use.
 *
 * Assisted-by: Claude:claude-fable-5
 */
#pragma once
/* Shared bits for the AVX2 kernels. */
#include <immintrin.h>
#include <cstdint>

/* Keep a vector in a register across this point. Used on freshly loaded
 * accumulators so GCC emits vfnmadd231 in place instead of vfnmadd213 with a
 * memory operand plus a register copy of the other factor. */
#define RECLU_PIN(v) __asm__("" : "+x"(v))

/* Lane mask "row >= k" for a 4-row group whose first row is 4*floor(k/4):
 * lane l is live iff l >= k % 4. */
static inline __m256d reclu_live_mask(int k) {
    alignas(32) static const int64_t tab[4][4] = {
        {-1, -1, -1, -1}, {0, -1, -1, -1}, {0, 0, -1, -1}, {0, 0, 0, -1}};
    return _mm256_load_pd(reinterpret_cast<const double *>(tab[k & 3]));
}
