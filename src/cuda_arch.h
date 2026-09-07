// cuda_arch.h -- shared-memory limits keyed by the cuSolverDx
//                SM<CC> architecture codes.
//
// Sources: CUDA C++ Programming Guide, "Compute Capabilities",
//          (Volta rows from the Volta Tuning Guide; not in the 13.x tables),
//          v13.3 (2026-05-27), Table 31 (memory information) and
//          Table 32 (SMEM capacity);
//          cuSolverDx "SM Operator" description.
//
// Co-developed-by: Claude:claude-fable-5.1

#ifndef CUDA_ARCH_H
#define CUDA_ARCH_H

#if __cplusplus < 201703L
#error "cuda_arch.h requires C++17"
#endif

#include <array>
#include <cstddef>
#include <optional>
#include <string_view>

namespace cuda_arch {

inline constexpr std::size_t KiB = 1024;

// Non-owning view of a static carveout list (a C++17 stand-in for std::span).
struct carveout_list {
    const unsigned* data = nullptr;
    std::size_t     size = 0;

    template <std::size_t N>
    constexpr carveout_list(const std::array<unsigned, N>& a) noexcept
        : data(a.data()), size(N) {}

    constexpr const unsigned* begin() const noexcept { return data; }
    constexpr const unsigned* end()   const noexcept { return data + size; }
    constexpr unsigned largest() const noexcept { return size ? data[size - 1] : 0u; }
};

struct smem_limits {
    unsigned cc;                    // cuSolverDx code: 750, 800, ..., 1210
    std::string_view arch;          // architecture family, as cuSolverDx names it
    std::string_view sm;            // nvcc target, e.g. "sm_90" for -arch=sm_90
    bool legacy;                    // dropped by cuSolverDx 0.4.0 (and CUDA 13)
    unsigned unified_cache_kb;      // L1 + SMEM unified data cache  (Table 32)
    unsigned max_per_sm_kb;         // configurable SMEM upper bound (Table 31)
    unsigned max_per_block_kb;      // per-block ceiling, opt-in     (Table 31)
    carveout_list carveouts_kb;     // legal SMEM capacities        (Table 32)

    // Static allocations are capped at 48 KB on every architecture. Above it,
    // the kernel must use dynamic shared memory *and* opt in via
    // cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, n).
    static constexpr std::size_t static_smem_limit = 48 * KiB;

    constexpr std::size_t max_per_block() const noexcept { return max_per_block_kb * KiB; }
    constexpr std::size_t max_per_sm()    const noexcept { return max_per_sm_kb    * KiB; }

    // System reservation per block: 1 KB on sm_80 and later (Ampere/Ada/
    // Hopper/Blackwell tuning guides), 0 on Volta and Turing.
    constexpr std::size_t reserved_per_block() const noexcept {
        return (max_per_sm_kb - max_per_block_kb) * KiB;
    }

    constexpr bool fits(std::size_t bytes) const noexcept { return bytes <= max_per_block(); }
    constexpr bool needs_dynamic_smem_opt_in(std::size_t bytes) const noexcept {
        return bytes > static_smem_limit;
    }

    // Smallest carveout (KB) that holds one block of `bytes`. Occupancy in
    // blocks per SM is not derivable from shared memory alone (Table 30 caps
    // resident blocks, warps and registers cap it further); use
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor for that.
    constexpr std::optional<unsigned> carveout_for(std::size_t bytes) const noexcept {
        const std::size_t need = bytes + reserved_per_block();
        for (unsigned c : carveouts_kb)
            if (c * KiB >= need) return c;
        return std::nullopt;
    }
};

namespace detail {

// Carveout lists, named by their largest entry. The largest carveout is
// 28 KB below the unified cache size: that remainder is the minimum L1.
inline constexpr std::array<unsigned,  6> cv_96     = {0, 8, 16, 32, 64, 96};
inline constexpr std::array<unsigned,  2> cv_turing = {32, 64};
inline constexpr std::array<unsigned,  8> cv_164    = {0, 8, 16, 32, 64, 100, 132, 164};
inline constexpr std::array<unsigned,  6> cv_100    = {0, 8, 16, 32, 64, 100};
inline constexpr std::array<unsigned, 10> cv_228    = {0, 8, 16, 32, 64, 100, 132, 164, 196, 228};

//                cc    arch          sm       legacy  cache  /SM  /blk  carveouts
inline constexpr std::array table = {
    smem_limits{  700, "Volta",     "sm_70",  true,   128,  96,  96, cv_96     },  // V100
    smem_limits{  720, "Volta",     "sm_72",  true,   128,  96,  96, cv_96     },  // Xavier
    smem_limits{  750, "Turing",    "sm_75",  false,   96,  64,  64, cv_turing },
    smem_limits{  800, "Ampere",    "sm_80",  false,  192, 164, 163, cv_164    },  // A100
    smem_limits{  860, "Ampere",    "sm_86",  false,  128, 100,  99, cv_100    },  // GA10x
    smem_limits{  870, "Ampere",    "sm_87",  false,  192, 164, 163, cv_164    },  // Orin
    smem_limits{  890, "Ada",       "sm_89",  false,  128, 100,  99, cv_100    },
    smem_limits{  900, "Hopper",    "sm_90",  false,  256, 228, 227, cv_228    },
    smem_limits{ 1000, "Blackwell", "sm_100", false,  256, 228, 227, cv_228    },  // B100/B200
    smem_limits{ 1030, "Blackwell", "sm_103", false,  256, 228, 227, cv_228    },  // B300 (Blackwell Ultra)
    smem_limits{ 1100, "Blackwell", "sm_110", false,  256, 228, 227, cv_228    },  // Thor (sm_101 before CUDA 13)
    smem_limits{ 1200, "Blackwell", "sm_120", false,  128, 100,  99, cv_100    },  // RTX 50 series
    smem_limits{ 1210, "Blackwell", "sm_121", false,  128, 100,  99, cv_100    },  // GB10 (DGX Spark)
};

} // namespace detail

// Runtime lookup, e.g. from cudaDeviceProp via cc_code(major, minor).
constexpr std::optional<smem_limits> find(unsigned cc) noexcept {
    for (const auto& e : detail::table)
        if (e.cc == cc) return e;
    return std::nullopt;
}

// Accepted by cuSolverDx's SM<CC> operator. Volta (700, 720) was removed in
// cuSolverDx 0.4.0; pass allow_legacy = true when building against 0.3.x.
constexpr bool is_cusolverdx_sm(unsigned cc, bool allow_legacy = false) noexcept {
    const auto r = find(cc);
    return r.has_value() && (allow_legacy || !r->legacy);
}

// Compile-time lookup; rejects codes cuSolverDx doesn't accept.
template <unsigned CC>
inline constexpr smem_limits limits_v = [] {
    constexpr auto r = find(CC);
    static_assert(r.has_value(), "not a cuSolverDx SM<CC> code");
    return *r;
}();

constexpr unsigned cc_code(int major, int minor) noexcept {
    return static_cast<unsigned>(major) * 100u + static_cast<unsigned>(minor) * 10u;
}

static_assert(limits_v<750>.max_per_block_kb == 64);
static_assert(limits_v<1100>.sm == "sm_110" && limits_v<1100>.arch == "Blackwell");
static_assert(limits_v<900>.max_per_block_kb == 227);
static_assert(limits_v<1200>.reserved_per_block() == KiB);
static_assert(limits_v<800>.carveout_for(60 * KiB) == 64u);
static_assert(limits_v<800>.carveout_for(64 * KiB) == 100u);   // 64 KB + 1 KB reserved
static_assert(!is_cusolverdx_sm(1010));
static_assert(!is_cusolverdx_sm(700) && is_cusolverdx_sm(700, true));
static_assert(limits_v<700>.reserved_per_block() == 0);
static_assert([] {
    for (const auto& e : detail::table)
        if (e.carveouts_kb.largest() != e.max_per_sm_kb) return false;
    return true;
    }(), "largest carveout must equal the per-SM maximum");

} // namespace cuda_arch

#endif // CUDA_ARCH_H