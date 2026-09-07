// lbm_streaming.h -- streaming operators, native kernels, time steppers
//
// The runtime side of streaming knows nothing about semi-Lagrangian or
// Lax-Wendroff: both schemes deliver, per lattice direction q, a sparse
// matrix A_q such that the streamed population is f_q <- A_q f_q. This
// header takes those matrices and applies them.
//
//   StreamingWeights   host exchange format produced by the assembly
//                      (lbm_schemes.h): 1 or Q sparsity patterns of fixed
//                      row length k, and one weight array per direction
//   EllOperator        the weights re-packed as column-major ELL in a
//                      memory space, the layout the native kernels want
//   kernels            per-row functors run through Exec<Space>
//   Stepper            the one virtual interface: advance one time step
//   FusedStepper       stream + collide in a single pass (native only)
//   SplitStepper       any Streamer, then a collision pass
//
// Convention: the stored populations are post-collision. One step is
// "stream, then collide", so the free by-product of a step is the set of
// macroscopic moments at the new time, for both steppers. Because the
// collision conserves rho and rho*u, macros() of the stored state equal
// those of the pre-collision populations at the same time.

#ifndef LBM_STREAMING_H
#define LBM_STREAMING_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "lbm_space.h"
#include "lbm_lattice.h"

namespace lbm {

// ---------------------------------------------------------------------
// Exchange format
// ---------------------------------------------------------------------

// Row-major fixed-k patterns, as NodeSet::stencils and the assembly
// kernels produce them: the k column indices of row s are ja[s*k .. s*k+k),
// and a[q][s*k + j] is the weight of column ja[s*k + j] in direction q.
template<typename T, typename I = std::int32_t>
struct StreamingWeights {
    using value_type = T;
    using index_type = I;

    I n = 0;
    int k = 0;
    std::vector<std::vector<I>> patterns;  // 1 (shared) or one per streamed direction
    std::vector<int> pattern_of;           // per direction: index into patterns, -1 = identity
    std::vector<std::vector<T>> a;         // per direction: n*k weights, empty if identity

    int num_directions() const { return static_cast<int>(pattern_of.size()); }
    bool is_identity(int q) const { return pattern_of[q] < 0; }
    bool shared_pattern() const { return patterns.size() == 1; }

    void validate() const {
        if (pattern_of.size() != a.size())
            throw std::invalid_argument("StreamingWeights: pattern_of/a size mismatch");
        for (const auto& p : patterns)
            if (p.size() != static_cast<std::size_t>(n) * k)
                throw std::invalid_argument("StreamingWeights: pattern size != n*k");
        for (std::size_t q = 0; q < a.size(); ++q) {
            if (is_identity(static_cast<int>(q))) {
                if (!a[q].empty())
                    throw std::invalid_argument("StreamingWeights: identity direction with weights");
            } else {
                if (pattern_of[q] >= static_cast<int>(patterns.size()))
                    throw std::invalid_argument("StreamingWeights: pattern index out of range");
                if (a[q].size() != static_cast<std::size_t>(n) * k)
                    throw std::invalid_argument("StreamingWeights: weight size != n*k");
            }
        }
    }
};

// ---------------------------------------------------------------------
// Native operator: column-major ELL
// ---------------------------------------------------------------------

// Storage order of a fixed-k ELL block, a per-space policy:
//
//   row_major   entry (s, j) at s*k + j: the k entries of a row are
//               contiguous. One thread walks its row sequentially, which
//               is what a CPU core and its prefetcher want. This is also
//               the layout the assembly produces, so packing is a copy.
//   col_major   entry (s, j) at j*ld + s, ld >= n padded: consecutive rows
//               are consecutive addresses, so a warp of threads handling
//               consecutive rows reads coalesced. The GPU layout.
//
// Kernels index through EllView::at(), so they are written once.
enum class EllLayout { row_major, col_major };

template<class Space>
struct default_ell_layout {
    static constexpr EllLayout value = EllLayout::row_major;
};
#if defined(__CUDACC__)
template<>
struct default_ell_layout<space::Cuda> {
    static constexpr EllLayout value = EllLayout::col_major;
};
#endif

// Non-owning view passed by value into kernels. a[q] == nullptr marks an
// identity direction; with a shared pattern all ja[q] are the same
// pointer.
template<class L, typename I, EllLayout Layout>
struct EllView {
    using lattice = L;
    using index_type = I;
    using T = typename L::value_type;
    static constexpr int Q = L::Q;
    static constexpr EllLayout layout = Layout;
    I n, ld;
    int k;
    bool shared;
    const I* ja[Q];
    const T* a[Q];

    LBM_HD std::size_t at(I s, int j) const {
        if constexpr (Layout == EllLayout::row_major) return std::size_t(s) * k + j;
        else return std::size_t(j) * ld + s;
    }
};

template<class L, typename I, class Space,
         EllLayout Layout = default_ell_layout<Space>::value>
class EllOperator {
public:
    using lattice = L;
    using T = typename L::value_type;
    using index_type = I;
    using space_type = Space;
    using view_type = EllView<L, I, Layout>;
    static constexpr int Q = L::Q;
    static constexpr EllLayout layout = Layout;

    // Re-packs (pads and, for col_major, transposes) the host weights and
    // moves them to Space.
    explicit EllOperator(const StreamingWeights<T, I>& w, I align = 64)
        : n_(w.n), k_(w.k),
          ld_(Layout == EllLayout::col_major ? padded(w.n, align) : w.n),
          shared_(w.shared_pattern())
    {
        w.validate();
        if (w.num_directions() != Q)
            throw std::invalid_argument("EllOperator: weights are for a different lattice");
        for (int q = 0; q < Q; ++q)
            if (w.is_identity(q) != (q == L::rest))
                throw std::invalid_argument("EllOperator: identity directions must be exactly the rest direction");

        const std::size_t plane = static_cast<std::size_t>(ld_) * k_;
        const int npat = static_cast<int>(w.patterns.size());
        const int nstream = Q - (L::rest >= 0 ? 1 : 0);

        view_.n = n_; view_.ld = ld_; view_.k = k_; view_.shared = shared_;

        // padded rows point at column 0 with weight 0 so they are harmless
        std::vector<I> hja(plane * npat, I(0));
        for (int p = 0; p < npat; ++p)
            for (I s = 0; s < n_; ++s)
                for (int j = 0; j < k_; ++j)
                    hja[p * plane + view_.at(s, j)] = w.patterns[p][std::size_t(s) * k_ + j];

        std::vector<T> ha(plane * nstream, T(0));
        std::vector<int> slot(Q, -1);
        int used = 0;
        for (int q = 0; q < Q; ++q) {
            if (w.is_identity(q)) continue;
            slot[q] = used++;
            for (I s = 0; s < n_; ++s)
                for (int j = 0; j < k_; ++j)
                    ha[slot[q] * plane + view_.at(s, j)] = w.a[q][std::size_t(s) * k_ + j];
        }

        ja_ = to_space<Space>(hja);
        a_ = to_space<Space>(ha);

        for (int q = 0; q < Q; ++q) {
            view_.ja[q] = w.is_identity(q) ? nullptr : ja_.data() + w.pattern_of[q] * plane;
            view_.a[q]  = w.is_identity(q) ? nullptr : a_.data() + slot[q] * plane;
        }
    }

    const view_type& view() const { return view_; }
    I n() const { return n_; }
    I ld() const { return ld_; }
    int k() const { return k_; }
    bool shared_pattern() const { return shared_; }

    // bytes touched by one streaming pass, for bandwidth estimates
    std::size_t bytes_per_step() const {
        const std::size_t nstream = Q - (L::rest >= 0 ? 1 : 0);
        const std::size_t idx = (shared_ ? 1 : nstream) * k_ * sizeof(I);
        return static_cast<std::size_t>(n_) * (idx + nstream * k_ * sizeof(T) + 2 * Q * sizeof(T));
    }

private:
    I n_;
    int k_;
    I ld_;
    bool shared_;
    Array<I, Space> ja_;
    Array<T, Space> a_;
    view_type view_;
};

// ---------------------------------------------------------------------
// Per-row kernels (host and device)
// ---------------------------------------------------------------------

// g[q] = sum_j A_q(s, j) f_q(ja_q(s, j)); identity directions copy.
// With Shared the column indices are loaded once per j and reused for
// every direction, which removes (Q-2)/(Q-1) of the index traffic.
template<bool Shared, class View, typename I>
LBM_HD inline void ell_gather_row(const View& A, I s,
                                  const typename View::T* __restrict__ f, I ldf,
                                  typename View::T (&g)[View::Q])
{
    using T = typename View::T;
    constexpr int Q = View::Q;
    constexpr int R = View::lattice::rest;

    LBM_UNROLL
    for (int q = 0; q < Q; ++q) g[q] = T(0);
    if constexpr (R >= 0) g[R] = f[std::size_t(R) * ldf + s];

    if constexpr (Shared) {
        const I* __restrict__ ja = A.ja[R == 0 ? 1 : 0];
        for (int j = 0; j < A.k; ++j) {
            const std::size_t e = A.at(s, j);
            const I c = ja[e];
            LBM_UNROLL
            for (int q = 0; q < Q; ++q) {
                if (q == R) continue;
                g[q] += A.a[q][e] * f[std::size_t(q) * ldf + c];
            }
        }
    } else {
        LBM_UNROLL
        for (int q = 0; q < Q; ++q) {
            if (q == R) continue;
            const I* __restrict__ ja = A.ja[q];
            const T* __restrict__ a = A.a[q];
            const T* __restrict__ fq = f + std::size_t(q) * ldf;
            T acc = T(0);
            for (int j = 0; j < A.k; ++j) {
                const std::size_t e = A.at(s, j);
                acc += a[e] * fq[ja[e]];
            }
            g[q] = acc;
        }
    }
}

// Optional macro output written by the collision kernels
template<typename T>
struct MacroPtrs {
    T* rho = nullptr;
    T* ux = nullptr;
    T* uy = nullptr;
    LBM_HD bool enabled() const { return rho != nullptr; }
};

template<class View, class Collision, bool Shared>
struct StreamCollideKernel {
    using L = typename View::lattice;
    using I = typename View::index_type;
    using T = typename L::value_type;
    View A;
    const T* f_in;
    T* f_out;
    I ldf;
    Collision collide;
    MacroPtrs<T> out;

    LBM_HD void operator()(std::ptrdiff_t i) const {
        const I s = static_cast<I>(i);
        T g[L::Q];
        ell_gather_row<Shared>(A, s, f_in, ldf, g);
        typename L::Macros m;
        collide(g, m);
        LBM_UNROLL
        for (int q = 0; q < L::Q; ++q) f_out[std::size_t(q) * ldf + s] = g[q];
        if (out.enabled()) { out.rho[s] = m.rho; out.ux[s] = m.ux; out.uy[s] = m.uy; }
    }
};

template<class View, bool Shared>
struct StreamKernel {
    using L = typename View::lattice;
    using I = typename View::index_type;
    using T = typename L::value_type;
    View A;
    const T* f_in;
    T* f_out;
    I ldf;

    LBM_HD void operator()(std::ptrdiff_t i) const {
        const I s = static_cast<I>(i);
        T g[L::Q];
        ell_gather_row<Shared>(A, s, f_in, ldf, g);
        LBM_UNROLL
        for (int q = 0; q < L::Q; ++q) f_out[std::size_t(q) * ldf + s] = g[q];
    }
};

template<class L, typename I, class Collision>
struct CollideKernel {
    using T = typename L::value_type;
    T* f;
    I ldf;
    Collision collide;
    MacroPtrs<T> out;

    LBM_HD void operator()(std::ptrdiff_t i) const {
        const I s = static_cast<I>(i);
        T g[L::Q];
        LBM_UNROLL
        for (int q = 0; q < L::Q; ++q) g[q] = f[std::size_t(q) * ldf + s];
        typename L::Macros m;
        collide(g, m);
        LBM_UNROLL
        for (int q = 0; q < L::Q; ++q) f[std::size_t(q) * ldf + s] = g[q];
        if (out.enabled()) { out.rho[s] = m.rho; out.ux[s] = m.ux; out.uy[s] = m.uy; }
    }
};

template<class L, typename I>
struct MacrosKernel {
    using T = typename L::value_type;
    const T* f;
    I ldf;
    MacroPtrs<T> out;

    LBM_HD void operator()(std::ptrdiff_t i) const {
        const I s = static_cast<I>(i);
        T g[L::Q];
        LBM_UNROLL
        for (int q = 0; q < L::Q; ++q) g[q] = f[std::size_t(q) * ldf + s];
        const auto m = L::macros(g);
        out.rho[s] = m.rho; out.ux[s] = m.ux; out.uy[s] = m.uy;
    }
};

template<class L, typename I>
struct EquilibriumKernel {
    using T = typename L::value_type;
    T* f;
    I ldf;
    const T* rho;
    const T* ux;
    const T* uy;

    LBM_HD void operator()(std::ptrdiff_t i) const {
        const I s = static_cast<I>(i);
        T g[L::Q];
        L::equilibrium(rho[s], ux[s], uy[s], g);
        LBM_UNROLL
        for (int q = 0; q < L::Q; ++q) f[std::size_t(q) * ldf + s] = g[q];
    }
};

// Convenience launchers
template<class L, typename I, class Space>
void compute_macros(I n, const typename L::value_type* f, I ldf,
                    MacroPtrs<typename L::value_type> out) {
    Exec<Space>::parallel_for(n, MacrosKernel<L, I>{f, ldf, out});
}

template<class L, typename I, class Space>
void set_equilibrium(I n, typename L::value_type* f, I ldf,
                     const typename L::value_type* rho,
                     const typename L::value_type* ux,
                     const typename L::value_type* uy) {
    Exec<Space>::parallel_for(n, EquilibriumKernel<L, I>{f, ldf, rho, ux, uy});
}

// ---------------------------------------------------------------------
// Streamers
// ---------------------------------------------------------------------
//
// A Streamer applies all Q matrices: stream(f_in, f_out, ldf). This is the
// seam for vendor libraries: an Eigen, MKL or cuSPARSE streamer is a class
// that owns the library's handles, is built once from StreamingWeights,
// and exposes the same call (see lbm_streamer_eigen.h). Backends that
// only offer SpMV are used through SplitStepper; the fused path needs the
// native operator.

// Op is an EllOperator<L, I, Space, Layout>
template<class Op>
class NativeStreamer {
public:
    using L = typename Op::lattice;
    using I = typename Op::index_type;
    using Space = typename Op::space_type;
    using View = typename Op::view_type;
    using T = typename L::value_type;

    explicit NativeStreamer(const Op& A) : A_(A) {}
    static constexpr const char* name() { return "native"; }

    void stream(const T* f_in, T* f_out, I ldf) const {
        const View& v = A_.view();
        if (v.shared)
            Exec<Space>::parallel_for(v.n, StreamKernel<View, true>{v, f_in, f_out, ldf});
        else
            Exec<Space>::parallel_for(v.n, StreamKernel<View, false>{v, f_in, f_out, ldf});
    }
    std::size_t bytes_per_step() const { return A_.bytes_per_step(); }

private:
    const Op& A_;
};

// ---------------------------------------------------------------------
// Steppers
// ---------------------------------------------------------------------

// The single runtime-polymorphic interface: one virtual call per time
// step, host side. Everything inside step() is statically dispatched.
template<class L, typename I, class Space>
struct Stepper {
    using T = typename L::value_type;
    virtual ~Stepper() = default;

    // f_out <- collide(stream(f_in)); if out.enabled(), also the macros at the new time
    virtual void step(const T* f_in, T* f_out, I ldf, MacroPtrs<T> out) = 0;
    virtual std::string name() const = 0;
    virtual std::size_t bytes_per_step() const = 0;  // estimate, for bandwidth reporting
};

// Op is an EllOperator<L, I, Space, Layout>
template<class Op, class Collision>
class FusedStepper final
    : public Stepper<typename Op::lattice, typename Op::index_type, typename Op::space_type> {
public:
    using L = typename Op::lattice;
    using I = typename Op::index_type;
    using Space = typename Op::space_type;
    using View = typename Op::view_type;
    using T = typename L::value_type;

    FusedStepper(const Op& A, Collision c) : A_(A), c_(c) {}

    void step(const T* f_in, T* f_out, I ldf, MacroPtrs<T> out) override {
        const View& v = A_.view();
        if (v.shared)
            Exec<Space>::parallel_for(v.n, StreamCollideKernel<View, Collision, true>{v, f_in, f_out, ldf, c_, out});
        else
            Exec<Space>::parallel_for(v.n, StreamCollideKernel<View, Collision, false>{v, f_in, f_out, ldf, c_, out});
    }
    std::string name() const override { return std::string("fused/native/") + Collision::name(); }
    std::size_t bytes_per_step() const override { return A_.bytes_per_step(); }

private:
    const Op& A_;
    Collision c_;
};

template<class L, typename I, class Space, class Collision, class Streamer>
class SplitStepper final : public Stepper<L, I, Space> {
public:
    using T = typename L::value_type;
    SplitStepper(I n, const Streamer& s, Collision c) : n_(n), s_(s), c_(c) {}

    void step(const T* f_in, T* f_out, I ldf, MacroPtrs<T> out) override {
        s_.stream(f_in, f_out, ldf);
        Exec<Space>::parallel_for(n_, CollideKernel<L, I, Collision>{f_out, ldf, c_, out});
    }
    std::string name() const override {
        return std::string("split/") + Streamer::name() + "/" + Collision::name();
    }
    std::size_t bytes_per_step() const override {
        // streaming pass plus a read-modify-write of the populations
        return s_.bytes_per_step() + static_cast<std::size_t>(n_) * 2 * L::Q * sizeof(T);
    }

private:
    I n_;
    const Streamer& s_;
    Collision c_;
};

} // namespace lbm

#endif // LBM_STREAMING_H
