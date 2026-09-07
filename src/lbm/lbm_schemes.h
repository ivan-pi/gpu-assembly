// lbm_schemes.h -- streaming schemes as assembly recipes
//
// A streaming scheme decides which sparse matrices A_q to build; it does
// not exist at run time. Both schemes here produce StreamingWeights:
//
//   semi-Lagrangian   A_q interpolates the population at the departure
//                     point x - c_q dt (point-value functional)
//   Lax-Wendroff      A_q = I - dt (c_q . D) + dt^2/2 (c_q . D)^2, built
//                     from the first and second derivative operators at
//                     the node (five functionals, combined per direction)
//
// and two stencil choices:
//
//   arrival           one stencil per node, centred on the node; shared
//                     by all directions (one pattern, Q-1 weight arrays)
//   departure         one stencil per node and direction, centred on the
//                     departure point (Q-1 patterns); semi-Lagrangian only
//
// The recipes are generic over the assembler (rbf::HostAssembler, or a
// device assembler with the same call) and the neighbour search.

#ifndef LBM_SCHEMES_H
#define LBM_SCHEMES_H

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "rbf_assembly.h"
#include "rbf_periodic.h"
#include "lbm_streaming.h"

namespace lbm {

enum class Scheme { semi_lagrangian, lax_wendroff };
enum class Stencils { arrival, departure };

inline const char* to_string(Scheme s) {
    return s == Scheme::semi_lagrangian ? "semi-lagrangian" : "lax-wendroff";
}
inline const char* to_string(Stencils s) {
    return s == Stencils::arrival ? "arrival" : "departure";
}

// Knn: callable (qx, qy, k) -> row-major n*k indices of the k nearest
// nodes of each query point. Assemble: see rbf_assembly.h.
template<class L, class Assemble, class Knn,
         typename T = typename L::value_type, typename I = std::int32_t>
StreamingWeights<T, I> build_streaming_weights(
        Scheme scheme, Stencils stencils, T dt,
        std::span<const T> x, std::span<const T> y, int k,
        Assemble&& assemble, Knn&& knn,
        const rbf::PeriodicBox<T>* box = nullptr)
{
    constexpr int Q = L::Q;
    constexpr int R = L::rest;
    const I n = static_cast<I>(x.size());

    StreamingWeights<T, I> w;
    w.n = n;
    w.k = k;
    w.pattern_of.assign(Q, -1);
    w.a.resize(Q);

    if (scheme == Scheme::semi_lagrangian && stencils == Stencils::arrival) {
        w.patterns.push_back(knn(x, y, k));
        std::vector<rbf::Functional<T>> ops;
        std::vector<int> dir;
        for (int q = 0; q < Q; ++q) {
            if (q == R) continue;
            ops.push_back({rbf::functional::value, -dt * T(L::cx[q]), -dt * T(L::cy[q])});
            dir.push_back(q);
        }
        auto vals = assemble(x, y, std::span<const I>{w.patterns[0]}, std::span<const rbf::Functional<T>>{ops});
        for (std::size_t r = 0; r < dir.size(); ++r) {
            w.pattern_of[dir[r]] = 0;
            w.a[dir[r]] = std::move(vals[r]);
        }
        return w;
    }

    if (scheme == Scheme::semi_lagrangian && stencils == Stencils::departure) {
        std::vector<T> xd(n), yd(n);
        const rbf::Functional<T> ops[1] = {{rbf::functional::value, T(0), T(0)}};
        for (int q = 0; q < Q; ++q) {
            if (q == R) continue;
            for (I i = 0; i < n; ++i) {
                xd[i] = x[i] - dt * T(L::cx[q]);
                yd[i] = y[i] - dt * T(L::cy[q]);
                if (box) { xd[i] = box->wrap_x(xd[i]); yd[i] = box->wrap_y(yd[i]); }
            }
            w.patterns.push_back(knn(std::span<const T>{xd}, std::span<const T>{yd}, k));
            auto vals = assemble(std::span<const T>{xd}, std::span<const T>{yd},
                                 std::span<const I>{w.patterns.back()},
                                 std::span<const rbf::Functional<T>>{ops});
            w.pattern_of[q] = static_cast<int>(w.patterns.size()) - 1;
            w.a[q] = std::move(vals[0]);
        }
        return w;
    }

    if (scheme == Scheme::lax_wendroff && stencils == Stencils::arrival) {
        w.patterns.push_back(knn(x, y, k));
        const auto& ja = w.patterns[0];
        const rbf::Functional<T> ops[5] = {
            {rbf::functional::dx}, {rbf::functional::dy},
            {rbf::functional::dxx}, {rbf::functional::dxy}, {rbf::functional::dyy}};
        auto D = assemble(x, y, std::span<const I>{ja}, std::span<const rbf::Functional<T>>{ops});
        const auto& Dx = D[0]; const auto& Dy = D[1];
        const auto& Dxx = D[2]; const auto& Dxy = D[3]; const auto& Dyy = D[4];

        // position of the diagonal in each row, for the identity
        std::vector<int> diag(n, -1);
        for (I s = 0; s < n; ++s)
            for (int j = 0; j < k; ++j)
                if (ja[std::size_t(s) * k + j] == s) { diag[s] = j; break; }
        for (I s = 0; s < n; ++s)
            if (diag[s] < 0)
                throw std::runtime_error("lax_wendroff: node " + std::to_string(s) + " not in its own stencil");

        for (int q = 0; q < Q; ++q) {
            if (q == R) continue;
            const T cx = T(L::cx[q]), cy = T(L::cy[q]);
            const T h = T(0.5) * dt * dt;
            auto& a = w.a[q];
            a.resize(std::size_t(n) * k);
            for (std::size_t e = 0; e < a.size(); ++e) {
                a[e] = -dt * (cx * Dx[e] + cy * Dy[e])
                     + h * (cx * cx * Dxx[e] + T(2) * cx * cy * Dxy[e] + cy * cy * Dyy[e]);
            }
            for (I s = 0; s < n; ++s) a[std::size_t(s) * k + diag[s]] += T(1);
            w.pattern_of[q] = 0;
        }
        return w;
    }

    throw std::invalid_argument(std::string("build_streaming_weights: ") + to_string(scheme)
                                + " with " + to_string(stencils) + " stencils is not defined");
}

} // namespace lbm

#endif // LBM_SCHEMES_H
