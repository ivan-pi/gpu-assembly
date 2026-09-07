// rbf_assembly.h -- host RBF-FD assembly: the assembler contract and a
//                   reference implementation
//
// The streaming-scheme recipes in lbm_schemes.h need an assembler, any
// callable with
//
//     weights = assemble(xc, yc, ja, functionals)
//
//     xc, yc        stencil centres, length n (the nodes themselves for
//                   arrival-point stencils, the departure points for
//                   departure-point stencils)
//     ja            n*k column indices, row-major; k = ja.size() / n
//     functionals   what to evaluate, each with a point in the local
//                   frame of the stencil centre
//     weights       one n*k array per functional, laid out like ja
//
// There are two assembly APIs by design. The device one
// (rbf_operators.h) is static: stencil size, degree and operator set are
// template parameters, because a GPU pays for specialisation once and
// gains throughput on every stencil. The host one is dynamic: those are
// run-time values, because a CPU has the latency to branch once per
// column and gains flexibility. An existing host API is adapted to the
// contract above with a lambda; HostAssembler below is a self-contained
// stand-in for it, written for clarity rather than speed (dense LU per
// stencil, no reuse of the factorisation across functionals).

#ifndef RBF_ASSEMBLY_H
#define RBF_ASSEMBLY_H

#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rbf_periodic.h"

namespace rbf {

// Linear functionals applied to the interpolant: the value at a point,
// or a derivative at a point. Semi-Lagrangian streaming needs `value`
// at the departure point; Lax-Wendroff needs the derivatives at the node.
enum class functional { value, dx, dy, dxx, dxy, dyy, laplace };

template<typename T>
struct Functional {
    functional op = functional::value;
    T x = 0, y = 0;   // evaluation point, local frame
};

// number of 2-D monomials of degree <= P
constexpr int npoly(int P) { return (P + 1) * (P + 2) / 2; }

namespace detail {

// --- polyharmonic spline phi(r) = r^m, m odd >= 3, in terms of r^2 ---
//
// With s = m r^(m-2) and t = m (m-2) r^(m-4):
//
//     phi_x  = s x         phi_xx = s + t x^2       phi_xy = t x y
//     phi_y  = s y         phi_yy = s + t y^2       lap    = m s
//
// Every derivative tends to 0 as r -> 0 for m >= 3, which is where
// r^(m-4) would be singular for m = 3, so the callers guard r2 == 0.

template<typename T>
T rpow(int m, T r2) {              // r^m for integer m >= 0, from r^2
    T p = T(1);
    for (int i = 0; i < m / 2; ++i) p *= r2;
    return (m % 2) ? p * std::sqrt(r2) : p;
}

template<typename T>
T phs(int m, T r2) { return rpow(m, r2); }

template<typename T>
T phs_s(int m, T r2) { return T(m) * rpow(m - 2, r2); }

template<typename T>
T phs_t(int m, T r2) { return T(m) * T(m - 2) * rpow(m - 2, r2) / r2; }

// --- monomials x^a y^c, a + c = d, for d = 0..P and c = 0..d ---
//
// Callers pass power tables X[i] = x^i, Y[i] = y^i (X[0] = Y[0] = 1)
// and a functor f(a, c) -> value; visit_monomials applies it in order.

template<class F>
void visit_monomials(int P, F&& f) {
    for (int d = 0; d <= P; ++d)
        for (int c = 0; c <= d; ++c) f(d - c, c);
}

template<typename T>
void powers(int P, T x, T* X) {
    X[0] = T(1);
    for (int i = 1; i <= P; ++i) X[i] = X[i-1] * x;
}

// Dense LU with partial pivoting, column-major A (n x n, lda = n),
// B (n x nrhs, ldb = n) overwritten with the solution. Returns 0 or the
// 1-based index of the first zero pivot, as LAPACK's dgesv would.
template<typename T>
int lu_solve(int n, T* A, int* piv, T* B, int nrhs) {
    for (int c = 0; c < n; ++c) {
        int p = c;
        T best = std::abs(A[c + c * n]);
        for (int r = c + 1; r < n; ++r) {
            const T v = std::abs(A[r + c * n]);
            if (v > best) { best = v; p = r; }
        }
        piv[c] = p;
        if (best == T(0)) return c + 1;
        if (p != c) {
            for (int j = 0; j < n; ++j) std::swap(A[c + j * n], A[p + j * n]);
            for (int j = 0; j < nrhs; ++j) std::swap(B[c + j * n], B[p + j * n]);
        }
        const T inv = T(1) / A[c + c * n];
        for (int r = c + 1; r < n; ++r) A[r + c * n] *= inv;
        for (int j = c + 1; j < n; ++j) {
            const T a = A[c + j * n];
            if (a == T(0)) continue;
            for (int r = c + 1; r < n; ++r) A[r + j * n] -= A[r + c * n] * a;
        }
    }
    for (int j = 0; j < nrhs; ++j) {
        T* b = B + j * n;
        for (int c = 0; c < n; ++c) {          // forward, unit lower
            const T bc = b[c];
            if (bc == T(0)) continue;
            for (int r = c + 1; r < n; ++r) b[r] -= A[r + c * n] * bc;
        }
        for (int c = n - 1; c >= 0; --c) {     // backward, upper
            b[c] /= A[c + c * n];
            const T bc = b[c];
            for (int r = 0; r < c; ++r) b[r] -= A[r + c * n] * bc;
        }
    }
    return 0;
}

} // namespace detail

// Fill the augmented collocation matrix of one stencil, nt x nt,
// col-major, nt = k + npoly(P):
//
//     [ A    Pm ]     A(i,j)  = phi(|| p_j - p_i ||)
//     [ Pm^T  0 ]     Pm(i,l) = monomial_l(p_i)
//
template<typename T>
void fill_collocation_matrix(int m, int P, int k, const T* xs, const T* ys, T* M) {
    using namespace detail;
    const int nt = k + npoly(P);
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < k; ++i) {
            const T dx = xs[j] - xs[i], dy = ys[j] - ys[i];
            M[i + j * nt] = phs(m, dx * dx + dy * dy);
        }
    std::vector<T> X(P + 1), Y(P + 1);
    for (int i = 0; i < k; ++i) {
        powers(P, xs[i], X.data());
        powers(P, ys[i], Y.data());
        int l = k;
        visit_monomials(P, [&](int a, int c) {
            M[i + l * nt] = M[l + i * nt] = X[a] * Y[c];
            ++l;
        });
    }
    for (int j = k; j < nt; ++j)
        for (int i = k; i < nt; ++i) M[i + j * nt] = T(0);
}

// Fill one right-hand side, length nt, for functional L at (ex, ey):
// rows 0..k-1 are L applied to phi(| p - p_i |) as a function of p, rows
// k.. are L applied to the monomials. The functional is chosen once per
// column; each case is the plain formula for that functional.
template<typename T>
void fill_rhs_column(functional L, int m, int P, int k,
                     const T* xs, const T* ys, T ex, T ey, T* b) {
    using namespace detail;
    std::vector<T> X(P + 1), Y(P + 1);
    powers(P, ex, X.data());
    powers(P, ey, Y.data());

    // rows for the RBF part, then the monomials
    auto fill = [&](auto phi_part, auto mono_part) {
        for (int i = 0; i < k; ++i) {
            const T x = ex - xs[i], y = ey - ys[i], r2 = x * x + y * y;
            b[i] = (L != functional::value && r2 == T(0)) ? T(0) : phi_part(x, y, r2);
        }
        int l = k;
        visit_monomials(P, [&](int a, int c) { b[l++] = mono_part(a, c); });
    };
    // x^a y^c and its derivatives from the power tables; negative powers are 0
    auto mono = [&](int a, int c) { return (a < 0 || c < 0) ? T(0) : X[a] * Y[c]; };

    switch (L) {
    case functional::value:
        fill([&](T, T, T r2)   { return phs(m, r2); },
             [&](int a, int c) { return mono(a, c); });
        break;
    case functional::dx:
        fill([&](T x, T, T r2) { return phs_s(m, r2) * x; },
             [&](int a, int c) { return T(a) * mono(a - 1, c); });
        break;
    case functional::dy:
        fill([&](T, T y, T r2) { return phs_s(m, r2) * y; },
             [&](int a, int c) { return T(c) * mono(a, c - 1); });
        break;
    case functional::dxx:
        fill([&](T x, T, T r2) { return phs_s(m, r2) + phs_t(m, r2) * x * x; },
             [&](int a, int c) { return T(a) * T(a - 1) * mono(a - 2, c); });
        break;
    case functional::dyy:
        fill([&](T, T y, T r2) { return phs_s(m, r2) + phs_t(m, r2) * y * y; },
             [&](int a, int c) { return T(c) * T(c - 1) * mono(a, c - 2); });
        break;
    case functional::dxy:
        fill([&](T x, T y, T r2) { return phs_t(m, r2) * x * y; },
             [&](int a, int c)   { return T(a) * T(c) * mono(a - 1, c - 1); });
        break;
    case functional::laplace:
        fill([&](T, T, T r2)   { return T(m) * phs_s(m, r2); },
             [&](int a, int c) { return T(a) * T(a - 1) * mono(a - 2, c)
                                      + T(c) * T(c - 1) * mono(a, c - 2); });
        break;
    }
}

// Reference host assembler; see the file header for the contract.
template<typename T = double, typename I = std::int32_t>
class HostAssembler {
public:
    using value_type = T;
    using index_type = I;

    // x, y: the cloud that stencil indices refer to; box: periodic
    // wrap-around for the displacements, or nullptr
    HostAssembler(std::span<const T> x, std::span<const T> y,
                  int poly_degree, int phs_exponent = 3,
                  const PeriodicBox<T>* box = nullptr)
        : x_(x), y_(y), P_(poly_degree), m_(phs_exponent), box_(box) {
        if (x_.size() != y_.size()) throw std::invalid_argument("HostAssembler: x/y size mismatch");
        if (P_ < 0) throw std::invalid_argument("HostAssembler: negative polynomial degree");
        if (m_ < 3 || m_ % 2 == 0) throw std::invalid_argument("HostAssembler: PHS exponent must be odd and >= 3");
    }

    int poly_degree() const { return P_; }
    int phs_exponent() const { return m_; }

    std::vector<std::vector<T>>
    operator()(std::span<const T> xc, std::span<const T> yc,
               std::span<const I> ja, std::span<const Functional<T>> L) const
    {
        const std::size_t n = xc.size();
        if (n == 0 || yc.size() != n || ja.size() % n != 0)
            throw std::invalid_argument("HostAssembler: centres/ja sizes do not match");
        const int k = static_cast<int>(ja.size() / n);
        const int nrhs = static_cast<int>(L.size());
        const int nt = k + npoly(P_);
        if (k < npoly(P_))
            throw std::invalid_argument("HostAssembler: stencil size " + std::to_string(k)
                                        + " below the polynomial count " + std::to_string(npoly(P_)));

        std::vector<std::vector<T>> out(nrhs, std::vector<T>(n * k));
        std::size_t singular = 0;

        #pragma omp parallel reduction(+:singular)
        {
            std::vector<T> A(std::size_t(nt) * nt), B(std::size_t(nt) * nrhs), xs(k), ys(k);
            std::vector<int> piv(nt);

            #pragma omp for schedule(static)
            for (std::ptrdiff_t s = 0; s < static_cast<std::ptrdiff_t>(n); ++s) {
                const T x0 = xc[s], y0 = yc[s];
                for (int j = 0; j < k; ++j) {
                    const I c = ja[s * k + j];
                    T dx = x_[c] - x0, dy = y_[c] - y0;
                    if (box_) { dx = box_->wrap_dx(dx); dy = box_->wrap_dy(dy); }
                    xs[j] = dx; ys[j] = dy;
                }
                fill_collocation_matrix(m_, P_, k, xs.data(), ys.data(), A.data());
                for (int r = 0; r < nrhs; ++r)
                    fill_rhs_column(L[r].op, m_, P_, k, xs.data(), ys.data(),
                                    L[r].x, L[r].y, B.data() + std::size_t(r) * nt);
                if (detail::lu_solve(nt, A.data(), piv.data(), B.data(), nrhs)) {
                    ++singular;
                    continue;
                }
                for (int r = 0; r < nrhs; ++r)
                    for (int j = 0; j < k; ++j)
                        out[r][s * k + j] = B[std::size_t(r) * nt + j];
            }
        }
        if (singular)
            throw std::runtime_error("HostAssembler: " + std::to_string(singular) + " singular stencil(s)");
        return out;
    }

private:
    std::span<const T> x_, y_;
    int P_, m_;
    const PeriodicBox<T>* box_;
};

} // namespace rbf

#endif // RBF_ASSEMBLY_H
