// rbf_rbffd.h -- RBF-FD building blocks shared by the host and the CUDA
//                assembly: PHS kernels, the monomial basis, the fill of the
//                augmented collocation matrix and of right-hand sides.
//
// Plain C++ when compiled by a host compiler; under nvcc every function is
// also __device__, so rbf_operators.h (cuSolverDx kernels) and
// rbf_assembly.h (host reference assembly) fill their systems with the
// same code and differ only in the dense solver.
//
// Local frame: every stencil is expressed relative to its centre
// (xs, ys are neighbour coordinates minus the centre), and functionals
// are evaluated at points given in that frame.

#ifndef RBF_RBFFD_H
#define RBF_RBFFD_H

#include <cmath>

#if defined(__CUDACC__)
#define RBF_HD __host__ __device__
#define RBF_UNROLL _Pragma("unroll")
#elif defined(__clang__)
#define RBF_HD
#define RBF_UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define RBF_HD
#define RBF_UNROLL _Pragma("GCC unroll 16")
#else
#define RBF_HD
#define RBF_UNROLL
#endif

namespace rbf_operators {

// Polyharmonic spline phi(r) = r^Q, Q odd, given r^2
template<int Q>
struct PHS {
    static_assert(Q >= 3 && Q % 2 == 1,
                  "2-D polyharmonic splines need an odd exponent >= 3");
    template<typename T>
    RBF_HD inline T operator()(T r2) const {
        using std::sqrt;
        const T r = sqrt(r2);
        T p = r;
        RBF_UNROLL
        for (int i = 0; i < (Q - 1) / 2; ++i) p *= r2;
        return p;
    }
};

template<> struct PHS<3> {
    template<typename T>
    RBF_HD inline T operator()(T r2) const {
        using std::sqrt;
        return r2 * sqrt(r2);
    }
};

template<> struct PHS<5> {
    template<typename T>
    RBF_HD inline T operator()(T r2) const {
        using std::sqrt;
        return r2 * r2 * sqrt(r2);
    }
};

template<> struct PHS<7> {
    template<typename T>
    RBF_HD inline T operator()(T r2) const {
        using std::sqrt;
        const T r4 = r2 * r2;
        return r4 * r2 * sqrt(r2);
    }
};

// Linear functionals applied to the interpolant, in the stencil-local
// frame: the value at (x, y), or a derivative at (x, y). Semi-Lagrangian
// streaming needs `value` at the departure point; Lax-Wendroff streaming
// needs the derivatives at the centre.
enum class functional {
    value, dx, dy, dxx, dxy, dyy, laplace
};

// L applied to phi(| p - p_k |) as a function of p, evaluated at p = (x, y)
// relative to p_k; r2 = x*x + y*y. All derivatives vanish at r = 0 for
// Q >= 3, which is where r^(Q-4) would be singular for Q = 3.
template<int Q, typename T>
RBF_HD inline T phs_apply(functional L, T x, T y, T r2) {
    using std::sqrt;
    if (L == functional::value) return PHS<Q>{}(r2);
    if (r2 == T(0)) return T(0);
    const T r = sqrt(r2);
    T rq2 = T(1);                      // r^(Q-2)
    for (int i = 0; i < Q - 2; ++i) rq2 *= r;
    const T rq4 = rq2 / r2;            // r^(Q-4)
    constexpr T q = T(Q), qq2 = T(Q) * T(Q - 2);
    switch (L) {
        case functional::dx:      return q * rq2 * x;
        case functional::dy:      return q * rq2 * y;
        case functional::dxx:     return q * rq2 + qq2 * rq4 * x * x;
        case functional::dyy:     return q * rq2 + qq2 * rq4 * y * y;
        case functional::dxy:     return qq2 * rq4 * x * y;
        case functional::laplace: return q * q * rq2;
        default:                  return T(0);
    }
}

// number of 2-D monomials of degree <= P
constexpr int npoly(int P) { return (P + 1) * (P + 2) / 2; }

// Monomials x^(d-j) y^j, d = 0..P, j = 0..d, in that order
template<int P> struct PolyBasis {
    static constexpr int np = npoly(P);

    template<typename T>
    RBF_HD inline void operator()(T x, T y, T* b, int inc = 1) const {
        T X[P+1], Y[P+1];
        X[0] = Y[0] = T(1);
        RBF_UNROLL
        for (int i = 1; i <= P; ++i) {
            X[i] = X[i-1]*x;
            Y[i] = Y[i-1]*y;
        }
        int k = 0;
        RBF_UNROLL
        for (int d = 0; d <= P; ++d)
            RBF_UNROLL
            for (int j = 0; j <= d; ++j)
                b[(k++)*inc] = X[d-j] * Y[j];
    }

    // L applied to each monomial, evaluated at (x, y)
    template<typename T>
    RBF_HD inline void apply(functional L, T x, T y, T* b, int inc = 1) const {
        if (L == functional::value) { (*this)(x, y, b, inc); return; }
        T X[P+1], Y[P+1];
        X[0] = Y[0] = T(1);
        for (int i = 1; i <= P; ++i) {
            X[i] = X[i-1]*x;
            Y[i] = Y[i-1]*y;
        }
        // x^a y^c and its derivatives; pw(v, e) = v^e with v^negative = 0
        auto pw = [&](const T* V, int e) { return e < 0 ? T(0) : V[e]; };
        int k = 0;
        for (int d = 0; d <= P; ++d) {
            for (int j = 0; j <= d; ++j) {
                const int a = d - j, c = j;
                const T fa = T(a), fc = T(c);
                T v;
                switch (L) {
                    case functional::dx:  v = fa * pw(X,a-1) * pw(Y,c); break;
                    case functional::dy:  v = fc * pw(X,a) * pw(Y,c-1); break;
                    case functional::dxx: v = fa * T(a-1) * pw(X,a-2) * pw(Y,c); break;
                    case functional::dyy: v = fc * T(c-1) * pw(X,a) * pw(Y,c-2); break;
                    case functional::dxy: v = fa * fc * pw(X,a-1) * pw(Y,c-1); break;
                    case functional::laplace:
                        v = fa * T(a-1) * pw(X,a-2) * pw(Y,c)
                          + fc * T(c-1) * pw(X,a) * pw(Y,c-2); break;
                    default: v = T(0);
                }
                b[(k++)*inc] = v;
            }
        }
    }
};

template<>
struct PolyBasis<2> {
    static constexpr int np = npoly(2); // np = 6
    template<typename T>
    RBF_HD void operator()(T x, T y, T* b, int inc = 1) const {
        b[0*inc] = T(1);
        b[1*inc] = x;
        b[2*inc] = y;
        b[3*inc] = x*x;
        b[4*inc] = x*y;
        b[5*inc] = y*y;
    }
    template<typename T>
    RBF_HD void apply(functional L, T x, T y, T* b, int inc = 1) const {
        switch (L) {
            case functional::value: (*this)(x, y, b, inc); return;
            case functional::dx:
                b[0*inc] = 0; b[1*inc] = 1; b[2*inc] = 0;
                b[3*inc] = 2*x; b[4*inc] = y; b[5*inc] = 0; return;
            case functional::dy:
                b[0*inc] = 0; b[1*inc] = 0; b[2*inc] = 1;
                b[3*inc] = 0; b[4*inc] = x; b[5*inc] = 2*y; return;
            case functional::dxx:
                b[0*inc] = 0; b[1*inc] = 0; b[2*inc] = 0;
                b[3*inc] = 2; b[4*inc] = 0; b[5*inc] = 0; return;
            case functional::dyy:
                b[0*inc] = 0; b[1*inc] = 0; b[2*inc] = 0;
                b[3*inc] = 0; b[4*inc] = 0; b[5*inc] = 2; return;
            case functional::dxy:
                b[0*inc] = 0; b[1*inc] = 0; b[2*inc] = 0;
                b[3*inc] = 0; b[4*inc] = 1; b[5*inc] = 0; return;
            case functional::laplace:
                b[0*inc] = 0; b[1*inc] = 0; b[2*inc] = 0;
                b[3*inc] = 2; b[4*inc] = 0; b[5*inc] = 2; return;
        }
    }
};

// Fill the RBF-FD approximation matrix
//
// M is nt x nt, col-major, ldm = nt = N + npoly(P)
//
//     [ A    Pm ]     A(k,col) = phi(|| p_col - p_k ||^2)
//     [ Pm^T  0 ]     Pm(k,i)  = monomial_i(p_k)
//
// Work is strided over [tid, tid + nthreads, ...) so the same function
// serves a serial host call (tid = 0, nthreads = 1) and a thread block.
template<int N, int P, int Q = 3, typename T>
RBF_HD void fill_matrix(T* M, const T* xs, const T* ys,
                        int tid, int nthreads) {

    constexpr PolyBasis<P> monomials{};
    constexpr int NP = decltype(monomials)::np;
    constexpr int ldm = N + NP;

    constexpr PHS<Q> phi{};

    // RBF block: consecutive threads take consecutive rows
    for (int col = 0; col < N; ++col) {
        const T xc = xs[col], yc = ys[col];
        for (int k = tid; k < N; k += nthreads) {
            const T dx = xc - xs[k];
            const T dy = yc - ys[k];
            M[k + col*ldm] = phi(dx*dx + dy*dy);
        }
    }

    // Both polynomial blocks in one pass. Row k of the upper-right block
    // and column k of the lower-left block hold the same NP basis values.
    for (int k = tid; k < N; k += nthreads) {
        monomials(xs[k], ys[k], &M[k + N*ldm], ldm);  // upper-right
        monomials(xs[k], ys[k], &M[N + k*ldm], 1);    // lower-left
    }

    // Zero block
    for (int k = tid; k < NP*NP; k +=nthreads) {
        const int row = k % NP;
        const int col = k / NP;
        M[(N+row)+(N+col)*ldm] = T(0);
    }

}

// B is nt x NRHS, col-major, ldb = nt. Column j is the right-hand side for
// evaluation point (xc[j], yc[j]) in the same local frame as xs/ys.
template<int N, int P, int NRHS, int Q = 3, typename T>
RBF_HD void fill_rhs(T* B, const T* xs, const T* ys,
                     const T* xc, const T* yc,
                     int tid, int nthreads) {

    constexpr PolyBasis<P> monomials{};
    constexpr int NP = decltype(monomials)::np;
    constexpr int ldb = N + NP;

    constexpr PHS<Q> phi{};

    // RBF rows, one thread per row
    for (int col = 0; col < NRHS; ++col) {
        const T xe = xc[col], ye = yc[col];
        for (int k = tid; k < N; k += nthreads) {
            const T dx = xe - xs[k], dy = ye - ys[k];
            B[k + col*ldb] = phi(dx*dx + dy*dy);
        }
    }

    // Polynomial rows, one thread per column, NP contiguous entries each
    for (int col = tid; col < NRHS; col += nthreads) {
        monomials(xc[col], yc[col], &B[N + col*ldb], 1);
    }

}

// General right-hand sides: column j is functional L[j] evaluated at
// (xc[j], yc[j]). With L[j] == value for all j this is fill_rhs.
template<int N, int P, int Q = 3, typename T>
RBF_HD void fill_rhs_functionals(T* B, const T* xs, const T* ys,
                                 const functional* L,
                                 const T* xc, const T* yc, int nrhs,
                                 int tid, int nthreads) {

    constexpr PolyBasis<P> monomials{};
    constexpr int NP = decltype(monomials)::np;
    constexpr int ldb = N + NP;

    for (int col = 0; col < nrhs; ++col) {
        const T xe = xc[col], ye = yc[col];
        for (int k = tid; k < N; k += nthreads) {
            const T dx = xe - xs[k], dy = ye - ys[k];
            B[k + col*ldb] = phs_apply<Q>(L[col], dx, dy, dx*dx + dy*dy);
        }
    }

    for (int col = tid; col < nrhs; col += nthreads) {
        monomials.apply(L[col], xc[col], yc[col], &B[N + col*ldb], 1);
    }
}

} // namespace rbf_operators

#endif // RBF_RBFFD_H
