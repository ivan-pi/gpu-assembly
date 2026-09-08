#ifndef RBF_OPS_H
#define RBF_OPS_H

// The RBF-FD operators for the device kernels, prototyped on the host:
// the PHS derivatives and the monomial derivatives of rbf_fd.f90, and
// three ways for a caller to say which operators the right-hand sides
// of a stencil carry. Compiles as plain C++20; RBF_HD expands to
// __host__ __device__ under nvcc.
//
//   1. Op codes at run time, one per column, as in the Fortran module:
//      the kernel switches on the code inside the fill loop.
//
//        Op ops[] = {Op::laplace, Op::dx, Op::value};
//        fill_rhs_dynamic<N, P, Q>(B, xs, ys, ops, xc, yc, 3, tid, nthreads);
//
//   2. Op codes at compile time, as a pack: NRHS = the size of the pack,
//      every column its own specialized loop, no dispatch.
//
//        fill_rhs_static<N, P, Q>(ops_list<Op::laplace, Op::dx, Op::value>{},
//                                 B, xs, ys, xc, yc, tid, nthreads);
//
//   3. Operator objects, composable: a tag per code, and sums and
//      multiples of them for operators the codes do not name, such as
//      an anisotropic Laplacian or a directional derivative.
//
//        auto ops = pack(2.0 * Dxx{} + 0.5 * Dyy{}, Dx{} * 0.6 + Dy{} * 0.8, Value{});
//        fill_rhs_objects<N, P, Q>(ops, B, xs, ys, xc, yc, tid, nthreads);
//
// All three produce the same columns: column j is L phi and L m at the
// evaluation point (xc[j], yc[j]) of the stencil-local frame, the
// derivatives at (0, 0) and the value at the interpolation point. The
// same PHS<Q>::apply and Poly<P>::apply serve all three, once as a
// template on the code and once through a switch that calls it, so the
// formulas exist in one place.
//
// The layout is that of rbf_operators.h: B is nt x nrhs column-major
// with ldb = nt = N + npoly(P), rows 0..N-1 the PHS part and the rest
// the monomials; the fill loops stride the rows by tid/nthreads as the
// kernels do.

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#ifdef __CUDACC__
#define RBF_HD __host__ __device__
#define RBF_UNROLL _Pragma("unroll")
#else
#define RBF_HD
#define RBF_UNROLL
#endif

namespace rbf_ops {

// The operator codes, with the values of OP_* in rbf_fd.f90
enum class Op : int { value = 0, dx = 1, dy = 2, dxx = 3, dxy = 4, dyy = 5, laplace = 6 };

// number of 2-D monomials of degree <= P
constexpr int npoly(int P) { return (P + 1) * (P + 2) / 2; }

// The smallest normal number, without <limits> in device code
template <typename T>
struct tiny;
template <>
struct tiny<double> {
    static constexpr double value = 2.2250738585072014e-308;
};
template <>
struct tiny<float> {
    static constexpr float value = 1.17549435e-38f;
};

// phi(r) = r^Q, Q odd, and its derivatives at the displacement
// d = (dx, dy) = evaluation point - node, the formulas of rbf_fd.f90:
//
//     phi_x   = Q r^(Q-2) dx           phi_xx  = Q r^(Q-4) ((Q-1) dx^2 + dy^2)
//     phi_xy  = Q (Q-2) r^(Q-4) dx dy  lap phi = Q^2 r^(Q-2)
//
// all zero at r = 0; the r^(Q-4) ones divide by r^2 clamped to the
// smallest normal number, which makes the 0/0 there a 0.
template <int Q>
struct PHS {
    static_assert(Q >= 3 && Q % 2 == 1, "2-D polyharmonic splines need an odd exponent >= 3");

    // r^(Q-2) from r^2
    template <typename T>
    RBF_HD static T rp(T r2) {
        T p = sqrt(r2);
        RBF_UNROLL
        for (int i = 0; i < (Q - 3) / 2; ++i)
            p *= r2;
        return p;
    }

    // The operator known at compile time
    template <Op op, typename T>
    RBF_HD static T apply(T dx, T dy) {
        constexpr T q = T(Q);
        const T r2 = dx * dx + dy * dy;
        const T p = rp(r2);
        if constexpr (op == Op::value) {
            return p * r2;
        } else if constexpr (op == Op::dx) {
            return q * p * dx;
        } else if constexpr (op == Op::dy) {
            return q * p * dy;
        } else if constexpr (op == Op::laplace) {
            return q * q * p;
        } else {
            const T s = p / fmax(r2, tiny<T>::value);  // r^(Q-4)
            if constexpr (op == Op::dxx) {
                return q * s * ((q - 1) * dx * dx + dy * dy);
            } else if constexpr (op == Op::dxy) {
                return q * (q - 2) * s * dx * dy;
            } else {
                static_assert(op == Op::dyy, "unknown operator");
                return q * s * (dx * dx + (q - 1) * dy * dy);
            }
        }
    }

    // The operator known at run time: the same code, one switch away
    template <typename T>
    RBF_HD static T apply(Op op, T dx, T dy) {
        switch (op) {
            case Op::value:
                return apply<Op::value>(dx, dy);
            case Op::dx:
                return apply<Op::dx>(dx, dy);
            case Op::dy:
                return apply<Op::dy>(dx, dy);
            case Op::dxx:
                return apply<Op::dxx>(dx, dy);
            case Op::dxy:
                return apply<Op::dxy>(dx, dy);
            case Op::dyy:
                return apply<Op::dyy>(dx, dy);
            default:
                return apply<Op::laplace>(dx, dy);
        }
    }
};

// The monomials x^i y^j of degree <= P by total degree (the order of
// PolyBasis<P> in rbf_operators.h and of rbf_fd.f90), with the
// operator applied: b[k * inc] = L m_k(x, y).
template <int P>
struct Poly {
    static constexpr int np = npoly(P);

    // x^i y^j from the power tables, zero for a negative exponent (whose
    // coefficient is zero anyway)
    template <typename T>
    RBF_HD static T mono(int i, int j, const T* X, const T* Y) {
        return (i < 0 || j < 0) ? T(0) : X[i] * Y[j];
    }

    template <Op op, typename T>
    RBF_HD static T term(int i, int j, const T* X, const T* Y) {
        if constexpr (op == Op::value) {
            return X[i] * Y[j];
        } else if constexpr (op == Op::dx) {
            return T(i) * mono(i - 1, j, X, Y);
        } else if constexpr (op == Op::dy) {
            return T(j) * mono(i, j - 1, X, Y);
        } else if constexpr (op == Op::dxx) {
            return T(i * (i - 1)) * mono(i - 2, j, X, Y);
        } else if constexpr (op == Op::dxy) {
            return T(i * j) * mono(i - 1, j - 1, X, Y);
        } else if constexpr (op == Op::dyy) {
            return T(j * (j - 1)) * mono(i, j - 2, X, Y);
        } else {
            static_assert(op == Op::laplace, "unknown operator");
            return T(i * (i - 1)) * mono(i - 2, j, X, Y) + T(j * (j - 1)) * mono(i, j - 2, X, Y);
        }
    }

    template <typename T>
    RBF_HD static T term(Op op, int i, int j, const T* X, const T* Y) {
        switch (op) {
            case Op::value:
                return term<Op::value>(i, j, X, Y);
            case Op::dx:
                return term<Op::dx>(i, j, X, Y);
            case Op::dy:
                return term<Op::dy>(i, j, X, Y);
            case Op::dxx:
                return term<Op::dxx>(i, j, X, Y);
            case Op::dxy:
                return term<Op::dxy>(i, j, X, Y);
            case Op::dyy:
                return term<Op::dyy>(i, j, X, Y);
            default:
                return term<Op::laplace>(i, j, X, Y);
        }
    }

    template <typename T>
    RBF_HD static void powers(T x, T y, T* X, T* Y) {
        X[0] = Y[0] = T(1);
        RBF_UNROLL
        for (int i = 1; i <= P; ++i) {
            X[i] = X[i - 1] * x;
            Y[i] = Y[i - 1] * y;
        }
    }

    // F(i, j, X, Y) gives the term; the two apply's below pass term<op>
    // or the run-time term
    template <typename T, typename F>
    RBF_HD static void apply_with(F f, T x, T y, T* b, int inc = 1) {
        T X[P + 1], Y[P + 1];
        powers(x, y, X, Y);
        int k = 0;
        RBF_UNROLL
        for (int d = 0; d <= P; ++d)
            RBF_UNROLL
        for (int j = 0; j <= d; ++j)
            b[(k++) * inc] = f(d - j, j, X, Y);
    }

    template <Op op, typename T>
    RBF_HD static void apply(T x, T y, T* b, int inc = 1) {
        apply_with([](int i, int j, const T* X, const T* Y) { return term<op>(i, j, X, Y); }, x, y,
                   b, inc);
    }

    template <typename T>
    RBF_HD static void apply(Op op, T x, T y, T* b, int inc = 1) {
        apply_with([op](int i, int j, const T* X, const T* Y) { return term(op, i, j, X, Y); }, x,
                   y, b, inc);
    }
};

// ---------------------------------------------------------------------
// 1. Op codes at run time
// ---------------------------------------------------------------------

template <int N, int P, int Q, typename T>
RBF_HD void fill_rhs_dynamic(T* B,
                             const T* xs,
                             const T* ys,
                             const Op* ops,
                             const T* xc,
                             const T* yc,
                             int nrhs,
                             int tid,
                             int nthreads) {
    constexpr int ldb = N + npoly(P);
    for (int col = 0; col < nrhs; ++col) {
        const Op op = ops[col];
        const T xe = xc[col], ye = yc[col];
        for (int k = tid; k < N; k += nthreads)
            B[k + col * ldb] = PHS<Q>::apply(op, xe - xs[k], ye - ys[k]);
    }
    for (int col = tid; col < nrhs; col += nthreads)
        Poly<P>::apply(ops[col], xc[col], yc[col], &B[N + col * ldb], 1);
}

// ---------------------------------------------------------------------
// 2. Op codes at compile time
// ---------------------------------------------------------------------

template <Op... ops>
struct ops_list {
    static constexpr int size = sizeof...(ops);
};

namespace detail {

template <int N, int P, int Q, Op op, typename T>
RBF_HD void fill_column(T* b, const T* xs, const T* ys, T xe, T ye, int tid, int nthreads) {
    for (int k = tid; k < N; k += nthreads)
        b[k] = PHS<Q>::template apply<op>(xe - xs[k], ye - ys[k]);
    // one thread per column for the np monomial rows, as rbf_operators.h
    if (tid == 0) Poly<P>::template apply<op>(xe, ye, b + N, 1);
}

}  // namespace detail

template <int N, int P, int Q, Op... ops, typename T>
RBF_HD void fill_rhs_static(ops_list<ops...>,
                            T* B,
                            const T* xs,
                            const T* ys,
                            const T* xc,
                            const T* yc,
                            int tid,
                            int nthreads) {
    constexpr int ldb = N + npoly(P);
    int col = 0;
    // a fold over the pack: column col gets operator ops[col]
    ((detail::fill_column<N, P, Q, ops>(B + col * ldb, xs, ys, xc[col], yc[col], tid, nthreads),
      ++col),
     ...);
}

// ---------------------------------------------------------------------
// 3. Operator objects, composable
// ---------------------------------------------------------------------
//
// An operator object answers phs<Q>(dx, dy) and poly<P>(i, j, X, Y);
// Basic<op> wraps a code, Scaled and Sum combine objects. The
// arithmetic operators build the expression, so 2 * Dxx{} + Dyy{} is an
// object of type Sum<Scaled<Basic<dxx>>, Basic<dyy>>, trivially
// copyable, a kernel argument.

template <Op op>
struct Basic {
    template <int Q, typename T>
    RBF_HD T phs(T dx, T dy) const {
        return PHS<Q>::template apply<op>(dx, dy);
    }
    template <int P, typename T>
    RBF_HD T poly(int i, int j, const T* X, const T* Y) const {
        return Poly<P>::template term<op>(i, j, X, Y);
    }
};

using Value = Basic<Op::value>;
using Dx = Basic<Op::dx>;
using Dy = Basic<Op::dy>;
using Dxx = Basic<Op::dxx>;
using Dxy = Basic<Op::dxy>;
using Dyy = Basic<Op::dyy>;
using Laplace = Basic<Op::laplace>;

template <typename A>
struct Scaled {
    A a;
    double c;
    template <int Q, typename T>
    RBF_HD T phs(T dx, T dy) const {
        return T(c) * a.template phs<Q>(dx, dy);
    }
    template <int P, typename T>
    RBF_HD T poly(int i, int j, const T* X, const T* Y) const {
        return T(c) * a.template poly<P>(i, j, X, Y);
    }
};

template <typename A, typename B>
struct Sum {
    A a;
    B b;
    template <int Q, typename T>
    RBF_HD T phs(T dx, T dy) const {
        return a.template phs<Q>(dx, dy) + b.template phs<Q>(dx, dy);
    }
    template <int P, typename T>
    RBF_HD T poly(int i, int j, const T* X, const T* Y) const {
        return a.template poly<P>(i, j, X, Y) + b.template poly<P>(i, j, X, Y);
    }
};

template <typename A>
struct is_operator : std::false_type {};
template <Op op>
struct is_operator<Basic<op>> : std::true_type {};
template <typename A>
struct is_operator<Scaled<A>> : std::true_type {};
template <typename A, typename B>
struct is_operator<Sum<A, B>> : std::true_type {};

template <typename A, typename = std::enable_if_t<is_operator<A>::value>>
RBF_HD constexpr Scaled<A> operator*(double c, A a) {
    return {a, c};
}
template <typename A, typename = std::enable_if_t<is_operator<A>::value>>
RBF_HD constexpr Scaled<A> operator*(A a, double c) {
    return {a, c};
}
template <typename A,
          typename B,
          typename = std::enable_if_t<is_operator<A>::value && is_operator<B>::value>>
RBF_HD constexpr Sum<A, B> operator+(A a, B b) {
    return {a, b};
}

// A tuple that device code can hold: the operators, by position
template <typename... Ops>
struct Pack;
template <>
struct Pack<> {};
template <typename Head, typename... Tail>
struct Pack<Head, Tail...> {
    Head head;
    Pack<Tail...> tail;
    static constexpr int size = 1 + sizeof...(Tail);
};

template <typename... Ops>
RBF_HD constexpr Pack<Ops...> pack(Ops... ops) {
    if constexpr (sizeof...(Ops) == 0) {
        return {};
    } else {
        return pack_impl(ops...);
    }
}
template <typename Head, typename... Tail>
RBF_HD constexpr Pack<Head, Tail...> pack_impl(Head head, Tail... tail) {
    return {head, pack(tail...)};
}

namespace detail {

template <int N, int P, int Q, typename A, typename T>
RBF_HD void fill_column(const A& a,
                        T* b,
                        const T* xs,
                        const T* ys,
                        T xe,
                        T ye,
                        int tid,
                        int nthreads) {
    for (int k = tid; k < N; k += nthreads)
        b[k] = a.template phs<Q>(xe - xs[k], ye - ys[k]);
    if (tid == 0)
        Poly<P>::apply_with(
            [&a](int i, int j, const T* X, const T* Y) { return a.template poly<P>(i, j, X, Y); },
            xe, ye, b + N, 1);
}

template <int N, int P, int Q, typename T>
RBF_HD void fill_pack(const Pack<>&, T*, const T*, const T*, const T*, const T*, int, int) {}

template <int N, int P, int Q, typename Head, typename... Tail, typename T>
RBF_HD void fill_pack(const Pack<Head, Tail...>& ops,
                      T* B,
                      const T* xs,
                      const T* ys,
                      const T* xc,
                      const T* yc,
                      int tid,
                      int nthreads) {
    constexpr int ldb = N + npoly(P);
    fill_column<N, P, Q>(ops.head, B, xs, ys, xc[0], yc[0], tid, nthreads);
    fill_pack<N, P, Q>(ops.tail, B + ldb, xs, ys, xc + 1, yc + 1, tid, nthreads);
}

}  // namespace detail

template <int N, int P, int Q, typename... Ops, typename T>
RBF_HD void fill_rhs_objects(const Pack<Ops...>& ops,
                             T* B,
                             const T* xs,
                             const T* ys,
                             const T* xc,
                             const T* yc,
                             int tid,
                             int nthreads) {
    detail::fill_pack<N, P, Q>(ops, B, xs, ys, xc, yc, tid, nthreads);
}

}  // namespace rbf_ops

#endif  // RBF_OPS_H
