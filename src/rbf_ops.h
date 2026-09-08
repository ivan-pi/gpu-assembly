#ifndef RBF_OPS_H
#define RBF_OPS_H

// The RBF-FD operators for the device kernels, prototyped on the host:
// what rbf_fd.f90 does with its OP_* codes, as C++ that compiles for
// the host and, under nvcc, for the device (RBF_HD).
//
// An operator is a partial derivative D<A, B> = d^A/dx^A d^B/dy^B of
// order A + B <= 2, or a linear combination of those (Scaled, Sum,
// built by * and +). The value is D<0, 0>, the Laplacian is
// D<2, 0> + D<0, 2>. Every operator answers two questions, on the PHS
// phi(r) = r^q at a displacement d = (dx, dy) and on a monomial
// x^i y^j at a point whose powers are tabulated:
//
//     op.phs<Q>(dx, dy)          op.poly(i, j, X, Y)
//
// and the column of a right-hand side is those two, rows 0..N-1 the
// PHS part and the rest the monomials by total degree, the order of
// PolyBasis<P> in rbf_operators.h and of rbf_fd.f90.
//
// Three ways for a caller to say which operators the right-hand sides
// carry, all filling the same columns for the same evaluation points
// (xc[j], yc[j]) of the stencil-local frame:
//
//   1. codes at run time, as in the Fortran module:
//
//        Op ops[] = {Op::laplace, Op::dx, Op::value};
//        fill_rhs_dynamic<N, P, Q>(B, xs, ys, ops, xc, yc, 3, tid, nthreads);
//
//   2. codes at compile time, NRHS the size of the pack, each column
//      its own loop:
//
//        fill_rhs_static<N, P, Q>(ops_list<Op::laplace, Op::dx, Op::value>{},
//                                 B, xs, ys, xc, yc, tid, nthreads);
//
//   3. operator objects, for what the codes do not name:
//
//        auto ops = pack(2.0 * Dxx{} + 0.5 * Dyy{}, 0.6 * Dx{} + 0.8 * Dy{}, Value{});
//        fill_rhs_objects<N, P, Q>(ops, B, xs, ys, xc, yc, tid, nthreads);
//
// 2 is 3 with the codes mapped to their tags; 1 is one switch on the
// code that calls 3 for the tag. B is nt x nrhs column-major with
// ldb = nt = N + npoly(P); the loops stride the rows by tid/nthreads
// as the kernels do.

#include <cmath>
#include <type_traits>

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

// ---------------------------------------------------------------------
// The basis
// ---------------------------------------------------------------------

// phi(r) = r^Q, Q odd: the powers of r the derivatives are made of
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

    // r^(Q-4) from r^2, the 0/0 at r = 0 clamped to 0: the numerator it
    // multiplies vanishes there for Q >= 3
    template <typename T>
    RBF_HD static T rs(T r2) {
        return rp(r2) / fmax(r2, tiny<T>::value);
    }
};

// The monomials x^i y^j of degree <= P by total degree
template <int P>
struct Poly {
    static constexpr int np = npoly(P);

    template <typename T>
    RBF_HD static void powers(T x, T y, T* X, T* Y) {
        X[0] = Y[0] = T(1);
        RBF_UNROLL
        for (int i = 1; i <= P; ++i) {
            X[i] = X[i - 1] * x;
            Y[i] = Y[i - 1] * y;
        }
    }

    // b[k * inc] = op applied to the k-th monomial at (x, y)
    template <typename A, typename T>
    RBF_HD static void apply(const A& op, T x, T y, T* b, int inc = 1) {
        T X[P + 1], Y[P + 1];
        powers(x, y, X, Y);
        int k = 0;
        RBF_UNROLL
        for (int d = 0; d <= P; ++d)
            RBF_UNROLL
        for (int j = 0; j <= d; ++j)
            b[(k++) * inc] = op.poly(d - j, j, X, Y);
    }
};

// ---------------------------------------------------------------------
// The operators
// ---------------------------------------------------------------------

// n (n-1) ... (n-k+1), the coefficient of the k-th derivative of x^n
RBF_HD constexpr int falling(int n, int k) {
    int f = 1;
    for (int i = 0; i < k; ++i)
        f *= n - i;
    return f;
}

// The partial derivative d^A/dx^A d^B/dy^B, A + B <= 2
template <int A, int B>
struct D {
    static_assert(A >= 0 && B >= 0 && A + B <= 2, "derivatives up to second order");
    static constexpr int order = A + B;

    // On phi = r^Q at the displacement (dx, dy) from the node, with
    // p = r^(Q-2) and s = r^(Q-4):
    //
    //     phi           = p r^2
    //     D_a phi       = Q p d_a
    //     D_a D_b phi   = Q s ((Q-2) d_a d_b + delta_ab r^2)
    template <int Q, typename T>
    RBF_HD T phs(T dx, T dy) const {
        constexpr T q = T(Q);
        const T r2 = dx * dx + dy * dy;
        if constexpr (order == 0) {
            return PHS<Q>::rp(r2) * r2;
        } else if constexpr (order == 1) {
            return q * PHS<Q>::rp(r2) * (A ? dx : dy);
        } else {
            const T da = A ? dx : dy, db = B ? dy : dx;  // the two axes differentiated
            constexpr bool same_axis = (A == 2 || B == 2);
            return q * PHS<Q>::rs(r2) * ((q - 2) * da * db + (same_axis ? r2 : T(0)));
        }
    }

    // On x^i y^j, from the power tables: i (i-1) ... j (j-1) ... x^(i-A) y^(j-B)
    template <typename T>
    RBF_HD T poly(int i, int j, const T* X, const T* Y) const {
        if (i < A || j < B) return T(0);
        return T(falling(i, A) * falling(j, B)) * X[i - A] * Y[j - B];
    }
};

template <typename A>
struct Scaled {
    A a;
    double c;
    template <int Q, typename T>
    RBF_HD T phs(T dx, T dy) const {
        return T(c) * a.template phs<Q>(dx, dy);
    }
    template <typename T>
    RBF_HD T poly(int i, int j, const T* X, const T* Y) const {
        return T(c) * a.poly(i, j, X, Y);
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
    template <typename T>
    RBF_HD T poly(int i, int j, const T* X, const T* Y) const {
        return a.poly(i, j, X, Y) + b.poly(i, j, X, Y);
    }
};

using Value = D<0, 0>;
using Dx = D<1, 0>;
using Dy = D<0, 1>;
using Dxx = D<2, 0>;
using Dxy = D<1, 1>;
using Dyy = D<0, 2>;
using Laplace = Sum<Dxx, Dyy>;

template <typename A>
struct is_operator : std::false_type {};
template <int A, int B>
struct is_operator<D<A, B>> : std::true_type {};
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

// The tag of a code
template <Op op>
using tag_of = std::conditional_t<
    op == Op::value,
    Value,
    std::conditional_t<
        op == Op::dx,
        Dx,
        std::conditional_t<
            op == Op::dy,
            Dy,
            std::conditional_t<
                op == Op::dxx,
                Dxx,
                std::conditional_t<op == Op::dxy,
                                   Dxy,
                                   std::conditional_t<op == Op::dyy, Dyy, Laplace>>>>>>;

// ---------------------------------------------------------------------
// The columns
// ---------------------------------------------------------------------

// Column b of a right-hand side: the operator at (xe, ye), rows
// 0..N-1 on the PHS of every node, then the np monomial rows
template <int N, int P, int Q, typename A, typename T>
RBF_HD void fill_column(const A& op,
                        T* b,
                        const T* xs,
                        const T* ys,
                        T xe,
                        T ye,
                        int tid,
                        int nthreads) {
    for (int k = tid; k < N; k += nthreads)
        b[k] = op.template phs<Q>(xe - xs[k], ye - ys[k]);
    if (tid == 0) Poly<P>::apply(op, xe, ye, b + N, 1);
}

// 3. operator objects. Pack is the tuple device code can hold, a
// head-and-tail struct; std::tuple would do on the host and
// cuda::std::tuple on the device.
template <typename... Ops>
struct Pack;
template <>
struct Pack<> {
    static constexpr int size = 0;
};
template <typename Head, typename... Tail>
struct Pack<Head, Tail...> {
    Head head;
    Pack<Tail...> tail;
    static constexpr int size = 1 + sizeof...(Tail);
};

RBF_HD constexpr Pack<> pack() { return {}; }
template <typename Head, typename... Tail>
RBF_HD constexpr Pack<Head, Tail...> pack(Head head, Tail... tail) {
    return {head, pack(tail...)};
}

template <int N, int P, int Q, typename T>
RBF_HD void fill_rhs_objects(const Pack<>&, T*, const T*, const T*, const T*, const T*, int, int) {}

template <int N, int P, int Q, typename Head, typename... Tail, typename T>
RBF_HD void fill_rhs_objects(const Pack<Head, Tail...>& ops,
                             T* B,
                             const T* xs,
                             const T* ys,
                             const T* xc,
                             const T* yc,
                             int tid,
                             int nthreads) {
    constexpr int ldb = N + npoly(P);
    fill_column<N, P, Q>(ops.head, B, xs, ys, xc[0], yc[0], tid, nthreads);
    fill_rhs_objects<N, P, Q>(ops.tail, B + ldb, xs, ys, xc + 1, yc + 1, tid, nthreads);
}

// 2. codes at compile time: their tags, packed
template <Op... ops>
struct ops_list {
    static constexpr int size = sizeof...(ops);
};

template <int N, int P, int Q, Op... ops, typename T>
RBF_HD void fill_rhs_static(ops_list<ops...>,
                            T* B,
                            const T* xs,
                            const T* ys,
                            const T* xc,
                            const T* yc,
                            int tid,
                            int nthreads) {
    fill_rhs_objects<N, P, Q>(pack(tag_of<ops>{}...), B, xs, ys, xc, yc, tid, nthreads);
}

// 1. codes at run time: one switch per column, onto the tag
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
        T* b = B + col * ldb;
        const T xe = xc[col], ye = yc[col];
        switch (ops[col]) {
            case Op::value:
                fill_column<N, P, Q>(Value{}, b, xs, ys, xe, ye, tid, nthreads);
                break;
            case Op::dx:
                fill_column<N, P, Q>(Dx{}, b, xs, ys, xe, ye, tid, nthreads);
                break;
            case Op::dy:
                fill_column<N, P, Q>(Dy{}, b, xs, ys, xe, ye, tid, nthreads);
                break;
            case Op::dxx:
                fill_column<N, P, Q>(Dxx{}, b, xs, ys, xe, ye, tid, nthreads);
                break;
            case Op::dxy:
                fill_column<N, P, Q>(Dxy{}, b, xs, ys, xe, ye, tid, nthreads);
                break;
            case Op::dyy:
                fill_column<N, P, Q>(Dyy{}, b, xs, ys, xe, ye, tid, nthreads);
                break;
            default:
                fill_column<N, P, Q>(Laplace{}, b, xs, ys, xe, ye, tid, nthreads);
        }
    }
}

}  // namespace rbf_ops

#endif  // RBF_OPS_H
