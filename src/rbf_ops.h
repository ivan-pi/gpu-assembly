#ifndef RBF_OPS_H
#define RBF_OPS_H

// The RBF-FD operators of rbf_fd.f90 for the device kernels, prototyped
// on the host (RBF_HD is __host__ __device__ under nvcc).
//
// An operator is a partial derivative D<A, B> = d^A/dx^A d^B/dy^B of
// order A + B <= 2, or a linear combination of those (* and +). The
// value is D<0, 0>, the Laplacian D<2, 0> + D<0, 2>. An operator
// answers for the PHS phi = r^Q at a displacement and for a monomial
// x^i y^j from the powers of the evaluation point, and a column of a
// right-hand side is those two, rows 0..N-1 the nodes and then the
// npoly(P) monomials by total degree, the order of rbf_operators.h.
// B is nt x nrhs column-major, ldb = nt = N + npoly(P); the loops
// stride the rows by tid/nthreads as the kernels do.
//
// Three ways to name the operators of the columns, one function each:
//
//   fill_rhs<N, P, Q>(B, xs, ys, xc, yc, tid, nthreads,          // objects
//                     2.0 * Dxx{} + 0.5 * Dyy{}, Dx{}, Value{});
//   fill_rhs_static<N, P, Q>(ops_list<Op::laplace, Op::dx>{},    // codes, compile time
//                            B, xs, ys, xc, yc, tid, nthreads);
//   fill_rhs_dynamic<N, P, Q>(B, xs, ys, ops, xc, yc, nrhs,      // codes, run time
//                             tid, nthreads);

#include <cmath>
#include <type_traits>

#ifdef __CUDACC__
#define RBF_HD __host__ __device__
#else
#define RBF_HD
#endif

namespace rbf_ops {

// The codes, with the values of OP_* in rbf_fd.f90
enum class Op : int { value = 0, dx, dy, dxx, dxy, dyy, laplace };

constexpr int npoly(int P) { return (P + 1) * (P + 2) / 2; }

// n (n-1) ... (n-k+1), the coefficient of the k-th derivative of x^n
RBF_HD constexpr int falling(int n, int k) {
    int f = 1;
    for (int i = 0; i < k; ++i)
        f *= n - i;
    return f;
}

// d^A/dx^A d^B/dy^B
template <int A, int B>
struct D {
    static_assert(A >= 0 && B >= 0 && A + B <= 2, "derivatives up to second order");

    // On phi = r^Q at the displacement (dx, dy), with p = r^(Q-2):
    //   phi = p r^2,  D_a phi = Q p d_a,  D_a D_b phi = Q p/r^2 ((Q-2) d_a d_b + delta_ab r^2)
    template <int Q, typename T>
    RBF_HD T phs(T dx, T dy) const {
        static_assert(Q >= 3 && Q % 2 == 1, "an odd exponent >= 3");
        const T r2 = dx * dx + dy * dy;
        T p = sqrt(r2);
        for (int i = 0; i < (Q - 3) / 2; ++i)
            p *= r2;
        if constexpr (A + B == 0) return p * r2;
        if constexpr (A + B == 1) return Q * p * (A ? dx : dy);
        if constexpr (A + B == 2)
            return r2 > 0
                       ? Q * p / r2 * ((Q - 2) * (A ? dx : dy) * (B ? dy : dx) + (A * B ? 0 : r2))
                       : T(0);
    }

    // On x^i y^j, from the power tables X, Y of the evaluation point
    template <typename T>
    RBF_HD T poly(int i, int j, const T* X, const T* Y) const {
        return (i < A || j < B) ? T(0) : falling(i, A) * falling(j, B) * X[i - A] * Y[j - B];
    }
};

template <typename A>
concept Operator = requires(const A a, double d, const double* t) {
    a.template phs<3>(d, d);
    a.poly(0, 0, t, t);
};

template <Operator A>
struct Scaled {
    A a;
    double c;
    template <int Q, typename T>
    RBF_HD T phs(T dx, T dy) const {
        return c * a.template phs<Q>(dx, dy);
    }
    template <typename T>
    RBF_HD T poly(int i, int j, const T* X, const T* Y) const {
        return c * a.poly(i, j, X, Y);
    }
};

template <Operator A, Operator B>
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

template <Operator A>
RBF_HD constexpr Scaled<A> operator*(double c, A a) {
    return {a, c};
}
template <Operator A, Operator B>
RBF_HD constexpr Sum<A, B> operator+(A a, B b) {
    return {a, b};
}

using Value = D<0, 0>;
using Dx = D<1, 0>;
using Dy = D<0, 1>;
using Dxx = D<2, 0>;
using Dxy = D<1, 1>;
using Dyy = D<0, 2>;
using Laplace = Sum<Dxx, Dyy>;

// The orders (A, B) of the codes; the Laplacian is the sum of two
constexpr int order_x[] = {0, 1, 0, 2, 1, 0, 0}, order_y[] = {0, 0, 1, 0, 1, 2, 0};
template <Op op>
using tag_of =
    std::conditional_t<op == Op::laplace, Laplace, D<order_x[int(op)], order_y[int(op)]>>;

// The tag of a run-time code, handed to f
template <typename F>
RBF_HD void with_tag(Op op, F f) {
    switch (op) {
        case Op::value:
            return f(Value{});
        case Op::dx:
            return f(Dx{});
        case Op::dy:
            return f(Dy{});
        case Op::dxx:
            return f(Dxx{});
        case Op::dxy:
            return f(Dxy{});
        case Op::dyy:
            return f(Dyy{});
        default:
            return f(Laplace{});
    }
}

// One column: the operator at (xe, ye) on the PHS of every node, then
// on the monomials by total degree
template <int N, int P, int Q, typename T>
RBF_HD void fill_column(Operator auto const& op,
                        T* b,
                        const T* xs,
                        const T* ys,
                        T xe,
                        T ye,
                        int tid,
                        int nthreads) {
    for (int k = tid; k < N; k += nthreads)
        b[k] = op.template phs<Q>(xe - xs[k], ye - ys[k]);
    if (tid != 0) return;
    T X[P + 1], Y[P + 1];
    X[0] = Y[0] = 1;
    for (int i = 1; i <= P; ++i) {
        X[i] = X[i - 1] * xe;
        Y[i] = Y[i - 1] * ye;
    }
    for (int d = 0, k = N; d <= P; ++d)
        for (int j = 0; j <= d; ++j)
            b[k++] = op.poly(d - j, j, X, Y);
}

// Column j is the j-th operator at (xc[j], yc[j])
template <int N, int P, int Q, typename T>
RBF_HD void fill_rhs(T* B,
                     const T* xs,
                     const T* ys,
                     const T* xc,
                     const T* yc,
                     int tid,
                     int nthreads,
                     Operator auto const&... ops) {
    int j = 0;
    ((fill_column<N, P, Q>(ops, B + j * (N + npoly(P)), xs, ys, xc[j], yc[j], tid, nthreads), ++j),
     ...);
}

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
    fill_rhs<N, P, Q>(B, xs, ys, xc, yc, tid, nthreads, tag_of<ops>{}...);
}

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
    for (int j = 0; j < nrhs; ++j)
        with_tag(ops[j], [&](auto op) {
            fill_column<N, P, Q>(op, B + j * (N + npoly(P)), xs, ys, xc[j], yc[j], tid, nthreads);
        });
}

}  // namespace rbf_ops

#endif  // RBF_OPS_H
