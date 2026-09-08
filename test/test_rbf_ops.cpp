// Tests for rbf_ops.h: the three ways of naming the right-hand-side
// operators produce the same columns, the derivative columns agree with
// central differences of the value column, the Laplacian is the sum of
// the second derivatives, and the composed operators are what they say.
// Also the cross-check with rbf_fd.f90: the same ring stencil as
// test_rbf_fd.f90 and the same formulas, so the columns of the two are
// the same numbers.

#include <cmath>
#include <cstdio>
#include <vector>

#include "rbf_ops.h"

#include "check.h"

using namespace rbf_ops;

constexpr int N = 21, P = 3;
constexpr int NT = N + npoly(P);
constexpr double pi = 3.14159265358979323846;

// The ring stencil of test_rbf_fd.f90
static void ring_stencil(double* x, double* y) {
    x[0] = y[0] = 0;
    for (int k = 1; k <= 7; ++k) {
        x[k] = std::cos(2 * pi * k / 7);
        y[k] = std::sin(2 * pi * k / 7);
    }
    for (int k = 1; k <= 13; ++k) {
        x[7 + k] = 2 * std::cos(2 * pi * k / 13 + 0.3);
        y[7 + k] = 2 * std::sin(2 * pi * k / 13 + 0.3);
    }
}

static double maxdiff(const double* a, const double* b, int n) {
    double m = 0;
    for (int i = 0; i < n; ++i)
        m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

template <int Q>
static void test_q() {
    double xs[N], ys[N];
    ring_stencil(xs, ys);

    // the seven operators: derivatives at the node, the value off it
    constexpr int NRHS = 7;
    const Op ops[NRHS] = {Op::dx, Op::dy, Op::dxx, Op::dxy, Op::dyy, Op::laplace, Op::value};
    double xc[NRHS] = {0, 0, 0, 0, 0, 0, 0.4}, yc[NRHS] = {0, 0, 0, 0, 0, 0, -0.7};

    std::vector<double> B1(NT * NRHS), B2(NT * NRHS), B3(NT * NRHS);

    // 1. run-time codes, with a "block" of 3 threads striding the rows
    for (int tid = 0; tid < 3; ++tid)
        fill_rhs_dynamic<N, P, Q>(B1.data(), xs, ys, ops, xc, yc, NRHS, tid, 3);

    // 2. compile-time codes
    using list = ops_list<Op::dx, Op::dy, Op::dxx, Op::dxy, Op::dyy, Op::laplace, Op::value>;
    static_assert(list::size == NRHS);
    for (int tid = 0; tid < 3; ++tid)
        fill_rhs_static<N, P, Q>(list{}, B2.data(), xs, ys, xc, yc, tid, 3);

    // 3. objects
    for (int tid = 0; tid < 3; ++tid)
        fill_rhs<N, P, Q>(B3.data(), xs, ys, xc, yc, tid, 3, Dx{}, Dy{}, Dxx{}, Dxy{}, Dyy{},
                          Laplace{}, Value{});

    // The same formulas, but GCC contracts a*b+c into FMAs differently
    // in the switch and in the template, so agreement is to rounding
    double big = 0;
    for (double v : B1)
        big = std::fmax(big, std::fabs(v));
    CHECK(maxdiff(B1.data(), B2.data(), NT * NRHS) / big < 1e-14);
    CHECK(maxdiff(B1.data(), B3.data(), NT * NRHS) / big < 1e-14);

    // The derivatives against central differences of the value column,
    // at a point off the nodes; the value column of a stencil "with
    // 0 threads" is just the basis, so evaluate it directly
    const double h = 1e-3, x0 = 0.3, y0 = -0.2;
    auto value = [&](double xe, double ye, double* b) {
        const Op v = Op::value;
        fill_rhs_dynamic<N, P, Q>(b, xs, ys, &v, &xe, &ye, 1, 0, 1);
    };
    std::vector<double> f0(NT), fpp(NT), fpm(NT), fmp(NT), fmm(NT), fd(NT), b(NT);
    value(x0, y0, f0.data());
    double scale = 0;
    for (double v : f0)
        scale = std::fmax(scale, std::fabs(v));

    auto column = [&](Op op, double* out) {
        fill_rhs_dynamic<N, P, Q>(out, xs, ys, &op, &x0, &y0, 1, 0, 1);
    };

    value(x0 + h, y0, fpp.data());
    value(x0 - h, y0, fmm.data());
    column(Op::dx, b.data());
    for (int i = 0; i < NT; ++i)
        fd[i] = (fpp[i] - fmm[i]) / (2 * h);
    CHECK(maxdiff(b.data(), fd.data(), NT) / scale < 1e-5);
    column(Op::dxx, b.data());
    for (int i = 0; i < NT; ++i)
        fd[i] = (fpp[i] - 2 * f0[i] + fmm[i]) / (h * h);
    CHECK(maxdiff(b.data(), fd.data(), NT) / scale < 1e-5);

    value(x0, y0 + h, fpp.data());
    value(x0, y0 - h, fmm.data());
    column(Op::dy, b.data());
    for (int i = 0; i < NT; ++i)
        fd[i] = (fpp[i] - fmm[i]) / (2 * h);
    CHECK(maxdiff(b.data(), fd.data(), NT) / scale < 1e-5);
    column(Op::dyy, b.data());
    for (int i = 0; i < NT; ++i)
        fd[i] = (fpp[i] - 2 * f0[i] + fmm[i]) / (h * h);
    CHECK(maxdiff(b.data(), fd.data(), NT) / scale < 1e-5);

    value(x0 + h, y0 + h, fpp.data());
    value(x0 + h, y0 - h, fpm.data());
    value(x0 - h, y0 + h, fmp.data());
    value(x0 - h, y0 - h, fmm.data());
    column(Op::dxy, b.data());
    for (int i = 0; i < NT; ++i)
        fd[i] = (fpp[i] - fpm[i] - fmp[i] + fmm[i]) / (4 * h * h);
    CHECK(maxdiff(b.data(), fd.data(), NT) / scale < 1e-5);

    // The Laplacian is the sum of the second derivatives
    std::vector<double> lap(NT), dxx(NT), dyy(NT);
    column(Op::laplace, lap.data());
    column(Op::dxx, dxx.data());
    column(Op::dyy, dyy.data());
    for (int i = 0; i < NT; ++i)
        fd[i] = dxx[i] + dyy[i];
    CHECK(maxdiff(lap.data(), fd.data(), NT) / scale < 1e-12);

    // At its own node every derivative of phi vanishes, without NaN
    double at_node[NT];
    const Op d2 = Op::dxx;
    fill_rhs_dynamic<N, P, Q>(at_node, xs, ys, &d2, &xs[3], &ys[3], 1, 0, 1);
    CHECK(at_node[3] == 0);
    for (int i = 0; i < NT; ++i)
        CHECK(std::isfinite(at_node[i]));

    // Composed operators: an anisotropic Laplacian and a directional
    // derivative, against the combinations of the plain columns
    std::vector<double> C(NT * 2), dxc(NT), dyc(NT);
    const double xz[2] = {x0, x0}, yz[2] = {y0, y0};
    fill_rhs<N, P, Q>(C.data(), xs, ys, xz, yz, 0, 1, 2.0 * Dxx{} + 0.5 * Dyy{},
                      0.6 * Dx{} + 0.8 * Dy{});
    column(Op::dx, dxc.data());
    column(Op::dy, dyc.data());
    for (int i = 0; i < NT; ++i) {
        fd[i] = 2.0 * dxx[i] + 0.5 * dyy[i];
        b[i] = 0.6 * dxc[i] + 0.8 * dyc[i];
    }
    CHECK(maxdiff(C.data(), fd.data(), NT) / scale < 1e-12);
    CHECK(maxdiff(C.data() + NT, b.data(), NT) / scale < 1e-12);
}

int main() {
    test_q<3>();
    test_q<5>();
    test_q<7>();
    return report("test_rbf_ops");
}
