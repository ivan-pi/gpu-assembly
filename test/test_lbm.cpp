// Tests for the LBM layer: assembly exactness, streaming weights,
// collision invariants, agreement between steppers and backends, and a
// short Taylor-Green run against the analytic decay.

#include <cmath>
#include <cstdio>
#include <random>
#include <span>
#include <vector>

#include "check.h"

#include "rbf_periodic.h"
#include "rbf_assembly.h"
#include "rbf_flow_benchmarks.h"
#include "lbm_lattice.h"
#include "lbm_collision.h"
#include "lbm_streaming.h"
#include "lbm_schemes.h"
#if defined(LBM_WITH_EIGEN)
#include "lbm_streamer_eigen.h"
#endif

using T = double;
using I = std::int32_t;
using L = lbm::D2Q9<T>;
using Space = lbm::space::Host;
using Op = lbm::EllOperator<L, I, Space>;

constexpr int K = 21, P = 4, PHS = 3;

struct Cloud {
    std::vector<T> x, y;
    rbf::PeriodicBox<T> box;
};

static Cloud jittered(int n, T jitter, unsigned seed = 7) {
    Cloud c;
    c.box = {0, 0, T(n), T(n)};
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> u(-jitter, jitter);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            c.x.push_back(T(i) + T(0.5) + u(gen));
            c.y.push_back(T(j) + T(0.5) + u(gen));
        }
    return c;
}

static T max_abs_diff(const T* a, const T* b, std::size_t n) {
    T m = 0;
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

// --- assembly: exact on polynomials of degree <= P -----------------------

static void test_assembly_polynomial() {
    // a non-periodic setting: a box so large that no image is ever a neighbour
    Cloud c = jittered(12, 0.3);
    const rbf::PeriodicBox<T> huge{-1e6, -1e6, 2e6, 2e6};
    const I n = static_cast<I>(c.x.size());
    auto ja = rbf::periodic_knn<T, I>(c.x, c.y, huge, K);

    auto p   = [](T x, T y) { return 1 + 2*x - y + 0.5*x*x - x*y + 0.25*y*y + 0.1*x*x*y; };
    auto px  = [](T x, T y) { return 2 + x - y + 0.2*x*y; };
    auto py  = [](T x, T y) { return -1 - x + 0.5*y + 0.1*x*x; };
    auto pxx = [](T, T y)   { return 1 + 0.2*y; };
    auto pxy = [](T x, T)   { return -1 + 0.2*x; };
    auto pyy = [](T, T)     { return 0.5; };

    rbf::HostAssembler<T, I> assemble(c.x, c.y, P, PHS, nullptr);
    const T ox = 0.37, oy = -0.21;
    const rbf::Functional<T> ops[6] = {
        {rbf::functional::value, ox, oy},
        {rbf::functional::dx}, {rbf::functional::dy},
        {rbf::functional::dxx}, {rbf::functional::dxy}, {rbf::functional::dyy}};
    auto W = assemble(c.x, c.y, std::span<const I>{ja}, std::span<const rbf::Functional<T>>{ops});
    CHECK(W.size() == 6);

    T e[6] = {0, 0, 0, 0, 0, 0};
    for (I s = 0; s < n; ++s) {
        T v[6] = {0, 0, 0, 0, 0, 0};
        for (int j = 0; j < K; ++j) {
            const I col = ja[s * K + j];
            const T f = p(c.x[col], c.y[col]);
            for (int r = 0; r < 6; ++r) v[r] += W[r][s * K + j] * f;
        }
        const T xs = c.x[s], ys = c.y[s];
        e[0] = std::max(e[0], std::abs(v[0] - p(xs + ox, ys + oy)));
        e[1] = std::max(e[1], std::abs(v[1] - px(xs, ys)));
        e[2] = std::max(e[2], std::abs(v[2] - py(xs, ys)));
        e[3] = std::max(e[3], std::abs(v[3] - pxx(xs, ys)));
        e[4] = std::max(e[4], std::abs(v[4] - pxy(xs, ys)));
        e[5] = std::max(e[5], std::abs(v[5] - pyy(xs, ys)));
    }
    for (int r = 0; r < 6; ++r) {
        CHECK(e[r] < 1e-8);
        if (e[r] >= 1e-8) std::printf("  functional %d: max error %.3e\n", r, e[r]);
    }
}

// --- streaming weights: constants are preserved ---------------------------

static void check_row_sums(const lbm::StreamingWeights<T, I>& w, const char* what) {
    T worst = 0;
    for (int q = 0; q < L::Q; ++q) {
        if (w.is_identity(q)) continue;
        for (I s = 0; s < w.n; ++s) {
            T sum = 0;
            for (int j = 0; j < w.k; ++j) sum += w.a[q][s * w.k + j];
            worst = std::max(worst, std::abs(sum - T(1)));
        }
    }
    CHECK(worst < 1e-9);
    if (worst >= 1e-9) std::printf("  %s: worst row-sum deviation %.3e\n", what, worst);
}

template<class Assemble, class Knn>
static lbm::StreamingWeights<T, I> weights_for(lbm::Scheme sc, lbm::Stencils st, T dt,
        const Cloud& c, Assemble& assemble, Knn& knn) {
    return lbm::build_streaming_weights<L>(sc, st, dt, std::span<const T>{c.x},
                                           std::span<const T>{c.y}, K, assemble, knn, &c.box);
}

static void test_streaming_weights() {
    Cloud c = jittered(16, 0.3);
    rbf::HostAssembler<T, I> assemble(c.x, c.y, P, PHS, &c.box);
    auto knn = [&](std::span<const T> qx, std::span<const T> qy, int k) {
        return rbf::periodic_knn<T, I>(c.x, c.y, c.box, qx, qy, k);
    };
    const T dt = 0.4;
    auto sl_a = weights_for(lbm::Scheme::semi_lagrangian, lbm::Stencils::arrival, dt, c, assemble, knn);
    auto sl_d = weights_for(lbm::Scheme::semi_lagrangian, lbm::Stencils::departure, dt, c, assemble, knn);
    auto lw   = weights_for(lbm::Scheme::lax_wendroff, lbm::Stencils::arrival, dt, c, assemble, knn);
    CHECK(sl_a.shared_pattern());
    CHECK(sl_d.patterns.size() == 8);
    CHECK(lw.shared_pattern());
    check_row_sums(sl_a, "sl/arrival");
    check_row_sums(sl_d, "sl/departure");
    check_row_sums(lw, "lw/arrival");

    // arrival- and departure-centred stencils are two different high-order
    // interpolants of the same field; on 16 nodes per wavelength they
    // agree to well below the discretisation error of either
    const I n = sl_a.n;
    std::vector<T> g(n);
    const T pi = std::acos(T(-1));
    for (I i = 0; i < n; ++i) g[i] = std::sin(2 * pi * c.x[i] / c.box.Lx) * std::cos(2 * pi * c.y[i] / c.box.Ly);
    T diff = 0;
    for (int q = 1; q < L::Q; ++q) {
        for (I s = 0; s < n; ++s) {
            T va = 0, vd = 0;
            for (int j = 0; j < K; ++j) {
                va += sl_a.a[q][s * K + j] * g[sl_a.patterns[0][s * K + j]];
                vd += sl_d.a[q][s * K + j] * g[sl_d.patterns[sl_d.pattern_of[q]][s * K + j]];
            }
            diff = std::max(diff, std::abs(va - vd));
        }
    }
    CHECK(diff < 1e-3);
    if (diff >= 1e-4) std::printf("  arrival vs departure: %.3e\n", diff);
}

// --- collision invariants -------------------------------------------------

static void test_collision() {
    T f[L::Q];
    L::equilibrium(1.02, 0.03, -0.01, f);
    for (int q = 0; q < L::Q; ++q) f[q] *= 1 + 0.05 * std::sin(1.7 * q);  // perturb
    const auto m0 = L::macros(f);

    T fb[L::Q], ft[L::Q], ft2[L::Q];
    for (int q = 0; q < L::Q; ++q) fb[q] = ft[q] = ft2[q] = f[q];
    L::Macros mb, mt, mt2;
    lbm::collision::BGK<L>{1.3}(fb, mb);
    lbm::collision::TRT<L>{1.3, 1.3}(ft, mt);
    lbm::collision::TRT<L>::from_magic(1.3, 0.25)(ft2, mt2);

    CHECK(std::abs(mb.rho - m0.rho) < 1e-14 && std::abs(mb.ux - m0.ux) < 1e-14);
    CHECK(max_abs_diff(fb, ft, L::Q) < 1e-14);          // TRT with equal rates is BGK
    const auto m1 = L::macros(fb), m2 = L::macros(ft2);
    CHECK(std::abs(m1.rho - m0.rho) < 1e-14 && std::abs(m1.ux - m0.ux) < 1e-14 && std::abs(m1.uy - m0.uy) < 1e-14);
    CHECK(std::abs(m2.rho - m0.rho) < 1e-14 && std::abs(m2.ux - m0.ux) < 1e-14 && std::abs(m2.uy - m0.uy) < 1e-14);
}

// --- steppers and backends agree; Taylor-Green decays -------------------

struct Run {
    std::vector<T> f, rho, ux, uy;
};

template<class MakeStepper>
static Run run(const lbm::StreamingWeights<T, I>& w, const Op& A,
               int steps, T omega, const Cloud& c, T u0, T nu, MakeStepper make) {
    const I n = w.n;
    const I ldf = lbm::padded<I>(n, 64);
    lbm::Array<T, Space> f0(std::size_t(ldf) * L::Q), f1(std::size_t(ldf) * L::Q);
    f0.fill(0); f1.fill(0);
    Run r;
    r.rho.resize(n); r.ux.resize(n); r.uy.resize(n);
    const T pi = std::acos(T(-1));
    flow_benchmarks::taylor_green<T> tg{2 * pi / c.box.Lx, 2 * pi / c.box.Ly, nu, u0};
    for (I i = 0; i < n; ++i) {
        auto [p, vx, vy] = tg({c.x[i], c.y[i]}, T(0));
        r.rho[i] = 1 + p / L::cs2; r.ux[i] = vx; r.uy[i] = vy;
    }
    lbm::set_equilibrium<L, I, Space>(n, f0.data(), ldf, r.rho.data(), r.ux.data(), r.uy.data());

    auto stepper = make(w, A, omega);
    lbm::Array<T, Space>* f[2] = {&f0, &f1};
    int cur = 0;
    for (int s = 0; s < steps; ++s) {
        stepper->step(f[cur]->data(), f[1 - cur]->data(), ldf,
                      lbm::MacroPtrs<T>{r.rho.data(), r.ux.data(), r.uy.data()});
        cur = 1 - cur;
    }
    r.f = lbm::to_host(*f[cur]);
    return r;
}

static void test_steppers_and_decay() {
    Cloud c = jittered(24, 0.25);
    rbf::HostAssembler<T, I> assemble(c.x, c.y, P, PHS, &c.box);
    auto knn = [&](std::span<const T> qx, std::span<const T> qy, int k) {
        return rbf::periodic_knn<T, I>(c.x, c.y, c.box, qx, qy, k);
    };

    const T cs = std::sqrt(L::cs2), u0 = 0.05 * cs;
    const T re = 10, nu = u0 * c.box.Lx / re, tau = nu / L::cs2;
    const T dt = 0.5 / std::sqrt(T(2)), omega = dt / (tau + 0.5 * dt);
    const int steps = 400;

    auto w = weights_for(lbm::Scheme::semi_lagrangian, lbm::Stencils::arrival, dt, c, assemble, knn);
    Op A(w);

    using lbm::collision::BGK;
    auto fused = [](const auto&, const auto& A, T om) -> std::unique_ptr<lbm::Stepper<L, I, Space>> {
        return std::make_unique<lbm::FusedStepper<Op, BGK<L>>>(A, BGK<L>{om});
    };
    lbm::NativeStreamer<Op> native(A);
    auto split = [&](const auto&, const auto& A, T om) -> std::unique_ptr<lbm::Stepper<L, I, Space>> {
        return std::make_unique<lbm::SplitStepper<L, I, Space, BGK<L>, lbm::NativeStreamer<Op>>>(
            A.n(), native, BGK<L>{om});
    };

    Run rf = run(w, A, steps, omega, c, u0, nu, fused);
    Run rs = run(w, A, steps, omega, c, u0, nu, split);
    const T d_fs = max_abs_diff(rf.f.data(), rs.f.data(), rf.f.size());
    CHECK(d_fs < 1e-12);
    if (d_fs >= 1e-12) std::printf("  fused vs split: %.3e\n", d_fs);

#if defined(LBM_WITH_EIGEN)
    lbm::EigenStreamer<L, I> eigen(w);
    auto split_eigen = [&](const auto&, const auto& A, T om) -> std::unique_ptr<lbm::Stepper<L, I, Space>> {
        return std::make_unique<lbm::SplitStepper<L, I, Space, BGK<L>, lbm::EigenStreamer<L, I>>>(
            A.n(), eigen, BGK<L>{om});
    };
    Run re_ = run(w, A, steps, omega, c, u0, nu, split_eigen);
    const T d_fe = max_abs_diff(rf.f.data(), re_.f.data(), rf.f.size());
    CHECK(d_fe < 1e-12);
    if (d_fe >= 1e-12) std::printf("  fused vs eigen: %.3e\n", d_fe);
#endif

    // decay of the velocity amplitude against exp(-t/td)
    const T pi = std::acos(T(-1));
    flow_benchmarks::taylor_green<T> tg{2 * pi / c.box.Lx, 2 * pi / c.box.Ly, nu, u0};
    const T time = steps * dt;
    const T expected = std::exp(-time / tg.time_constant());
    T umax = 0, err2 = 0, ref2 = 0;
    for (I i = 0; i < w.n; ++i) {
        umax = std::max(umax, std::sqrt(rf.ux[i] * rf.ux[i] + rf.uy[i] * rf.uy[i]));
        auto [p, vx, vy] = tg({c.x[i], c.y[i]}, time);
        err2 += (rf.ux[i] - vx) * (rf.ux[i] - vx) + (rf.uy[i] - vy) * (rf.uy[i] - vy);
        ref2 += vx * vx + vy * vy;
    }
    const T decay = umax / u0, rel = std::sqrt(err2 / ref2);
    std::printf("  Taylor-Green: t/td = %.3f, umax/u0 = %.4f, expected %.4f, rel. L2 error %.3e\n",
                time / tg.time_constant(), decay, expected, rel);
    CHECK(std::abs(decay - expected) < 0.05 * expected);
    CHECK(rel < 0.05);

    // Lax-Wendroff and TRT run stably on the same case
    auto wlw = weights_for(lbm::Scheme::lax_wendroff, lbm::Stencils::arrival, dt, c, assemble, knn);
    Op Alw(wlw);
    using lbm::collision::TRT;
    auto fused_trt = [](const auto&, const auto& A, T om) -> std::unique_ptr<lbm::Stepper<L, I, Space>> {
        return std::make_unique<lbm::FusedStepper<Op, TRT<L>>>(A, TRT<L>::from_magic(om));
    };
    Run rl = run(wlw, Alw, steps, omega, c, u0, nu, fused_trt);
    T umax_lw = 0;
    for (I i = 0; i < w.n; ++i)
        umax_lw = std::max(umax_lw, std::sqrt(rl.ux[i] * rl.ux[i] + rl.uy[i] * rl.uy[i]));
    std::printf("  Lax-Wendroff/TRT: umax/u0 = %.4f, expected %.4f\n", umax_lw / u0, expected);
    CHECK(std::abs(umax_lw / u0 - expected) < 0.05 * expected);
}

int main() {
    test_assembly_polynomial();
    test_streaming_weights();
    test_collision();
    test_steppers_and_decay();
    return report("lbm");
}
