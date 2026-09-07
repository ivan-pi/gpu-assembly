// lbm_bench.cpp -- periodic flow benchmark driver
//
// Generates (or reads) a point cloud in a doubly periodic box, assembles
// the streaming operator for the chosen scheme, and runs the LBM time
// loop with the chosen collision model, stepper and backend. Reports
// MLUPS, an effective bandwidth, and for the Taylor-Green vortex the
// velocity error against the analytic decay.
//
// Every axis of variation is a command-line option; see usage(). The
// choices that change types (collision, stepper, backend) are resolved
// once in make_stepper(); the time loop itself makes one virtual call per
// step.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "rbf_io.h"
#include "rbf_io_vtk.h"
#include "rbf_reorder.h"
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

struct Options {
    std::string flow = "tg";          // tg | shear
    int nodes = 64;                   // nodes per side; box side = nodes (dx = 1)
    double jitter = 0.25;             // jitter amplitude in units of dx
    unsigned seed = 1;
    std::string nodefile;             // .points file instead of generated nodes
    double length = 0;                // box side for a nodefile (0: nodes)
    std::string order = "none";       // none | morton | hilbert
    int k = 21, poly = 4, phs = 3;
    std::string scheme = "sl";        // sl | lw
    std::string stencils = "arrival"; // arrival | departure
    std::string collision = "bgk";    // bgk | trt
    double magic = 0.25;
    std::string stepper = "fused";    // fused | split
    std::string backend = "native";   // native | eigen
    double re = 100, ma = 0.05, cfl = 0.5;
    int steps = 1000, io = 100;
    std::string out;                  // VTK prefix; empty = no files
};

static void usage() {
    std::printf(
        "lbm_bench [options]\n"
        "  --flow tg|shear          benchmark (default tg)\n"
        "  --nodes N                N x N jittered nodes, box side N (default 64)\n"
        "  --jitter a               jitter amplitude / dx (default 0.25; 0 = grid)\n"
        "  --seed s                 jitter seed\n"
        "  --nodefile f --length L  read a .points file in [0,L)^2 instead\n"
        "  --order none|morton|hilbert  node renumbering (default none)\n"
        "  --k K --poly P --phs Q   stencil size, polynomial degree, PHS exponent (21 4 3)\n"
        "  --scheme sl|lw           semi-Lagrangian | Lax-Wendroff (default sl)\n"
        "  --stencils arrival|departure  stencil centres (default arrival)\n"
        "  --collision bgk|trt      collision model (default bgk)\n"
        "  --magic v                TRT magic parameter (default 0.25)\n"
        "  --stepper fused|split    stream+collide in one pass, or two (default fused)\n"
        "  --backend native|eigen   streaming backend for the split stepper (default native)\n"
        "  --re Re --ma Ma --cfl c  Reynolds, Mach (u0 = Ma cs), CFL (100 0.05 0.5)\n"
        "  --steps n --io m         time steps, output interval (1000 100)\n"
        "  --out prefix             write prefix_<step>.vtk\n");
}

static Options parse(int argc, char** argv) {
    Options o;
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); std::exit(2); }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(); std::exit(0); }
        else if (a == "--flow") o.flow = need(i);
        else if (a == "--nodes") o.nodes = std::atoi(need(i));
        else if (a == "--jitter") o.jitter = std::atof(need(i));
        else if (a == "--seed") o.seed = static_cast<unsigned>(std::atoi(need(i)));
        else if (a == "--nodefile") o.nodefile = need(i);
        else if (a == "--length") o.length = std::atof(need(i));
        else if (a == "--order") o.order = need(i);
        else if (a == "--k") o.k = std::atoi(need(i));
        else if (a == "--poly") o.poly = std::atoi(need(i));
        else if (a == "--phs") o.phs = std::atoi(need(i));
        else if (a == "--scheme") o.scheme = need(i);
        else if (a == "--stencils") o.stencils = need(i);
        else if (a == "--collision") o.collision = need(i);
        else if (a == "--magic") o.magic = std::atof(need(i));
        else if (a == "--stepper") o.stepper = need(i);
        else if (a == "--backend") o.backend = need(i);
        else if (a == "--re") o.re = std::atof(need(i));
        else if (a == "--ma") o.ma = std::atof(need(i));
        else if (a == "--cfl") o.cfl = std::atof(need(i));
        else if (a == "--steps") o.steps = std::atoi(need(i));
        else if (a == "--io") o.io = std::atoi(need(i));
        else if (a == "--out") o.out = need(i);
        else { std::fprintf(stderr, "unknown option %s\n", argv[i]); usage(); std::exit(2); }
    }
    return o;
}

// --- nodes ---------------------------------------------------------------

static void make_nodes(const Options& o, std::vector<T>& x, std::vector<T>& y, T& length) {
    if (!o.nodefile.empty()) {
        std::tie(x, y) = rbf::io::read_points<T>(o.nodefile);
        length = o.length > 0 ? o.length : T(o.nodes);
        return;
    }
    const int n = o.nodes;
    length = T(n);
    x.resize(std::size_t(n) * n);
    y.resize(std::size_t(n) * n);
    std::mt19937 gen(o.seed);
    std::uniform_real_distribution<T> u(-o.jitter, o.jitter);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const std::size_t s = std::size_t(j) * n + i;
            x[s] = T(i) + T(0.5) + (o.jitter > 0 ? u(gen) : T(0));
            y[s] = T(j) + T(0.5) + (o.jitter > 0 ? u(gen) : T(0));
        }
}

// --- assembly -------------------------------------------------------------

// The reference host assembler; an existing host assembly API plugs in
// here through a lambda with the same four arguments.
static lbm::StreamingWeights<T, I> assemble_weights(const Options& o, T dt,
        std::span<const T> x, std::span<const T> y, const rbf::PeriodicBox<T>& box) {
    rbf::HostAssembler<T, I> assemble(x, y, o.poly, o.phs, &box);
    auto knn = [&](std::span<const T> qx, std::span<const T> qy, int k) {
        return rbf::periodic_knn<T, I>(x, y, box, qx, qy, k);
    };
    const auto scheme = o.scheme == "lw" ? lbm::Scheme::lax_wendroff : lbm::Scheme::semi_lagrangian;
    const auto st = o.stencils == "departure" ? lbm::Stencils::departure : lbm::Stencils::arrival;
    return lbm::build_streaming_weights<L>(scheme, st, dt, x, y, o.k, assemble, knn, &box);
}

// --- stepper factory -----------------------------------------------------

using Op = lbm::EllOperator<L, I, Space>;   // row-major on the host by default

struct Backends {
    const Op& A;
    std::unique_ptr<lbm::NativeStreamer<Op>> native;
#if defined(LBM_WITH_EIGEN)
    std::unique_ptr<lbm::EigenStreamer<L, I>> eigen;
#endif
};

template<class Collision>
static std::unique_ptr<lbm::Stepper<L, I, Space>>
make_stepper_for(const Options& o, Backends& b, Collision c) {
    using namespace lbm;
    if (o.stepper == "fused") {
        if (o.backend != "native") {
            std::fprintf(stderr, "the fused stepper needs the native backend\n");
            std::exit(2);
        }
        return std::make_unique<FusedStepper<Op, Collision>>(b.A, c);
    }
    if (o.backend == "native") {
        b.native = std::make_unique<NativeStreamer<Op>>(b.A);
        return std::make_unique<SplitStepper<L, I, Space, Collision, NativeStreamer<Op>>>(
            b.A.n(), *b.native, c);
    }
#if defined(LBM_WITH_EIGEN)
    if (o.backend == "eigen") {
        return std::make_unique<SplitStepper<L, I, Space, Collision, EigenStreamer<L, I>>>(
            b.A.n(), *b.eigen, c);
    }
#endif
    std::fprintf(stderr, "unknown or unavailable backend %s\n", o.backend.c_str());
    std::exit(2);
}

static std::unique_ptr<lbm::Stepper<L, I, Space>>
make_stepper(const Options& o, Backends& b, T omega) {
    if (o.collision == "bgk") return make_stepper_for(o, b, lbm::collision::BGK<L>{omega});
    if (o.collision == "trt") return make_stepper_for(o, b, lbm::collision::TRT<L>::from_magic(omega, o.magic));
    std::fprintf(stderr, "unknown collision %s\n", o.collision.c_str());
    std::exit(2);
}

// --- main ----------------------------------------------------------------

int main(int argc, char** argv) {
    const Options o = parse(argc, argv);
    using clock = std::chrono::steady_clock;
    auto seconds = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

    // nodes
    std::vector<T> x, y;
    T length;
    make_nodes(o, x, y, length);
    const I n = static_cast<I>(x.size());
    const rbf::PeriodicBox<T> box{0, 0, length, length};

    if (o.order != "none") {
        rbf::BBox2<double> bb{0, 0, length, length};
        auto p = o.order == "hilbert" ? rbf::hilbert_order<I>(x, y, 16, bb)
                                      : rbf::morton_order<I>(x, y, 16, bb);
        p.permute(std::span{x});
        p.permute(std::span{y});
    }

    // physics, lattice units dx = 1
    const T cs = std::sqrt(L::cs2);
    const T u0 = o.ma * cs;
    const T nu = u0 * length / o.re;
    const T tau = nu / L::cs2;
    const T dt = o.cfl / std::sqrt(T(2));
    const T omega = dt / (tau + T(0.5) * dt);

    std::printf("nodes    = %d (%s)\n", n, o.nodefile.empty() ? "generated" : o.nodefile.c_str());
    std::printf("box      = %g x %g, periodic\n", length, length);
    std::printf("stencil  = k %d, poly %d, phs %d\n", o.k, o.poly, o.phs);
    std::printf("scheme   = %s, %s stencils\n", o.scheme.c_str(), o.stencils.c_str());
    std::printf("u0 = %g  nu = %g  tau = %g  dt = %g  dt/tau = %g  omega = %g\n",
                u0, nu, tau, dt, dt / tau, omega);

    // assembly
    auto t0 = clock::now();
    const auto weights = assemble_weights(o, dt, x, y, box);
    auto t1 = clock::now();
    Op A(weights);
    auto t2 = clock::now();
    std::printf("assembly = %.3f s, packing = %.3f s, %zu pattern(s)\n",
                seconds(t0, t1), seconds(t1, t2), weights.patterns.size());

    Backends backends{A, nullptr
#if defined(LBM_WITH_EIGEN)
        , nullptr
#endif
    };
#if defined(LBM_WITH_EIGEN)
    if (o.backend == "eigen") backends.eigen = std::make_unique<lbm::EigenStreamer<L, I>>(weights);
#endif
    auto stepper = make_stepper(o, backends, omega);
    std::printf("stepper  = %s\n", stepper->name().c_str());

    // fields
    const I ldf = lbm::padded<I>(n, 64);
    lbm::Array<T, Space> f[2] = {lbm::Array<T, Space>(std::size_t(ldf) * L::Q),
                                 lbm::Array<T, Space>(std::size_t(ldf) * L::Q)};
    f[0].fill(T(0)); f[1].fill(T(0));
    lbm::Array<T, Space> rho(n), ux(n), uy(n);
    const lbm::MacroPtrs<T> macros{rho.data(), ux.data(), uy.data()};

    // initial condition
    const T pi = std::acos(T(-1));
    flow_benchmarks::taylor_green<T> tg{2 * pi / length, 2 * pi / length, nu, u0};
    flow_benchmarks::shear_layer<T> shear{u0, T(80), T(0.05), length};
    const bool is_tg = o.flow == "tg";
    for (I i = 0; i < n; ++i) {
        if (is_tg) {
            auto [p, vx, vy] = tg({x[i], y[i]}, T(0));
            rho[i] = T(1) + p / L::cs2; ux[i] = vx; uy[i] = vy;
        } else {
            auto [vx, vy] = shear({x[i], y[i]});
            rho[i] = T(1); ux[i] = vx; uy[i] = vy;
        }
    }
    lbm::set_equilibrium<L, I, Space>(n, f[0].data(), ldf, rho.data(), ux.data(), uy.data());
    if (is_tg) std::printf("Taylor-Green decay time = %g (%g steps)\n",
                           tg.time_constant(), tg.time_constant() / dt);

    auto diagnostics = [&](int step, T time) {
        T rmin = rho[0], rmax = rho[0], umax = 0, err2 = 0, ref2 = 0;
        for (I i = 0; i < n; ++i) {
            rmin = std::min(rmin, rho[i]); rmax = std::max(rmax, rho[i]);
            umax = std::max(umax, std::sqrt(ux[i] * ux[i] + uy[i] * uy[i]));
            if (is_tg) {
                auto [p, vx, vy] = tg({x[i], y[i]}, time);
                err2 += (ux[i] - vx) * (ux[i] - vx) + (uy[i] - vy) * (uy[i] - vy);
                ref2 += vx * vx + vy * vy;
            }
        }
        std::printf("step %7d  t = %10.3f  rho [%.6f, %.6f]  umax/u0 = %.5f",
                    step, time, rmin, rmax, umax / u0);
        if (is_tg) std::printf("  decay %.5f  rel. L2 err %.3e",
                               std::exp(-time / tg.time_constant()), std::sqrt(err2 / ref2));
        std::printf("\n");
        if (!o.out.empty()) {
            rbf::io::write_vtk_polydata(o.out + "_" + std::to_string(step) + ".vtk", n,
                x.data(), y.data(), {{"rho", rho.data()}}, {{"U", ux.data(), uy.data()}});
        }
    };

    // time loop
    int cur = 0;
    T time = 0;
    diagnostics(0, time);
    auto tstart = clock::now();
    for (int step = 1; step <= o.steps; ++step) {
        const bool want = (o.io > 0 && step % o.io == 0) || step == o.steps;
        stepper->step(f[cur].data(), f[1 - cur].data(), ldf,
                      want ? macros : lbm::MacroPtrs<T>{});
        cur = 1 - cur;
        time += dt;
        if (want) diagnostics(step, time);
    }
    lbm::Exec<Space>::synchronize();
    auto tend = clock::now();

    const double elapsed = seconds(tstart, tend);
    const double updates = double(n) * o.steps;
    std::printf("elapsed  = %.3f s\n", elapsed);
    std::printf("MLUPS    = %.2f\n", updates * 1e-6 / elapsed);
    std::printf("GB/s     = %.1f (estimate from bytes touched per step)\n",
                double(stepper->bytes_per_step()) * o.steps / elapsed * 1e-9);
    return 0;
}
