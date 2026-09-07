// Tests for rbf_spatial.h / rbf_spatial.cpp: the periodic box and the
// ckdtree-backed k-nearest-neighbour search.
//
// Every search result is checked against an O(n*nq) brute-force search
// using the same minimum-image metric, on pseudo-random clouds. That is
// the only reference that catches a wrong box convention, which is what
// there is to get wrong here.
//
// Build and run via CMake, from the repository root:
//
//   cmake -B build
//   cmake --build build
//   ctest --test-dir build

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

#include "rbf_spatial.h"

#include "check.h"

using rbf::spatial::KdTree;
using rbf::spatial::PeriodicBox;

// A cheap reproducible generator; the clouds only need to be irregular.
struct Rng {
    std::uint64_t s;
    double next() {  // in [0, 1)
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<double>(s >> 11) * 0x1.0p-53;
    }
};

using Box = std::optional<PeriodicBox<double>>;

// Points spread over the box, or over the unit square when there is none.
static void cloud(Rng& rng, std::size_t n, const Box& b,
                  std::vector<double>& x, std::vector<double>& y) {
    const PeriodicBox<double> f = b ? *b : PeriodicBox<double>{};
    x.resize(n);
    y.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = f.x0 + f.Lx * rng.next();
        y[i] = f.y0 + f.Ly * rng.next();
    }
}

static double dist(const Box& b, double ax, double ay, double bx, double by) {
    const double dx = b ? b->minimum_image_x(ax - bx) : ax - bx;
    const double dy = b ? b->minimum_image_y(ay - by) : ay - by;
    return std::sqrt(dx*dx + dy*dy);
}

// The k nearest, by sorting every point of the cloud.
static std::vector<int> brute_knn(const Box& b,
                                  std::span<const double> x,
                                  std::span<const double> y,
                                  double qx, double qy, int k) {
    std::vector<int> p(x.size());
    std::iota(p.begin(), p.end(), 0);
    std::stable_sort(p.begin(), p.end(), [&](int a, int c) {
        return dist(b, x[a], y[a], qx, qy) < dist(b, x[c], y[c], qx, qy);
    });
    p.resize(static_cast<std::size_t>(k));
    return p;
}

// Compare a stencil row against the brute-force answer. Ties in distance
// are broken differently by the two, so the distances are what must
// agree; the indices only have to agree where the distance is distinct.
static void check_row(const Box& b,
                      std::span<const double> x, std::span<const double> y,
                      double qx, double qy, std::span<const std::int32_t> got) {
    const auto k = static_cast<int>(got.size());
    const auto want = brute_knn(b, x, y, qx, qy, k);
    for (int j = 0; j < k; ++j) {
        const double dg = dist(b, x[got[j]], y[got[j]], qx, qy);
        const double dw = dist(b, x[want[j]], y[want[j]], qx, qy);
        CHECK(std::abs(dg - dw) <= 1e-12 * (1.0 + dw));
        if (j > 0) {  // sorted by distance
            const double dprev = dist(b, x[got[j-1]], y[got[j-1]], qx, qy);
            CHECK(dprev <= dg + 1e-12);
        }
    }
}

static void test_box() {
    const PeriodicBox<double> b{-1.0, 2.0, 4.0, 3.0};

    // wrap lands in [x0, x0+L), from either side and from far away
    CHECK(std::abs(b.wrap_x(-1.0) - (-1.0)) < 1e-15);
    CHECK(std::abs(b.wrap_x(3.5) - (-0.5)) < 1e-15);
    CHECK(std::abs(b.wrap_x(-1.25) - 2.75) < 1e-15);
    CHECK(std::abs(b.wrap_x(-1.0 - 1000 * 4.0) - (-1.0)) < 1e-12);
    CHECK(b.wrap_x(-1.0 - 1e-18) >= -1.0 && b.wrap_x(-1.0 - 1e-18) < 3.0);
    CHECK(std::abs(b.wrap_y(2.0 + 3.0) - 2.0) < 1e-15);

    // minimum image is the shortest of the images, half a period away
    // from zero going to -L/2 as Fortran's anint does
    CHECK(std::abs(b.minimum_image_x(0.5) - 0.5) < 1e-15);
    CHECK(std::abs(b.minimum_image_x(3.0) - (-1.0)) < 1e-15);
    CHECK(std::abs(b.minimum_image_x(2.0) - (-2.0)) < 1e-15);
    CHECK(std::abs(b.minimum_image_y(-2.0) - 1.0) < 1e-15);

    // the default is the unit box at the origin
    const PeriodicBox<double> unit{};
    CHECK(std::abs(unit.wrap_x(-0.25) - 0.75) < 1e-15);
    CHECK(std::abs(unit.minimum_image_y(0.75) - (-0.25)) < 1e-15);
}

// The whole search API on one box: self-stencils, arbitrary queries,
// query points from outside the box, and the returned distances.
static void search_on(const char* what, const Box& b, std::size_t n, int k) {
    std::printf("  %s\n", what);
    Rng rng{0x9e3779b97f4a7c15ull ^ n};
    std::vector<double> x, y;
    cloud(rng, n, b, x, y);
    const KdTree tree(x, y, b);
    CHECK(tree.size() == n);
    CHECK(tree.periodic() == b.has_value());

    // stencils on the cloud itself: every node is its own first neighbour
    const auto ja = tree.stencils(k);
    CHECK(ja.size() == n * static_cast<std::size_t>(k));
    for (std::size_t s = 0; s < n; ++s) {
        CHECK(ja[s*k] == static_cast<std::int32_t>(s));
        check_row(b, x, y, x[s], y[s],
                  std::span{ja}.subspan(s*k, static_cast<std::size_t>(k)));
    }

    // arbitrary query points, deliberately shifted a whole period out of
    // the box so that the wrapping of the query is exercised too
    const std::size_t nq = 64;
    std::vector<double> qx, qy;
    cloud(rng, nq, b, qx, qy);
    if (b)
        for (std::size_t s = 0; s < nq; ++s) {
            qx[s] += 3 * b->Lx;
            qy[s] -= 2 * b->Ly;
        }
    const auto jq = tree.stencils(std::span<const double>{qx},
                                  std::span<const double>{qy}, k);
    CHECK(jq.size() == nq * static_cast<std::size_t>(k));
    for (std::size_t s = 0; s < nq; ++s)
        check_row(b, x, y, qx[s], qy[s],
                  std::span{jq}.subspan(s*k, static_cast<std::size_t>(k)));

    // the distances that come with the indices are the true ones
    std::vector<std::intptr_t> idx(nq * static_cast<std::size_t>(k));
    std::vector<double> d(idx.size());
    tree.knn(qx, qy, k, idx, d);
    for (std::size_t i = 0; i < idx.size(); ++i) {
        const auto s = i / static_cast<std::size_t>(k);
        const double want = dist(b, x[idx[i]], y[idx[i]], qx[s], qy[s]);
        CHECK(std::abs(d[i] - want) <= 1e-12 * (1.0 + want));
        CHECK(idx[i] == jq[i]);
    }
}

static void test_search() {
    search_on("open plane", std::nullopt, 500, 8);
    search_on("unit box", PeriodicBox<double>{}, 500, 8);
    search_on("shifted, anisotropic box",
              PeriodicBox<double>{-3.0, 1.5, 2.0, 5.0}, 700, 12);
    // k = n: the stencil is the whole cloud, the hardest case for the
    // pruning, and a thin box where a neighbour may be reached by more
    // than one image
    search_on("k = n on a thin box",
              PeriodicBox<double>{0.0, 0.0, 1.0, 0.05}, 40, 40);
}

// A cloud whose neighbours differ between the open plane and the box:
// the four corners of the unit square sit 0.04 apart across the
// boundary but 0.68 from the point in the middle, so identifying the
// sides changes who every corner's neighbour is.
static void test_periodicity_matters() {
    const std::vector<double> x{0.02, 0.98, 0.02, 0.98, 0.5};
    const std::vector<double> y{0.02, 0.02, 0.98, 0.98, 0.5};
    const PeriodicBox<double> box{};

    const auto open = KdTree(x, y).stencils(2);
    const auto wrapped = KdTree(x, y, box).stencils(2);

    // Each corner has two neighbours at 0.04, one across each side, so
    // which one comes back is a tie; the distance is the assertion.
    for (int s = 0; s < 4; ++s) {
        CHECK(open[s*2 + 1] == 4);  // in the plane, the middle point
        const int nb = wrapped[s*2 + 1];
        CHECK(nb != 4);
        CHECK(std::abs(dist(box, x[s], y[s], x[nb], y[nb]) - 0.04) < 1e-12);
    }
}

// The small clouds, where the tree is a single leaf and the build's
// degenerate paths are the ones taken.
static void test_degenerate() {
    const PeriodicBox<double> box{};

    const std::vector<double> one_x{0.25}, one_y{0.75};
    const KdTree one(one_x, one_y, box);
    CHECK(one.size() == 1);
    CHECK(one.stencils(1) == (std::vector<std::int32_t>{0}));

    // every point identical: the build bails out of splitting, and all
    // 16 tie at distance 0, so no node is guaranteed to lead its own row
    const std::vector<double> same_x(16, 0.5), same_y(16, 0.5);
    const auto ja = KdTree(same_x, same_y, box).stencils(4);
    CHECK(ja.size() == 16 * 4);
    for (auto j : ja) CHECK(j >= 0 && j < 16);

    // collinear points, so one dimension has no spread at all
    std::vector<double> line_x(32), line_y(32, 0.5);
    for (int i = 0; i < 32; ++i) line_x[i] = i / 32.0;
    const auto jl = KdTree(line_x, line_y, box).stencils(3);
    for (int s = 0; s < 32; ++s) {
        CHECK(jl[s*3] == s);
        check_row(box, line_x, line_y, line_x[s], line_y[s],
                  std::span{jl}.subspan(static_cast<std::size_t>(s)*3, 3));
    }
}

// The T-taking entry points: the same answers through the one-shot
// functions, and through a float cloud.
static void test_free_functions() {
    using rbf::spatial::knn_stencils;
    const PeriodicBox<double> box{};
    Rng rng{12345};
    std::vector<double> x, y;
    cloud(rng, 300, box, x, y);

    const auto direct = KdTree(x, y, box).stencils(5);
    const auto once = knn_stencils<double>(x, y, 5, box);
    CHECK(direct == once);

    const auto self = knn_stencils<double>(x, y, std::span<const double>{x},
                                           std::span<const double>{y}, 5, box);
    CHECK(direct == self);

    // a float cloud: the same neighbours, since the coordinates survive
    // the widening exactly
    std::vector<float> xf(x.begin(), x.end()), yf(y.begin(), y.end());
    std::vector<double> xw(xf.begin(), xf.end()), yw(yf.begin(), yf.end());
    const PeriodicBox<float> boxf{};
    const auto as_float = knn_stencils<float>(xf, yf, 5, boxf);
    const auto as_double = KdTree(xw, yw, box).stencils(5);
    CHECK(as_float == as_double);

    // a narrower index type still round-trips
    const auto narrow = knn_stencils<double, std::int64_t>(x, y, 5, box);
    CHECK(narrow.size() == direct.size());
    for (std::size_t i = 0; i < narrow.size(); ++i)
        CHECK(narrow[i] == direct[i]);
}

int main() {
    test_box();
    test_search();
    test_periodicity_matters();
    test_degenerate();
    test_free_functions();
    return report("spatial");
}
