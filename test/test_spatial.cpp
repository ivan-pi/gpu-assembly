// Tests for rbf_spatial.h / rbf_spatial.cpp: the periodic box and the
// ckdtree-backed k-nearest-neighbour search.
//
// Every search result is checked against an O(n*nq) brute-force search
// using the same minimum-image metric, on pseudo-random clouds. That is
// the only reference that catches a wrong box convention, which is what
// there is to get wrong here. The searches run in 1, 2 and 3 dimensions,
// since the tree carries the dimension at run time.
//
// Build and run via CMake, from the repository root:
//
//   cmake -B build
//   cmake --build build
//   ctest --test-dir build

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

#include "rbf_spatial.h"

#include "check.h"

using rbf::spatial::interleave;
using rbf::spatial::KdTree;
using rbf::spatial::PeriodicBox;

template<std::size_t D>
using Box = std::optional<PeriodicBox<double, D>>;

// A cheap reproducible generator; the clouds only need to be irregular.
struct Rng {
    std::uint64_t s;
    double next() {  // in [0, 1)
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return static_cast<double>(s >> 11) * 0x1.0p-53;
    }
};

// Points spread over the box, or over the unit cube when there is none.
template<std::size_t D>
static std::vector<double> cloud(Rng& rng, std::size_t n, const Box<D>& b) {
    std::vector<double> p(n * D);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t d = 0; d < D; ++d)
            p[i*D + d] = (b ? b->period[d] : 1.0) * rng.next();
    return p;
}

template<std::size_t D>
static double dist(const Box<D>& b, const double* p, const double* q) {
    double s = 0;
    for (std::size_t d = 0; d < D; ++d) {
        const double e = b ? b->minimum_image(d, p[d] - q[d]) : p[d] - q[d];
        s += e * e;
    }
    return std::sqrt(s);
}

// The k nearest, by sorting every point of the cloud.
template<std::size_t D>
static std::vector<int> brute_knn(const Box<D>& b, std::span<const double> pts,
                                  const double* q, int k) {
    std::vector<int> p(pts.size() / D);
    std::iota(p.begin(), p.end(), 0);
    std::stable_sort(p.begin(), p.end(), [&](int a, int c) {
        return dist<D>(b, &pts[a*D], q) < dist<D>(b, &pts[c*D], q);
    });
    p.resize(static_cast<std::size_t>(k));
    return p;
}

// Compare a stencil row against the brute-force answer. Ties in distance
// are broken differently by the two, so the distances are what must
// agree; the indices only have to agree where the distance is distinct.
// The brute-force row is sorted, so matching it entry by entry also
// checks that the tree's row is.
template<std::size_t D>
static void check_row(const Box<D>& b, std::span<const double> pts,
                      const double* q, const std::int32_t* got, int k) {
    const auto want = brute_knn<D>(b, pts, q, k);
    for (int j = 0; j < k; ++j) {
        const double dg = dist<D>(b, &pts[got[j]*D], q);
        const double dw = dist<D>(b, &pts[want[j]*D], q);
        CHECK(std::abs(dg - dw) <= 1e-12 * (1.0 + dw));
    }
}

static void test_box() {
    // an aggregate: braces, or a designated initialiser
    const PeriodicBox<double, 2> b{{4.0, 3.0}};
    const PeriodicBox<double, 2> same{.period = {4.0, 3.0}};
    CHECK(b.period == same.period);

    // wrap lands in [0, L), from either side and from far away
    CHECK(std::abs(b.wrap(0, 0.0)) < 1e-15);
    CHECK(std::abs(b.wrap(0, 4.5) - 0.5) < 1e-15);
    CHECK(std::abs(b.wrap(0, -0.25) - 3.75) < 1e-15);
    CHECK(std::abs(b.wrap(0, -1000 * 4.0)) < 1e-12);
    // the case fold's guard is for: a hair below zero
    CHECK(b.wrap(0, -1e-30) >= 0.0 && b.wrap(0, -1e-30) < 4.0);
    CHECK(std::abs(b.wrap(1, 3.0)) < 1e-15);

    // minimum image is the shortest of the images
    CHECK(std::abs(b.minimum_image(0, 0.5) - 0.5) < 1e-15);
    CHECK(std::abs(b.minimum_image(0, 3.0) - (-1.0)) < 1e-15);
    CHECK(std::abs(b.minimum_image(1, -2.0) - 1.0) < 1e-15);
    // exactly half a period: the two images are equidistant, so only
    // the magnitude is pinned down
    CHECK(std::abs(std::abs(b.minimum_image(0, 2.0)) - 2.0) < 1e-15);

    // the whole-point forms agree with the per-axis ones
    const std::array<double, 2> p{4.5, 5.0};
    const auto w = b.wrap(p);
    CHECK(std::abs(w[0] - b.wrap(0, p[0])) < 1e-15);
    CHECK(std::abs(w[1] - b.wrap(1, p[1])) < 1e-15);
    const auto mi = b.minimum_image(std::array<double, 2>{3.0, -2.0});
    CHECK(std::abs(mi[0] - (-1.0)) < 1e-15);
    CHECK(std::abs(mi[1] - 1.0) < 1e-15);

    // nothing about it is 2-d
    const PeriodicBox<double, 3> c{{1.0, 2.0, 4.0}};
    CHECK(std::abs(c.wrap(2, 4.25) - 0.25) < 1e-15);
    CHECK(std::abs(c.minimum_image(2, 3.0) - (-1.0)) < 1e-15);
}

static void test_interleave() {
    const std::vector<double> x{0.0, 1.0, 2.0};
    const std::vector<double> y{10.0, 11.0, 12.0};
    const std::vector<float> z{20.0f, 21.0f, 22.0f};

    const auto xy = interleave(x, y);
    CHECK(xy == (std::vector<double>{0, 10, 1, 11, 2, 12}));

    // mixed coordinate types are widened to double
    const auto xyz = interleave(x, y, z);
    CHECK(xyz == (std::vector<double>{0, 10, 20, 1, 11, 21, 2, 12, 22}));

    // spans work as well as containers
    const auto only_x = interleave(std::span<const double>{x});
    CHECK(only_x == x);
}

// The whole search API on one box: self-stencils, arbitrary queries,
// query points from outside the box, and the returned distances.
template<std::size_t D>
static void search_on(const char* what, const Box<D>& b, std::size_t n, int k) {
    std::printf("  %zud: %s\n", D, what);
    Rng rng{0x9e3779b97f4a7c15ull ^ (n * 31 + D)};
    const auto pts = cloud<D>(rng, n, b);
    const KdTree tree = b ? KdTree(pts, *b) : KdTree(pts, static_cast<int>(D));
    CHECK(tree.size() == n);
    CHECK(tree.ndim() == static_cast<int>(D));
    CHECK(tree.periodic() == b.has_value());

    // stencils on the cloud itself: every node is its own first neighbour
    const auto ja = tree.stencils(k);
    CHECK(ja.size() == n * static_cast<std::size_t>(k));
    for (std::size_t s = 0; s < n; ++s) {
        CHECK(ja[s*k] == static_cast<std::int32_t>(s));
        check_row<D>(b, pts, &pts[s*D], &ja[s*k], k);
    }

    // arbitrary query points, deliberately shifted whole periods out of
    // the box so that the wrapping of the query is exercised too
    const std::size_t nq = 64;
    auto q = cloud<D>(rng, nq, b);
    if (b)
        for (std::size_t s = 0; s < nq; ++s)
            for (std::size_t d = 0; d < D; ++d)
                q[s*D + d] += (d % 2 ? -2.0 : 3.0) * b->period[d];

    const auto jq = tree.stencils(q, k);
    CHECK(jq.size() == nq * static_cast<std::size_t>(k));
    for (std::size_t s = 0; s < nq; ++s)
        check_row<D>(b, pts, &q[s*D], &jq[s*k], k);

    // the distances that come with the indices are the true ones
    std::vector<std::intptr_t> idx(nq * static_cast<std::size_t>(k));
    std::vector<double> d(idx.size());
    tree.query(q, k, idx, d);
    for (std::size_t i = 0; i < idx.size(); ++i) {
        const auto s = i / static_cast<std::size_t>(k);
        const double want = dist<D>(b, &pts[idx[i]*D], &q[s*D]);
        CHECK(std::abs(d[i] - want) <= 1e-12 * (1.0 + want));
        CHECK(idx[i] == jq[i]);
    }
}

static void test_search() {
    search_on<1>("open line", std::nullopt, 200, 5);
    search_on<1>("ring", PeriodicBox<double, 1>{{5.0}}, 200, 5);

    search_on<2>("open plane", std::nullopt, 500, 8);
    search_on<2>("unit box", PeriodicBox<double, 2>{{1.0, 1.0}}, 500, 8);
    search_on<2>("anisotropic box",
                 PeriodicBox<double, 2>{{2.0, 5.0}}, 700, 12);
    // k = n: the stencil is the whole cloud, the hardest case for the
    // pruning, and a thin box where a neighbour may be reached by more
    // than one image
    search_on<2>("k = n on a thin box",
                 PeriodicBox<double, 2>{{1.0, 0.05}}, 40, 40);

    search_on<3>("open space", std::nullopt, 600, 10);
    search_on<3>("unit cube",
                 PeriodicBox<double, 3>{{1.0, 1.0, 1.0}}, 600, 10);
    search_on<3>("anisotropic cell",
                 PeriodicBox<double, 3>{{1.0, 3.0, 0.5}}, 600, 16);
}

// A cloud whose neighbours differ between the open plane and the box:
// the four corners of the unit square sit 0.04 apart across the
// boundary but 0.68 from the point in the middle, so identifying the
// sides changes who every corner's neighbour is.
static void test_periodicity_matters() {
    const std::vector<double> x{0.02, 0.98, 0.02, 0.98, 0.5};
    const std::vector<double> y{0.02, 0.02, 0.98, 0.98, 0.5};
    const auto pts = interleave(x, y);
    const Box<2> box = PeriodicBox<double, 2>{{1.0, 1.0}};

    const auto open = KdTree(pts, 2).stencils(2);
    const auto wrapped = KdTree(pts, *box).stencils(2);

    // Each corner has two neighbours at 0.04, one across each side, so
    // which one comes back is a tie; the distance is the assertion.
    for (int s = 0; s < 4; ++s) {
        CHECK(open[s*2 + 1] == 4);  // in the plane, the middle point
        const int nb = wrapped[s*2 + 1];
        CHECK(nb != 4);
        CHECK(std::abs(dist<2>(box, &pts[s*2], &pts[nb*2]) - 0.04) < 1e-12);
    }
}

// The small clouds, where the tree is a single leaf and the build's
// degenerate paths are the ones taken.
static void test_degenerate() {
    const Box<2> box = PeriodicBox<double, 2>{{1.0, 1.0}};

    const std::vector<double> one{0.25, 0.75};
    const KdTree tree(one, *box);
    CHECK(tree.size() == 1);
    CHECK(tree.stencils(1) == (std::vector<std::int32_t>{0}));

    // every point identical: the build bails out of splitting, and all
    // 16 tie at distance 0, so no node is guaranteed to lead its own row
    const std::vector<double> same(32, 0.5);
    const auto ja = KdTree(same, *box).stencils(4);
    CHECK(ja.size() == 16 * 4);
    for (auto j : ja) CHECK(j >= 0 && j < 16);

    // collinear points, so one dimension has no spread at all
    std::vector<double> line_x(32), line_y(32, 0.5);
    for (int i = 0; i < 32; ++i) line_x[i] = i / 32.0;
    const auto pts = interleave(line_x, line_y);
    const auto jl = KdTree(pts, *box).stencils(3);
    for (int s = 0; s < 32; ++s) {
        CHECK(jl[s*3] == s);
        check_row<2>(box, pts, &pts[s*2], &jl[s*3], 3);
    }
}

// The two index types give the same stencils, and the raw query gives
// the same indices as either.
static void test_index_types() {
    Rng rng{12345};
    const Box<2> box = PeriodicBox<double, 2>{{1.0, 1.0}};
    const auto pts = cloud<2>(rng, 300, box);
    const KdTree tree(pts, *box);

    const auto narrow = tree.stencils<std::int32_t>(5);
    const auto wide = tree.stencils<std::int64_t>(5);
    CHECK(std::equal(narrow.begin(), narrow.end(), wide.begin(), wide.end()));

    std::vector<std::intptr_t> raw(300 * 5);
    tree.query(5, raw);
    CHECK(std::equal(narrow.begin(), narrow.end(), raw.begin(), raw.end()));
}

int main() {
    test_box();
    test_interleave();
    test_search();
    test_periodicity_matters();
    test_degenerate();
    test_index_types();
    return report("spatial");
}
