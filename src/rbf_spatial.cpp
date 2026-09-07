// The ckdtree glue behind rbf::spatial::KdTree.
//
// SciPy drives ckdtree from Python: the ckdtree struct is a plain view
// onto numpy arrays owned by the wrapper class, and the wrapper is
// responsible for the invariants the C++ side assumes. Impl below is
// that wrapper. The three obligations worth naming:
//
//   - raw_boxsize_data is 2m long, the m periods followed by their
//     halves, and the data must already lie in [0, L). Hence the wrap
//     in the constructor. Query points need no such care: query_knn
//     wraps them itself.
//   - build_ckdtree overwrites the maxes/mins it is given, so it gets
//     copies; raw_maxes and raw_mins must survive as the root node's
//     bounding box for the queries.
//   - the node buffer grows during the build, so every ckdtreenode's
//     less/greater pointers are stale by the end of it. They have to be
//     rebuilt from the _less/_greater indices, which stay valid. See
//     link_nodes below; SciPy does this in cKDTree._post_init.

#include "rbf_spatial.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>

#include "ckdtree_decl.h"

static_assert(std::is_same_v<ckdtree_intp_t, std::intptr_t>,
              "ckdtree's index type must match the one KdTree exposes");

namespace rbf::spatial {

namespace {

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "rbf::spatial: %s\n", what);
    std::exit(1);
}

// Turn the _less/_greater node indices back into pointers, after the
// buffer has stopped moving. SciPy walks the tree from the root; a flat
// pass is the same set of nodes, because the one place the build
// abandons a node (the retry when all points tie in the split
// dimension) pops it off the buffer first.
void link_nodes(std::vector<ckdtreenode>& nodes) {
    ckdtreenode* root = nodes.data();
    const auto count = static_cast<std::intptr_t>(nodes.size());
    for (auto& n : nodes) {
        if (n.split_dim == -1) {  // leaf
            n.less = nullptr;
            n.greater = nullptr;
        } else {
            assert(n._less >= 0 && n._less < count && n._greater >= 0 && n._greater < count &&
                   "child index outside the node buffer");
            n.less = root + n._less;
            n.greater = root + n._greater;
        }
    }
}

}  // namespace

struct KdTree::Impl {
    std::vector<double> data;            // n*m, interleaved, inside the box
    std::vector<std::intptr_t> indices;  // permuted by the build
    std::vector<double> mins, maxes;     // the root node's bounding box
    std::vector<double> boxsize;         // 2m; empty in the open plane
    std::vector<ckdtreenode> nodes;
    ckdtree tree{};  // n, m and the pointers into the above
};

KdTree::KdTree(std::span<const double> points,
               int ndim,
               std::span<const double> period,
               KdTreeParams params)
    : impl_(std::make_unique<Impl>()) {
    if (ndim < 1) fail("KdTree: ndim must be positive");
    const auto m = static_cast<std::size_t>(ndim);
    if (points.empty()) fail("KdTree: empty point cloud");
    if (points.size() % m != 0) fail("KdTree: points is not a whole number of ndim-vectors");
    if (params.leafsize < 1) fail("KdTree: leafsize must be positive");
    // As SciPy does, and for a sharper reason than tidiness: a NaN makes
    // the build's split comparator inconsistent, which is undefined
    // behaviour in nth_element, not merely a bad tree.
    for (const double v : points)
        if (!std::isfinite(v)) fail("KdTree: coordinates must be finite");

    const bool periodic = !period.empty();
    if (periodic && period.size() != m) fail("KdTree: box and points have different dimensions");

    const std::size_t n = points.size() / m;
    auto& im = *impl_;

    im.data.assign(points.begin(), points.end());
    if (periodic) {
        for (const double L : period)
            if (L <= 0) fail("KdTree: box sides must be positive");

        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t d = 0; d < m; ++d)
                im.data[i * m + d] = detail::fold(im.data[i * m + d], period[d]);

        im.boxsize.resize(2 * m);
        for (std::size_t d = 0; d < m; ++d) {
            im.boxsize[d] = period[d];
            im.boxsize[m + d] = 0.5 * period[d];
        }
    }

    im.indices.resize(n);
    std::iota(im.indices.begin(), im.indices.end(), std::intptr_t{0});

    im.mins.assign(im.data.begin(), im.data.begin() + m);
    im.maxes = im.mins;
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t d = 0; d < m; ++d) {
            im.mins[d] = std::min(im.mins[d], im.data[i * m + d]);
            im.maxes[d] = std::max(im.maxes[d], im.data[i * m + d]);
        }

    auto& t = im.tree;
    t.tree_buffer = &im.nodes;
    t.raw_data = im.data.data();
    t.n = static_cast<std::intptr_t>(n);
    t.m = static_cast<std::intptr_t>(m);
    t.leafsize = params.leafsize;
    t.raw_maxes = im.maxes.data();
    t.raw_mins = im.mins.data();
    t.raw_indices = im.indices.data();
    t.raw_boxsize_data = im.boxsize.empty() ? nullptr : im.boxsize.data();

    std::vector<double> build_maxes = im.maxes, build_mins = im.mins;
    build_ckdtree(&t, 0, t.n, build_maxes.data(), build_mins.data(), params.balanced,
                  params.compact);

    link_nodes(im.nodes);
    t.ctree = im.nodes.data();
    t.size = static_cast<std::intptr_t>(im.nodes.size());
}

KdTree::~KdTree() = default;
KdTree::KdTree(KdTree&&) noexcept = default;
KdTree& KdTree::operator=(KdTree&&) noexcept = default;

std::size_t KdTree::size() const { return static_cast<std::size_t>(impl_->tree.n); }

int KdTree::ndim() const { return static_cast<int>(impl_->tree.m); }

bool KdTree::periodic() const { return !impl_->boxsize.empty(); }

std::span<const double> KdTree::points() const { return impl_->data; }

namespace {

// The checks shared by every query entry point; returns the number of
// query points.
std::size_t checked_query(const ckdtree& tree,
                          std::span<const double> q,
                          int k,
                          std::size_t idx_size,
                          std::size_t dist_size) {
    const auto m = static_cast<std::size_t>(tree.m);
    if (q.size() % m != 0) fail("KdTree::query: q is not a whole number of ndim-vectors");
    if (k < 1 || static_cast<std::size_t>(k) > static_cast<std::size_t>(tree.n))
        fail("KdTree::query: k outside [1, number of points]");
    const std::size_t nq = q.size() / m;
    const std::size_t want = nq * static_cast<std::size_t>(k);
    if (idx_size != want) fail("KdTree::query: idx must hold nq*k indices");
    if (dist_size != 0 && dist_size != want)
        fail("KdTree::query: dist must be empty or hold nq*k distances");
    return nq;
}

// Runs query_knn over blocks of query points, one thread per block,
// writing the indices as I. ckdtree writes intptr_t; for a narrower I
// each block goes through a per-thread staging row and is narrowed
// from there, so the result never exists as an intptr_t array.
//
// Safe because query_knn takes the tree by const pointer and keeps
// every scrap of query state local -- the node pool, both heaps and the
// wrapped query point are all locals of query_single_point, and the
// only static in the file is a function-local const infinity. SciPy
// leans on the same property: cKDTree.query's `workers` splits the
// query points into contiguous ranges and calls query_knn on the shared
// tree from several threads with the GIL released.
//
// The block is scheduling granularity, nothing more: query_single_point
// allocates its own scratch per point whatever we do, so blocking
// amortises only query_knn's own row buffer and the loop overhead.
template <typename I>
void query_many(const ckdtree& tree, const double* q, std::size_t nq, int k, I* out, double* dist) {
    // query_knn selects ranks out of the sorted neighbours; the k
    // nearest are ranks 1..k.
    std::vector<std::intptr_t> ranks(static_cast<std::size_t>(k));
    std::iota(ranks.begin(), ranks.end(), std::intptr_t{1});
    const auto m = static_cast<std::size_t>(tree.m);
    const auto kk = static_cast<std::size_t>(k);
    const double inf = std::numeric_limits<double>::infinity();

    constexpr std::size_t block = 256;
    const auto nblocks = static_cast<std::ptrdiff_t>((nq + block - 1) / block);
    constexpr bool direct = std::is_same_v<I, std::intptr_t>;

#pragma omp parallel
    {
        // Per-thread staging: distances the caller did not ask for, and
        // indices in ckdtree's type when the caller's is narrower.
        std::vector<double> scratch(dist ? 0 : block * kk);
        std::vector<std::intptr_t> stage(direct ? 0 : block * kk);
#pragma omp for schedule(static)
        for (std::ptrdiff_t b = 0; b < nblocks; ++b) {
            const std::size_t s0 = static_cast<std::size_t>(b) * block;
            const std::size_t cnt = std::min(block, nq - s0);
            std::intptr_t* idx;
            if constexpr (direct)
                idx = out + s0 * kk;
            else
                idx = stage.data();
            query_knn(&tree, dist ? dist + s0 * kk : scratch.data(), idx, q + s0 * m,
                      static_cast<std::intptr_t>(cnt), ranks.data(), k, k, /*eps=*/0.0, /*p=*/2.0,
                      /*distance_upper_bound=*/inf);
            if constexpr (!direct)
                for (std::size_t i = 0; i < cnt * kk; ++i) {
                    assert(static_cast<std::intptr_t>(static_cast<I>(stage[i])) == stage[i] &&
                           "stencil index does not fit the requested index type");
                    out[s0 * kk + i] = static_cast<I>(stage[i]);
                }
        }
    }
}

}  // namespace

void KdTree::query(std::span<const double> q,
                   int k,
                   std::span<std::intptr_t> idx,
                   std::span<double> dist) const {
    const auto& tree = impl_->tree;
    const std::size_t nq = checked_query(tree, q, k, idx.size(), dist.size());
    if (nq == 0) return;
    query_many(tree, q.data(), nq, k, idx.data(), dist.empty() ? nullptr : dist.data());
}

void KdTree::query(int k, std::span<std::intptr_t> idx, std::span<double> dist) const {
    query(points(), k, idx, dist);
}

std::vector<std::int32_t> KdTree::knn_stencils(std::span<const double> q, int k) const {
    const auto& tree = impl_->tree;
    const std::size_t nq = q.size() / static_cast<std::size_t>(tree.m);
    std::vector<std::int32_t> ja(nq * static_cast<std::size_t>(k));
    checked_query(tree, q, k, ja.size(), 0);
    if (nq > 0) query_many(tree, q.data(), nq, k, ja.data(), nullptr);
    return ja;
}

std::vector<std::int32_t> KdTree::knn_stencils(int k) const { return knn_stencils(points(), k); }

}  // namespace rbf::spatial
