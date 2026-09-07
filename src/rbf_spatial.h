#ifndef RBF_SPATIAL_H
#define RBF_SPATIAL_H

// rbf::spatial -- the point-cloud search structures the rest of the
// library builds on, and the periodic box they search in.
//
// The k-d tree is SciPy's ckdtree, vendored under third_party/ckdtree
// and built as the `ckdtree` target. It was chosen over a tiled search
// because it handles periodicity in the metric itself: a tree built
// with a box measures distances under the minimum-image convention, so
// a query costs the same as in the open plane instead of 9x (3^d - 1
// images and the cloud) the points. Queries may be arbitrary points,
// not just nodes, which is what departure-point-centred stencils need.
//
// ckdtree carries the dimension at run time, and so does KdTree, in the
// interleaved layout ckdtree itself uses: the library is 2-d today, and
// 3-d costs a different ndim and nothing else. Coordinates are double,
// because ckdtree's are. This is the place for a quadtree or a ball
// tree, should one be needed.

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <vector>

namespace rbf::spatial {

namespace detail {

// x reduced into [0, L). The guard is not paranoia and not about large
// x: for x a hair below zero, say -1e-30, the difference rounds up to L
// itself and lands outside the interval. One pass fixes it; ckdtree's
// own wrap_position guards the same way.
template <typename T>
T fold(T x, T L) {
    x -= L * std::floor(x / L);
    while (x >= L)
        x -= L;
    while (x < 0)
        x += L;
    return x;
}

}  // namespace detail

// The box [0, period[d]) along each of D axes, with opposite faces
// identified. Anchored at the origin, as SciPy's boxsize is and as the
// Fortran type periodic_box in src/rbf_periodic_box.f90 is; a cloud
// that lives somewhere else is translated by the caller, once, rather
// than by the box on every wrap.
template <typename T, std::size_t D>
struct PeriodicBox {
    static_assert(D >= 1, "a box needs at least one axis");

    std::array<T, D> period{};  // side lengths, all positive

    // The coordinate mapped into the box.
    T wrap(std::size_t d, T x) const { return detail::fold(x, period[d]); }

    // The shortest of the displacement and its periodic images.
    //
    // rint, not round: this runs in the assembly inner loop. std::round
    // is a libm call on gcc at every -march, std::rint an instruction
    // on gcc at the baseline ISA and on both compilers from SSE4.1
    // (-march=x86-64-v2) up. They differ only for a displacement of
    // exactly half a period, where the two images are equidistant and
    // either answer is as good -- rint takes the even one, Fortran's
    // anint the one away from zero.
    T minimum_image(std::size_t d, T dx) const {
        return dx - period[d] * std::rint(dx / period[d]);
    }

    std::array<T, D> wrap(std::array<T, D> p) const {
        for (std::size_t d = 0; d < D; ++d)
            p[d] = wrap(d, p[d]);
        return p;
    }

    std::array<T, D> minimum_image(std::array<T, D> dx) const {
        for (std::size_t d = 0; d < D; ++d)
            dx[d] = minimum_image(d, dx[d]);
        return dx;
    }
};

// Per-axis coordinate arrays into the interleaved n*ndim layout KdTree
// takes, widened to double on the way, since ckdtree works in double:
//
//     auto points = rbf::spatial::interleave(x, y);      // 2-d
//     auto points = rbf::spatial::interleave(x, y, z);   // 3-d
template <std::ranges::contiguous_range Axis, std::ranges::contiguous_range... Rest>
std::vector<double> interleave(const Axis& x, const Rest&... rest) {
    constexpr std::size_t D = 1 + sizeof...(Rest);
    const std::size_t n = std::ranges::size(x);
    assert(((std::ranges::size(rest) == n) && ...) && "axes of different lengths");

    std::vector<double> out(n * D);
    std::size_t d = 0;
    const auto take = [&](const auto& axis) {
        const auto* a = std::ranges::data(axis);
        for (std::size_t i = 0; i < n; ++i)
            out[i * D + d] = static_cast<double>(a[i]);
        ++d;
    };
    take(x);
    (take(rest), ...);
    return out;
}

// Build parameters, named after SciPy's; they trade build time against
// query time and never change the answer.
struct KdTreeParams {
    int leafsize = 16;     // points below which a node is searched by brute force
    bool balanced = true;  // split at the median rather than at the midpoint
    bool compact = true;   // shrink each node's box onto its points
};

// A k-d tree over a fixed point cloud, in any number of dimensions,
// periodic or not.
//
// Points come in ckdtree's interleaved layout: coordinate d of point i
// at points[i*ndim + d]. interleave() builds it from per-axis arrays.
// The cloud is copied in, so the caller's arrays need not outlive the
// tree, and the indices the queries return refer to it in the order it
// was passed. The tree is immutable once built, so const queries are
// thread-safe.
class KdTree {
public:
    // In the open plane.
    KdTree(std::span<const double> points, int ndim, KdTreeParams params = {})
        : KdTree(points, ndim, std::span<const double>{}, params) {}

    // In a periodic box, which is also where the dimension comes from.
    // Points outside the box are wrapped into it, so the cloud need not
    // be pre-wrapped; the indices are unaffected either way.
    template <std::size_t D>
    KdTree(std::span<const double> points,
           const PeriodicBox<double, D>& box,
           KdTreeParams params = {})
        : KdTree(points, static_cast<int>(D), std::span<const double>{box.period}, params) {}

    ~KdTree();
    KdTree(KdTree&&) noexcept;
    KdTree& operator=(KdTree&&) noexcept;

    std::size_t size() const;  // points in the cloud
    int ndim() const;
    bool periodic() const;

    // The k nearest neighbours of each query point, sorted by distance
    // -- cKDTree.query, with its eps, p and distance_upper_bound fixed
    // at the exact Euclidean search.
    //
    // q is nq*ndim interleaved; idx and dist are nq*k row-major, so the
    // neighbours of query s sit at idx[s*k]. dist holds true Euclidean
    // distances (minimum-image in a periodic box) and may be left empty
    // when only the indices are wanted. Query points are wrapped into
    // the box. OpenMP-parallel over the queries, which is what SciPy's
    // `workers` does too.
    void query(std::span<const double> q,
               int k,
               std::span<std::intptr_t> idx,
               std::span<double> dist = {}) const;

    // The same, centred on the cloud's own points, in its own order.
    void query(int k, std::span<std::intptr_t> idx, std::span<double> dist = {}) const;

    // Fixed-k stencils as ja(k, nq) in Fortran order: the k nearest
    // neighbours of query s are contiguous at ja[s*k], sorted by
    // distance, as int32_t. ckdtree produces intptr_t; the narrowing
    // happens per block inside the parallel query, so the result never
    // exists in the wider type. A caller who wants intptr_t has query().
    std::vector<std::int32_t> knn_stencils(std::span<const double> q, int k) const;

    // Stencils centred on the cloud's own points, so ja[s*k] == s as
    // long as the cloud has no coincident points: a node is its own
    // nearest neighbour at distance zero, and a duplicate ties with it.
    std::vector<std::int32_t> knn_stencils(int k) const;

private:
    // period is empty in the open plane, else one side length per axis.
    KdTree(std::span<const double> points,
           int ndim,
           std::span<const double> period,
           KdTreeParams params);

    std::span<const double> points() const;  // the cloud, as stored

    // The ckdtree struct lives behind this pointer for two reasons: its
    // header defines global names (struct ckdtree, ckdtree_fabs and
    // friends) that should not reach every includer, and it points into
    // buffers Impl owns, so Impl must never move.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rbf::spatial

#endif  // RBF_SPATIAL_H
