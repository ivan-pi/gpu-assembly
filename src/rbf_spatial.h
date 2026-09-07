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

// x reduced into [0, L). The loops catch the case where x is so far
// outside that x - L*floor(x/L) rounds up to L itself.
template<typename T>
T fold(T x, T L) {
    x -= L * std::floor(x / L);
    while (x >= L) x -= L;
    while (x < 0) x += L;
    return x;
}

} // namespace detail

// The box [origin[d], origin[d] + period[d]) along each of D axes, with
// opposite faces identified. The Fortran type periodic_box in
// src/rbf_periodic_box.f90 is the 2-d case with the origin at zero.
template<typename T, std::size_t D>
struct PeriodicBox {
    static_assert(D >= 1, "a box needs at least one axis");

    std::array<T, D> origin{};  // lower corner
    std::array<T, D> period{};  // side lengths, all positive

    static constexpr std::size_t ndim = D;

    // The coordinate mapped into the box.
    T wrap(std::size_t d, T x) const {
        return origin[d] + detail::fold(x - origin[d], period[d]);
    }

    // The shortest of the displacement and its periodic images. Matches
    // Fortran's anint (half away from zero), so a displacement of
    // exactly half a period maps to -L/2.
    T minimum_image(std::size_t d, T dx) const {
        return dx - period[d] * std::round(dx / period[d]);
    }

    std::array<T, D> wrap(std::array<T, D> p) const {
        for (std::size_t d = 0; d < D; ++d) p[d] = wrap(d, p[d]);
        return p;
    }

    std::array<T, D> minimum_image(std::array<T, D> dx) const {
        for (std::size_t d = 0; d < D; ++d) dx[d] = minimum_image(d, dx[d]);
        return dx;
    }
};

// Per-axis coordinate arrays into the interleaved n*ndim layout KdTree
// takes, widened to double on the way, since ckdtree works in double:
//
//     auto points = rbf::spatial::interleave(x, y);      // 2-d
//     auto points = rbf::spatial::interleave(x, y, z);   // 3-d
template<std::ranges::contiguous_range Axis, std::ranges::contiguous_range... Rest>
std::vector<double> interleave(const Axis& x, const Rest&... rest) {
    constexpr std::size_t D = 1 + sizeof...(Rest);
    const std::size_t n = std::ranges::size(x);
    assert(((std::ranges::size(rest) == n) && ...) && "axes of different lengths");

    std::vector<double> out(n * D);
    std::size_t d = 0;
    const auto take = [&](const auto& axis) {
        const auto* a = std::ranges::data(axis);
        for (std::size_t i = 0; i < n; ++i)
            out[i*D + d] = static_cast<double>(a[i]);
        ++d;
    };
    take(x);
    (take(rest), ...);
    return out;
}

// Build parameters, named after SciPy's; they trade build time against
// query time and never change the answer.
struct KdTreeParams {
    int leafsize = 16;    // points below which a node is searched by brute force
    bool balanced = true; // split at the median rather than at the midpoint
    bool compact = true;  // shrink each node's box onto its points
};

namespace detail {

// How a box reaches the tree, whose dimension is a run-time value: two
// ndim-long ranges, or empty for the open plane. Read in the
// constructor and never held.
struct BoxView {
    std::span<const double> origin, period;
};

} // namespace detail

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
        : KdTree(points, ndim, detail::BoxView{}, params) {}

    // In a periodic box, which is also where the dimension comes from.
    // Points outside the box are wrapped into it, so the cloud need not
    // be pre-wrapped; the indices are unaffected either way.
    template<std::size_t D>
    KdTree(std::span<const double> points, const PeriodicBox<double, D>& box,
           KdTreeParams params = {})
        : KdTree(points, static_cast<int>(D),
                 detail::BoxView{box.origin, box.period}, params) {}

    ~KdTree();
    KdTree(KdTree&&) noexcept;
    KdTree& operator=(KdTree&&) noexcept;

    std::size_t size() const;  // points in the cloud
    int ndim() const;
    bool periodic() const;

    // The k nearest neighbours of each query point, sorted by distance.
    // q is nq*ndim interleaved; idx and dist are nq*k row-major, so the
    // neighbours of query s sit at idx[s*k]. dist holds true Euclidean
    // distances (minimum-image in a periodic box) and may be left empty
    // when only the indices are wanted. Query points are wrapped into
    // the box. OpenMP-parallel over the queries.
    void knn(std::span<const double> q, int k,
             std::span<std::intptr_t> idx, std::span<double> dist = {}) const;

    // The same, centred on the cloud's own points, in its own order.
    void knn(int k, std::span<std::intptr_t> idx,
             std::span<double> dist = {}) const;

    // Fixed-k stencils as ja(k, nq) in Fortran order: the k neighbours
    // of query s are contiguous at ja[s*k], sorted by distance. I is the
    // index type of the CsrMatrix<T, I> they will feed.
    template<typename I = std::int32_t>
    std::vector<I> stencils(std::span<const double> q, int k) const {
        const std::size_t nq = q.size() / static_cast<std::size_t>(ndim());
        std::vector<std::intptr_t> idx(nq * static_cast<std::size_t>(k));
        knn(q, k, idx);
        return narrow<I>(idx);
    }

    // Stencils centred on the cloud's own points, so ja[s*k] == s as
    // long as the cloud has no coincident points: a node is its own
    // nearest neighbour at distance zero, and a duplicate ties with it.
    template<typename I = std::int32_t>
    std::vector<I> stencils(int k) const {
        std::vector<std::intptr_t> idx(size() * static_cast<std::size_t>(k));
        knn(k, idx);
        return narrow<I>(idx);
    }

private:
    KdTree(std::span<const double> points, int ndim, detail::BoxView box,
           KdTreeParams params);

    template<typename I>
    static std::vector<I> narrow(const std::vector<std::intptr_t>& idx) {
        std::vector<I> ja(idx.size());
        for (std::size_t i = 0; i < idx.size(); ++i) {
            assert(static_cast<std::intptr_t>(static_cast<I>(idx[i])) == idx[i] &&
                   "stencil index does not fit the requested index type");
            ja[i] = static_cast<I>(idx[i]);
        }
        return ja;
    }

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rbf::spatial

#endif // RBF_SPATIAL_H
