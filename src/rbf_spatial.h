#ifndef RBF_SPATIAL_H
#define RBF_SPATIAL_H

// rbf::spatial -- the point-cloud search structures the rest of the
// library builds on, and the periodic box they search in.
//
// The k-d tree is SciPy's ckdtree, vendored under third_party/ckdtree
// and built as the `ckdtree` target. It was chosen over a tiled search
// because it handles periodicity in the metric itself: a tree built
// with a PeriodicBox measures distances under the minimum-image
// convention, so a query costs the same as in the open plane instead of
// 9x the points. Queries may be arbitrary points, not just nodes, which
// is what departure-point-centred stencils need.
//
// The tree carries double coordinates because ckdtree does; the T-taking
// entry points below convert. This is the place for a quadtree or a
// ball tree, should one be needed.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace rbf::spatial {

// The rectangle [x0, x0+Lx) x [y0, y0+Ly) with opposite sides
// identified, periodic in both directions.
//
// The C++ twin of the Fortran type periodic_box in
// src/rbf_periodic_box.f90, which has its origin fixed at zero: wrap
// maps a point into the box, minimum_image shortens a displacement to
// the nearest of its periodic images.
template<typename T>
struct PeriodicBox {
    T x0 = 0, y0 = 0;  // lower-left corner
    T Lx = 1, Ly = 1;  // periods

    // The shortest of the displacement and its periodic images. Matches
    // Fortran's anint (half away from zero), so a displacement of
    // exactly half a period maps to -L/2.
    T minimum_image_x(T dx) const { return dx - Lx * std::round(dx / Lx); }
    T minimum_image_y(T dy) const { return dy - Ly * std::round(dy / Ly); }

    // The point mapped into the box. Lands in [x0, x0+Lx) even when x
    // is so far outside that x - L*floor(x/L) rounds to the upper edge.
    T wrap_x(T x) const { return x0 + fold(x - x0, Lx); }
    T wrap_y(T y) const { return y0 + fold(y - y0, Ly); }

private:
    static T fold(T d, T L) {
        d -= L * std::floor(d / L);
        while (d >= L) d -= L;
        while (d < 0) d += L;
        return d;
    }
};

// Build parameters, named after SciPy's; they trade build time against
// query time and never change the answer.
struct KdTreeParams {
    int leafsize = 16;    // points below which a node is searched by brute force
    bool balanced = true; // split at the median rather than at the midpoint
    bool compact = true;  // shrink each node's box onto its points
};

// A 2-d k-d tree over a fixed point cloud, periodic or not.
//
// The cloud is copied in, so the caller's arrays need not outlive the
// tree. Indices returned by the queries refer to the cloud in the order
// it was passed. The tree is immutable once built, and const queries
// are thread-safe.
class KdTree {
public:
    // Without a box the cloud sits in the open plane. With one, points
    // outside it are wrapped in, so the cloud need not be pre-wrapped;
    // the indices are unaffected either way.
    KdTree(std::span<const double> x, std::span<const double> y,
           std::optional<PeriodicBox<double>> box = std::nullopt,
           KdTreeParams params = {});

    ~KdTree();
    KdTree(KdTree&&) noexcept;
    KdTree& operator=(KdTree&&) noexcept;

    std::size_t size() const;
    const std::optional<PeriodicBox<double>>& box() const;
    bool periodic() const { return box().has_value(); }

    // The k nearest neighbours of each query point, sorted by distance:
    // idx and dist are nq*k, row-major, so the neighbours of query s sit
    // at idx[s*k]. dist holds true Euclidean distances (minimum-image in
    // a periodic box) and may be left empty when only the indices are
    // wanted. Query points are wrapped into the box. OpenMP-parallel
    // over the queries.
    void knn(std::span<const double> qx, std::span<const double> qy, int k,
             std::span<std::intptr_t> idx, std::span<double> dist = {}) const;

    // The same, centred on the cloud's own points, in its own order.
    void knn(int k, std::span<std::intptr_t> idx,
             std::span<double> dist = {}) const;

    // Fixed-k stencils as ja(k, nq) in Fortran order: the k neighbours
    // of query s are contiguous at ja[s*k], sorted by distance. I is the
    // index type of the CsrMatrix<T, I> they will feed.
    template<typename I = std::int32_t>
    std::vector<I> stencils(std::span<const double> qx,
                            std::span<const double> qy, int k) const {
        std::vector<std::intptr_t> idx(qx.size() * static_cast<std::size_t>(k));
        knn(qx, qy, k, idx);
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

namespace detail {

// ckdtree is double-only; a cloud of another coordinate type is
// converted on the way in. The double case is a no-op that keeps the
// caller's storage.
template<typename T>
struct AsDoubles {
    explicit AsDoubles(std::span<const T> v) : buf(v.begin(), v.end()) {}
    std::span<const double> get() const { return buf; }
    std::vector<double> buf;
};

template<>
struct AsDoubles<double> {
    explicit AsDoubles(std::span<const double> v) : v_(v) {}
    std::span<const double> get() const { return v_; }
    std::span<const double> v_;
};

template<typename T>
std::optional<PeriodicBox<double>> widen(const std::optional<PeriodicBox<T>>& b) {
    if (!b) return std::nullopt;
    return PeriodicBox<double>{
        static_cast<double>(b->x0), static_cast<double>(b->y0),
        static_cast<double>(b->Lx), static_cast<double>(b->Ly)};
}

} // namespace detail

// One-shot k-nearest-neighbour stencils of the query points (qx, qy) in
// the cloud (x, y): ja(k, nq) in Fortran order, as KdTree::stencils.
// Without a box the cloud sits in the open plane.
//
// These build a tree per call. Hold a KdTree instead when the same
// cloud is queried more than once.
template<typename T, typename I = std::int32_t>
std::vector<I> knn_stencils(std::span<const T> x, std::span<const T> y,
                            std::span<const T> qx, std::span<const T> qy,
                            int k, std::optional<PeriodicBox<T>> box = std::nullopt) {
    const detail::AsDoubles<T> xd(x), yd(y), qxd(qx), qyd(qy);
    return KdTree(xd.get(), yd.get(), detail::widen(box))
        .stencils<I>(qxd.get(), qyd.get(), k);
}

// Stencils centred on the cloud's own points, as KdTree::stencils(k).
template<typename T, typename I = std::int32_t>
std::vector<I> knn_stencils(std::span<const T> x, std::span<const T> y,
                            int k, std::optional<PeriodicBox<T>> box = std::nullopt) {
    const detail::AsDoubles<T> xd(x), yd(y);
    return KdTree(xd.get(), yd.get(), detail::widen(box)).stencils<I>(k);
}

} // namespace rbf::spatial

#endif // RBF_SPATIAL_H
