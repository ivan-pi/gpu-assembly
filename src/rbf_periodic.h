// rbf_periodic.h -- periodic box and k-nearest-neighbour search in it
//
// The flow benchmarks run on a doubly periodic box. Periodicity enters in
// two places, and both use the same PeriodicBox: the stencil search
// (neighbours across the boundary) and the assembly (minimum-image
// displacement of a neighbour from the stencil centre).
//
// periodic_knn tiles the cloud with its 8 images and runs nanoflann on
// the 9n points, so queries may be arbitrary points in the box, which is
// what departure-point-centred stencils need. Memory is 9x the cloud,
// which is negligible next to the weights.

#ifndef RBF_PERIODIC_H
#define RBF_PERIODIC_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include <nanoflann.hpp>

namespace rbf {

template<typename T>
struct PeriodicBox {
    T x0 = 0, y0 = 0;   // lower-left corner
    T Lx = 1, Ly = 1;   // periods

    // minimum-image displacement
    T wrap_dx(T d) const { return d - Lx * std::nearbyint(d / Lx); }
    T wrap_dy(T d) const { return d - Ly * std::nearbyint(d / Ly); }

    // point mapped into [x0, x0+Lx) x [y0, y0+Ly)
    T wrap_x(T x) const { x -= x0; x -= Lx * std::floor(x / Lx); return x + x0; }
    T wrap_y(T y) const { y -= y0; y -= Ly * std::floor(y / Ly); return y + y0; }
};

namespace detail {
template<typename T>
struct TiledCloud {
    std::vector<T> x, y;
    std::size_t kdtree_get_point_count() const { return x.size(); }
    T kdtree_get_pt(std::size_t i, std::size_t d) const { return d == 0 ? x[i] : y[i]; }
    template<class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};
}

// k nearest neighbours in the periodic box of each query point (qx, qy),
// as row-major fixed-k stencils: ja[s*k + j], sorted by distance. Indices
// refer to the original cloud (x, y). A query that coincides with a node
// gets that node first.
template<typename T, typename I = std::int32_t>
std::vector<I> periodic_knn(std::span<const T> x, std::span<const T> y,
                            const PeriodicBox<T>& box,
                            std::span<const T> qx, std::span<const T> qy, int k)
{
    const std::size_t n = x.size();
    assert(y.size() == n && qx.size() == qy.size());
    if (k < 1 || static_cast<std::size_t>(k) > n)
        throw std::invalid_argument("periodic_knn: k out of range");

    // image-major tiling: tiled index t refers to node t % n
    detail::TiledCloud<T> cloud;
    cloud.x.reserve(9 * n);
    cloud.y.reserve(9 * n);
    for (int sy = -1; sy <= 1; ++sy)
        for (int sx = -1; sx <= 1; ++sx)
            for (std::size_t i = 0; i < n; ++i) {
                cloud.x.push_back(x[i] + sx * box.Lx);
                cloud.y.push_back(y[i] + sy * box.Ly);
            }

    using Tree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<T, detail::TiledCloud<T>>,
        detail::TiledCloud<T>, 2, std::size_t>;
    Tree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    const std::size_t nq = qx.size();
    std::vector<I> ja(nq * k);
    #pragma omp parallel
    {
        std::vector<std::size_t> idx(k);
        std::vector<T> d2(k);
        #pragma omp for schedule(static)
        for (std::ptrdiff_t s = 0; s < static_cast<std::ptrdiff_t>(nq); ++s) {
            const T q[2] = {box.wrap_x(qx[s]), box.wrap_y(qy[s])};
            [[maybe_unused]] const auto found = tree.knnSearch(q, k, idx.data(), d2.data());
            assert(found == static_cast<std::size_t>(k));
            for (int j = 0; j < k; ++j) ja[s * k + j] = static_cast<I>(idx[j] % n);
        }
    }
    return ja;
}

// stencils centred on the nodes themselves
template<typename T, typename I = std::int32_t>
std::vector<I> periodic_knn(std::span<const T> x, std::span<const T> y,
                            const PeriodicBox<T>& box, int k) {
    return periodic_knn<T, I>(x, y, box, x, y, k);
}

} // namespace rbf

#endif // RBF_PERIODIC_H
