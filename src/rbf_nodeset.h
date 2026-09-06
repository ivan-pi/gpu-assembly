#ifndef RBF_NODESET_H
#define RBF_NODESET_H

#include <cstdint>
#include <vector>
#include <iostream>
#include <string>

#include <nanoflann.hpp>

namespace rbf {

// Node set with its own k-d tree.
//
// nanoflann's tree keeps a reference to the dataset, so the tree
// here refers to the enclosing object.
//
// The class is non-copyable.
//
template<typename T = double, typename I = int32_t>
class NodeSet {
    static_assert(std::is_integral_v<I>, "index type must be integral");
public:
    using value_type = T;
    using index_type= I;

    std::vector<T> x, y;
    std::vector<int> flag;        // per-node tag; 0 = interior, nonzero = boundary
    std::vector<index_type> bnd;  // indices of nonzero-flag nodes (boundary)

    explicit NodeSet(const std::string& fname, size_t leaf_max_size = 10,
                     unsigned n_thread_build = 1)
        : tree_(2, *this, nanoflann::KDTreeSingleIndexAdaptorParams(
                leaf_max_size
                nanoflann:KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex,
                n_thread_build))
    {
        std::ifstream in{fname};
        if (!in) { std::cerr << "cannot open " << fname << '\n'; std::exit(1); }
        double px, py; int f;
        index_type n = 0; nb = 0;
        while (in >> px >> py >> f) {
            if (f) {
                bnd.push_back(n);
                ++nb;
            }
            x.push_back(px);
            y.push_back(py);
            flag.push_back(f);
            ++n;
        }
        num_points_ = n;
        num_boundary_ = nb;
        tree_.buildIndex(); // now that x, y and num_points_ are final
    }

    NodeSet(const NodeSet&) = delete;
    NodeSet& operator=(const NodeSet&) = delete;

    size_t num_points() const { return num_points_; }
    size_t num_boundary() const { return num_boundary_; }
    size_t num_interior() const { return num_points_ - num_boundary_; }

    // Indices of nodes with a particular flag value
    //
    // Useful for problems with several boundary kinds
    std::vector<index_type> indices_with(int value) const {
        std::vector<I> out;
        for (index_type i = 0; i < num_points_; ++i) {
            if (flag[i] == value) out.push_back(i);
        }
        return out;
    }

    // k-nearest-neighbour stencils as ja(k, ntot) in Fortran order: the k
    // neighbours of node s are contiguous at ja[s*k], sorted by distance, so
    // ja[s*k] == s. Values are 0-based, of the index type I chosen to match
    // the CsrMatrix<T, I> they will feed (int32_t by default).
    auto stencils(int k) const {
        const int n = num_points_;
        if (k < 1 || k > n) {
            std::cerr << "error: NodeSet::stencils: k=" << k
                      << " with " << n << " nodes\n";
            std::exit(1);
        }
        std::vector<index_type> ja(n * k);
        #pragma omp parallel
        {
            std::vector<T> d2(k);
            #pragma omp for schedule(static)
            for (index_type s = 0; s < n; ++s) {
                const T q[2] = {x[s], y[s]};
                tree_.knnSearch(q, k, &ja[s*k], d2.data());
            }
        }
        return ja;
    }

    // nanoflann dataset-adaptor interface
    size_t kdtree_get_point_count() const { return num_points_; }
    T kdtree_get_pt(size_t i, size_t d) const { return d == 0 ? x[i] : y[i]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }

private:
    using Tree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<T, NodeSet>, NodeSet, 2, I>;
    Tree tree_;
    size_t num_points_{0}, num_boundary_{0};
};

} // namespace rbf

#endif // RBF_NODESET_H
