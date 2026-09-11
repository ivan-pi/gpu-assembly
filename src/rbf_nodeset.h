#ifndef RBF_NODESET_H
#define RBF_NODESET_H

#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nanoflann.hpp>

#include "rbf_grid.h"
#include "rbf_io.h"
#include "rbf_reorder.h"

namespace rbf {

// Parameters for the k-d tree build; see nanoflann's
// KDTreeSingleIndexAdaptorParams. They take effect when the tree is
// (re)built, i.e. on the first stencils() call after construction or
// after a renumber().
struct TreeParams {
    size_t leaf_max_size = 10;
    unsigned n_thread_build = 1;
};

// Node set with a lazily-built k-d tree.
//
// nanoflann's tree keeps a reference to the dataset, so the tree here
// refers to the enclosing object. It is built on first use (stencils())
// and invalidated by renumber(), so the natural flow
//
//     NodeSet ns("case.nodes");
//     ns.renumber(morton_order(ns.x, ns.y))
//       .renumber(boundary_last_by_flag(ns.flag));
//     auto ja = ns.stencils(k);
//
// builds the tree exactly once, in the final numbering.
//
// The class is non-copyable.
//
template <typename T = double, typename I = int32_t>
class NodeSet {
    static_assert(std::is_integral_v<I>, "index type must be integral");

public:
    using value_type = T;
    using index_type = I;

    std::vector<T> x, y;
    std::vector<int> flag;        // per-node tag; 0 = interior, nonzero = boundary
    std::vector<index_type> bnd;  // indices of nonzero-flag nodes (boundary)

    // Reads a node file (.node, the Triangle format; see rbf::io::read_nodes).
    // The boundary marker becomes the flag; a file without markers gives
    // all-interior nodes. Attributes are skipped: read them with
    // rbf::io::read_nodes directly and permute with file_order() if needed.
    explicit NodeSet(const std::string& fname) {
        num_points_ = io::read_nodes(fname, x, y, flag);
        rebuild_bnd();
        file_order_ = Permutation<I>::identity(num_points_);
    }

    // Nodes given directly: the coordinates and a flag per node (0 =
    // interior), moved in. The three must have the same length.
    NodeSet(std::vector<T> x_, std::vector<T> y_, std::vector<int> flag_)
        : x(std::move(x_)), y(std::move(y_)), flag(std::move(flag_)) {
        assert(y.size() == x.size() && flag.size() == x.size() && "x, y and flag differ in length");
        num_points_ = x.size();
        rebuild_bnd();
        file_order_ = Permutation<I>::identity(num_points_);
    }

    // The nodes of an UnstructuredGrid (a grid file, rbf::io::read_grid)
    // as a node set: the connectivity is dropped, and the flag is
    // markers(), the first boundary part listing each node, whatever base
    // the grid keeps its indices in. The grid is taken by value, so
    //
    //     NodeSet<double> ns(std::move(g));    // moves the coordinates out of g
    //     NodeSet<double> ns(g);               // copies, g stays usable
    template <class IG>
    explicit NodeSet(UnstructuredGrid<T, IG> g) {
        flag = g.markers();  // before the coordinates move out of g
        x = std::move(g).x();
        y = std::move(g).y();
        num_points_ = x.size();
        rebuild_bnd();
        file_order_ = Permutation<I>::identity(num_points_);
    }

    NodeSet(const NodeSet&) = delete;
    NodeSet& operator=(const NodeSet&) = delete;

    // Writes the nodes in the current numbering as a node file with the
    // flag as boundary marker, so a renumbered set can be saved and read
    // back as is. Only x, y and flag go to the file: file_order() and the
    // k-d tree are not part of it, so the re-read set starts from the
    // identity order.
    void write(const std::string& fname) const {
        io::write_nodes(fname, num_points_, x.data(), y.data(), flag.data());
    }

    size_t num_points() const { return num_points_; }
    size_t num_boundary() const { return bnd.size(); }
    size_t num_interior() const { return num_points_ - bnd.size(); }

    // Renumber the nodes: permutes x, y and flag, recomputes bnd, and
    // invalidates the k-d tree. Returns *this so orderings can be
    // chained.
    //
    // file_order() is extended, never reset: it always maps the
    // *current* numbering back to the original file order. New position
    // i holds the node that sat at position p.map()[i] before this
    // call, which in turn came from file position
    // file_order().map()[p.map()[i]] -- exactly the composition
    // file_order().then(p).
    //
    // Stencils extracted before a renumber are expressed in the old
    // numbering and are not updated (renumber_stencils() in
    // rbf_reorder.h does that if needed).
    NodeSet& renumber(const Permutation<I>& p) {
        assert(p.size() == num_points_);
        p.permute(std::span{x});
        p.permute(std::span{y});
        p.permute(std::span{flag});
        rebuild_bnd();
        file_order_ = file_order_.then(p);
        tree_.reset();
        return *this;
    }

    // Current numbering -> original file order; identity if renumber()
    // was never called. Use it to permute per-node data loaded in file
    // order, to unpermute results for output, or to save the renumbering
    // as an ordering file:
    //
    //     file_order().permute(std::span{u});    // file order -> current
    //     file_order().unpermute(std::span{u});  // current -> file order
    //     file_order().write("case.iperm");      // Permutation::read gets it back
    const Permutation<I>& file_order() const { return file_order_; }

    // Indices of nodes with a particular flag value
    //
    // Useful for problems with several boundary kinds
    std::vector<index_type> indices_with(int value) const {
        std::vector<I> out;
        for (size_t i = 0; i < num_points_; ++i) {
            if (flag[i] == value) out.push_back(static_cast<I>(i));
        }
        return out;
    }

    // k-nearest-neighbour stencils as ja(k, ntot) in Fortran order: the k
    // neighbours of node s are contiguous at ja[s*k], sorted by distance, so
    // ja[s*k] == s. Values are 0-based, of the index type I chosen to match
    // the CsrMatrix<T, I> they will feed (int32_t by default).
    //
    // Builds the k-d tree on first use; tp only takes effect when the
    // tree is actually (re)built.
    auto stencils(int k, const TreeParams& tp = {}) const {
        const auto n = static_cast<index_type>(num_points_);
        if (k < 1 || static_cast<size_t>(k) > num_points_) {
            std::cerr << "error: NodeSet::stencils: k=" << k << " with " << n << " nodes\n";
            std::exit(1);
        }
        ensure_tree(tp);
        std::vector<index_type> ja(num_points_ * k);
#pragma omp parallel
        {
            std::vector<T> d2(k);
#pragma omp for schedule(static)
            for (index_type s = 0; s < n; ++s) {
                const T q[2] = {x[s], y[s]};
                [[maybe_unused]] const auto found = tree_->knnSearch(q, k, &ja[s * k], d2.data());
                assert(found == static_cast<size_t>(k));
            }
        }
        return ja;
    }

    // nanoflann dataset-adaptor interface
    size_t kdtree_get_point_count() const { return num_points_; }
    T kdtree_get_pt(size_t i, size_t d) const { return d == 0 ? x[i] : y[i]; }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& bb) const {
        if (x.empty()) return false;
        const auto b = compute_bbox(std::span<const T>{x}, std::span<const T>{y});
        bb[0].low = b.xmin;
        bb[0].high = b.xmax;
        bb[1].low = b.ymin;
        bb[1].high = b.ymax;
        return true;
    }

private:
    using Tree = nanoflann::
        KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<T, NodeSet>, NodeSet, 2, I>;

    void rebuild_bnd() {
        bnd.clear();
        for (size_t i = 0; i < num_points_; ++i) {
            if (flag[i]) bnd.push_back(static_cast<I>(i));
        }
    }

    // Building the index is deferred to first use so that renumber()
    // can run beforehand (or in between) without paying for a rebuild.
    // optional<Tree> gives in-class storage with deferred construction
    // (the Tree has no default constructor: it wants the dataset
    // reference and the build parameters up front). mutable, because
    // stencils() is logically const: building the cache does not change
    // the observable node set.
    void ensure_tree(const TreeParams& tp) const {
        if (!tree_) {
            tree_.emplace(2, *this,
                          nanoflann::KDTreeSingleIndexAdaptorParams(
                              tp.leaf_max_size, nanoflann::KDTreeSingleIndexAdaptorFlags::None,
                              tp.n_thread_build));
        }
    }

    mutable std::optional<Tree> tree_;
    size_t num_points_{0};
    Permutation<I> file_order_;
};

}  // namespace rbf

#endif  // RBF_NODESET_H
