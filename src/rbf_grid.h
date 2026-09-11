#ifndef RBF_GRID_H
#define RBF_GRID_H

// A 2D unstructured grid: nodes, triangle and quad connectivity, and the
// boundary as node lists, one list per boundary part. This is what a
// grid file (.grid) holds; rbf_io.h reads and writes the format
// (read_grid, write_grid) and docs/file_formats.md#grid-file describes
// it, with the references.
//
// Assisted-by: Claude Fable 5

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rbf {

// The base a grid's node indices are kept in, in memory. The file format
// is always 1-based; what is chosen here is only whether reading keeps
// that (IndexBase::one) or shifts to 0-based (IndexBase::zero), the
// numbering of the graph file and the rest of the library.
enum class IndexBase { one, zero };

// The class owns its arrays and establishes the structural invariants at
// construction: coordinates of one length, whole elements, every index in
// range for the declared base, boundary parts of at least two nodes. The
// format is 1-based; base() says which IndexBase the indices are kept in
// here, chosen at construction (by read_grid's second argument).
template <class T = double, class I = std::int32_t>
class UnstructuredGrid {
public:
    // The arrays are moved in and the invariants asserted, as the writers
    // do: x and y of one length; tri whole triangles and quad whole quads,
    // row-major; bound the node list of each boundary part, in order along
    // the boundary, each of at least two nodes (a part marks itself closed
    // by repeating its first node last); every index in [1, nnodes] under
    // IndexBase::one, in [0, nnodes) under IndexBase::zero.
    UnstructuredGrid(std::vector<T> x,
                     std::vector<T> y,
                     std::vector<I> tri,
                     std::vector<I> quad,
                     std::vector<std::vector<I>> bound,
                     IndexBase base)
        : x_(std::move(x)), y_(std::move(y)), tri_(std::move(tri)), quad_(std::move(quad)),
          bound_(std::move(bound)), base_(base) {
        assert(y_.size() == x_.size() && "x and y differ in length");
        assert(tri_.size() % 3 == 0 && "tri is not whole triangles");
        assert(quad_.size() % 4 == 0 && "quad is not whole quadrilaterals");
#ifndef NDEBUG
        for (const I v : tri_)
            assert(in_range(v) && "triangle node out of range");
        for (const I v : quad_)
            assert(in_range(v) && "quadrilateral node out of range");
        for (const auto& part : bound_) {
            assert(part.size() >= 2 && "boundary part of fewer than 2 nodes");
            for (const I v : part)
                assert(in_range(v) && "boundary node out of range");
        }
#endif
    }

    std::size_t num_nodes() const { return x_.size(); }
    std::size_t num_triangles() const { return tri_.size() / 3; }
    std::size_t num_quads() const { return quad_.size() / 4; }
    std::size_t num_boundaries() const { return bound_.size(); }
    IndexBase base() const { return base_; }
    bool zero_based() const { return base_ == IndexBase::zero; }

    // The arrays, read-only. The && overloads of x() and y() move the
    // coordinates out of an expiring grid -- std::move(g).x() -- which is
    // how NodeSet takes them over without a copy.
    const std::vector<T>& x() const& { return x_; }
    const std::vector<T>& y() const& { return y_; }
    std::vector<T>&& x() && { return std::move(x_); }
    std::vector<T>&& y() && { return std::move(y_); }
    const std::vector<I>& tri() const& { return tri_; }
    const std::vector<I>& quad() const& { return quad_; }
    const std::vector<std::vector<I>>& bound() const& { return bound_; }

    // Whether boundary part b closes a loop, which the format marks by
    // repeating the node where the part closes; false for an open polyline.
    bool closed(std::size_t b) const {
        const auto& part = bound_[b];
        return part.size() > 1 && part.front() == part.back();
    }

    // Boundary marker per node in the convention of the node file: 0 for a
    // node no part lists, else the number (from 1) of the first part that
    // lists it. NodeSet takes a grid and uses this as the flag; the result
    // is indexed by node position, whatever the base.
    std::vector<int> markers() const {
        std::vector<int> m(num_nodes(), 0);
        for (std::size_t b = 0; b < bound_.size(); ++b)
            for (const I j : bound_[b])
                if (m[pos(j)] == 0) m[pos(j)] = static_cast<int>(b + 1);
        return m;
    }

    // The format's orientation conventions:
    //   - element nodes are ordered counterclockwise,
    //   - boundary node ordering is induced by the element node ordering,
    //   - the domain is always on your left while walking along a boundary.
    // Checked against the coordinates and connectivity: an element whose
    // signed area is not positive; a directed element edge used twice,
    // which no consistent counterclockwise numbering produces; a part edge
    // that is not an element edge, walks against one (domain on the
    // right), or is an interior edge; and element boundary edges no part
    // walks. One message per violation, in file order, empty when the grid
    // follows the conventions; node numbers in the messages are 1-based,
    // as in the file. read_grid does not call this: parsing accepts any
    // orientation.
    std::vector<std::string> orientation_report() const {
        const std::uint64_t n = num_nodes();
        const auto no = [&](I v) { return std::to_string(pos(v) + 1); };  // as in the file
        const auto key = [&](I u, I v) { return static_cast<std::uint64_t>(pos(u)) * n + pos(v); };
        std::vector<std::string> out;

        // the directed element edges, and the counterclockwise test
        std::unordered_map<std::uint64_t, int> edges;
        const auto add_elements = [&](const std::vector<I>& conn, std::size_t nv,
                                      const char* name) {
            for (std::size_t e = 0; e * nv < conn.size(); ++e) {
                const I* el = conn.data() + e * nv;
                double area2 = 0;  // twice the signed area, by the shoelace formula
                for (std::size_t i = 0; i < nv; ++i) {
                    const std::size_t u = pos(el[i]), v = pos(el[(i + 1) % nv]);
                    area2 +=
                        static_cast<double>(x_[u]) * y_[v] - static_cast<double>(x_[v]) * y_[u];
                    ++edges[key(el[i], el[(i + 1) % nv])];
                }
                if (!(area2 > 0))
                    out.push_back(std::string(name) + " " + std::to_string(e + 1) +
                                  " is not counterclockwise");
            }
        };
        add_elements(tri_, 3, "triangle");
        add_elements(quad_, 4, "quadrilateral");

        // every element edge again, in file order, for deterministic output
        const auto each_edge = [&](auto&& f) {
            const auto walk = [&](const std::vector<I>& conn, std::size_t nv) {
                for (std::size_t e = 0; e * nv < conn.size(); ++e)
                    for (std::size_t i = 0; i < nv; ++i)
                        f(conn[e * nv + i], conn[e * nv + (i + 1) % nv]);
            };
            walk(tri_, 3);
            walk(quad_, 4);
        };
        std::unordered_set<std::uint64_t> seen;
        each_edge([&](I u, I v) {
            if (edges[key(u, v)] > 1 && seen.insert(key(u, v)).second)
                out.push_back("element edge " + no(u) + " -> " + no(v) +
                              " is used twice in the same direction");
        });

        // a part edge must be an element edge whose reverse no element
        // uses: a mesh-boundary edge, walked with the domain on the left
        std::unordered_set<std::uint64_t> walked;
        for (std::size_t b = 0; b < bound_.size(); ++b) {
            const auto& part = bound_[b];
            for (std::size_t j = 0; j + 1 < part.size(); ++j) {
                const I u = part[j], v = part[j + 1];
                const bool fwd = edges.count(key(u, v)) > 0, rev = edges.count(key(v, u)) > 0;
                const std::string edge =
                    "boundary part " + std::to_string(b + 1) + ", edge " + no(u) + " -> " + no(v);
                if (fwd && !rev)
                    walked.insert(key(u, v));
                else if (!fwd && rev)
                    out.push_back(edge + " walks with the domain on the right");
                else if (fwd && rev)
                    out.push_back(edge + " is an interior edge");
                else
                    out.push_back(edge + " is not an element edge");
            }
        }

        // and the parts together must walk the whole mesh boundary
        seen.clear();
        each_edge([&](I u, I v) {
            if (edges.count(key(v, u)) == 0 && walked.count(key(u, v)) == 0 &&
                seen.insert(key(u, v)).second)
                out.push_back("element boundary edge " + no(u) + " -> " + no(v) +
                              " is not walked by any boundary part");
        });
        return out;
    }

private:
    // 0-based position of a stored index, in either base
    std::size_t pos(I v) const {
        assert(in_range(v) && "node index out of range");
        return static_cast<std::size_t>(v - (zero_based() ? 0 : 1));
    }
    bool in_range(I v) const {
        const I p = v - (zero_based() ? 0 : 1);
        return p >= 0 && static_cast<std::size_t>(p) < x_.size();
    }

    std::vector<T> x_, y_;
    std::vector<I> tri_;                 // 3 * num_triangles(), row-major
    std::vector<I> quad_;                // 4 * num_quads(), row-major
    std::vector<std::vector<I>> bound_;  // the node list of each boundary part
    IndexBase base_;
};

}  // namespace rbf

#endif  // RBF_GRID_H
