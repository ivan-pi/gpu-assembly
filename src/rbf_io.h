#ifndef RBF_IO_H
#define RBF_IO_H

// ASCII input and output for RBF-FD drivers. The formats are described in
// docs/file_formats.md.
//
//   read_points, read_points_aos     points file (.points): count, then "x y" -> SoA or AoS
//   read_graph_csr                   graph file (.graph): stencils -> (ia, ja)
//   read_nodes, write_nodes          node file (.node), Triangle's format; what NodeSet reads and
//   writes read_ordering, write_ordering    ordering file (.iperm): one new index per node
//   read_grid, write_grid            grid file (.grid): Nishikawa's 2D unstructured grid -> Grid
//   read_bcmap, read_mapbc           boundary conditions (.bcmap, .mapbc): part tag -> name/number
//   write_matrix_market              CSR matrix -> Matrix Market (real, or pattern)
//
// rbf_io_vtk.h and rbf_io_gnuplot.h add the plotting formats.
//
// Every reader checks that the file opened. Readers whose format carries a
// count also check that they got that many, exiting with a message rather
// than returning a short container.
// Every writer emits the shortest text that reads back to the same number,
// so a file written here reproduces the values it was given.
//
// Assisted-by: Claude Fable 5.1

#include <cassert>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace rbf::io {

// Helpers shared by the readers and writers.
namespace detail {

[[noreturn]] inline void fail(const std::string& fname, const std::string& what) {
    std::cerr << "rbf::io: " << fname << ": " << what << '\n';
    std::exit(1);
}

inline std::ifstream open_in(const std::string& fname) {
    std::ifstream in{fname};
    if (!in) fail(fname, "cannot open for reading");
    return in;
}

inline std::ofstream open_out(const std::string& fname) {
    std::ofstream out{fname};
    if (!out) fail(fname, "cannot open for writing");
    return out;
}

// A number as the shortest text that reads back exactly (std::to_chars):
// 0.1 is written as 0.1, not 0.10000000000000001, and integers as is.
//
//     out << num(x[i]) << ' ' << num(y[i]);
template <class V>
struct Num {
    V v;
};
template <class V>
Num<V> num(V v) {
    return {v};
}
template <class V>
std::ostream& operator<<(std::ostream& os, Num<V> n) {
    char buf[32];
    const auto r = std::to_chars(buf, buf + sizeof buf, n.v);
    return os.write(buf, r.ptr - buf);
}

// Tokens of one line, parsed with std::from_chars: no stream is built per
// line, integers are range-checked into their type, and a number must end
// at whitespace or the end of the line, so "1.5" is not the integer 1.
struct LineCursor {
    const char* p;
    const char* end;

    explicit LineCursor(std::string_view s) : p(s.data()), end(s.data() + s.size()) {}

    static bool ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
    void skip_ws() {
        while (p != end && ws(*p))
            ++p;
    }
    bool at_end() {
        skip_ws();
        return p == end;
    }

    // Reals are parsed as double and converted, so a value below the float
    // range reads as 0 rather than failing; integers are parsed in their own
    // type, so an out-of-range index is an error.
    template <class V>
    bool next(V& v) {
        skip_ws();
        const char* q;
        if constexpr (std::is_floating_point_v<V>) {
            double d;
            const auto r = std::from_chars(p, end, d);
            if (r.ec != std::errc{}) return false;
            v = static_cast<V>(d);
            q = r.ptr;
        } else {
            const auto r = std::from_chars(p, end, v);
            if (r.ec != std::errc{}) return false;
            q = r.ptr;
        }
        if (q != end && !ws(*q)) return false;
        p = q;
        return true;
    }

    // The next whitespace-delimited token, for names.
    bool next_token(std::string& t) {
        skip_ws();
        const char* q = p;
        while (q != end && !ws(*q))
            ++q;
        if (q == p) return false;
        t.assign(p, q);
        p = q;
        return true;
    }

    // The text at the cursor, for error messages.
    std::string here() const { return std::string(std::string_view(p, end - p).substr(0, 16)); }
};

// Next line with content: comment starts a comment ('#' for the node file,
// '!' for the boundary-condition files), blank lines are skipped. The line
// is truncated at the comment so the cursor sees data only.
inline bool next_data_line(std::istream& in,
                           std::string& line,
                           std::size_t& lineno,
                           char comment = '#') {
    while (std::getline(in, line)) {
        ++lineno;
        if (const auto at = line.find(comment); at != std::string::npos) line.resize(at);
        if (line.find_first_not_of(" \t\r") != std::string::npos) return true;
    }
    return false;
}

}  // namespace detail

// A named per-node column: v[i * stride] is the value at node i. The
// writers that take fields (VTK, gnuplot columns) take lists of these.
template <class T>
struct Column {
    std::string_view name;
    const T* v;
    std::size_t stride = 1;
};

// A list argument for those writers: a braced list at the call site, or any
// contiguous range of elements (vector, array, span).
template <class E>
struct List : std::span<const E> {
    List() = default;
    List(std::initializer_list<E> l) : std::span<const E>(l.begin(), l.size()) {}
    template <class R>
        requires std::constructible_from<std::span<const E>, const R&>
    List(const R& r) : std::span<const E>(r) {}
};

// ---------------------------------------------------------------------------
// Points file (.points)
// ---------------------------------------------------------------------------

namespace detail {

// The points file: the point count on the first line, then one coordinate
// pair per line. on_count(n) is called once, on_point(x, y) n times.
template <class T, class OnCount, class OnPoint>
void read_points_with(const std::string& fname, OnCount on_count, OnPoint on_point) {
    static_assert(std::is_floating_point_v<T>, "coordinates must be floating point");
    auto in = open_in(fname);
    std::string line;
    std::size_t n = 0;
    std::getline(in, line);
    if (LineCursor c{line}; !c.next(n) || !c.at_end())
        fail(fname, "missing or malformed point count");
    on_count(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::getline(in, line))
            fail(fname, "expected " + std::to_string(n) + " points, found " + std::to_string(i));
        LineCursor c{line};
        T px, py;
        if (!(c.next(px) && c.next(py)) || !c.at_end())
            fail(fname, "line " + std::to_string(i + 2) + ": expected \"x y\"");
        if (!std::isfinite(px) || !std::isfinite(py))
            fail(fname, "line " + std::to_string(i + 2) + ": coordinate is not finite");
        on_point(px, py);
    }
}

}  // namespace detail

// Coordinates as separate arrays (SoA), the layout the assembly kernels
// take. The points file is the companion of the graph file: one gives the
// point cloud, the other the stencils, in the same numbering.
template <class T = double>
std::pair<std::vector<T>, std::vector<T>> read_points(const std::string& fname) {
    std::vector<T> x, y;
    detail::read_points_with<T>(
        fname,
        [&](std::size_t n) {
            x.reserve(n);
            y.reserve(n);
        },
        [&](T px, T py) {
            x.push_back(px);
            y.push_back(py);
        });
    return {std::move(x), std::move(y)};
}

// Same file, into an array of structs: std::vector<std::array<T,2>>,
// std::vector<Point> with Point{T x, y;}, or any container whose value_type
// is brace-constructible from two T.
template <class ArrayOfStructs, class T = double>
ArrayOfStructs read_points_aos(const std::string& fname) {
    using Struct = typename ArrayOfStructs::value_type;
    ArrayOfStructs xy;
    detail::read_points_with<T>(
        fname, [&](std::size_t n) { xy.reserve(n); },
        [&](T px, T py) { xy.push_back(Struct{px, py}); });
    return xy;
}

// ---------------------------------------------------------------------------
// Node file (.node), the Triangle format
// ---------------------------------------------------------------------------

// Node file in the format of Triangle's .node file:
//
//     n 2 nattr nmark            # vertices, dimension, attributes, marker column (0 or 1)
//     i x y a0 ... a(nattr-1) [marker]
//     ...
//
// '#' starts a comment and blank lines are allowed anywhere; vertices are
// numbered consecutively from 0 or from 1. x, y and marker are cleared and
// filled in line order; without a marker column every marker is 0.
// Attributes are returned row-major, n x nattr, when a vector is passed for
// them, and skipped otherwise. Returns n. A malformed file is an error
// reported with its line number.
template <class T>
std::size_t read_nodes(const std::string& fname,
                       std::vector<T>& x,
                       std::vector<T>& y,
                       std::vector<int>& marker,
                       std::vector<T>* attributes = nullptr) {
    static_assert(std::is_floating_point_v<T>, "coordinates must be floating point");
    auto in = detail::open_in(fname);
    x.clear();
    y.clear();
    marker.clear();
    if (attributes) attributes->clear();

    std::string line;
    std::size_t lineno = 0;
    const auto fail_here = [&](const std::string& what) {
        detail::fail(fname, "line " + std::to_string(lineno) + ": " + what);
    };

    if (!detail::next_data_line(in, line, lineno)) detail::fail(fname, "empty file");
    std::size_t n, dim, nattr, nmark;
    {
        detail::LineCursor c{line};
        if (!(c.next(n) && c.next(dim) && c.next(nattr) && c.next(nmark)) || !c.at_end())
            fail_here("expected header \"n dim nattr nmark\"");
        if (dim != 2) fail_here("dimension " + std::to_string(dim) + ", expected 2");
        if (nmark > 1) fail_here("marker column count must be 0 or 1");
    }
    x.reserve(n);
    y.reserve(n);
    marker.reserve(n);
    if (attributes) attributes->reserve(n * nattr);

    std::size_t first = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (!detail::next_data_line(in, line, lineno))
            detail::fail(fname, "header says " + std::to_string(n) + " vertices, found " +
                                    std::to_string(i));
        detail::LineCursor c{line};
        std::size_t idx;
        double px, py;
        if (!(c.next(idx) && c.next(px) && c.next(py))) fail_here("expected \"index x y ...\"");
        if (!std::isfinite(px) || !std::isfinite(py)) fail_here("coordinate is not finite");
        if (i == 0) {
            if (idx > 1) fail_here("vertex numbering must start at 0 or 1");
            first = idx;
        } else if (idx != first + i) {
            fail_here("vertex number " + std::to_string(idx) + ", expected " +
                      std::to_string(first + i));
        }
        for (std::size_t a = 0; a < nattr; ++a) {
            double v;
            if (!c.next(v)) fail_here("expected " + std::to_string(nattr) + " attributes");
            if (attributes) attributes->push_back(static_cast<T>(v));
        }
        int m = 0;
        if (nmark && !c.next(m)) fail_here("expected a boundary marker");
        if (!c.at_end()) fail_here("unexpected data after the record");
        x.push_back(static_cast<T>(px));
        y.push_back(static_cast<T>(py));
        marker.push_back(m);
    }
    if (detail::next_data_line(in, line, lineno))
        fail_here("unexpected data after vertex " + std::to_string(n - 1));
    return n;
}

// Inverse of read_nodes, numbered from 0. A null marker leaves the marker
// column out (nmark = 0); attributes, if given, are row-major n x nattr.
template <class T>
void write_nodes(const std::string& fname,
                 std::size_t n,
                 const T* x,
                 const T* y,
                 const int* marker = nullptr,
                 std::size_t nattr = 0,
                 const T* attributes = nullptr) {
    static_assert(std::is_floating_point_v<T>, "coordinates must be floating point");
    using detail::num;
    assert(nattr == 0 || attributes);
    auto out = detail::open_out(fname);
    out << n << " 2 " << nattr << ' ' << (marker ? 1 : 0) << '\n';
    for (std::size_t i = 0; i < n; ++i) {
        out << i << ' ' << num(x[i]) << ' ' << num(y[i]);
        for (std::size_t a = 0; a < nattr; ++a)
            out << ' ' << num(attributes[i * nattr + a]);
        if (marker) out << ' ' << marker[i];
        out << '\n';
    }
}

// ---------------------------------------------------------------------------
// Grid file (.grid), Nishikawa's 2D unstructured grid
// ---------------------------------------------------------------------------

// A 2D unstructured grid as a grid file holds it: nodes, triangle and quad
// connectivity, and the boundary as node lists, one list per boundary part.
// The file is 1-based; zero_based says which base the indices are kept in
// here, chosen by read_grid's second argument -- native 1-based, or
// 0-based like the graph file and the rest of the library.
template <class T = double, class I = std::int32_t>
struct Grid {
    std::vector<T> x, y;
    std::vector<I> tri;                 // 3 * num_triangles(), row-major
    std::vector<I> quad;                // 4 * num_quads(), row-major
    std::vector<std::vector<I>> bound;  // node indices of each boundary part, in
                                        // order along the boundary; a part marks
                                        // itself closed by repeating its first
                                        // node last
    bool zero_based = false;            // base of tri, quad and bound: the file's
                                        // 1-based by default, 0-based if shifted

    std::size_t num_nodes() const { return x.size(); }
    std::size_t num_triangles() const { return tri.size() / 3; }
    std::size_t num_quads() const { return quad.size() / 4; }
    std::size_t num_boundaries() const { return bound.size(); }

    // Whether boundary part b closes a loop, which the format marks by
    // repeating the node where the part closes; false for an open polyline.
    bool closed(std::size_t b) const {
        const auto& part = bound[b];
        return part.size() > 1 && part.front() == part.back();
    }

    // Boundary marker per node in the convention of the node file: 0 for a
    // node no part lists, else the number (from 1) of the first part that
    // lists it. NodeSet takes a Grid and uses this as the flag; the result
    // is indexed by node position, whatever the base.
    std::vector<int> markers() const {
        const I off = zero_based ? 0 : 1;
        std::vector<int> m(num_nodes(), 0);
        for (std::size_t b = 0; b < bound.size(); ++b)
            for (const I j : bound[b])
                if (m[j - off] == 0) m[j - off] = static_cast<int>(b + 1);
        return m;
    }
};

// Grid file, the custom 2D format of Nishikawa's grid-generation and
// EDU2D solver codes (docs/file_formats.md#grid-file has the references):
//
//     nnodes ntria nquad
//     x y                  (nnodes lines)
//     a b c                (ntria triangles)
//     a b c d              (nquad quads)
//     nbound
//     nb                   (the node count of every part, one per line)
//     b1                   (then the node lists, part after part,
//     ...                   one index per line)
//
// Node indices in the file are 1-based; zero_based = false keeps them
// that way, zero_based = true shifts them to 0-based, the numbering of
// the graph file, and the Grid records the choice for markers() and
// write_grid. Every count must be met exactly, an index must lie in
// [1, nnodes] in the file, an element's nodes must be distinct, and a
// boundary part must list at least two nodes, consecutive ones distinct.
// The format itself has no blank lines; the reader skips any it meets,
// so a file spaced apart for readability reads the same. The reference's
// orientation conventions (elements counterclockwise, parts with the
// interior on their left) are not checked.
template <class T = double, class I = std::int32_t>
Grid<T, I> read_grid(const std::string& fname, bool zero_based) {
    static_assert(std::is_floating_point_v<T>, "coordinates must be floating point");
    static_assert(std::is_integral_v<I> && std::is_signed_v<I>,
                  "index type must be a signed integer");
    auto in = detail::open_in(fname);
    std::string line;
    std::size_t lineno = 0;

    const auto fail_here = [&](const std::string& what) {
        detail::fail(fname, "line " + std::to_string(lineno) + ": " + what);
    };
    const auto next_line = [&](const std::string& what) {
        do {
            if (!std::getline(in, line)) detail::fail(fname, "unexpected end of file: " + what);
            ++lineno;
        } while (line.find_first_not_of(" \t\r") == std::string::npos);
    };
    const auto read_count = [&](const std::string& what) {
        next_line(what);
        detail::LineCursor c{line};
        std::size_t n;
        if (!c.next(n) || !c.at_end()) fail_here("expected " + what + " on a line of its own");
        return n;
    };

    Grid<T, I> g;
    g.zero_based = zero_based;
    std::size_t nnodes = 0, ntria = 0, nquad = 0;
    next_line("the header");
    {
        detail::LineCursor c{line};
        if (!(c.next(nnodes) && c.next(ntria) && c.next(nquad)) || !c.at_end())
            fail_here("expected the header \"nnodes ntria nquad\"");
    }
    g.x.reserve(nnodes);
    g.y.reserve(nnodes);
    for (std::size_t i = 0; i < nnodes; ++i) {
        next_line("node " + std::to_string(i + 1) + " of " + std::to_string(nnodes));
        detail::LineCursor c{line};
        T px, py;
        if (!(c.next(px) && c.next(py)) || !c.at_end()) fail_here("expected \"x y\"");
        if (!std::isfinite(px) || !std::isfinite(py)) fail_here("coordinate is not finite");
        g.x.push_back(px);
        g.y.push_back(py);
    }

    // an element's nodes, or one boundary node: 1-based in the file, checked
    // against nnodes and shifted only if a 0-based grid was asked for
    const I shift = zero_based ? 1 : 0;
    const auto next_index = [&](detail::LineCursor& c) {
        I v;
        if (!c.next(v)) fail_here("bad node index at \"" + c.here() + "\"");
        if (v < 1 || static_cast<std::size_t>(v) > nnodes)
            fail_here("node index " + std::to_string(v) + " outside [1, " + std::to_string(nnodes) +
                      "]; grid files are 1-based");
        return static_cast<I>(v - shift);
    };
    const auto read_elements = [&](std::vector<I>& conn, std::size_t ne, std::size_t nv,
                                   const char* name) {
        conn.reserve(nv * ne);
        for (std::size_t e = 0; e < ne; ++e) {
            next_line(name + std::string(" ") + std::to_string(e + 1) + " of " +
                      std::to_string(ne));
            detail::LineCursor c{line};
            const std::size_t at = conn.size();
            for (std::size_t v = 0; v < nv; ++v)
                conn.push_back(next_index(c));
            if (!c.at_end())
                fail_here(std::string("expected the ") + std::to_string(nv) + " nodes of a " +
                          name);
            for (std::size_t v = 0; v < nv; ++v)
                for (std::size_t w = v + 1; w < nv; ++w)
                    if (conn[at + v] == conn[at + w])
                        fail_here(std::string(name) + " node " +
                                  std::to_string(conn[at + v] + shift) + " listed twice");
        }
    };
    read_elements(g.tri, ntria, 3, "triangle");
    read_elements(g.quad, nquad, 4, "quadrilateral");

    const std::size_t nbound = read_count("the boundary count");
    g.bound.resize(nbound);
    const auto part_name = [](std::size_t b) { return "boundary part " + std::to_string(b + 1); };
    const auto read_part_count = [&](std::size_t b) {
        const std::size_t nb = read_count("the node count of " + part_name(b));
        if (nb < 2)
            fail_here(part_name(b) + " has " + std::to_string(nb) + " nodes, at least 2 needed");
        return nb;
    };
    const auto read_part_nodes = [&](std::size_t b, std::size_t nb) {
        auto& nodes = g.bound[b];
        nodes.reserve(nb);
        for (std::size_t j = 0; j < nb; ++j) {
            next_line("node " + std::to_string(j + 1) + " of " + part_name(b));
            detail::LineCursor c{line};
            const I v = next_index(c);
            if (!c.at_end()) fail_here("expected one boundary node per line");
            if (j > 0 && v == nodes.back())
                fail_here(part_name(b) + " lists node " + std::to_string(v + shift) +
                          " twice in a row");
            nodes.push_back(v);
        }
    };
    // every part's count first, then the lists, part after part
    std::vector<std::size_t> nb(nbound);
    for (std::size_t b = 0; b < nbound; ++b)
        nb[b] = read_part_count(b);
    for (std::size_t b = 0; b < nbound; ++b)
        read_part_nodes(b, nb[b]);
    in >> std::ws;
    if (!in.eof()) detail::fail(fname, "unexpected data after the last boundary part");
    return g;
}

// Inverse of read_grid, writing the canonical layout: the counts on the
// header line, boundary part counts before the lists, and indices
// 1-based, as the format requires, shifted back if the Grid says it is
// 0-based. The connectivity must be whole elements of valid indices in
// the Grid's base, and every boundary part needs the two nodes the
// reader requires; all of it is asserted, since a violation writes a
// file read_grid rejects.
template <class T, class I>
void write_grid(const std::string& fname, const Grid<T, I>& g) {
    using detail::num;
    const I shift = g.zero_based ? 1 : 0;
    assert(g.x.size() == g.y.size() && "x and y differ in length");
    assert(g.tri.size() % 3 == 0 && "tri is not whole triangles");
    assert(g.quad.size() % 4 == 0 && "quad is not whole quadrilaterals");
#ifndef NDEBUG
    const auto in_range = [&](I v) {
        return v >= 1 - shift && static_cast<std::size_t>(v + shift) <= g.x.size();
    };
    for (const I v : g.tri)
        assert(in_range(v) && "triangle node out of range");
    for (const I v : g.quad)
        assert(in_range(v) && "quadrilateral node out of range");
    for (const auto& part : g.bound) {
        assert(part.size() >= 2 && "boundary part of fewer than 2 nodes");
        for (const I v : part)
            assert(in_range(v) && "boundary node out of range");
    }
#endif
    auto out = detail::open_out(fname);
    out << g.num_nodes() << ' ' << g.num_triangles() << ' ' << g.num_quads() << '\n';
    for (std::size_t i = 0; i < g.num_nodes(); ++i)
        out << num(g.x[i]) << ' ' << num(g.y[i]) << '\n';
    for (std::size_t e = 0; e < g.num_triangles(); ++e)
        out << (g.tri[3 * e] + shift) << ' ' << (g.tri[3 * e + 1] + shift) << ' '
            << (g.tri[3 * e + 2] + shift) << '\n';
    for (std::size_t e = 0; e < g.num_quads(); ++e)
        out << (g.quad[4 * e] + shift) << ' ' << (g.quad[4 * e + 1] + shift) << ' '
            << (g.quad[4 * e + 2] + shift) << ' ' << (g.quad[4 * e + 3] + shift) << '\n';
    out << g.num_boundaries() << '\n';
    for (const auto& part : g.bound)
        out << part.size() << '\n';
    for (const auto& part : g.bound)
        for (const I v : part)
            out << (v + shift) << '\n';
}

// ---------------------------------------------------------------------------
// Boundary-condition files (.bcmap, .mapbc)
// ---------------------------------------------------------------------------

// The condition on one boundary part of a grid file, by its tag -- the
// part number Grid::markers() assigns. Which fields carry it depends on
// the dialect: the .bcmap of the EDU2D solvers names the condition (bc
// stays 0), FUN3D's .mapbc numbers it in bc, with name the optional
// family name, empty when absent.
struct BoundaryCondition {
    int tag = 0;
    int bc = 0;
    std::string name;
};

namespace detail {

inline void check_unique_tags(const std::string& fname, const std::vector<BoundaryCondition>& bcs) {
    for (std::size_t i = 0; i < bcs.size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            if (bcs[j].tag == bcs[i].tag)
                fail(fname, "boundary tag " + std::to_string(bcs[i].tag) + " listed twice");
}

}  // namespace detail

// Boundary-condition file of the EDU2D/3D solvers (.bcmap; the grid-file
// reference describes it): one boundary part per line, its tag and the
// name of its condition, read to end of file.
//
//     ! Boundary tag  BC name
//     1 freestream
//     2 subsonic_outflow
//     3 viscous_wall
//
// '!' starts a comment and blank lines are skipped. A tag listed twice is
// an error; the returned records keep the file's order.
inline std::vector<BoundaryCondition> read_bcmap(const std::string& fname) {
    auto in = detail::open_in(fname);
    std::vector<BoundaryCondition> bcs;
    std::string line;
    std::size_t lineno = 0;
    while (detail::next_data_line(in, line, lineno, '!')) {
        detail::LineCursor c{line};
        BoundaryCondition r;
        if (!c.next(r.tag) || !c.next_token(r.name) || !c.at_end())
            detail::fail(fname, "line " + std::to_string(lineno) + ": expected \"tag name\"");
        bcs.push_back(std::move(r));
    }
    detail::check_unique_tags(fname, bcs);
    return bcs;
}

// FUN3D's boundary-condition file (.mapbc; the FUN3D manual, appendix B):
// the number of boundary groups on the first line, then one line per part
// with its tag, the FUN3D boundary-condition number, and optionally a
// family name.
//
//     13
//     1 6662 box_ymin
//     2 5025 box_zmax
//     ...
//
// '!' starts a comment and blank lines are skipped, which also accepts the
// commented header some variants carry. The count must be met exactly with
// nothing after it, and a tag listed twice is an error; the returned
// records keep the file's order, name empty where no family is given.
inline std::vector<BoundaryCondition> read_mapbc(const std::string& fname) {
    auto in = detail::open_in(fname);
    std::string line;
    std::size_t lineno = 0;

    if (!detail::next_data_line(in, line, lineno, '!')) detail::fail(fname, "empty file");
    std::size_t n;
    if (detail::LineCursor c{line}; !c.next(n) || !c.at_end())
        detail::fail(fname, "line " + std::to_string(lineno) +
                                ": expected the boundary-group count on a line of its own");

    std::vector<BoundaryCondition> bcs;
    bcs.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!detail::next_data_line(in, line, lineno, '!'))
            detail::fail(fname, "header says " + std::to_string(n) + " boundary groups, found " +
                                    std::to_string(i));
        detail::LineCursor c{line};
        BoundaryCondition r;
        if (!c.next(r.tag) || !c.next(r.bc))
            detail::fail(fname,
                         "line " + std::to_string(lineno) + ": expected \"tag bc [family]\"");
        c.next_token(r.name);  // the family name is optional
        if (!c.at_end())
            detail::fail(fname, "line " + std::to_string(lineno) +
                                    ": unexpected data after the family name");
        bcs.push_back(std::move(r));
    }
    if (detail::next_data_line(in, line, lineno, '!'))
        detail::fail(fname, "line " + std::to_string(lineno) + ": unexpected data after " +
                                std::to_string(n) + " boundary groups");
    detail::check_unique_tags(fname, bcs);
    return bcs;
}

// ---------------------------------------------------------------------------
// Graph file (.graph)
// ---------------------------------------------------------------------------

// Adjacency graph in compressed sparse row form, from a graph file: a
// "n nnz" header, then one line of 0-based stencil indices per node:
//
//     n nnz
//     j00 j01 j02 ...
//     j10 j11 ...
//     ...
//
// Rows may have different lengths, but every row must have at least one
// entry (a node with no neighbours gives a singular system), every index
// must lie in [0, n) and appear at most once per row, the entry count must
// match nnz, and nothing may follow the n-th row. With k > 0 every row must hold exactly k entries
// and the header must satisfy nnz == n * k. Returns {ia, ja} with ia 0-based.
template <class I = std::int32_t>
std::pair<std::vector<I>, std::vector<I>> read_graph_csr(const std::string& fname, int k = 0) {
    static_assert(std::is_integral_v<I> && std::is_signed_v<I>,
                  "index type must be a signed integer");
    assert(k >= 0 && "k must be 0 (ragged rows) or the row length");
    auto in = detail::open_in(fname);

    std::string line;
    std::size_t n = 0, nnz = 0;
    {
        std::getline(in, line);
        detail::LineCursor c{line};
        if (!(c.next(n) && c.next(nnz)) || !c.at_end())
            detail::fail(fname, "malformed header, expected: n nnz");
    }
    if (k > 0 && nnz != n * static_cast<std::size_t>(k))
        detail::fail(fname, "header says " + std::to_string(n) + " rows and " +
                                std::to_string(nnz) +
                                " entries, inconsistent with k=" + std::to_string(k));

    std::vector<I> ia, ja;
    ia.reserve(n + 1);
    ia.push_back(0);
    ja.reserve(nnz);
    std::vector<std::size_t> last_row(n, n);  // row in which each index was last seen

    for (std::size_t i = 0; i < n; ++i) {
        if (!std::getline(in, line))
            detail::fail(fname,
                         "header says " + std::to_string(n) + " rows, found " + std::to_string(i));
        detail::LineCursor c{line};
        std::size_t len = 0;
        for (I j; !c.at_end(); ++len) {
            if (!c.next(j))
                detail::fail(fname,
                             "row " + std::to_string(i) + ": bad index at \"" + c.here() + "\"");
            if (j < 0 || static_cast<std::size_t>(j) >= n)
                detail::fail(fname, "row " + std::to_string(i) + ": index " + std::to_string(j) +
                                        " outside [0, " + std::to_string(n) +
                                        "); indices are 0-based");
            if (last_row[j] == i)
                detail::fail(fname, "row " + std::to_string(i) + ": index " + std::to_string(j) +
                                        " listed twice");
            last_row[j] = i;
            ja.push_back(j);
        }
        if (len == 0) detail::fail(fname, "row " + std::to_string(i) + " has no entries");
        if (k > 0 && len != static_cast<std::size_t>(k))
            detail::fail(fname, "row " + std::to_string(i) + " has " + std::to_string(len) +
                                    " entries, expected k=" + std::to_string(k));
        ia.push_back(static_cast<I>(ja.size()));
    }
    if (ja.size() != nnz)
        detail::fail(fname, "header says " + std::to_string(nnz) + " entries, found " +
                                std::to_string(ja.size()));
    in >> std::ws;
    if (!in.eof()) detail::fail(fname, "unexpected data after row " + std::to_string(n - 1));
    return {std::move(ia), std::move(ja)};
}

// ---------------------------------------------------------------------------
// Ordering file (.iperm)
// ---------------------------------------------------------------------------

// A permutation of the nodes in the METIS ordering-file format: one integer
// per line, no header, read to end of file. Line i holds the new index of
// node i, so the file is the inverse permutation iperm, old -> new, 0-based.
// Permutation::read and write in rbf_reorder.h wrap these two.
//
// The values must form a permutation of 0 .. n-1: an index out of range or
// listed twice is an error.
template <class I = std::int32_t>
std::vector<I> read_ordering(const std::string& fname) {
    static_assert(std::is_integral_v<I> && std::is_signed_v<I>,
                  "index type must be a signed integer");
    auto in = detail::open_in(fname);
    std::vector<I> iperm;
    for (I v; in >> v;)
        iperm.push_back(v);
    if (!in.eof())
        detail::fail(fname, "line " + std::to_string(iperm.size() + 1) + ": not an integer");

    const std::size_t n = iperm.size();
    std::vector<char> seen(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const I v = iperm[i];
        if (v < 0 || static_cast<std::size_t>(v) >= n)
            detail::fail(fname, "line " + std::to_string(i + 1) + ": index " + std::to_string(v) +
                                    " outside [0, " + std::to_string(n) + ")");
        if (seen[v])
            detail::fail(fname, "line " + std::to_string(i + 1) + ": index " + std::to_string(v) +
                                    " appears twice");
        seen[v] = 1;
    }
    return iperm;
}

// Inverse of read_ordering: iperm[i] is the new index of node i.
template <class I>
void write_ordering(const std::string& fname, std::size_t n, const I* iperm) {
    static_assert(std::is_integral_v<I>, "index type must be integral");
#ifndef NDEBUG
    std::vector<char> seen(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        assert(iperm[i] >= 0 && static_cast<std::size_t>(iperm[i]) < n &&
               "iperm index out of range");
        assert(!seen[iperm[i]] && "iperm index listed twice: not a permutation");
        seen[iperm[i]] = 1;
    }
#endif
    auto out = detail::open_out(fname);
    for (std::size_t i = 0; i < n; ++i)
        out << iperm[i] << '\n';
}

// ---------------------------------------------------------------------------
// Matrix Market
// ---------------------------------------------------------------------------

// CSR -> Matrix Market coordinate format, for inspection in MATLAB/Python or
// as input to a reference solver.
//
// The output is always 1-based, as the format requires. csr_base is the base
// of the ia and ja arrays passed in (0 or 1) and is removed before the
// conversion. ia[0] must equal it, ia must be non-decreasing, and every
// column must lie in [csr_base, csr_base + cols); all of this is asserted,
// since a violation would write a file that solvers reject or misread.
// The entry count is taken from ia, so the header can never disagree with
// the body.
//
// Pass a == nullptr to write the sparsity pattern alone, as Matrix Market's
// "pattern" value type: (row, col) entries with no values. Useful before
// assembly, or to compare stencil graphs without the weights.
template <class T, class I>
void write_matrix_market(const std::string& fname,
                         std::size_t rows,
                         std::size_t cols,
                         const I* ia,
                         const I* ja,
                         const T* a,
                         int csr_base = 0) {
    static_assert(std::is_integral_v<I>, "index type must be integral");
    static_assert(std::is_floating_point_v<T>, "value type must be floating point");
    using detail::num;
    assert((csr_base == 0 || csr_base == 1) && "csr_base must be 0 or 1");
    assert(ia[0] == csr_base && "ia does not start at the declared csr_base");
#ifndef NDEBUG
    for (std::size_t i = 0; i < rows; ++i) {
        assert(ia[i + 1] >= ia[i] && "ia is not non-decreasing");
        for (I k = ia[i] - csr_base; k < ia[i + 1] - csr_base; ++k)
            assert(ja[k] >= csr_base && static_cast<std::size_t>(ja[k] - csr_base) < cols &&
                   "ja column out of range");
    }
#endif
    auto out = detail::open_out(fname);
    out << "%%MatrixMarket matrix coordinate " << (a ? "real" : "pattern") << " general\n";
    out << rows << ' ' << cols << ' ' << (ia[rows] - ia[0]) << '\n';
    for (std::size_t i = 0; i < rows; ++i)
        for (I k = ia[i] - csr_base; k < ia[i + 1] - csr_base; ++k) {
            out << (i + 1) << ' ' << (ja[k] - csr_base + 1);
            if (a) out << ' ' << num(a[k]);
            out << '\n';
        }
}

// Sparsity pattern only, without having to name a value type at the call site.
template <class I>
void write_matrix_market_pattern(const std::string& fname,
                                 std::size_t rows,
                                 std::size_t cols,
                                 const I* ia,
                                 const I* ja,
                                 int csr_base = 0) {
    write_matrix_market<double, I>(fname, rows, cols, ia, ja, nullptr, csr_base);
}

}  // namespace rbf::io

#endif  // RBF_IO_H
