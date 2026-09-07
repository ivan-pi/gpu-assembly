#ifndef RBF_IO_H
#define RBF_IO_H

// ASCII input and output for RBF-FD drivers. The formats are described in
// docs/file_formats.md.
//
//   read_points, read_points_aos     points file (.points): count, then "x y" -> SoA or AoS
//   read_graph_csr                   graph file (.graph): stencils -> (ia, ja)
//   read_nodes, write_nodes          node file (.node), Triangle's format; what NodeSet reads and
//   writes read_ordering, write_ordering    ordering file (.iperm): one new index per node
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

    // The text at the cursor, for error messages.
    std::string here() const { return std::string(std::string_view(p, end - p).substr(0, 16)); }
};

// Next line with content: '#' starts a comment, blank lines are skipped.
// The line is truncated at the comment so the cursor sees data only.
inline bool next_data_line(std::istream& in, std::string& line, std::size_t& lineno) {
    while (std::getline(in, line)) {
        ++lineno;
        if (const auto hash = line.find('#'); hash != std::string::npos) line.resize(hash);
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
