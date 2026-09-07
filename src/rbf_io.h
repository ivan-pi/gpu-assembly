#ifndef RBF_IO_H
#define RBF_IO_H

// ASCII input and output for RBF-FD drivers. The formats are described in
// docs/file_formats.md.
//
//   read_points, read_points_aos     points file (.points): count, then "x y" -> SoA or AoS
//   read_graph_csr                   graph file (.graph): stencils -> (ia, ja)
//   read_nodes, write_nodes          node file (.node), Triangle's format; what NodeSet reads and writes
//   read_ordering, write_ordering    ordering file (.iperm): one new index per node
//   write_matrix_market              CSR matrix -> Matrix Market (real, or pattern)
//
// Every reader checks that the file opened. Readers whose format carries a
// count also check that they got that many, exiting with a message rather
// than returning a short container.
// Every writer emits full round-trip precision for floating point, so a file
// written here reproduces the values it was given.
//
// Assisted-by: Claude Fable 5.1

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace rbf::io {

// Helpers shared by the readers and writers.
namespace detail {

inline std::ifstream open_in(const std::string& fname) {
    std::ifstream in{fname};
    if (!in) {
        std::cerr << "rbf::io: cannot open " << fname << " for reading\n";
        std::exit(1);
    }
    return in;
}

inline std::ofstream open_out(const std::string& fname) {
    std::ofstream out{fname};
    if (!out) {
        std::cerr << "rbf::io: cannot open " << fname << " for writing\n";
        std::exit(1);
    }
    return out;
}

[[noreturn]] inline void fail(const std::string& fname, const std::string& what) {
    std::cerr << "rbf::io: " << fname << ": " << what << '\n';
    std::exit(1);
}

// Enough significant digits to round-trip a T exactly.
template <class T>
std::ostream& full_precision(std::ostream& os) {
    return os << std::setprecision(std::numeric_limits<T>::max_digits10);
}

// Nothing but whitespace remains: readers call this after consuming what
// the header announced, so trailing data is reported, not ignored.
inline void expect_end(std::istream& in, const std::string& fname,
                       const std::string& what) {
    in >> std::ws;
    if (!in.eof()) detail::fail(fname, "unexpected data after " + what);
}

} // namespace detail

// ---------------------------------------------------------------------------
// Points file (.points)
// ---------------------------------------------------------------------------

// Coordinates as separate arrays (SoA), from a points file: the point count
// on the first line, then one coordinate pair per line:
//
//     n
//     x0 y0
//     x1 y1
//     ...
//
// Returns {x, y}, the layout the assembly kernels take. The points file is
// the companion of the graph file (read_graph_csr): one gives the point
// cloud, the other the stencils, in the same numbering.
template <class T = double>
std::pair<std::vector<T>, std::vector<T>> read_points(const std::string& fname)
{
    auto in = detail::open_in(fname);
    std::size_t n = 0;
    if (!(in >> n)) detail::fail(fname, "missing or malformed point count");

    std::vector<T> x, y;
    x.reserve(n); y.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        T px, py;
        in >> px >> py;
        x.push_back(px); y.push_back(py);
    }
    // One check: stream failure is sticky, so a short or malformed file fails
    // here whether it broke on the first record or the last.
    if (!in) detail::fail(fname, "expected " + std::to_string(n) + " points");
    return {std::move(x), std::move(y)};
}

// Same file, into an array of structs: std::vector<std::array<T,2>>,
// std::vector<Point> with Point{T x, y;}, or any container whose value_type
// is brace-constructible from two T. Preferred where a point is passed
// around as a unit; the SoA form suits device upload.
template <class ArrayOfStructs, class T = double>
ArrayOfStructs read_points_aos(const std::string& fname)
{
    using Struct = typename ArrayOfStructs::value_type;
    auto in = detail::open_in(fname);
    std::size_t n = 0;
    if (!(in >> n)) detail::fail(fname, "missing or malformed point count");

    ArrayOfStructs xy;
    xy.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        T px, py;
        in >> px >> py;
        xy.push_back(Struct{px, py});
    }
    if (!in) detail::fail(fname, "expected " + std::to_string(n) + " points");
    return xy;
}

// ---------------------------------------------------------------------------
// Node file (.node), the Triangle format
// ---------------------------------------------------------------------------

namespace detail {

// Tokens of one line, parsed with strtod/strtoll so that a line is one
// null-terminated buffer and no stream is built per line.
struct LineCursor {
    const char* p;

    void skip_ws() { while (*p == ' ' || *p == '\t' || *p == '\r') ++p; }
    bool at_end() { skip_ws(); return *p == '\0'; }

    bool real(double& v) {
        char* e;
        v = std::strtod(p, &e);
        if (e == p) return false;
        p = e;
        return true;
    }
    // An integer must end at whitespace or end of line, so "1.5" is not 1.
    bool integer(long long& v) {
        char* e;
        v = std::strtoll(p, &e, 10);
        if (e == p || (*e != '\0' && *e != ' ' && *e != '\t' && *e != '\r')) return false;
        p = e;
        return true;
    }
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

} // namespace detail

// Node file in the format of Triangle's .node file
// (https://www.cs.cmu.edu/~quake/triangle.node.html):
//
//     n 2 nattr nmark            # vertices, dimension, attributes, marker column (0 or 1)
//     i x y a0 ... a(nattr-1) [marker]
//     ...
//
// '#' starts a comment and blank lines are allowed anywhere. Vertices are
// numbered consecutively from 0 or from 1; either is accepted and the line
// order is the node numbering. The marker is 0 for an interior node and
// nonzero for a boundary node, which is NodeSet's flag; without a marker
// column every node is interior. Attributes are per-node reals, returned
// row-major, n x nattr, when a vector is passed for them, and skipped
// otherwise. Returns n.
template <class T>
std::size_t read_nodes(const std::string& fname,
                       std::vector<T>& x, std::vector<T>& y, std::vector<int>& marker,
                       std::vector<T>* attributes = nullptr)
{
    auto in = detail::open_in(fname);
    x.clear(); y.clear(); marker.clear();
    if (attributes) attributes->clear();

    std::string line;
    std::size_t lineno = 0;
    const auto where = [&] { return "line " + std::to_string(lineno) + ": "; };

    if (!detail::next_data_line(in, line, lineno)) detail::fail(fname, "empty file");
    long long n, dim, nattr, nmark;
    {
        detail::LineCursor c{line.c_str()};
        if (!(c.integer(n) && c.integer(dim) && c.integer(nattr) && c.integer(nmark)) || !c.at_end())
            detail::fail(fname, where() + "expected header \"n dim nattr nmark\"");
        if (dim != 2) detail::fail(fname, where() + "dimension " + std::to_string(dim) + ", expected 2");
        if (n < 0 || nattr < 0 || nmark < 0 || nmark > 1)
            detail::fail(fname, where() + "bad header values");
    }
    x.reserve(n); y.reserve(n); marker.reserve(n);
    if (attributes) attributes->reserve(n * nattr);

    long long first = 0;
    for (long long i = 0; i < n; ++i) {
        if (!detail::next_data_line(in, line, lineno))
            detail::fail(fname, "header says " + std::to_string(n) + " vertices, found "
                                + std::to_string(i));
        detail::LineCursor c{line.c_str()};
        long long idx;
        double px, py;
        if (!(c.integer(idx) && c.real(px) && c.real(py)))
            detail::fail(fname, where() + "expected \"index x y ...\"");
        if (i == 0) {
            if (idx != 0 && idx != 1)
                detail::fail(fname, where() + "vertex numbering must start at 0 or 1");
            first = idx;
        } else if (idx != first + i) {
            detail::fail(fname, where() + "vertex number " + std::to_string(idx)
                                + ", expected " + std::to_string(first + i));
        }
        for (long long a = 0; a < nattr; ++a) {
            double v;
            if (!c.real(v)) detail::fail(fname, where() + "expected " + std::to_string(nattr) + " attributes");
            if (attributes) attributes->push_back(static_cast<T>(v));
        }
        long long m = 0;
        if (nmark && !c.integer(m)) detail::fail(fname, where() + "expected a boundary marker");
        if (!c.at_end()) detail::fail(fname, where() + "unexpected data after the record");
        x.push_back(static_cast<T>(px));
        y.push_back(static_cast<T>(py));
        marker.push_back(static_cast<int>(m));
    }
    if (detail::next_data_line(in, line, lineno))
        detail::fail(fname, where() + "unexpected data after vertex " + std::to_string(n - 1));
    return static_cast<std::size_t>(n);
}

// Inverse of read_nodes, numbered from 0. A null marker leaves the marker
// column out (nmark = 0); attributes, if given, are row-major n x nattr.
template <class T>
void write_nodes(const std::string& fname, std::size_t n,
                 const T* x, const T* y, const int* marker = nullptr,
                 std::size_t nattr = 0, const T* attributes = nullptr)
{
    assert(nattr == 0 || attributes);
    auto out = detail::open_out(fname);
    detail::full_precision<T>(out);
    out << n << " 2 " << nattr << ' ' << (marker ? 1 : 0) << '\n';
    for (std::size_t i = 0; i < n; ++i) {
        out << i << ' ' << x[i] << ' ' << y[i];
        for (std::size_t a = 0; a < nattr; ++a) out << ' ' << attributes[i * nattr + a];
        if (marker) out << ' ' << marker[i];
        out << '\n';
    }
}

// ---------------------------------------------------------------------------
// Graph file (.graph)
// ---------------------------------------------------------------------------

namespace detail {

// Parse every integer in text into ja; returns how many were found.
// Anything that is not whitespace or an integer is an error, reported
// under the given context ("row 3").
template <class I>
std::size_t parse_ints(std::string_view text, std::vector<I>& ja,
                       const std::string& fname, const std::string& context)
{
    std::size_t count = 0;
    const char* p = text.data();
    const char* const end = p + text.size();
    for (;;) {
        while (p != end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p;
        if (p == end) return count;
        I j;
        const auto [next, ec] = std::from_chars(p, end, j);
        if (ec != std::errc{})
            fail(fname, context + ": bad index at \""
                        + std::string(p, std::min<std::size_t>(end - p, 16)) + "\"");
        ja.push_back(j);
        ++count;
        p = next;
    }
}

} // namespace detail

// Adjacency graph in compressed sparse row form, from a graph file: a
// "rows nnz" header followed by one line of stencil indices per node:
//
//     n nnz
//     j00 j01 j02 ...
//     j10 j11 ...
//     ...
//
// Each line is one row; rows may have different lengths. Every row must
// list at least one neighbour: a
// node with no neighbours gives a singular system, so an empty row is an
// error, as are a short file, an entry count that disagrees with nnz, and
// data after the n-th row.
//
// With k > 0 every row is required to hold exactly k entries, as for
// k-nearest-neighbour stencils; the header must satisfy nnz == n * k and a
// row of any other length is an error. Same file format, stronger checks.
//
// Indices are 0-based, as in every graph file produced so far: a value
// outside [0, n) is an error, which also catches a 1-based file. Returns
// {ia, ja} with ia 0-based.
template <class I = std::int32_t>
std::pair<std::vector<I>, std::vector<I>> read_graph_csr(const std::string& fname,
                                                         int k = 0)
{
    assert(k >= 0 && "k must be 0 (ragged rows) or the row length");
    auto in = detail::open_in(fname);

    std::size_t n = 0, nnz = 0;
    if (!(in >> n >> nnz)) detail::fail(fname, "malformed header, expected: rows nnz");

    std::vector<I> ia, ja;
    ia.reserve(n + 1);
    ia.push_back(0);

    if (k > 0 && nnz != n * static_cast<std::size_t>(k))
        detail::fail(fname, "header says " + std::to_string(n) + " rows and "
                            + std::to_string(nnz) + " entries, inconsistent with k="
                            + std::to_string(k));
    ja.reserve(nnz);
    std::string line;
    std::getline(in, line);   // rest of the header line
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::getline(in, line))
            detail::fail(fname, "header says " + std::to_string(n)
                                + " rows, found " + std::to_string(i));
        const std::size_t len = detail::parse_ints(line, ja, fname, "row " + std::to_string(i));
        if (len == 0)
            detail::fail(fname, "row " + std::to_string(i) + " has no entries");
        if (k > 0 && len != static_cast<std::size_t>(k))
            detail::fail(fname, "row " + std::to_string(i) + " has " + std::to_string(len)
                                + " entries, expected k=" + std::to_string(k));
        ia.push_back(static_cast<I>(ja.size()));
    }
    if (ja.size() != nnz)
        detail::fail(fname, "header says " + std::to_string(nnz) + " entries, found "
                            + std::to_string(ja.size()));
    detail::expect_end(in, fname, "row " + std::to_string(n - 1));
    for (std::size_t p = 0; p < ja.size(); ++p)
        if (ja[p] < 0 || static_cast<std::size_t>(ja[p]) >= n)
            detail::fail(fname, "entry " + std::to_string(p) + " is index "
                                + std::to_string(ja[p]) + ", outside [0, "
                                + std::to_string(n) + "); indices are 0-based");
    return {std::move(ia), std::move(ja)};
}

// ---------------------------------------------------------------------------
// Ordering file (.iperm)
// ---------------------------------------------------------------------------

// A permutation of the nodes in the METIS ordering-file format (manual
// 5.1.0, section 4.2.2): one integer per line, no header, read to end of
// file. Line i holds the new index of node i, so the file is the inverse
// permutation iperm, old -> new, 0-based. Permutation::read and write in
// rbf_reorder.h wrap these two for a Permutation.
//
// The values must form a permutation of 0 .. n-1: an index out of range or
// listed twice is an error.
template <class I = std::int32_t>
std::vector<I> read_ordering(const std::string& fname)
{
    auto in = detail::open_in(fname);
    std::vector<I> iperm;
    for (I v; in >> v; ) iperm.push_back(v);
    if (!in.eof())
        detail::fail(fname, "line " + std::to_string(iperm.size() + 1) + ": not an integer");

    const std::size_t n = iperm.size();
    std::vector<char> seen(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const I v = iperm[i];
        if (v < 0 || static_cast<std::size_t>(v) >= n)
            detail::fail(fname, "line " + std::to_string(i + 1) + ": index "
                                + std::to_string(v) + " outside [0, " + std::to_string(n) + ")");
        if (seen[v])
            detail::fail(fname, "line " + std::to_string(i + 1) + ": index "
                                + std::to_string(v) + " appears twice");
        seen[v] = 1;
    }
    return iperm;
}

// Inverse of read_ordering: iperm[i] is the new index of node i.
template <class I>
void write_ordering(const std::string& fname, std::size_t n, const I* iperm)
{
    auto out = detail::open_out(fname);
    for (std::size_t i = 0; i < n; ++i) out << iperm[i] << '\n';
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
// the body. I is deduced from ia and ja alone, so rows, cols and csr_base
// can be plain integer literals whatever the index type.
//
// Pass a == nullptr to write the sparsity pattern alone, as Matrix Market's
// "pattern" value type: (row, col) entries with no values. Useful before
// assembly, or to compare stencil graphs without the weights.
//
// Values, when written, carry max_digits10 significant digits, so the file
// round-trips exactly; the stream default of 6 would quietly truncate
// weights well above the accuracy the assembly is tested to.
template <class T, class I>
void write_matrix_market(const std::string& fname,
                         std::type_identity_t<I> rows, std::type_identity_t<I> cols,
                         const I* ia, const I* ja, const T* a,
                         std::type_identity_t<I> csr_base = 0)
{
    assert((csr_base == 0 || csr_base == 1) && "csr_base must be 0 or 1");
    assert(rows >= 0 && cols >= 0);
    assert(ia[0] == csr_base && "ia does not start at the declared csr_base");
#ifndef NDEBUG
    for (I i = 0; i < rows; ++i) {
        assert(ia[i + 1] >= ia[i] && "ia is not non-decreasing");
        for (I k = ia[i] - csr_base; k < ia[i + 1] - csr_base; ++k)
            assert(ja[k] >= csr_base && ja[k] - csr_base < cols && "ja column out of range");
    }
#endif
    auto out = detail::open_out(fname);
    const I nnz = ia[rows] - ia[0];
    out << "%%MatrixMarket matrix coordinate " << (a ? "real" : "pattern")
        << " general\n";
    out << rows << ' ' << cols << ' ' << nnz << '\n';
    if (a) {
        detail::full_precision<T>(out);
        for (I i = 0; i < rows; ++i)
            for (I k = ia[i] - csr_base; k < ia[i + 1] - csr_base; ++k)
                out << (i + 1) << ' ' << (ja[k] - csr_base + 1) << ' ' << a[k] << '\n';
    } else {
        for (I i = 0; i < rows; ++i)
            for (I k = ia[i] - csr_base; k < ia[i + 1] - csr_base; ++k)
                out << (i + 1) << ' ' << (ja[k] - csr_base + 1) << '\n';
    }
}

// Sparsity pattern only, without having to name a value type at the call site.
template <class I>
void write_matrix_market_pattern(const std::string& fname,
                                 std::type_identity_t<I> rows, std::type_identity_t<I> cols,
                                 const I* ia, const I* ja,
                                 std::type_identity_t<I> csr_base = 0)
{
    write_matrix_market<double, I>(fname, rows, cols, ia, ja, nullptr, csr_base);
}

} // namespace rbf::io

#endif // RBF_IO_H
