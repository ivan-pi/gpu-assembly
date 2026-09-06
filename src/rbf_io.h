#ifndef RBF_IO_H
#define RBF_IO_H

// ASCII input and output for RBF-FD drivers.
//
//   read_points, read_points_aos   count-prefixed "x y" list -> SoA or AoS
//   read_nodes, write_nodes        "x y flag" list, the format NodeSet reads
//   read_graph_csr                 variable-length adjacency list -> (ia, ja)
//   write_matrix_market            CSR matrix -> Matrix Market (real, or pattern)
//
// Every reader checks that the file opened. Readers whose format carries a
// count also check that they got that many, exiting with a message rather
// than returning a short container.
// Every writer emits full round-trip precision for floating point, so a file
// written here reproduces the values it was given.
//
// These are ASCII formats: convenient, diffable, and slow. If reading becomes
// a bottleneck the answer is a binary format, not a faster parser.

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

// Everything from the current position to the end of the stream, in one
// read; the size comes from seeking, so this needs a real file.
inline std::string read_rest(std::istream& in) {
    const auto here = in.tellg();
    assert(here != std::istream::pos_type(-1) && "stream is not seekable");
    in.seekg(0, std::ios::end);
    std::string body(static_cast<std::size_t>(in.tellg() - here), '\0');
    in.seekg(here);
    in.read(body.data(), static_cast<std::streamsize>(body.size()));
    body.resize(static_cast<std::size_t>(in.gcount()));
    return body;
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
// Points
// ---------------------------------------------------------------------------

// Coordinates as separate arrays (SoA), from a file whose first line is the
// point count:
//
//     n
//     x0 y0
//     x1 y1
//     ...
//
// Returns {x, y}, the layout the assembly kernels take. This is a different
// format from the one NodeSet reads (see read_nodes): no count header there,
// and a third flag column per line. Use this reader for plain point clouds
// that carry no per-node tag, such as node generator output.
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
// Nodes: coordinates plus a per-node flag, the format NodeSet reads
// ---------------------------------------------------------------------------

// One node per line, no header, read to end of file:
//
//     x0 y0 flag0
//     x1 y1 flag1
//     ...
//
// x, y and flag are cleared and filled, and the node count is returned. The
// flag is 0 for interior nodes and nonzero for boundary nodes. The file is
// taken to be complete: reading stops at the first record that does not
// parse as three numbers, whether that is the end of the file or not.
// NodeSet's constructor delegates to this.
template <class T>
std::size_t read_nodes(const std::string& fname,
                       std::vector<T>& x, std::vector<T>& y, std::vector<int>& flag)
{
    auto in = detail::open_in(fname);
    x.clear(); y.clear(); flag.clear();
    T px, py; int f;
    while (in >> px >> py >> f) {
        x.push_back(px);
        y.push_back(py);
        flag.push_back(f);
    }
    return x.size();
}

// Inverse of read_nodes. A null flag writes 0 (interior) for every node.
template <class T>
void write_nodes(const std::string& fname, std::size_t n,
                 const T* x, const T* y, const int* flag = nullptr)
{
    auto out = detail::open_out(fname);
    detail::full_precision<T>(out);
    for (std::size_t i = 0; i < n; ++i)
        out << x[i] << ' ' << y[i] << ' ' << (flag ? flag[i] : 0) << '\n';
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

namespace detail {

// Parse every integer in text into ja; returns how many were found.
// Anything that is not whitespace or an integer is an error, reported
// under the given context ("row 3", "body").
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

// Adjacency graph in compressed sparse row form, from a file with a
// "rows nnz" header followed by one line of neighbour indices per row:
//
//     n nnz
//     j00 j01 j02 ...
//     j10 j11 ...
//     ...
//
// With k == 0 (the default) row lengths may differ, and the line structure
// says where each row ends. Every row must list at least one neighbour: a
// node with no neighbours gives a singular system, so an empty row is an
// error, as are a short file, an entry count that disagrees with nnz, and
// data after the n-th row.
//
// With k > 0 every row is known to hold exactly k entries, as for k-nearest
// neighbour stencils. The header must satisfy nnz == n * k, and the body is
// then read as a flat list of n * k indices, line breaks carrying no
// meaning. Use it when k is known: the check against the header is the
// stronger one.
//
// Returns {ia, ja}. Index values are passed through unchanged, so they must
// already be in the base (0 or 1) the consumer expects; ia is built 0-based.
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

    if (k > 0) {
        if (nnz != n * static_cast<std::size_t>(k))
            detail::fail(fname, "header says " + std::to_string(n) + " rows and "
                                + std::to_string(nnz) + " entries, inconsistent with k="
                                + std::to_string(k));
        ja.reserve(nnz);
        const std::string body = detail::read_rest(in);
        const std::size_t found = detail::parse_ints(body, ja, fname, "body");
        if (found != nnz)
            detail::fail(fname, "header says " + std::to_string(nnz) + " entries, found "
                                + std::to_string(found));
        for (std::size_t i = 1; i <= n; ++i) ia.push_back(static_cast<I>(i * k));
    } else {
        ja.reserve(nnz);
        std::string line;
        std::getline(in, line);   // rest of the header line
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::getline(in, line))
                detail::fail(fname, "header says " + std::to_string(n)
                                    + " rows, found " + std::to_string(i));
            if (detail::parse_ints(line, ja, fname, "row " + std::to_string(i)) == 0)
                detail::fail(fname, "row " + std::to_string(i) + " has no entries");
            ia.push_back(static_cast<I>(ja.size()));
        }
        if (ja.size() != nnz)
            detail::fail(fname, "header says " + std::to_string(nnz) + " entries, found "
                                + std::to_string(ja.size()));
        // the flat path consumed the whole file above; only this one can
        // leave data behind
        detail::expect_end(in, fname, "row " + std::to_string(n - 1));
    }
    return {std::move(ia), std::move(ja)};
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
//
// Values, when written, carry max_digits10 significant digits, so the file
// round-trips exactly; the stream default of 6 would quietly truncate
// weights well above the accuracy the assembly is tested to.
template <class T, class I>
void write_matrix_market(const std::string& fname,
                         I rows, I cols,
                         const I* ia, const I* ja, const T* a,
                         I csr_base = 0)
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
                                 I rows, I cols,
                                 const I* ia, const I* ja,
                                 I csr_base = 0)
{
    write_matrix_market<double, I>(fname, rows, cols, ia, ja, nullptr, csr_base);
}

} // namespace rbf::io

#endif // RBF_IO_H
