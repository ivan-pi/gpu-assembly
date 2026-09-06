#ifndef RBF_IO_H
#define RBF_IO_H

// ASCII input and output for RBF-FD drivers.
//
//   read_points, read_points_aos   count-prefixed "x y" list -> SoA or AoS
//   read_nodes, write_nodes        "x y flag" list, the format NodeSet reads
//   read_graph_csr                 variable-length adjacency list -> (ia, ja)
//   write_matrix_market            CSR matrix -> Matrix Market (real, or pattern)
//   write_field, read_field        one value per node <-> plain list
//
// Every reader checks that the file opened and that the data it returns is
// complete, exiting with a message rather than returning a short container.
// Every writer emits full round-trip precision for floating point, so a file
// written here reproduces the values it was given.
//
// These are ASCII formats: convenient, diffable, and slow. If reading becomes
// a bottleneck the answer is a binary format, not a faster parser.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rbf::io {

// Helpers shared by the readers and writers. A named namespace rather than an
// anonymous one: an anonymous namespace in a header gives every translation
// unit its own copy, and the templates below would then refer to different
// functions in different units, which the one-definition rule forbids.
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

// True if the line holds nothing but whitespace.
inline bool blank(const std::string& line) {
    return line.find_first_not_of(" \t\r\n") == std::string::npos;
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

template <class T = double>
struct Nodes {
    std::vector<T> x, y;
    std::vector<int> flag;   // 0 = interior, nonzero = boundary
    std::size_t size() const { return x.size(); }
};

// One node per line, no header, blank lines ignored:
//
//     x0 y0 flag0
//     x1 y1 flag1
//     ...
//
// A line that does not parse as "x y flag" is an error, reported with its
// line number. NodeSet's constructor delegates to this.
template <class T = double>
Nodes<T> read_nodes(const std::string& fname)
{
    auto in = detail::open_in(fname);
    Nodes<T> nodes;
    std::string line;
    for (std::size_t lineno = 1; std::getline(in, line); ++lineno) {
        if (detail::blank(line)) continue;
        std::istringstream ls(line);
        T px, py; int f;
        if (!(ls >> px >> py >> f))
            detail::fail(fname, "line " + std::to_string(lineno)
                                + ": expected \"x y flag\"");
        nodes.x.push_back(px);
        nodes.y.push_back(py);
        nodes.flag.push_back(f);
    }
    return nodes;
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

template <class T>
void write_nodes(const std::string& fname, const Nodes<T>& nodes)
{
    write_nodes(fname, nodes.size(), nodes.x.data(), nodes.y.data(),
                nodes.flag.data());
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

// Adjacency graph in compressed sparse row form, from a file with a
// "rows nnz" header followed by one line of neighbour indices per row:
//
//     n nnz
//     j00 j01 j02 ...
//     j10 j11 ...
//     ...
//
// Row lengths need not be equal, which is the point: a fixed-k stencil set
// does not need a file at all. Exactly n rows are read and each must list at
// least one neighbour; a node with no neighbours would give a singular
// system, so an empty row is reported as an error rather than stored. The
// entry count is checked against nnz, and anything but whitespace after the
// n-th row is an error too.
//
// Returns {ia, ja}. Index values are passed through unchanged, so they must
// already be in the base (0 or 1) the consumer expects; ia is built 0-based.
template <class I = std::int32_t>
std::pair<std::vector<I>, std::vector<I>> read_graph_csr(const std::string& fname)
{
    auto in = detail::open_in(fname);

    std::string header;
    if (!std::getline(in, header)) detail::fail(fname, "empty file");
    std::istringstream hs(header);
    std::size_t n = 0, nnz = 0;
    if (!(hs >> n >> nnz)) detail::fail(fname, "malformed header, expected: rows nnz");

    std::vector<I> ia, ja;
    ia.reserve(n + 1); ja.reserve(nnz);
    ia.push_back(0);

    std::string line;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::getline(in, line))
            detail::fail(fname, "header says " + std::to_string(n) + " rows, found "
                                + std::to_string(i));
        std::istringstream rs(line);
        const std::size_t before = ja.size();
        for (I j; rs >> j; ) ja.push_back(j);
        if (ja.size() == before)
            detail::fail(fname, "row " + std::to_string(i) + " has no entries");
        ia.push_back(static_cast<I>(ja.size()));
    }
    while (std::getline(in, line))
        if (!detail::blank(line))
            detail::fail(fname, "data after row " + std::to_string(n - 1)
                                + ", header says " + std::to_string(n) + " rows");

    if (ja.size() != nnz)
        detail::fail(fname, "header says " + std::to_string(nnz) + " entries, found "
                            + std::to_string(ja.size()));
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
// conversion. The entry count is taken from ia, so the header can never
// disagree with the body.
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

// ---------------------------------------------------------------------------
// Fields
// ---------------------------------------------------------------------------

// One value per line, with an "n" header: solutions, residuals, errors.
template <class T>
void write_field(const std::string& fname, std::size_t n, const T* v)
{
    auto out = detail::open_out(fname);
    detail::full_precision<T>(out);
    out << n << '\n';
    for (std::size_t i = 0; i < n; ++i) out << v[i] << '\n';
}

template <class T>
void write_field(const std::string& fname, const std::vector<T>& v)
{
    write_field(fname, v.size(), v.data());
}

// Inverse of write_field: reference solutions, restart data.
template <class T = double>
std::vector<T> read_field(const std::string& fname)
{
    auto in = detail::open_in(fname);
    std::size_t n = 0;
    if (!(in >> n)) detail::fail(fname, "missing or malformed value count");
    std::vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) in >> v[i];
    if (!in) detail::fail(fname, "expected " + std::to_string(n) + " values");
    return v;
}

} // namespace rbf::io

#endif // RBF_IO_H
