// Round-trip tests for rbf_io.h and rbf_io_vtk.h.
//
// Build and run via CMake, from the repository root:
//
//   cmake -B build
//   cmake --build build
//   ctest --test-dir build
//
// Only the success paths are covered: the readers exit on a bad file, which
// cannot be observed from inside the process.

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "rbf_io.h"
#include "rbf_io_vtk.h"
#include "rbf_nodeset.h"

static int failures = 0;

#define CHECK(cond) \
    do { if (!(cond)) { \
        std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } } while (0)

// Values that do not survive the stream default of six digits.
static const std::vector<double> awkward{
    0.1, 1.0 / 3.0, 2.0 / 3.0, 1e-300, 123456.789012345, -9.87654321e-7};

static void write_text(const std::string& fname, const std::string& text) {
    std::ofstream f(fname);
    f << text;
}

struct Point { double x, y; };
struct PointF { float x, y; };

static void test_read_points() {
    write_text("pts.txt", "3\n0.1 0.2\n1.5 -2.5\n1e-300 3\n");

    auto [x, y] = rbf::io::read_points("pts.txt");
    CHECK(x == (std::vector<double>{0.1, 1.5, 1e-300}));
    CHECK(y == (std::vector<double>{0.2, -2.5, 3.0}));

    // AoS into std::array, a struct of doubles, and a struct of floats
    auto a = rbf::io::read_points_aos<std::vector<std::array<double, 2>>>("pts.txt");
    CHECK(a.size() == 3);
    CHECK(a[1][0] == 1.5 && a[1][1] == -2.5);

    auto p = rbf::io::read_points_aos<std::vector<Point>>("pts.txt");
    CHECK(p.size() == 3);
    CHECK(p[2].x == 1e-300 && p[2].y == 3.0);

    auto pf = rbf::io::read_points_aos<std::vector<PointF>, float>("pts.txt");
    CHECK(pf.size() == 3);
    CHECK(pf[1].x == 1.5f && pf[1].y == -2.5f);

    // the count header bounds the read: extra lines are ignored
    write_text("pts.txt", "1\n7 8\n9 10\n");
    auto [x1, y1] = rbf::io::read_points("pts.txt");
    CHECK(x1.size() == 1 && y1.size() == 1);
    CHECK(x1[0] == 7 && y1[0] == 8);

    std::remove("pts.txt");
}

static void test_nodes_roundtrip() {
    std::vector<double> x = awkward, y = awkward;
    for (auto& v : y) v = -v;
    const std::vector<int> flag{0, 1, 0, 2, 0, 1};

    rbf::io::write_nodes("nodes.txt", x.size(), x.data(), y.data(), flag.data());
    std::vector<double> rx, ry; std::vector<int> rflag;
    CHECK(rbf::io::read_nodes("nodes.txt", rx, ry, rflag) == x.size());
    CHECK(rx == x);
    CHECK(ry == y);
    CHECK(rflag == flag);

    // the output buffers are replaced, not appended to
    CHECK(rbf::io::read_nodes("nodes.txt", rx, ry, rflag) == x.size());
    CHECK(rx.size() == x.size());

    // NodeSet reads the same file and agrees on the tags
    rbf::NodeSet<double> ns("nodes.txt");
    CHECK(ns.num_points() == 6);
    CHECK(ns.num_boundary() == 3);
    CHECK(ns.indices_with(2) == (std::vector<std::int32_t>{3}));
    CHECK(ns.x == x);

    // null flag writes interior everywhere; a missing final newline is fine
    rbf::io::write_nodes("nodes.txt", x.size(), x.data(), y.data());
    write_text("nodes2.txt", "1 2 0\n3 4 1");
    CHECK(rbf::io::read_nodes("nodes.txt", rx, ry, rflag) == x.size());
    CHECK(rflag == std::vector<int>(x.size(), 0));
    CHECK(rbf::io::read_nodes("nodes2.txt", rx, ry, rflag) == 2);
    CHECK(rx == (std::vector<double>{1, 3}) && rflag == (std::vector<int>{0, 1}));

    std::remove("nodes.txt");
    std::remove("nodes2.txt");
}

static void test_read_graph_csr() {
    // ragged rows, extra whitespace, CRLF line ends, trailing newline
    write_text("graph.txt", "3 6\r\n0 1 2\r\n  1 0 \r\n2 \r\n");
    auto [ia, ja] = rbf::io::read_graph_csr<std::int32_t>("graph.txt");
    CHECK(ia == (std::vector<std::int32_t>{0, 3, 5, 6}));
    CHECK(ja == (std::vector<std::int32_t>{0, 1, 2, 1, 0, 2}));

    // indices are passed through unchanged (here 1-based); no final newline
    write_text("graph.txt", "2 3\n1 2\n2");
    auto [ia1, ja1] = rbf::io::read_graph_csr<std::int64_t>("graph.txt");
    CHECK(ia1 == (std::vector<std::int64_t>{0, 2, 3}));
    CHECK(ja1 == (std::vector<std::int64_t>{1, 2, 2}));

    // fixed k: same file read both ways gives the same result
    write_text("graph.txt", "3 6\n0 1\n1 2\n2 0\n");
    auto [ia2, ja2] = rbf::io::read_graph_csr<std::int32_t>("graph.txt");
    auto [ia3, ja3] = rbf::io::read_graph_csr<std::int32_t>("graph.txt", 2);
    CHECK(ia2 == (std::vector<std::int32_t>{0, 2, 4, 6}));
    CHECK(ia3 == ia2 && ja3 == ja2);

    // with k the body is a flat list: line breaks carry no meaning
    write_text("graph.txt", "3 6\n0 1 1 2\n2 0\n");
    auto [ia4, ja4] = rbf::io::read_graph_csr<std::int32_t>("graph.txt", 2);
    CHECK(ia4 == ia2 && ja4 == ja2);

    std::remove("graph.txt");
}

// Parse a Matrix Market coordinate file back into header + triplets.
struct MtxFile {
    std::string banner;
    long rows = 0, cols = 0, nnz = 0;
    std::vector<long> i, j;
    std::vector<double> v;
    bool pattern = false;
};

static MtxFile read_mtx(const std::string& fname) {
    MtxFile m;
    std::ifstream f(fname);
    std::getline(f, m.banner);
    m.pattern = m.banner.find("pattern") != std::string::npos;
    f >> m.rows >> m.cols >> m.nnz;
    for (long k = 0; k < m.nnz; ++k) {
        long i, j; double v = 0;
        f >> i >> j;
        if (!m.pattern) f >> v;
        m.i.push_back(i); m.j.push_back(j); m.v.push_back(v);
    }
    return m;
}

static void test_matrix_market() {
    // 3x4 rectangular CSR, 0-based
    const std::vector<int> ia{0, 2, 3, 6};
    const std::vector<int> ja{0, 3, 1, 0, 2, 3};
    const std::vector<double> a(awkward.begin(), awkward.begin() + 6);

    rbf::io::write_matrix_market("a.mtx", 3, 4, ia.data(), ja.data(), a.data());
    auto m = read_mtx("a.mtx");
    CHECK(m.banner == "%%MatrixMarket matrix coordinate real general");
    CHECK(m.rows == 3 && m.cols == 4 && m.nnz == 6);
    CHECK(m.i == (std::vector<long>{1, 1, 2, 3, 3, 3}));
    CHECK(m.j == (std::vector<long>{1, 4, 2, 1, 3, 4}));
    CHECK(m.v == a);   // exact: full precision was written

    // same matrix with 1-based ia/ja gives the same file
    std::vector<int> ia1 = ia, ja1 = ja;
    for (auto& v : ia1) ++v;
    for (auto& v : ja1) ++v;
    rbf::io::write_matrix_market("b.mtx", 3, 4, ia1.data(), ja1.data(), a.data(), 1);
    auto m1 = read_mtx("b.mtx");
    CHECK(m1.i == m.i && m1.j == m.j && m1.v == m.v);
    CHECK(m1.nnz == 6);

    // pattern: same structure, no values
    rbf::io::write_matrix_market_pattern("p.mtx", 3, 4, ia.data(), ja.data());
    auto mp = read_mtx("p.mtx");
    CHECK(mp.banner == "%%MatrixMarket matrix coordinate pattern general");
    CHECK(mp.pattern);
    CHECK(mp.rows == 3 && mp.cols == 4 && mp.nnz == 6);
    CHECK(mp.i == m.i && mp.j == m.j);

    // float values round-trip at float precision
    const std::vector<float> af{0.1f, 1.0f / 3.0f, 1e-30f, 123456.79f, -9.8765e-7f, 2.0f};
    rbf::io::write_matrix_market("f.mtx", 3, 4, ia.data(), ja.data(), af.data());
    {
        std::ifstream f("f.mtx");
        std::string banner; std::getline(f, banner);
        int r, c, nnz; f >> r >> c >> nnz;
        for (int k = 0; k < nnz; ++k) {
            int i, j; float v;
            f >> i >> j >> v;
            CHECK(v == af[k]);
        }
    }

    std::remove("a.mtx"); std::remove("b.mtx");
    std::remove("p.mtx"); std::remove("f.mtx");
}

static void test_vtk_compiles_and_writes() {
    // Checks the header is usable, the layout, and that the type label
    // follows T.
    const std::vector<double> x{0, 1}, y{0, 1}, rho{1, 2}, ux{3, 4}, uy{5, 6};
    rbf::io::write_lbm_vtk_polydata("out.vtk", 2, x.data(), y.data(),
                                 rho.data(), ux.data(), uy.data());
    std::ifstream f("out.vtk");
    std::string line;
    int points = 0, vectors = 0, scalars = 0;
    while (std::getline(f, line)) {
        if (line == "POINTS 2 double") ++points;
        if (line.rfind("VECTORS Velocity", 0) == 0) ++vectors;
        if (line.rfind("SCALARS Density", 0) == 0) ++scalars;
    }
    CHECK(points == 1 && vectors == 1 && scalars == 1);
    std::remove("out.vtk");
}

int main() {
    test_read_points();
    test_nodes_roundtrip();
    test_read_graph_csr();
    test_matrix_market();
    test_vtk_compiles_and_writes();

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all io tests passed\n");
    return 0;
}
