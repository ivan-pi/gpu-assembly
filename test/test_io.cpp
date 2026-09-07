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
//
// Assisted-by: Claude Fable 5.1

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
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

static void test_read_nodes() {
    write_text("pts.nodes", "3\n0.1 0.2\n1.5 -2.5\n1e-300 3\n");

    auto [x, y] = rbf::io::read_nodes("pts.nodes");
    CHECK(x == (std::vector<double>{0.1, 1.5, 1e-300}));
    CHECK(y == (std::vector<double>{0.2, -2.5, 3.0}));

    // AoS into std::array, a struct of doubles, and a struct of floats
    auto a = rbf::io::read_nodes_aos<std::vector<std::array<double, 2>>>("pts.nodes");
    CHECK(a.size() == 3);
    CHECK(a[1][0] == 1.5 && a[1][1] == -2.5);

    auto p = rbf::io::read_nodes_aos<std::vector<Point>>("pts.nodes");
    CHECK(p.size() == 3);
    CHECK(p[2].x == 1e-300 && p[2].y == 3.0);

    auto pf = rbf::io::read_nodes_aos<std::vector<PointF>, float>("pts.nodes");
    CHECK(pf.size() == 3);
    CHECK(pf[1].x == 1.5f && pf[1].y == -2.5f);

    // the count header bounds the read: extra lines are ignored
    write_text("pts.nodes", "1\n7 8\n9 10\n");
    auto [x1, y1] = rbf::io::read_nodes("pts.nodes");
    CHECK(x1.size() == 1 && y1.size() == 1);
    CHECK(x1[0] == 7 && y1[0] == 8);

    std::remove("pts.nodes");
}

static void test_nodeset_roundtrip() {
    std::vector<double> x = awkward, y = awkward;
    for (auto& v : y) v = -v;
    const std::vector<int> flag{0, 1, 0, 2, 0, 1};

    rbf::io::write_nodeset("a.nodeset", x.size(), x.data(), y.data(), flag.data());
    std::vector<double> rx, ry; std::vector<int> rflag;
    CHECK(rbf::io::read_nodeset("a.nodeset", rx, ry, rflag) == x.size());
    CHECK(rx == x);
    CHECK(ry == y);
    CHECK(rflag == flag);

    // the output buffers are replaced, not appended to
    CHECK(rbf::io::read_nodeset("a.nodeset", rx, ry, rflag) == x.size());
    CHECK(rx.size() == x.size());

    // NodeSet reads the same file and agrees on the tags
    rbf::NodeSet<double> ns("a.nodeset");
    CHECK(ns.num_points() == 6);
    CHECK(ns.num_boundary() == 3);
    CHECK(ns.indices_with(2) == (std::vector<std::int32_t>{3}));
    CHECK(ns.x == x);

    // renumber, write, read back: the file carries the new numbering
    ns.renumber(rbf::boundary_last_by_flag(ns.flag));
    ns.write("c.nodeset");
    rbf::NodeSet<double> ns2("c.nodeset");
    CHECK(ns2.x == ns.x && ns2.y == ns.y && ns2.flag == ns.flag);
    CHECK(ns2.num_boundary() == 3 && ns2.bnd == ns.bnd);
    CHECK(ns2.flag[0] == 0 && ns2.flag[5] != 0);
    std::remove("c.nodeset");

    // null flag writes interior everywhere; a missing final newline is fine
    rbf::io::write_nodeset("a.nodeset", x.size(), x.data(), y.data());
    write_text("b.nodeset", "1 2 0\n3 4 1");
    CHECK(rbf::io::read_nodeset("a.nodeset", rx, ry, rflag) == x.size());
    CHECK(rflag == std::vector<int>(x.size(), 0));
    CHECK(rbf::io::read_nodeset("b.nodeset", rx, ry, rflag) == 2);
    CHECK(rx == (std::vector<double>{1, 3}) && rflag == (std::vector<int>{0, 1}));

    std::remove("a.nodeset");
    std::remove("b.nodeset");
}

static void test_read_graph_csr() {
    // ragged rows, extra whitespace, CRLF line ends, trailing newline
    write_text("g.graph", "3 6\r\n0 1 2\r\n  1 0 \r\n2 \r\n");
    auto [ia, ja] = rbf::io::read_graph_csr<std::int32_t>("g.graph");
    CHECK(ia == (std::vector<std::int32_t>{0, 3, 5, 6}));
    CHECK(ja == (std::vector<std::int32_t>{0, 1, 2, 1, 0, 2}));

    // 64-bit index type; no final newline; the largest valid index is n - 1
    write_text("g.graph", "2 3\n0 1\n1");
    auto [ia1, ja1] = rbf::io::read_graph_csr<std::int64_t>("g.graph");
    CHECK(ia1 == (std::vector<std::int64_t>{0, 2, 3}));
    CHECK(ja1 == (std::vector<std::int64_t>{0, 1, 1}));

    // fixed k: same file read both ways gives the same result
    write_text("g.graph", "3 6\n0 1\n1 2\n2 0\n");
    auto [ia2, ja2] = rbf::io::read_graph_csr<std::int32_t>("g.graph");
    auto [ia3, ja3] = rbf::io::read_graph_csr<std::int32_t>("g.graph", 2);
    CHECK(ia2 == (std::vector<std::int32_t>{0, 2, 4, 6}));
    CHECK(ia3 == ia2 && ja3 == ja2);

    std::remove("g.graph");
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

    // 64-bit indices with literal shape arguments
    const std::vector<std::int64_t> ia64(ia.begin(), ia.end()), ja64(ja.begin(), ja.end());
    rbf::io::write_matrix_market("c.mtx", 3, 4, ia64.data(), ja64.data(), a.data());
    auto m64 = read_mtx("c.mtx");
    CHECK(m64.i == m.i && m64.j == m.j && m64.v == m.v);
    rbf::io::write_matrix_market_pattern("c.mtx", 3, 4, ia64.data(), ja64.data());
    CHECK(read_mtx("c.mtx").pattern);

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

    std::remove("a.mtx"); std::remove("b.mtx"); std::remove("c.mtx");
    std::remove("p.mtx"); std::remove("f.mtx");
}

// Read a legacy POLYDATA file back: point count, vertex count, and the
// POINT_DATA blocks in order as (kind, name, values flattened).
struct VtkBlock { std::string kind, name; std::vector<double> v; };
struct VtkFile { std::size_t npoints = 0, nverts = 0; std::vector<double> xyz; std::vector<VtkBlock> blocks; };

static VtkFile read_vtk(const std::string& fname) {
    VtkFile f;
    std::ifstream in(fname);
    std::string tok;
    while (in >> tok) {
        if (tok == "POINTS") {
            std::string type; in >> f.npoints >> type;
            f.xyz.resize(3 * f.npoints);
            for (auto& v : f.xyz) in >> v;
        } else if (tok == "VERTICES") {
            std::size_t size; in >> f.nverts >> size;
            for (std::size_t i = 0; i < size; ++i) in >> tok;
        } else if (tok == "SCALARS" || tok == "VECTORS") {
            VtkBlock b; b.kind = tok;
            std::string type; in >> b.name >> type;
            if (b.kind == "SCALARS") { in >> tok >> tok >> tok; }   // "1", LOOKUP_TABLE, default
            b.v.resize((b.kind == "SCALARS" ? 1 : 3) * f.npoints);
            for (auto& v : b.v) in >> v;
            f.blocks.push_back(std::move(b));
        }
    }
    return f;
}

static bool same_file(const std::string& a, const std::string& b) {
    std::ifstream fa(a), fb(b);
    std::string sa((std::istreambuf_iterator<char>(fa)), {}), sb((std::istreambuf_iterator<char>(fb)), {});
    return !sa.empty() && sa == sb;
}

static void test_vtk_polydata() {
    const std::vector<double> x = awkward, y{1, 2, 3, 4, 5, 6};
    const std::vector<double> u{10, 11, 12, 13, 14, 15}, r{0.5, 0.25, 0.125, 1.0 / 3.0, 1e-300, 0};
    const std::vector<double> ux{7, 8, 9, 10, 11, 12}, uy{-1, -2, -3, -4, -5, -6};
    const std::size_t n = x.size();

    // two scalars and a vector, listed in place
    rbf::io::write_vtk_polydata("f.vtk", n, x.data(), y.data(),
                                {{"u", u.data()}, {"residual", r.data()}},
                                {{"U", ux.data(), uy.data()}});
    auto f = read_vtk("f.vtk");
    CHECK(f.npoints == n && f.nverts == n);
    for (std::size_t i = 0; i < n; ++i)
        CHECK(f.xyz[3*i] == x[i] && f.xyz[3*i + 1] == y[i] && f.xyz[3*i + 2] == 0);
    CHECK(f.blocks.size() == 3);
    CHECK(f.blocks[0].kind == "SCALARS" && f.blocks[0].name == "u" && f.blocks[0].v == u);
    CHECK(f.blocks[1].kind == "SCALARS" && f.blocks[1].name == "residual" && f.blocks[1].v == r);
    CHECK(f.blocks[2].kind == "VECTORS" && f.blocks[2].name == "U");
    for (std::size_t i = 0; i < n; ++i)
        CHECK(f.blocks[2].v[3*i] == ux[i] && f.blocks[2].v[3*i + 1] == uy[i]);

    // scalars only, from a runtime-built list; no vectors at all
    std::vector<rbf::io::VtkScalar<double>> fields{{"u", u.data()}, {"r", r.data()}};
    rbf::io::write_vtk_polydata("g.vtk", n, x.data(), y.data(), fields);
    auto g = read_vtk("g.vtk");
    CHECK(g.blocks.size() == 2 && g.blocks[1].name == "r" && g.blocks[1].v == r);

    // vectors only, and no fields at all
    rbf::io::write_vtk_polydata("h.vtk", n, x.data(), y.data(), {}, {{"U", ux.data(), uy.data()}});
    CHECK(read_vtk("h.vtk").blocks.size() == 1);
    rbf::io::write_vtk_polydata("i.vtk", n, x.data(), y.data(), {});
    CHECK(read_vtk("i.vtk").blocks.empty() && read_vtk("i.vtk").npoints == n);

    // the LBM wrappers: SoA and interleaved give byte-identical files
    std::vector<double> p(2 * n), vel(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        p[2*i] = x[i]; p[2*i + 1] = y[i];
        vel[2*i] = ux[i]; vel[2*i + 1] = uy[i];
    }
    rbf::io::write_lbm_vtk_polydata("lbm1.vtk", n, x.data(), y.data(), u.data(), ux.data(), uy.data());
    rbf::io::write_lbm_vtk_polydata("lbm2.vtk", n, p.data(), u.data(), vel.data());
    CHECK(same_file("lbm1.vtk", "lbm2.vtk"));
    auto l = read_vtk("lbm1.vtk");
    CHECK(l.blocks.size() == 2 && l.blocks[0].name == "Density" && l.blocks[1].name == "Velocity");
    CHECK(l.blocks[0].v == u);

    // float labels its arrays as float
    const std::vector<float> xf{0, 1}, yf{0, 1}, uf{0.1f, 1.0f / 3.0f};
    rbf::io::write_vtk_polydata("k.vtk", 2, xf.data(), yf.data(), {{"u", uf.data()}});
    {
        std::ifstream in("k.vtk"); std::string line; int hits = 0;
        while (std::getline(in, line))
            if (line == "POINTS 2 float" || line == "SCALARS u float 1") ++hits;
        CHECK(hits == 2);
    }

    for (const char* fn : {"f.vtk", "g.vtk", "h.vtk", "i.vtk", "lbm1.vtk", "lbm2.vtk", "k.vtk"})
        std::remove(fn);
}

int main() {
    test_read_nodes();
    test_nodeset_roundtrip();
    test_read_graph_csr();
    test_matrix_market();
    test_vtk_polydata();

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all io tests passed\n");
    return 0;
}
