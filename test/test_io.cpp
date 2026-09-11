// Round-trip tests for rbf_io.h, rbf_io_vtk.h and rbf_io_gnuplot.h.
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

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

#include "rbf_io.h"
#include "rbf_io_gnuplot.h"
#include "rbf_io_vtk.h"
#include "rbf_nodeset.h"

#include "check.h"

// Values that do not survive the stream default of six digits.
static const std::vector<double> awkward{0.1,    1.0 / 3.0,        2.0 / 3.0,
                                         1e-300, 123456.789012345, -9.87654321e-7};
static const std::vector<float> awkward_f{0.1f, 1.0f / 3.0f, 1e-30f, 123456.79f, -9.8765e-7f, 2.0f};

static void write_text(const std::string& fname, const std::string& text) {
    std::ofstream f(fname);
    f << text;
}

static std::string first_line(const std::string& fname) {
    std::ifstream in(fname);
    std::string line;
    std::getline(in, line);
    return line;
}

static bool same_file(const std::string& a, const std::string& b) {
    std::ifstream fa(a), fb(b);
    std::string sa((std::istreambuf_iterator<char>(fa)), {}),
        sb((std::istreambuf_iterator<char>(fb)), {});
    return !sa.empty() && sa == sb;
}

struct Point {
    double x, y;
};
struct PointF {
    float x, y;
};

static void test_read_points() {
    write_text("pts.points", "3\n0.1 0.2\n1.5 -2.5\n1e-300 3\n");

    auto [x, y] = rbf::io::read_points("pts.points");
    CHECK(x == (std::vector<double>{0.1, 1.5, 1e-300}));
    CHECK(y == (std::vector<double>{0.2, -2.5, 3.0}));

    // AoS into std::array, a struct of doubles, and a struct of floats
    auto a = rbf::io::read_points_aos<std::vector<std::array<double, 2>>>("pts.points");
    CHECK(a.size() == 3);
    CHECK(a[1][0] == 1.5 && a[1][1] == -2.5);

    auto p = rbf::io::read_points_aos<std::vector<Point>>("pts.points");
    CHECK(p.size() == 3);
    CHECK(p[2].x == 1e-300 && p[2].y == 3.0);

    auto pf = rbf::io::read_points_aos<std::vector<PointF>, float>("pts.points");
    CHECK(pf.size() == 3);
    CHECK(pf[1].x == 1.5f && pf[1].y == -2.5f);

    // the count header bounds the read: extra lines are ignored
    write_text("pts.points", "1\n7 8\n9 10\n");
    auto [x1, y1] = rbf::io::read_points("pts.points");
    CHECK(x1.size() == 1 && y1.size() == 1);
    CHECK(x1[0] == 7 && y1[0] == 8);

    std::remove("pts.points");
}

static void test_node_file() {
    std::vector<double> x = awkward, y = awkward;
    for (auto& v : y)
        v = -v;
    const std::vector<int> marker{0, 1, 0, 2, 0, 1};
    const std::vector<double> attr{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};  // 6 x 2

    // markers and two attributes round-trip at full precision
    rbf::io::write_nodes("a.node", x.size(), x.data(), y.data(), marker.data(), 2, attr.data());
    std::vector<double> rx, ry, rattr;
    std::vector<int> rm;
    CHECK(rbf::io::read_nodes("a.node", rx, ry, rm, &rattr) == x.size());
    CHECK(rx == x && ry == y && rm == marker && rattr == attr);

    // attributes are skipped when not asked for; buffers are replaced, not appended
    CHECK(rbf::io::read_nodes("a.node", rx, ry, rm) == x.size());
    CHECK(rx == x && rm == marker);

    // no marker column: every node interior
    rbf::io::write_nodes("a.node", x.size(), x.data(), y.data());
    CHECK(rbf::io::read_nodes("a.node", rx, ry, rm) == x.size());
    CHECK(rm == std::vector<int>(x.size(), 0) && rx == x);

    // Triangle conventions: comments, blank lines, 1-based numbering, CRLF
    write_text("b.node",
               "# node file\r\n\r\n3 2 0 1   # header\r\n"
               "1 0.5 1.5 1\r\n\r\n2 2.5 3.5 0 # interior\r\n3 4.5 5.5 7\r\n\r\n# end\r\n");
    CHECK(rbf::io::read_nodes("b.node", rx, ry, rm) == 3);
    CHECK(rx == (std::vector<double>{0.5, 2.5, 4.5}));
    CHECK(ry == (std::vector<double>{1.5, 3.5, 5.5}));
    CHECK(rm == (std::vector<int>{1, 0, 7}));

    // NodeSet reads it, renumbers, writes, and reads itself back
    rbf::io::write_nodes("a.node", x.size(), x.data(), y.data(), marker.data());
    rbf::NodeSet<double> ns("a.node");
    CHECK(ns.num_points() == 6 && ns.num_boundary() == 3);
    CHECK(ns.indices_with(2) == (std::vector<std::int32_t>{3}));
    CHECK(ns.x == x);
    ns.renumber(rbf::boundary_last_by_flag(ns.flag));
    ns.write("c.node");
    rbf::NodeSet<double> ns2("c.node");
    CHECK(ns2.x == ns.x && ns2.y == ns.y && ns2.flag == ns.flag);
    CHECK(ns2.num_boundary() == 3 && ns2.bnd == ns.bnd);
    CHECK(ns2.flag[0] == 0 && ns2.flag[5] != 0);

    // float coordinates round-trip too
    rbf::io::write_nodes("f.node", awkward_f.size(), awkward_f.data(), awkward_f.data());
    std::vector<float> fx, fy;
    std::vector<int> fm;
    CHECK(rbf::io::read_nodes("f.node", fx, fy, fm) == awkward_f.size());
    CHECK(fx == awkward_f && fy == awkward_f);

    for (const char* fn : {"a.node", "b.node", "c.node", "f.node"})
        std::remove(fn);
}

static void test_grid_file() {
    // the 9-node example of the format reference: a square domain with a
    // square hole, 5 triangles, 2 quads, the hole (B1) and the outer
    // boundary (B2) as closed loops. The counts share the header line and
    // the boundary node counts come before the lists.
    write_text("s.grid",
               "9 5 2\n"
               "0.0 0.0\n2.0 2.0\n0.0 3.0\n2.0 1.0\n2.0 0.0\n3.0 3.0\n1.3 1.0\n3.0 0.0\n"
               "1.3 2.0\n"
               "9 2 3\n5 8 4\n2 6 3\n1 7 9\n1 9 3\n"
               "1 5 4 7\n4 8 6 2\n"
               "2\n5\n6\n4\n7\n9\n2\n4\n1\n5\n8\n6\n3\n1\n");

    // native 1-based indices, as in the file
    auto n1 = rbf::io::read_grid("s.grid", false);
    CHECK(n1.num_nodes() == 9 && n1.num_triangles() == 5 && n1.num_quads() == 2 &&
          n1.num_boundaries() == 2);
    CHECK(!n1.zero_based);
    CHECK(n1.x[6] == 1.3 && n1.y[6] == 1.0);  // node 7 of the file
    CHECK(n1.tri == (std::vector<std::int32_t>{9, 2, 3, 5, 8, 4, 2, 6, 3, 1, 7, 9, 1, 9, 3}));
    CHECK(n1.quad == (std::vector<std::int32_t>{1, 5, 4, 7, 4, 8, 6, 2}));
    CHECK(n1.bound[0] == (std::vector<std::int32_t>{4, 7, 9, 2, 4}));  // 1st repeated: closed
    CHECK(n1.bound[1] == (std::vector<std::int32_t>{1, 5, 8, 6, 3, 1}));
    CHECK(n1.closed(0) && n1.closed(1));

    // 0-based on request, the numbering of the graph file
    auto g = rbf::io::read_grid("s.grid", true);
    CHECK(g.zero_based);
    CHECK(g.tri == (std::vector<std::int32_t>{8, 1, 2, 4, 7, 3, 1, 5, 2, 0, 6, 8, 0, 8, 2}));
    CHECK(g.quad == (std::vector<std::int32_t>{0, 4, 3, 6, 3, 7, 5, 1}));
    CHECK(g.bound[0] == (std::vector<std::int32_t>{3, 6, 8, 1, 3}));
    CHECK(g.bound[1] == (std::vector<std::int32_t>{0, 4, 7, 5, 2, 0}));
    CHECK(g.closed(0) && g.closed(1));

    // markers: the first part listing a node names it, in either base;
    // every node here is on the boundary
    const std::vector<int> expected_markers{2, 1, 2, 1, 2, 2, 1, 2, 1};
    CHECK(g.markers() == expected_markers);
    CHECK(n1.markers() == expected_markers);

    // the example follows the reference's orientation conventions:
    // elements counterclockwise, every part edge a mesh-boundary element
    // edge with the domain on its left, and the whole boundary walked
    CHECK(g.orientation_report().empty());
    CHECK(n1.orientation_report().empty());

    // blank lines between the sections (or anywhere else) change nothing
    write_text("b.grid",
               "9 5 2\n"
               "0.0 0.0\n2.0 2.0\n0.0 3.0\n2.0 1.0\n2.0 0.0\n3.0 3.0\n1.3 1.0\n3.0 0.0\n"
               "1.3 2.0\n"
               "\n9 2 3\n5 8 4\n2 6 3\n1 7 9\n1 9 3\n"
               "\n1 5 4 7\n4 8 6 2\n"
               "\n2\n\n5\n6\n\n4\n7\n9\n2\n4\n\n1\n5\n8\n6\n3\n1\n\n");
    auto gb = rbf::io::read_grid("b.grid", true);
    CHECK(gb.x == g.x && gb.y == g.y && gb.tri == g.tri && gb.quad == g.quad &&
          gb.bound == g.bound);

    // a NodeSet directly from the Grid: markers become the flag, tri and
    // quad are dropped. Copied from an lvalue, moved from an rvalue, and
    // the same node set from either base.
    rbf::NodeSet<double> ns(g);  // copy: g stays intact
    CHECK(!g.x.empty() && ns.num_points() == 9 && ns.num_boundary() == 9);
    CHECK(ns.x == g.x && ns.y == g.y && ns.flag == expected_markers);
    CHECK(ns.indices_with(1) == (std::vector<std::int32_t>{1, 3, 6, 8}));
    rbf::NodeSet<double> nm(std::move(n1));  // move: the coordinates leave n1
    CHECK(nm.x == ns.x && nm.y == ns.y && nm.flag == ns.flag);
    CHECK(n1.x.empty());

    // round trip at full precision in both bases; the written file does
    // not depend on the base
    for (std::size_t i = 0; i < awkward.size(); ++i) {
        g.x[i] = awkward[i];
        g.y[i] = -awkward[i];
    }
    rbf::io::write_grid("t.grid", g);
    auto r = rbf::io::read_grid("t.grid", true);
    CHECK(r.x == g.x && r.y == g.y && r.tri == g.tri && r.quad == g.quad && r.bound == g.bound);
    auto r1 = rbf::io::read_grid("t.grid", false);
    rbf::io::write_grid("t1.grid", r1);
    CHECK(same_file("t.grid", "t1.grid"));

    // float coordinates and 64-bit indices read the same file
    auto gf = rbf::io::read_grid<float, std::int64_t>("s.grid", true);
    CHECK(gf.x[6] == 1.3f && gf.tri.size() == 15 && gf.tri[0] == 8);

    // triangles only, no quads and no boundary; every node interior
    write_text("u.grid", "3 1 0\n0 0\n1 0\n0 1\n1 2 3\n0\n");
    auto u = rbf::io::read_grid("u.grid", false);
    CHECK(u.num_triangles() == 1 && u.num_quads() == 0 && u.num_boundaries() == 0);
    CHECK(u.markers() == std::vector<int>(3, 0));

    // open parts sharing an endpoint: the first part keeps the shared node
    write_text("v.grid", "3 1 0\n0 0\n1 0\n0 1\n1 2 3\n2\n2\n2\n1\n2\n2\n3\n");
    auto v = rbf::io::read_grid("v.grid", true);
    CHECK(v.bound[0] == (std::vector<std::int32_t>{0, 1}));
    CHECK(v.bound[1] == (std::vector<std::int32_t>{1, 2}));
    CHECK(!v.closed(0) && !v.closed(1));
    CHECK(v.markers() == (std::vector<int>{1, 1, 2}));

    for (const char* fn : {"s.grid", "b.grid", "t.grid", "t1.grid", "u.grid", "v.grid"})
        std::remove(fn);
}

static std::size_t msgs_with(const std::vector<std::string>& report, const std::string& what) {
    std::size_t n = 0;
    for (const auto& m : report)
        if (m.find(what) != std::string::npos) ++n;
    return n;
}

static void test_grid_orientation() {
    // one counterclockwise triangle, its boundary walked with the domain
    // on the left: nothing to report
    write_text("o.grid", "3 1 0\n0 0\n1 0\n0 1\n1 2 3\n1\n4\n1\n2\n3\n1\n");
    CHECK(rbf::io::read_grid("o.grid", true).orientation_report().empty());
    CHECK(rbf::io::read_grid("o.grid", false).orientation_report().empty());

    // the same triangle numbered clockwise: negative area, and every part
    // edge now runs against the element's edges
    write_text("o.grid", "3 1 0\n0 0\n1 0\n0 1\n1 3 2\n1\n4\n1\n2\n3\n1\n");
    auto r = rbf::io::read_grid("o.grid", true).orientation_report();
    CHECK(msgs_with(r, "triangle 1 is not counterclockwise") == 1);
    CHECK(msgs_with(r, "domain on the right") == 3);
    CHECK(msgs_with(r, "not walked by any boundary part") == 3);

    // a counterclockwise triangle whose boundary is walked backwards
    write_text("o.grid", "3 1 0\n0 0\n1 0\n0 1\n1 2 3\n1\n4\n1\n3\n2\n1\n");
    r = rbf::io::read_grid("o.grid", false).orientation_report();
    CHECK(msgs_with(r, "not counterclockwise") == 0);
    CHECK(msgs_with(r, "domain on the right") == 3);
    CHECK(msgs_with(r, "not walked by any boundary part") == 3);

    // two triangles of the unit square: a part along the shared diagonal
    // is an interior edge, and the real boundary goes unwalked
    write_text("o.grid", "4 2 0\n0 0\n1 0\n1 1\n0 1\n1 2 3\n1 3 4\n1\n2\n1\n3\n");
    r = rbf::io::read_grid("o.grid", true).orientation_report();
    CHECK(msgs_with(r, "boundary part 1, edge 1 -> 3 is an interior edge") == 1);
    CHECK(msgs_with(r, "not walked by any boundary part") == 4);

    // a part edge between nodes no element connects
    write_text("o.grid", "4 2 0\n0 0\n1 0\n1 1\n0 1\n1 2 3\n1 3 4\n1\n2\n2\n4\n");
    r = rbf::io::read_grid("o.grid", true).orientation_report();
    CHECK(msgs_with(r, "edge 2 -> 4 is not an element edge") == 1);

    std::remove("o.grid");
}

static void test_bc_files() {
    // .bcmap, the EDU2D example of the grid-file reference
    write_text("a.bcmap",
               "! Boundary tag  BC name\n1 freestream\n2 subsonic_outflow\n3 viscous_wall\n");
    auto b = rbf::io::read_bcmap("a.bcmap");
    CHECK(b.size() == 3);
    CHECK(b[0].tag == 1 && b[0].name == "freestream" && b[0].bc == 0);
    CHECK(b[1].tag == 2 && b[1].name == "subsonic_outflow");
    CHECK(b[2].tag == 3 && b[2].name == "viscous_wall");

    // .mapbc, the sample of the FUN3D manual: count, then tag, BC number
    // and family name
    write_text("a.mapbc",
               "13\n1 6662 box_ymin\n2 5025 box_zmax\n3 5050 box_xmin\n4 5025 box_ymax\n"
               "5 5025 box_zmin\n6 5025 box_xmax\n7 3000 wing_upper\n8 3000 wing_lower\n"
               "9 3000 wing_upper\n10 3000 wing_upper\n11 3000 wing_lower\n12 3000 wing_lower\n"
               "13 3000 wing_tip\n");
    auto m = rbf::io::read_mapbc("a.mapbc");
    CHECK(m.size() == 13);
    CHECK(m[0].tag == 1 && m[0].bc == 6662 && m[0].name == "box_ymin");
    CHECK(m[6].tag == 7 && m[6].bc == 3000 && m[6].name == "wing_upper");
    CHECK(m[12].tag == 13 && m[12].bc == 3000 && m[12].name == "wing_tip");

    // family names are optional; a commented header and blank lines are
    // skipped, as in the grid-file reference's variant
    write_text("b.mapbc", "! Boundary tag  BC #\n3\n\n1 5050\n2 5051 outflow\n3 4000\n");
    auto s = rbf::io::read_mapbc("b.mapbc");
    CHECK(s.size() == 3);
    CHECK(s[0].tag == 1 && s[0].bc == 5050 && s[0].name.empty());
    CHECK(s[1].tag == 2 && s[1].bc == 5051 && s[1].name == "outflow");
    CHECK(s[2].tag == 3 && s[2].bc == 4000 && s[2].name.empty());

    for (const char* fn : {"a.bcmap", "a.mapbc", "b.mapbc"})
        std::remove(fn);
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

static void test_ordering() {
    using rbf::Permutation;
    // p places old item p[i] at new position i; the file stores old -> new
    const Permutation<int> p(std::vector<int>{2, 0, 3, 1});
    p.write("o.iperm");
    CHECK(rbf::io::read_ordering<int>("o.iperm") == (std::vector<int>{1, 3, 0, 2}));
    const auto q = Permutation<int>::read("o.iperm");
    CHECK(std::equal(q.map().begin(), q.map().end(), p.map().begin()));
    CHECK(std::equal(q.inv().begin(), q.inv().end(), p.inv().begin()));

    // the raw functions and a 64-bit read of the same file
    rbf::io::write_ordering("o.iperm", p.size(), p.inv().data());
    CHECK(rbf::io::read_ordering<std::int64_t>("o.iperm") ==
          (std::vector<std::int64_t>{1, 3, 0, 2}));

    // NodeSet: the file maps file order to the current numbering
    write_text("d.node", "4 2 0 1\n0 0 0 1\n1 1 0 0\n2 2 0 1\n3 3 0 0\n");
    rbf::NodeSet<double> ns("d.node");
    ns.renumber(rbf::boundary_last_by_flag(ns.flag));
    ns.file_order().write("d.iperm");
    const auto fo = Permutation<int>::read("d.iperm");
    std::vector<double> u{0, 1, 2, 3};  // per-node data in file order: x
    fo.permute(std::span{u});
    CHECK(u == ns.x);

    for (const char* fn : {"o.iperm", "d.node", "d.iperm"})
        std::remove(fn);
}

// Parse a Matrix Market coordinate file back into header + triplets,
// reading the values as V so float files compare exactly.
template <class V = double>
struct MtxFile {
    std::string banner;
    long rows = 0, cols = 0, nnz = 0;
    std::vector<long> i, j;
    std::vector<V> v;
    bool pattern = false;
};

template <class V = double>
static MtxFile<V> read_mtx(const std::string& fname) {
    MtxFile<V> m;
    std::ifstream f(fname);
    std::getline(f, m.banner);
    m.pattern = m.banner.find("pattern") != std::string::npos;
    f >> m.rows >> m.cols >> m.nnz;
    for (long k = 0; k < m.nnz; ++k) {
        long i, j;
        V v = 0;
        f >> i >> j;
        if (!m.pattern) f >> v;
        m.i.push_back(i);
        m.j.push_back(j);
        m.v.push_back(v);
    }
    return m;
}

static void test_matrix_market() {
    // 3x4 rectangular CSR, 0-based
    const std::vector<int> ia{0, 2, 3, 6};
    const std::vector<int> ja{0, 3, 1, 0, 2, 3};
    const auto& a = awkward;

    rbf::io::write_matrix_market("a.mtx", 3, 4, ia.data(), ja.data(), a.data());
    auto m = read_mtx("a.mtx");
    CHECK(m.banner == "%%MatrixMarket matrix coordinate real general");
    CHECK(m.rows == 3 && m.cols == 4 && m.nnz == 6);
    CHECK(m.i == (std::vector<long>{1, 1, 2, 3, 3, 3}));
    CHECK(m.j == (std::vector<long>{1, 4, 2, 1, 3, 4}));
    CHECK(m.v == a);  // exact: full precision was written

    // same matrix with 1-based ia/ja gives the same file
    std::vector<int> ia1 = ia, ja1 = ja;
    for (auto& v : ia1)
        ++v;
    for (auto& v : ja1)
        ++v;
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
    rbf::io::write_matrix_market("f.mtx", 3, 4, ia.data(), ja.data(), awkward_f.data());
    CHECK(read_mtx<float>("f.mtx").v == awkward_f);

    std::remove("a.mtx");
    std::remove("b.mtx");
    std::remove("c.mtx");
    std::remove("p.mtx");
    std::remove("f.mtx");
}

// Read a legacy POLYDATA file back: point count, vertex count, and the
// POINT_DATA blocks in order as (kind, name, values flattened).
struct VtkBlock {
    std::string kind, name, type;
    std::vector<double> v;
};
struct VtkFile {
    std::size_t npoints = 0, nverts = 0;
    std::string type;
    std::vector<double> xyz;
    std::vector<VtkBlock> blocks;
};

static VtkFile read_vtk(const std::string& fname) {
    VtkFile f;
    std::ifstream in(fname);
    std::string tok;
    while (in >> tok) {
        if (tok == "POINTS") {
            in >> f.npoints >> f.type;
            f.xyz.resize(3 * f.npoints);
            for (auto& v : f.xyz)
                in >> v;
        } else if (tok == "VERTICES") {
            std::size_t size;
            in >> f.nverts >> size;
            for (std::size_t i = 0; i < size; ++i)
                in >> tok;
        } else if (tok == "SCALARS" || tok == "VECTORS") {
            VtkBlock b;
            b.kind = tok;
            in >> b.name >> b.type;
            if (b.kind == "SCALARS") {
                in >> tok >> tok >> tok;
            }  // "1", LOOKUP_TABLE, default
            b.v.resize((b.kind == "SCALARS" ? 1 : 3) * f.npoints);
            for (auto& v : b.v)
                in >> v;
            f.blocks.push_back(std::move(b));
        }
    }
    return f;
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
        CHECK(f.xyz[3 * i] == x[i] && f.xyz[3 * i + 1] == y[i] && f.xyz[3 * i + 2] == 0);
    CHECK(f.blocks.size() == 3);
    CHECK(f.blocks[0].kind == "SCALARS" && f.blocks[0].name == "u" && f.blocks[0].v == u);
    CHECK(f.blocks[1].kind == "SCALARS" && f.blocks[1].name == "residual" && f.blocks[1].v == r);
    CHECK(f.blocks[2].kind == "VECTORS" && f.blocks[2].name == "U");
    for (std::size_t i = 0; i < n; ++i)
        CHECK(f.blocks[2].v[3 * i] == ux[i] && f.blocks[2].v[3 * i + 1] == uy[i]);

    // scalars from a runtime-built list, vectors listed in place
    std::vector<rbf::io::Column<double>> fields{{"u", u.data()}, {"r", r.data()}};
    rbf::io::write_vtk_polydata("g.vtk", n, x.data(), y.data(), fields,
                                {{"U", ux.data(), uy.data()}});
    auto g = read_vtk("g.vtk");
    CHECK(g.blocks.size() == 3 && g.blocks[1].name == "r" && g.blocks[1].v == r);
    CHECK(g.blocks[2].kind == "VECTORS");

    // vectors only, and no fields at all
    rbf::io::write_vtk_polydata("h.vtk", n, x.data(), y.data(), {}, {{"U", ux.data(), uy.data()}});
    CHECK(read_vtk("h.vtk").blocks.size() == 1);
    rbf::io::write_vtk_polydata("i.vtk", n, x.data(), y.data(), {});
    const auto i = read_vtk("i.vtk");
    CHECK(i.blocks.empty() && i.npoints == n);

    // interleaved points and vectors through the strides give the same file
    std::vector<double> p(2 * n), vel(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        p[2 * i] = x[i];
        p[2 * i + 1] = y[i];
        vel[2 * i] = ux[i];
        vel[2 * i + 1] = uy[i];
    }
    rbf::io::write_vtk_polydata("s1.vtk", n, x.data(), y.data(), {{"u", u.data()}},
                                {{"U", ux.data(), uy.data()}});
    rbf::io::write_vtk_polydata("s2.vtk", n, p.data(), p.data() + 1, {{"u", u.data()}},
                                {{"U", vel.data(), vel.data() + 1, 2}}, 2);
    CHECK(same_file("s1.vtk", "s2.vtk"));

    // the title line is the second line; it may be given, or empty
    rbf::io::write_vtk_polydata("t.vtk", n, x.data(), y.data(), {}, {}, 1, "case 7, step 100");
    {
        std::ifstream in("t.vtk");
        std::string l1, l2, l3;
        std::getline(in, l1);
        std::getline(in, l2);
        std::getline(in, l3);
        CHECK(l2 == "case 7, step 100" && l3 == "ASCII");
    }
    rbf::io::write_vtk_polydata("t.vtk", n, x.data(), y.data(), {}, {}, 1, "");
    {
        std::ifstream in("t.vtk");
        std::string l1, l2, l3;
        std::getline(in, l1);
        std::getline(in, l2);
        std::getline(in, l3);
        CHECK(l2.empty() && l3 == "ASCII");
    }
    CHECK(read_vtk("t.vtk").npoints == n);

    // float labels its arrays as float and round-trips at float precision
    rbf::io::write_vtk_polydata("k.vtk", awkward_f.size(), awkward_f.data(), awkward_f.data(),
                                {{"u", awkward_f.data()}});
    const auto k = read_vtk("k.vtk");
    CHECK(k.type == "float" && k.blocks.size() == 1 && k.blocks[0].type == "float");
    for (std::size_t q = 0; q < awkward_f.size(); ++q)
        CHECK(static_cast<float>(k.blocks[0].v[q]) == awkward_f[q]);

    for (const char* fn :
         {"f.vtk", "g.vtk", "h.vtk", "i.vtk", "s1.vtk", "s2.vtk", "t.vtk", "k.vtk"})
        std::remove(fn);
}

static void test_gnuplot_columns() {
    const std::vector<double> x = awkward, y{1, 2, 3, 4, 5, 6};
    const std::vector<double> rho{10, 11, 12, 13, 14, 15},
        rc{0.5, 0.25, 0.125, 1.0 / 3.0, 1e-300, 0};
    const std::size_t n = x.size();

    rbf::io::write_columns("m.dat", n, x.data(), y.data(),
                           {{"rho", rho.data()}, {"rcond", rc.data()}});
    CHECK(first_line("m.dat") == "# x y rho rcond");
    {
        std::ifstream in("m.dat");
        std::string header;
        std::getline(in, header);
        for (std::size_t i = 0; i < n; ++i) {
            double a, b, c, d;
            in >> a >> b >> c >> d;
            CHECK(a == x[i] && b == y[i] && c == rho[i] && d == rc[i]);
        }
        std::string rest;
        in >> rest;
        CHECK(rest.empty() && in.eof());
    }

    // no columns beyond the coordinates; interleaved points through the stride
    std::vector<double> p(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        p[2 * i] = x[i];
        p[2 * i + 1] = y[i];
    }
    rbf::io::write_columns("a.dat", n, x.data(), y.data(), {});
    rbf::io::write_columns("b.dat", n, p.data(), p.data() + 1, {}, 2);
    CHECK(same_file("a.dat", "b.dat"));
    CHECK(first_line("a.dat") == "# x y");

    // a runtime-built column list
    std::vector<rbf::io::Column<double>> cols{{"rho", rho.data()}};
    rbf::io::write_columns("c.dat", n, x.data(), y.data(), cols);
    CHECK(first_line("c.dat") == "# x y rho");

    for (const char* fn : {"m.dat", "a.dat", "b.dat", "c.dat"})
        std::remove(fn);
}

int main() {
    test_read_points();
    test_node_file();
    test_grid_file();
    test_grid_orientation();
    test_bc_files();
    test_read_graph_csr();
    test_ordering();
    test_matrix_market();
    test_vtk_polydata();
    test_gnuplot_columns();

    return report("io");
}
