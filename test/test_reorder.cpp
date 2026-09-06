// Tests for rbf_reorder.h / rbf_ordering.F90 / NodeSet::renumber.
//
// Build and run via CMake, from the repository root:
//
//   cmake -B build
//   cmake --build build
//   ctest --test-dir build

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <span>
#include <vector>

#include "rbf_reorder.h"
#include "rbf_nodeset.h"

static int failures = 0;

#define CHECK(cond) \
    do { if (!(cond)) { \
        std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } } while (0)

// Reference Morton key: interleave the ndiv low bits of the cell
// coordinates, x in the even (low) positions.
static std::int64_t interleave(int ix, int iy, int ndiv) {
    std::int64_t z = 0;
    for (int b = ndiv - 1; b >= 0; --b)
        z = (z << 2) | (((iy >> b) & 1) << 1) | ((ix >> b) & 1);
    return z;
}

static void test_permutation() {
    using rbf::Permutation;

    auto id = Permutation<int>::identity(4);
    std::vector<double> a{10, 11, 12, 13};
    id.permute(std::span{a});
    CHECK(a == (std::vector<double>{10, 11, 12, 13}));

    // p places old item p[i] at new position i
    Permutation<int> p(std::vector<int>{2, 0, 3, 1});
    p.permute(std::span{a});
    CHECK(a == (std::vector<double>{12, 10, 13, 11}));
    p.unpermute(std::span{a});
    CHECK(a == (std::vector<double>{10, 11, 12, 13}));

    // inv() is the old -> new map
    for (int i = 0; i < 4; ++i)
        CHECK(p.inv()[p.map()[i]] == i);

    // then(): applying p then q in sequence equals applying p.then(q)
    Permutation<int> q(std::vector<int>{1, 3, 0, 2});
    std::vector<double> b1{10, 11, 12, 13}, b2 = b1;
    p.permute(std::span{b1});
    q.permute(std::span{b1});
    p.then(q).permute(std::span{b2});
    CHECK(b1 == b2);

    CHECK(p.then(p.inverse()).map()[0] == 0);
    CHECK(p.then(p.inverse()).map()[3] == 3);
}

// 4x4 grid of cell centres in bbox [0,4]x[0,4], row-major node order.
static void grid4(std::vector<double>& x, std::vector<double>& y) {
    x.clear(); y.clear();
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i) {
            x.push_back(i + 0.5);
            y.push_back(j + 0.5);
        }
}

static void test_morton_keys() {
    std::vector<double> x, y;
    grid4(x, y);
    const rbf::BBox2<double> bbox{0, 0, 4, 4};
    const auto keys = rbf::morton_keys(x, y, 2, bbox);
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            CHECK(keys[j*4 + i] == interleave(i, j, 2));
}

static void test_hilbert_keys() {
    std::vector<double> x, y;
    grid4(x, y);
    const rbf::BBox2<double> bbox{0, 0, 4, 4};
    const auto keys = rbf::hilbert_keys(x, y, 2, bbox);

    // Keys must be exactly 0..15, and walking them in order must visit
    // grid-adjacent cells (the defining property of the Hilbert curve).
    std::vector<int> cell_of_key(16, -1);
    for (int s = 0; s < 16; ++s) {
        CHECK(keys[s] >= 0 && keys[s] < 16);
        CHECK(cell_of_key[keys[s]] == -1);
        cell_of_key[keys[s]] = s;
    }
    for (int k = 1; k < 16; ++k) {
        const int a = cell_of_key[k-1], b = cell_of_key[k];
        const int dx = std::abs(a % 4 - b % 4);
        const int dy = std::abs(a / 4 - b / 4);
        CHECK(dx + dy == 1);
    }
}

static void test_boundary_last() {
    // flag and index-list variants must agree, and both must be stable
    const std::vector<int> flag{0, 1, 0, 1, 0, 0, 2};
    const std::vector<int> bnd{1, 3, 6};

    const auto pf = rbf::boundary_last_by_flag<int>(flag);
    const auto pi = rbf::boundary_last_by_index<int>(7, bnd);

    const std::vector<int> expect{0, 2, 4, 5, 1, 3, 6};
    for (int i = 0; i < 7; ++i) {
        CHECK(pf.map()[i] == expect[i]);
        CHECK(pi.map()[i] == expect[i]);
    }
}

static void test_renumber_stencils() {
    // 4 nodes, k = 2, self first
    std::vector<int> ja{0, 1,  1, 2,  2, 3,  3, 0};
    const rbf::Permutation<int> p(std::vector<int>{2, 0, 3, 1});

    rbf::renumber_stencils(std::span{ja}, 2, p);

    // self-first invariant survives
    for (int s = 0; s < 4; ++s)
        CHECK(ja[s*2] == s);
    // new node s (old p[s]) keeps its old neighbour, relabelled:
    // old rows were {i, (i+1)%4}; new second entry is inv[(p[s]+1)%4]
    for (int s = 0; s < 4; ++s)
        CHECK(ja[s*2 + 1] == p.inv()[(p.map()[s] + 1) % 4]);

    // general CSR path gives the same answer for the same graph
    std::vector<int> ja2{0, 1,  1, 2,  2, 3,  3, 0};
    auto ia = rbf::make_row_ptr<int>(4, 2);
    rbf::renumber_csr(std::span{ia}, std::span{ja2}, p);
    CHECK(ja2 == ja);
    CHECK(ia == (std::vector<int>{0, 2, 4, 6, 8}));
}

static void test_nodeset() {
    // 4x4 grid, edge nodes flagged as boundary
    {
        std::ofstream f("grid.nodes");
        for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i) {
                const int b = (i == 0 || i == 3 || j == 0 || j == 3);
                f << i + 0.5 << ' ' << j + 0.5 << ' ' << b << '\n';
            }
    }

    rbf::NodeSet<double> ns("grid.nodes");
    CHECK(ns.num_points() == 16);
    CHECK(ns.num_boundary() == 12);
    CHECK(ns.num_interior() == 4);

    const auto fx = ns.x, fy = ns.y; // file-order copies
    const auto fflag = ns.flag;

    ns.renumber(rbf::morton_order(ns.x, ns.y, 2, rbf::BBox2<double>{0, 0, 4, 4}))
      .renumber(rbf::boundary_last_by_flag(ns.flag));

    // boundary block sits at the end, bnd was rebuilt to match
    for (size_t i = 0; i < ns.num_interior(); ++i)
        CHECK(ns.flag[i] == 0);
    for (size_t i = ns.num_interior(); i < ns.num_points(); ++i)
        CHECK(ns.flag[i] != 0);
    CHECK(ns.bnd.size() == 12);
    CHECK(ns.bnd.front() == static_cast<int>(ns.num_interior()));

    // file_order() maps the chained renumbering back to file order
    const auto& fo = ns.file_order();
    for (size_t i = 0; i < ns.num_points(); ++i) {
        CHECK(ns.x[i] == fx[fo.map()[i]]);
        CHECK(ns.y[i] == fy[fo.map()[i]]);
        CHECK(ns.flag[i] == fflag[fo.map()[i]]);
    }

    // permute() carries file-order data into the current numbering
    auto u = fx;
    fo.permute(std::span{u});
    CHECK(u == ns.x);
    fo.unpermute(std::span{u});
    CHECK(u == fx);

    // tree is built here, once, in the final numbering
    const auto ja = ns.stencils(4);
    for (size_t s = 0; s < ns.num_points(); ++s)
        CHECK(ja[s*4] == static_cast<int>(s));

    std::remove("grid.nodes");
}

int main() {
    test_permutation();
    test_morton_keys();
    test_hilbert_keys();
    test_boundary_last();
    test_renumber_stencils();
    test_nodeset();

    if (failures == 0) {
        std::printf("all tests passed\n");
        return 0;
    }
    std::printf("%d check(s) failed\n", failures);
    return 1;
}
