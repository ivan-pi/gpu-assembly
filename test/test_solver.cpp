// Tests for rbf_solver.h: the C interface over Eigen's iterative
// solvers, on the five-point Laplacian of a square grid with Dirichlet
// boundary, which is symmetric positive definite, and on the same with
// an upwind convection term, which is not. The right-hand side is
// manufactured from a known solution, so every method and
// preconditioner is checked against it; the matrix-free path runs the
// same systems through a callback and must land on the same answers.
//
// Build and run via CMake, from the repository root:
//
//   cmake -B build -DGPU_ASSEMBLY_ENABLE_SOLVER=ON
//   cmake --build build
//   ctest --test-dir build

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

#include "rbf_solver.h"

#include "check.h"

// A sparse matrix in CSR, 0-based, as the solver takes it.
struct Csr {
    int n = 0;
    std::vector<int> ia, ja;
    std::vector<double> val;

    int nnz() const { return static_cast<int>(ja.size()); }

    // y += alpha * A * x
    void apply(double alpha, const double* x, double* y) const {
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int p = ia[i]; p < ia[i + 1]; ++p)
                s += val[p] * x[ja[p]];
            y[i] += alpha * s;
        }
    }
};

// The five-point Laplacian on an m-by-m grid, -u_xx - u_yy discretized
// with the grid spacing folded in, plus an upwind discretization of
// c u_x when c is nonzero. Diagonally dominant either way.
static Csr laplacian(int m, double c = 0) {
    Csr A;
    A.n = m * m;
    A.ia.push_back(0);
    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            const int row = i + j * m;
            // sorted by column, which is not required but keeps the
            // matrix in the shape the assembly produces
            if (j > 0) {
                A.ja.push_back(row - m);
                A.val.push_back(-1);
            }
            if (i > 0) {
                A.ja.push_back(row - 1);
                A.val.push_back(-1 - c);
            }
            A.ja.push_back(row);
            A.val.push_back(4 + c);
            if (i < m - 1) {
                A.ja.push_back(row + 1);
                A.val.push_back(-1);
            }
            if (j < m - 1) {
                A.ja.push_back(row + m);
                A.val.push_back(-1);
            }
            A.ia.push_back(A.nnz());
        }
    }
    return A;
}

// A smooth field on the grid, the solution the right-hand side is made from.
static std::vector<double> exact(int m) {
    std::vector<double> u(static_cast<std::size_t>(m) * m);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < m; ++i)
            u[i + j * m] = std::sin(0.3 * i) * std::cos(0.2 * j) + 0.1 * i;
    return u;
}

static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        d = std::fmax(d, std::fabs(a[i] - b[i]));
    return d;
}

static void callback(int nr, int nc, double alpha, const double* x, double* y, void* data) {
    const auto& A = *static_cast<const Csr*>(data);
    CHECK(nr == A.n && nc == A.n);
    A.apply(alpha, x, y);
}

// Every method and preconditioner on one matrix, against its known
// solution, from a zero initial guess.
static void test_csr(const Csr& A, const std::vector<double>& u, solver_method method) {
    std::vector<double> b(u.size(), 0);
    A.apply(1, u.data(), b.data());

    for (const auto precond : {PRECOND_NONE, PRECOND_JACOBI, PRECOND_ILU}) {
        std::vector<double> x(u.size(), 0);
        double err = -1;
        int iter = -1;
        const double tol = 1e-12;
        const int status =
            rbf_solve_csr_dp(A.n, A.nnz(), A.val.data(), A.ia.data(), A.ja.data(), b.data(),
                             x.data(), &err, &iter, &method, &precond, nullptr, &tol);
        CHECK(status == SOLVER_SUCCESS);
        CHECK(err >= 0 && err <= tol);
        CHECK(iter >= 0 && iter <= 2 * A.n);
        CHECK(max_abs_diff(x, u) < 1e-9);
    }
}

// The matrix-free solver on the same system, unpreconditioned.
static void test_matrix_free(const Csr& A, const std::vector<double>& u, solver_method method) {
    std::vector<double> b(u.size(), 0);
    A.apply(1, u.data(), b.data());

    std::vector<double> x(u.size(), 0);
    double err = -1;
    int iter = -1;
    const double tol = 1e-12;
    const int status = rbf_solve_mf_dp(A.n, A.n, callback, const_cast<Csr*>(&A), b.data(), x.data(),
                                       &err, &iter, &method, nullptr, &tol);
    CHECK(status == SOLVER_SUCCESS);
    CHECK(err >= 0 && err <= tol);
    CHECK(iter >= 0 && iter <= 2 * A.n);
    CHECK(max_abs_diff(x, u) < 1e-9);
}

static void test_defaults_and_guess() {
    const int m = 12;
    const Csr A = laplacian(m);
    const std::vector<double> u = exact(m);
    std::vector<double> b(u.size(), 0);
    A.apply(1, u.data(), b.data());

    // every optional argument left out: BiCGSTAB, Jacobi, 2n
    // iterations, epsilon
    std::vector<double> x(u.size(), 0);
    CHECK(rbf_solve_csr_dp(A.n, A.nnz(), A.val.data(), A.ia.data(), A.ja.data(), b.data(), x.data(),
                           nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) == SOLVER_SUCCESS);
    CHECK(max_abs_diff(x, u) < 1e-9);

    // the exact solution as initial guess needs no iteration at all
    x = u;
    int iter = -1;
    CHECK(rbf_solve_csr_dp(A.n, A.nnz(), A.val.data(), A.ia.data(), A.ja.data(), b.data(), x.data(),
                           nullptr, &iter, nullptr, nullptr, nullptr, nullptr) == SOLVER_SUCCESS);
    CHECK(iter == 0);
    CHECK(max_abs_diff(x, u) == 0);
}

// The product on its own, against the reference loop, with y not read
// when beta is zero.
static void test_csr_mv() {
    const int m = 9;
    const Csr A = laplacian(m, 0.25);
    const std::vector<double> u = exact(m);
    std::vector<double> ref(u.size(), 0);
    A.apply(1, u.data(), ref.data());

    std::vector<double> y(u.size(), std::nan(""));
    rbf_csr_mv_dp(A.n, A.n, A.nnz(), A.val.data(), A.ia.data(), A.ja.data(), 1, u.data(), 0,
                  y.data());
    CHECK(max_abs_diff(y, ref) < 1e-14);

    // y = -0.5 y + 2 A u
    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] = 0.1 * static_cast<double>(i);
    std::vector<double> expect(u.size());
    for (std::size_t i = 0; i < y.size(); ++i)
        expect[i] = -0.5 * y[i] + 2 * ref[i];
    rbf_csr_mv_dp(A.n, A.n, A.nnz(), A.val.data(), A.ia.data(), A.ja.data(), 2, u.data(), -0.5,
                  y.data());
    CHECK(max_abs_diff(y, expect) < 1e-14);
}

static void test_failures() {
    const int m = 20;
    const Csr A = laplacian(m);
    const std::vector<double> u = exact(m);
    std::vector<double> b(u.size(), 0);
    A.apply(1, u.data(), b.data());
    std::vector<double> x(u.size(), 0);

    // an iteration limit too low to converge: NO_CONVERGENCE, with the
    // last iterate and its residual reported
    const int one = 1;
    const double tol = 1e-14;
    double err = -1;
    int iter = -1;
    CHECK(rbf_solve_csr_dp(A.n, A.nnz(), A.val.data(), A.ia.data(), A.ja.data(), b.data(), x.data(),
                           &err, &iter, nullptr, nullptr, &one, &tol) == SOLVER_NO_CONVERGENCE);
    CHECK(iter == 1);
    CHECK(err > tol);

    // the same through the callback
    CHECK(rbf_solve_mf_dp(A.n, A.n, callback, const_cast<Csr*>(&A), b.data(), x.data(), &err, &iter,
                          nullptr, &one, &tol) == SOLVER_NO_CONVERGENCE);
    CHECK(iter == 1);

    // INVALID_INPUT: a null array, a row pointer that disagrees with
    // nnz, an enumerator outside the enum, a rectangular operator
    CHECK(rbf_solve_csr_dp(A.n, A.nnz(), nullptr, A.ia.data(), A.ja.data(), b.data(), x.data(),
                           nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr) == SOLVER_INVALID_INPUT);
    CHECK(rbf_solve_csr_dp(A.n, A.nnz() - 1, A.val.data(), A.ia.data(), A.ja.data(), b.data(),
                           x.data(), nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr) == SOLVER_INVALID_INPUT);
    const auto bad_method = static_cast<solver_method>(7);
    CHECK(rbf_solve_csr_dp(A.n, A.nnz(), A.val.data(), A.ia.data(), A.ja.data(), b.data(), x.data(),
                           nullptr, nullptr, &bad_method, nullptr, nullptr,
                           nullptr) == SOLVER_INVALID_INPUT);
    const auto bad_precond = static_cast<solver_precond>(7);
    CHECK(rbf_solve_csr_dp(A.n, A.nnz(), A.val.data(), A.ia.data(), A.ja.data(), b.data(), x.data(),
                           nullptr, nullptr, nullptr, &bad_precond, nullptr,
                           nullptr) == SOLVER_INVALID_INPUT);
    CHECK(rbf_solve_mf_dp(A.n, A.n + 1, callback, const_cast<Csr*>(&A), b.data(), x.data(), nullptr,
                          nullptr, nullptr, nullptr, nullptr) == SOLVER_INVALID_INPUT);
    CHECK(rbf_solve_mf_dp(A.n, A.n, nullptr, nullptr, b.data(), x.data(), nullptr, nullptr, nullptr,
                          nullptr, nullptr) == SOLVER_INVALID_INPUT);
}

int main() {
    const int m = 20;
    const std::vector<double> u = exact(m);

    // symmetric positive definite: both methods
    const Csr spd = laplacian(m);
    test_csr(spd, u, SOLVER_CG);
    test_csr(spd, u, SOLVER_BICGSTAB);
    test_matrix_free(spd, u, SOLVER_CG);
    test_matrix_free(spd, u, SOLVER_BICGSTAB);

    // nonsymmetric: BiCGSTAB only
    const Csr nonsym = laplacian(m, 0.5);
    test_csr(nonsym, u, SOLVER_BICGSTAB);
    test_matrix_free(nonsym, u, SOLVER_BICGSTAB);

    test_defaults_and_guess();
    test_csr_mv();
    test_failures();

    return report("test_solver");
}
