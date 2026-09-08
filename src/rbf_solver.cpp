// rbf_solver.cpp -- the Eigen side of rbf_solver.h.
//
// Eigen is confined to this translation unit: the header is plain C and
// the arrays cross it as pointers. The CSR arrays are wrapped in place
// by an Eigen::Map, which the solver's Ref binds to without a copy as
// long as the matrix is in compressed form, which CSR is; what the
// solver allocates is its Krylov vectors and the preconditioner. The
// matrix-free path follows Eigen's "matrix-free solvers" example: a
// minimal expression type whose product evaluates through the caller's
// callback.
//
// Eigen's own assertions follow NDEBUG, like the rest of the library.

#include "rbf_solver.h"

#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

namespace {

// The status codes of the header are Eigen's, by construction.
static_assert(SOLVER_SUCCESS == static_cast<int>(Eigen::Success));
static_assert(SOLVER_NUMERICAL_ISSUE == static_cast<int>(Eigen::NumericalIssue));
static_assert(SOLVER_NO_CONVERGENCE == static_cast<int>(Eigen::NoConvergence));
static_assert(SOLVER_INVALID_INPUT == static_cast<int>(Eigen::InvalidInput));

using Scalar = double;
using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
using VectorMap = Eigen::Map<Vector>;
using ConstVectorMap = Eigen::Map<const Vector>;

// The matrix type the CSR solvers are instantiated with. The Map over
// the caller's arrays is a different type, and the solver's compute()
// takes it through a Ref<const CsrMatrix>, which is where the in-place
// binding happens.
using CsrMatrix = Eigen::SparseMatrix<Scalar, Eigen::RowMajor, int>;
using CsrMap = Eigen::Map<const CsrMatrix>;

// Iteration limit and tolerance as Eigen would default them: NULL
// means Eigen's own choice, which setMaxIterations(-1) and the
// constructor's tolerance are.
struct Controls {
    const int* max_iter;
    const double* tolerance;
};

// Runs one solver type on A, x being the initial guess on input and
// the solution on output, and reports the way the C interface does.
template <class Solver, class Matrix>
int run(const Matrix& A,
        const double* b,
        double* x,
        Eigen::Index n,
        double* res_error,
        int* res_iter,
        Controls c) {
    Solver solver;
    if (c.max_iter) solver.setMaxIterations(*c.max_iter);
    if (c.tolerance) solver.setTolerance(*c.tolerance);

    solver.compute(A);

    if (solver.info() == Eigen::Success) {
        const ConstVectorMap bv(b, n);
        VectorMap xv(x, n);
        xv = solver.solveWithGuess(bv, xv);
    }

    if (res_error) *res_error = solver.error();
    if (res_iter) *res_iter = static_cast<int>(solver.iterations());
    return static_cast<int>(solver.info());
}

// --- CSR ---

template <class Precond>
using BiCGSTAB = Eigen::BiCGSTAB<CsrMatrix, Precond>;

template <class Precond>
using ConjugateGradient = Eigen::ConjugateGradient<CsrMatrix, Eigen::Lower | Eigen::Upper, Precond>;

// One method over the three preconditioners; Incomplete is the
// factorization that suits the method.
template <template <class> class Method, class Incomplete>
int solve_csr(solver_precond precond,
              const CsrMap& A,
              const double* b,
              double* x,
              double* res_error,
              int* res_iter,
              Controls c) {
    const auto n = A.rows();
    switch (precond) {
        case PRECOND_NONE:
            return run<Method<Eigen::IdentityPreconditioner>>(A, b, x, n, res_error, res_iter, c);
        case PRECOND_JACOBI:
            return run<Method<Eigen::DiagonalPreconditioner<Scalar>>>(A, b, x, n, res_error,
                                                                      res_iter, c);
        case PRECOND_ILU:
            return run<Method<Incomplete>>(A, b, x, n, res_error, res_iter, c);
    }
    return SOLVER_INVALID_INPUT;
}

// --- Matrix-free ---

// An n-by-m operator known only through the callback. Eigen wants a
// type derived from EigenBase with the traits of a sparse matrix and a
// product operator returning a Product expression; the evaluation of
// that expression is the generic_product_impl specialization below.
class Callback;

}  // namespace

namespace Eigen::internal {
template <>
struct traits<Callback> : public traits<SparseMatrix<double>> {};
}  // namespace Eigen::internal

namespace {

class Callback : public Eigen::EigenBase<Callback> {
public:
    using Scalar = double;
    using RealScalar = double;
    using StorageIndex = int;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    Callback(int nr, int nc, rbf_matvec mv, void* data) : nr_(nr), nc_(nc), mv_(mv), data_(data) {}

    Index rows() const { return nr_; }
    Index cols() const { return nc_; }

    template <class Rhs>
    Eigen::Product<Callback, Rhs, Eigen::AliasFreeProduct> operator*(
        const Eigen::MatrixBase<Rhs>& x) const {
        return Eigen::Product<Callback, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    // dst += alpha * A * rhs, the update the callback computes. The
    // solvers only ever multiply plain vectors, so both operands have
    // contiguous storage to hand over.
    template <class Rhs, class Dest>
    void scale_and_add_to(Dest& dst, const Rhs& rhs, double alpha) const {
        mv_(nr_, nc_, alpha, rhs.data(), dst.data(), data_);
    }

private:
    int nr_, nc_;
    rbf_matvec mv_;
    void* data_;
};

}  // namespace

namespace Eigen::internal {
template <class Rhs>
struct generic_product_impl<Callback, Rhs, SparseShape, DenseShape, GemvProduct>
    : generic_product_impl_base<Callback, Rhs, generic_product_impl<Callback, Rhs>> {
    template <class Dest>
    static void scaleAndAddTo(Dest& dst, const Callback& lhs, const Rhs& rhs, const double& alpha) {
        lhs.scale_and_add_to(dst, rhs, alpha);
    }
};
}  // namespace Eigen::internal

// --- The C interface ---

extern "C" {

void rbf_csr_mv_dp(int nr,
                   int nc,
                   int nnz,
                   const double* val,
                   const int* ia,
                   const int* ja,
                   double alpha,
                   const double* x,
                   double beta,
                   double* y) {
    const CsrMap A(nr, nc, nnz, ia, ja, val);
    const ConstVectorMap xv(x, nc);
    VectorMap yv(y, nr);
    // BLAS semantics: y is not read when beta is zero, so a NaN or an
    // uninitialized y cannot leak into the result
    if (beta == 0)
        yv.noalias() = alpha * (A * xv);
    else
        yv = beta * yv + alpha * (A * xv);
}

int rbf_solve_csr_dp(int n,
                     int nnz,
                     const double* val,
                     const int* ia,
                     const int* ja,
                     const double* b,
                     double* x,
                     double* res_error,
                     int* res_iter,
                     const solver_method* method,
                     const solver_precond* precond,
                     const int* max_iter,
                     const double* tolerance) {
    if (n <= 0 || nnz < 0 || !val || !ia || !ja || !b || !x || ia[n] != nnz)
        return SOLVER_INVALID_INPUT;

    const solver_method m = method ? *method : SOLVER_BICGSTAB;
    const solver_precond p = precond ? *precond : PRECOND_JACOBI;
    const Controls c{max_iter, tolerance};

    const CsrMap A(n, n, nnz, ia, ja, val);

    switch (m) {
        case SOLVER_BICGSTAB:
            return solve_csr<BiCGSTAB, Eigen::IncompleteLUT<Scalar>>(p, A, b, x, res_error,
                                                                     res_iter, c);
        case SOLVER_CG:
            return solve_csr<ConjugateGradient, Eigen::IncompleteCholesky<Scalar>>(
                p, A, b, x, res_error, res_iter, c);
    }
    return SOLVER_INVALID_INPUT;
}

int rbf_solve_mf_dp(int nr,
                    int nc,
                    rbf_matvec mv,
                    void* data,
                    const double* b,
                    double* x,
                    double* res_error,
                    int* res_iter,
                    const solver_method* method,
                    const int* max_iter,
                    const double* tolerance) {
    if (nr <= 0 || nc <= 0 || nr != nc || !mv || !b || !x) return SOLVER_INVALID_INPUT;

    const solver_method m = method ? *method : SOLVER_BICGSTAB;
    const Controls c{max_iter, tolerance};

    const Callback A(nr, nc, mv, data);

    switch (m) {
        case SOLVER_BICGSTAB:
            return run<Eigen::BiCGSTAB<Callback, Eigen::IdentityPreconditioner>>(
                A, b, x, nc, res_error, res_iter, c);
        case SOLVER_CG:
            return run<Eigen::ConjugateGradient<Callback, Eigen::Lower | Eigen::Upper,
                                                Eigen::IdentityPreconditioner>>(
                A, b, x, nc, res_error, res_iter, c);
    }
    return SOLVER_INVALID_INPUT;
}

}  // extern "C"
