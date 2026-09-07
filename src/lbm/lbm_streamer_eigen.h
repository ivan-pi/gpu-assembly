// lbm_streamer_eigen.h -- Eigen CSR backend for the split stepper
//
// Illustrates the backend seam: a Streamer owns whatever the library
// needs (here a row pointer and Q sparse-matrix maps over the row-major
// weights, which are already CSR with a fixed row length), is built once
// from StreamingWeights, and exposes stream(f_in, f_out, ldf). The MKL
// (mkl_sparse_?_create_csr + mkl_sparse_?_mv) and cuSPARSE
// (cusparseCreateCsr + cusparseSpMV with a preallocated buffer) streamers
// have the same shape; they differ only in what the constructor creates
// and the destructor releases.
//
// Host only. Requires Eigen 3.4.

#ifndef LBM_STREAMER_EIGEN_H
#define LBM_STREAMER_EIGEN_H

#include <cstdint>
#include <numeric>
#include <vector>

#include <Eigen/Sparse>

#include "lbm_streaming.h"

namespace lbm {

template<class L, typename I = std::int32_t>
class EigenStreamer {
public:
    using T = typename L::value_type;
    using SpMat = Eigen::SparseMatrix<T, Eigen::RowMajor, I>;
    using Map = Eigen::Map<const SpMat>;
    using Vec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    using VecOut = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

    static constexpr const char* name() { return "eigen"; }

    // keeps a reference to w: the maps point into its arrays
    explicit EigenStreamer(const StreamingWeights<T, I>& w) : w_(w), n_(w.n), nnz_(w.n * w.k) {
        w.validate();
        ia_.resize(w.n + 1);
        for (I s = 0; s <= w.n; ++s) ia_[s] = s * w.k;
    }

    void stream(const T* f_in, T* f_out, I ldf) const {
        for (int q = 0; q < L::Q; ++q) {
            const T* src = f_in + std::size_t(q) * ldf;
            T* dst = f_out + std::size_t(q) * ldf;
            if (w_.is_identity(q)) {
                for (I i = 0; i < n_; ++i) dst[i] = src[i];
                continue;
            }
            Map A(n_, n_, nnz_, ia_.data(), w_.patterns[w_.pattern_of[q]].data(), w_.a[q].data());
            VecOut(dst, n_).noalias() = A * Vec(src, n_);
        }
    }

    std::size_t bytes_per_step() const {
        const std::size_t nstream = L::Q - (L::rest >= 0 ? 1 : 0);
        return std::size_t(n_) * (nstream * w_.k * (sizeof(I) + sizeof(T)) + 2 * L::Q * sizeof(T));
    }

private:
    const StreamingWeights<T, I>& w_;
    I n_, nnz_;
    std::vector<I> ia_;
};

} // namespace lbm

#endif // LBM_STREAMER_EIGEN_H
