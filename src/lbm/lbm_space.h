// lbm_space.h -- memory and execution spaces
//
// A space is a tag type. Array<T, Space> owns a contiguous buffer in that
// space; Exec<Space>::parallel_for runs a per-index functor there. Every
// kernel in the LBM headers is written once as a functor with an
// LBM_HD operator()(I i) and run through Exec<Space>, so adding an
// execution space means adding one Array and one Exec specialisation
// here, not touching the kernels.
//
// Provided:
//   space::Host   aligned heap memory, OpenMP parallel-for
//   space::Cuda   device memory, one thread per index (nvcc only)
//
// Managed (unified) memory would be a third Array specialisation with
// the Cuda Exec; it is deliberately not the default, since explicit
// placement keeps host/device traffic visible.

#ifndef LBM_SPACE_H
#define LBM_SPACE_H

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define LBM_HD __host__ __device__
#define LBM_UNROLL _Pragma("unroll")
#elif defined(__clang__)
#define LBM_HD
#define LBM_UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define LBM_HD
#define LBM_UNROLL _Pragma("GCC unroll 27")
#else
#define LBM_HD
#define LBM_UNROLL
#endif

namespace lbm {

namespace space {
struct Host { static constexpr const char* name() { return "host"; } };
#if defined(__CUDACC__)
struct Cuda { static constexpr const char* name() { return "cuda"; } };
#endif
} // namespace space

template<typename T, class Space> class Array;

// --- Host --------------------------------------------------------------

template<typename T>
class Array<T, space::Host> {
public:
    using value_type = T;
    using space_type = space::Host;

    Array() = default;
    explicit Array(std::size_t n) : n_(n) {
        if (n_) {
            const std::size_t bytes = ((n_ * sizeof(T) + 63) / 64) * 64;
            p_ = static_cast<T*>(std::aligned_alloc(64, bytes));
            if (!p_) throw std::bad_alloc();
        }
    }
    ~Array() { std::free(p_); }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(Array&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    Array& operator=(Array&& o) noexcept {
        if (this != &o) { std::free(p_); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }

    T* data() { return p_; }
    const T* data() const { return p_; }
    std::size_t size() const { return n_; }
    T& operator[](std::size_t i) { return p_[i]; }
    const T& operator[](std::size_t i) const { return p_[i]; }

    void fill(const T& v) { for (std::size_t i = 0; i < n_; ++i) p_[i] = v; }

    // host <-> this space
    void upload(const T* src, std::size_t n, std::size_t offset = 0) {
        std::memcpy(p_ + offset, src, n * sizeof(T));
    }
    void download(T* dst, std::size_t n, std::size_t offset = 0) const {
        std::memcpy(dst, p_ + offset, n * sizeof(T));
    }

private:
    T* p_ = nullptr;
    std::size_t n_ = 0;
};

template<class Space> struct Exec;

template<>
struct Exec<space::Host> {
    // f(i) for i in [0, n); the functor is copied into the parallel region
    template<class F>
    static void parallel_for(std::size_t n, F f) {
        const std::ptrdiff_t m = static_cast<std::ptrdiff_t>(n);
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < m; ++i) f(i);
    }
    static void synchronize() {}
};

// --- CUDA (nvcc only) ---------------------------------------------------

#if defined(__CUDACC__)

inline void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template<typename T>
class Array<T, space::Cuda> {
public:
    using value_type = T;
    using space_type = space::Cuda;

    Array() = default;
    explicit Array(std::size_t n) : n_(n) {
        if (n_) cuda_check(cudaMalloc(&p_, n_ * sizeof(T)), "cudaMalloc");
    }
    ~Array() { if (p_) cudaFree(p_); }

    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;
    Array(Array&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    Array& operator=(Array&& o) noexcept {
        if (this != &o) { if (p_) cudaFree(p_); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
        return *this;
    }

    T* data() { return p_; }
    const T* data() const { return p_; }
    std::size_t size() const { return n_; }

    void fill(const T& v) {
        std::vector<T> h(n_, v);
        upload(h.data(), n_);
    }
    void upload(const T* src, std::size_t n, std::size_t offset = 0) {
        cuda_check(cudaMemcpy(p_ + offset, src, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    void download(T* dst, std::size_t n, std::size_t offset = 0) const {
        cuda_check(cudaMemcpy(dst, p_ + offset, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    }

private:
    T* p_ = nullptr;
    std::size_t n_ = 0;
};

namespace detail {
template<class F>
__global__ void for_kernel(std::size_t n, F f) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) f(static_cast<std::ptrdiff_t>(i));
}
}

template<>
struct Exec<space::Cuda> {
    static constexpr unsigned block = 128;
    template<class F>
    static void parallel_for(std::size_t n, F f) {
        if (!n) return;
        const unsigned grid = static_cast<unsigned>((n + block - 1) / block);
        detail::for_kernel<F><<<grid, block>>>(n, f);
        cuda_check(cudaGetLastError(), "parallel_for launch");
    }
    static void synchronize() { cuda_check(cudaDeviceSynchronize(), "synchronize"); }
};

#endif // __CUDACC__

// --- helpers ------------------------------------------------------------

template<class Space, typename T>
Array<T, Space> to_space(const std::vector<T>& v) {
    Array<T, Space> a(v.size());
    if (!v.empty()) a.upload(v.data(), v.size());
    return a;
}

template<typename T, class Space>
std::vector<T> to_host(const Array<T, Space>& a) {
    std::vector<T> v(a.size());
    if (a.size()) a.download(v.data(), a.size());
    return v;
}

// leading dimension padded to a multiple of `align` elements
template<typename I>
constexpr I padded(I n, I align) { return ((n + align - 1) / align) * align; }

} // namespace lbm

#endif // LBM_SPACE_H
