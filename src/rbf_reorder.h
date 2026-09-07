#ifndef RBF_REORDER_H
#define RBF_REORDER_H

// Node renumbering: permutations, space-filling-curve orderings, and
// boundary-last partitions.
//
// Deliberately independent of NodeSet: besides the main workflow, it
// can back offline tools that reorder node/graph files ahead of a run.
//
// The curve keys (Morton/Hilbert) are computed by the Fortran module
// rbf_ordering.F90; link it into any target that uses morton_order,
// hilbert_order, or the *_keys functions. Everything else in this
// header is self-contained, apart from Permutation::read and write,
// which use the ordering-file functions in rbf_io.h.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "rbf_io.h"

namespace rbf {

template<typename T>
struct BBox2 {
    T xmin, ymin, xmax, ymax;
};

// BBox2<double> is passed by reference through the bind(c) interface
// below and must stay layout-compatible with type bbox2d in
// rbf_ordering.F90.
static_assert(std::is_standard_layout_v<BBox2<double>> &&
              std::is_trivially_copyable_v<BBox2<double>>);
static_assert(sizeof(BBox2<double>) == 4 * sizeof(double));

} // namespace rbf

extern "C" {
void rbf_morton_keys(const std::int32_t* n,
                     const double* x, const double* y,
                     const rbf::BBox2<double>* bbox, const std::int32_t* ndiv,
                     std::int64_t* keys);
void rbf_hilbert_keys(const std::int32_t* n,
                      const double* x, const double* y,
                      const rbf::BBox2<double>* bbox, const std::int32_t* ndiv,
                      std::int64_t* keys);
}

namespace rbf {

template<typename T>
BBox2<T> compute_bbox(std::span<const T> x, std::span<const T> y) {
    assert(!x.empty() && x.size() == y.size());
    const auto [xlo, xhi] = std::minmax_element(x.begin(), x.end());
    const auto [ylo, yhi] = std::minmax_element(y.begin(), y.end());
    return {*xlo, *ylo, *xhi, *yhi};
}

// A permutation of n items. Convention: map()[i] is the OLD index of the
// item that lands at NEW position i (gather form):  new[i] = old[map()[i]].
// The inverse (old -> new) is computed once and cached, since column
// relabelling and joins against original-order data both need it.
template<typename I = std::int32_t>
class Permutation {
public:
    Permutation() = default;
    explicit Permutation(std::vector<I> new_to_old)
        : p_(std::move(new_to_old)), ip_(make_inverse(p_)) {}

    static Permutation identity(size_t n) {
        std::vector<I> v(n);
        std::iota(v.begin(), v.end(), I{0});
        return Permutation(std::move(v));
    }

    // From the old -> new map: from_inverse(p.inv()) == p.
    static Permutation from_inverse(std::vector<I> old_to_new) {
        Permutation p;
        p.p_ = make_inverse(old_to_new);
        p.ip_ = std::move(old_to_new);
        return p;
    }

    // Ordering file (.iperm), the METIS format: line i is the new index
    // of item i, i.e. the file holds inv(). read(f).write(g) copies f.
    static Permutation read(const std::string& fname) {
        return from_inverse(io::read_ordering<I>(fname));
    }
    void write(const std::string& fname) const {
        io::write_ordering(fname, ip_.size(), ip_.data());
    }

    size_t size() const { return p_.size(); }

    std::span<const I> map() const { return p_; }  // new -> old
    std::span<const I> inv() const { return ip_; } // old -> new

    // In-place through a temporary:  a_new[i] = a_old[map()[i]]
    template<typename T>
    void permute(std::span<T> a) const {
        assert(a.size() == p_.size());
        const std::vector<T> tmp(a.begin(), a.end());
        for (size_t i = 0; i < p_.size(); ++i)
            a[i] = tmp[p_[i]];
    }

    // Inverse application:  a_old[map()[i]] = a_new[i]
    template<typename T>
    void unpermute(std::span<T> a) const {
        assert(a.size() == p_.size());
        const std::vector<T> tmp(a.begin(), a.end());
        for (size_t i = 0; i < p_.size(); ++i)
            a[p_[i]] = tmp[i];
    }

    // Composition: apply *this first, then q.
    // (a.then(b)).map()[i] == a.map()[b.map()[i]]
    Permutation then(const Permutation& q) const {
        assert(size() == q.size());
        std::vector<I> r(p_.size());
        for (size_t i = 0; i < p_.size(); ++i)
            r[i] = p_[q.p_[i]];
        return Permutation(std::move(r));
    }

    Permutation inverse() const { return Permutation(ip_); }

private:
    static std::vector<I> make_inverse(const std::vector<I>& p) {
#ifndef NDEBUG
        std::vector<char> seen(p.size(), 0);
        for (const auto v : p) {
            assert(v >= 0 && static_cast<size_t>(v) < p.size() &&
                   "not a permutation: index out of range");
            assert(!seen[v] && "not a permutation: duplicate index");
            seen[v] = 1;
        }
#endif
        std::vector<I> ip(p.size());
        for (size_t i = 0; i < p.size(); ++i)
            ip[p[i]] = static_cast<I>(i);
        return ip;
    }

    std::vector<I> p_;  // new -> old
    std::vector<I> ip_; // old -> new
};

namespace detail {

using KeyFill = void (*)(const std::int32_t*, const double*, const double*,
                         const BBox2<double>*, const std::int32_t*,
                         std::int64_t*);

inline std::vector<std::int64_t> fill_keys(
        KeyFill fill,
        std::span<const double> x, std::span<const double> y,
        int ndiv, std::optional<BBox2<double>> bbox) {
    assert(x.size() == y.size());
    assert(ndiv >= 1 && ndiv <= 31); // 2 bits per level in an int64 key
    const auto bb = bbox ? *bbox : compute_bbox(x, y);
    const auto n = static_cast<std::int32_t>(x.size());
    const auto nd = static_cast<std::int32_t>(ndiv);
    std::vector<std::int64_t> keys(x.size());
    fill(&n, x.data(), y.data(), &bb, &nd, keys.data());
    return keys;
}

template<typename I>
Permutation<I> order_by_keys(std::span<const std::int64_t> keys) {
    std::vector<I> p(keys.size());
    std::iota(p.begin(), p.end(), I{0});
    std::stable_sort(p.begin(), p.end(),
        [keys](I a, I b) { return keys[a] < keys[b]; });
    return Permutation<I>(std::move(p));
}

} // namespace detail

// Curve keys, mainly for inspection and testing; the *_order builders
// below are the intended entry points. ndiv quadtree levels give 2*ndiv
// key bits (ndiv <= 31); the default resolves ~65k cells per axis.
// bbox is computed from the points when not supplied.

inline std::vector<std::int64_t> morton_keys(
        std::span<const double> x, std::span<const double> y,
        int ndiv = 16, std::optional<BBox2<double>> bbox = std::nullopt) {
    return detail::fill_keys(&rbf_morton_keys, x, y, ndiv, bbox);
}

inline std::vector<std::int64_t> hilbert_keys(
        std::span<const double> x, std::span<const double> y,
        int ndiv = 16, std::optional<BBox2<double>> bbox = std::nullopt) {
    return detail::fill_keys(&rbf_hilbert_keys, x, y, ndiv, bbox);
}

// Morton (Z-curve) order of the points. Stable: points falling into the
// same cell keep their relative order.
template<typename I = std::int32_t>
Permutation<I> morton_order(
        std::span<const double> x, std::span<const double> y,
        int ndiv = 16, std::optional<BBox2<double>> bbox = std::nullopt) {
    return detail::order_by_keys<I>(morton_keys(x, y, ndiv, bbox));
}

// Hilbert-curve order of the points; same contract as morton_order.
template<typename I = std::int32_t>
Permutation<I> hilbert_order(
        std::span<const double> x, std::span<const double> y,
        int ndiv = 16, std::optional<BBox2<double>> bbox = std::nullopt) {
    return detail::order_by_keys<I>(hilbert_keys(x, y, ndiv, bbox));
}

// Stable partition: nodes with pred(i)==false keep their relative order
// at the front; nodes with pred(i)==true keep theirs at the back.
template<typename I = std::int32_t, typename Pred>
Permutation<I> partition_last(I n, Pred&& pred) {
    std::vector<I> p(static_cast<size_t>(n));
    std::iota(p.begin(), p.end(), I{0});
    std::stable_partition(p.begin(), p.end(),
        [&pred](I i) { return !pred(i); });
    return Permutation<I>(std::move(p));
}

// Nodes with nonzero flag go last (stable). n is inferred: flag.size().
template<typename I = std::int32_t>
Permutation<I> boundary_last_by_flag(std::span<const int> flag) {
    return partition_last<I>(static_cast<I>(flag.size()),
        [flag](I i) { return flag[i] != 0; });
}

// The listed nodes go last (stable). n must be given: bnd has length nb,
// not n. (Deliberately not an overload of the flag version -- with
// I == int the two argument lists would be a silent mixup.)
template<typename I = std::int32_t>
Permutation<I> boundary_last_by_index(I n, std::span<const I> bnd) {
    std::vector<char> mask(static_cast<size_t>(n), 0);
    for (const auto b : bnd) {
        assert(b >= 0 && b < n && "boundary index out of range");
        mask[b] = 1;
    }
    return partition_last<I>(n, [&mask](I i) { return mask[i] != 0; });
}

// Renumber a fixed-width stencil graph in place. Layout: the k
// neighbours of node s are contiguous at ja[s*k].
//
// In matrix terms this is the graph of the symmetric renumbering
//
//     A' = P A P^T,   i.e.   A'(i,j) = A(p(i), p(j)),  p = map()
//
// where P is the permutation matrix with P(i, p(i)) = 1. Applying P
// from the left gathers old row p(s) into new row s, done here by
// copying the row blocks. Applying P^T from the right permutes the
// columns the same way; in index storage that means each stored column
// value j (an old node number) is relabelled to the new number of that
// node, inv()[j] -- the column relabelling IS the P^T factor, no data
// movement within the row is needed.
//
// Within-row order is preserved, so the invariant ja[s*k] == s
// survives. One pass over the graph with one temporary copy of ja;
// writes are sequential, reads gather.
template<typename I>
void renumber_stencils(std::span<I> ja, int k, const Permutation<I>& p) {
    const size_t n = p.size();
    assert(ja.size() == n * static_cast<size_t>(k));
    const std::vector<I> tmp(ja.begin(), ja.end());
    const auto pm = p.map();
    const auto ip = p.inv();
    for (size_t s = 0; s < n; ++s)
        for (size_t j = 0; j < static_cast<size_t>(k); ++j)
            ja[s*k + j] = ip[tmp[static_cast<size_t>(pm[s])*k + j]];
}

// Renumber a general CSR graph in place: A' = P A P^T as in
// renumber_stencils. Repacking old row p(i) into new row i applies P
// from the left; relabelling each stored column value through inv()
// applies P^T from the right.
template<typename I>
void renumber_csr(std::span<I> ia, std::span<I> ja, const Permutation<I>& p) {
    const size_t n = p.size();
    assert(ia.size() == n + 1);
    assert(static_cast<size_t>(ia[n]) == ja.size());
    const std::vector<I> ia_tmp(ia.begin(), ia.end());
    const std::vector<I> ja_tmp(ja.begin(), ja.end());
    const auto pm = p.map();
    const auto ip = p.inv();
    ia[0] = 0;
    for (size_t i = 0; i < n; ++i) {
        const I iaa = ia_tmp[pm[i]];
        const I iab = ia_tmp[pm[i] + 1];
        std::copy(ja_tmp.begin() + iaa, ja_tmp.begin() + iab,
                  ja.begin() + ia[i]);
        ia[i+1] = ia[i] + (iab - iaa);
    }
    for (auto& j : ja) j = ip[j];
}

// Explicit row pointer {0, k, 2k, ...} for a fixed-width graph -- the
// bridge to consumers that expect honest CSR.
template<typename I = std::int32_t>
std::vector<I> make_row_ptr(I n, int k) {
    std::vector<I> ia(static_cast<size_t>(n) + 1);
    for (size_t s = 0; s <= static_cast<size_t>(n); ++s)
        ia[s] = static_cast<I>(s * k);
    return ia;
}

} // namespace rbf

#endif // RBF_REORDER_H
