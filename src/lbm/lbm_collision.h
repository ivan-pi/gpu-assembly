// lbm_collision.h -- collision models
//
// A collision model is a functor over the populations of one node:
//
//     LBM_HD void operator()(T (&f)[Q], typename L::Macros& m) const;
//
// It relaxes f in place and reports the (conserved) macroscopic moments
// in m. Models hold their parameters by value so they can be copied into
// kernels. Everything a model needs from the lattice comes through the
// static interface of L (macros, equilibrium, opp, w, ...), which is what
// makes a model lattice-generic.
//
// Adding a model: write the functor here; nothing else changes, the
// steppers are templates on the model type.

#ifndef LBM_COLLISION_H
#define LBM_COLLISION_H

#include "lbm_lattice.h"

namespace lbm {
namespace collision {

// Single-relaxation-time (BGK)
template<class L>
struct BGK {
    using lattice = L;
    using T = typename L::value_type;
    static constexpr int Q = L::Q;

    T omega;

    static constexpr const char* name() { return "bgk"; }

    LBM_HD void operator()(T (&f)[Q], typename L::Macros& m) const {
        m = L::macros(f);
        T feq[Q];
        L::equilibrium(m.rho, m.ux, m.uy, feq);
        LBM_UNROLL
        for (int q = 0; q < Q; ++q) f[q] += omega * (feq[q] - f[q]);
    }
};

// Two-relaxation-time: even (symmetric) and odd (antisymmetric) parts of
// f - feq relax with omega_plus and omega_minus. omega_plus sets the
// viscosity; omega_minus follows from the magic parameter
// Lambda = (1/omega_plus - 1/2)(1/omega_minus - 1/2). With
// omega_minus == omega_plus the model reduces to BGK.
template<class L>
struct TRT {
    using lattice = L;
    using T = typename L::value_type;
    static constexpr int Q = L::Q;

    T omega_plus, omega_minus;

    static constexpr const char* name() { return "trt"; }

    static TRT from_magic(T omega_plus, T Lambda = T(0.25)) {
        const T tau_plus = T(1) / omega_plus - T(0.5);
        const T tau_minus = Lambda / tau_plus;
        return TRT{omega_plus, T(1) / (tau_minus + T(0.5))};
    }

    LBM_HD void operator()(T (&f)[Q], typename L::Macros& m) const {
        m = L::macros(f);
        T feq[Q];
        L::equilibrium(m.rho, m.ux, m.uy, feq);
        T g[Q];
        LBM_UNROLL
        for (int q = 0; q < Q; ++q) {
            const int p = L::opp[q];
            const T ne_q = f[q] - feq[q];
            const T ne_p = f[p] - feq[p];
            const T even = T(0.5) * (ne_q + ne_p);
            const T odd  = T(0.5) * (ne_q - ne_p);
            g[q] = f[q] - omega_plus * even - omega_minus * odd;
        }
        LBM_UNROLL
        for (int q = 0; q < Q; ++q) f[q] = g[q];
    }
};

} // namespace collision
} // namespace lbm

#endif // LBM_COLLISION_H
