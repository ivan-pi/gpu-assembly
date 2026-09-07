// lbm_lattice.h -- lattice definitions
//
// A lattice is a static class: velocity set, weights, speed of sound, and
// the two per-node maps every collision model needs, macros() and
// equilibrium(), written on register arrays so they inline into both
// host loops and device kernels.
//
// D2Q9 numbering and storage convention as in d2q9_kernels.cuf:
/*
      6  2  5       0           rest             w0 = 4/9
       \ | /        1, 2, 3, 4  E, N, W, S       ws = 1/9
      3--0--1       5, 6, 7, 8  NE, NW, SW, SE   wd = 1/36
       / | \
      7  4  8
*/
// Populations are stored planar: direction q of node i is f[i + ldf*q],
// with ldf >= n a padded leading dimension.

#ifndef LBM_LATTICE_H
#define LBM_LATTICE_H

#include "lbm_space.h"

namespace lbm {

template<typename T>
struct D2Q9 {
    using value_type = T;
    static constexpr int D = 2;
    static constexpr int Q = 9;
    static constexpr int rest = 0;          // index of the zero-velocity direction

    static constexpr T cs2 = T(1) / T(3);
    static constexpr int cx[Q]  = {0, 1, 0,-1, 0, 1,-1,-1, 1};
    static constexpr int cy[Q]  = {0, 0, 1, 0,-1, 1, 1,-1,-1};
    static constexpr int opp[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

    static constexpr T w0 = T(4) / T(9), ws = T(1) / T(9), wd = T(1) / T(36);
    static constexpr T w[Q] = {w0, ws, ws, ws, ws, wd, wd, wd, wd};

    struct Macros { T rho, ux, uy; };

    LBM_HD static Macros macros(const T (&f)[Q]) {
        Macros m;
        m.rho = (((f[5] + f[7]) + (f[6] + f[8])) + ((f[1] + f[3]) + (f[2] + f[4]))) + f[0];
        const T invrho = T(1) / m.rho;
        m.ux = invrho * (((f[5] - f[7]) + (f[8] - f[6])) + (f[1] - f[3]));
        m.uy = invrho * (((f[5] - f[7]) + (f[6] - f[8])) + (f[2] - f[4]));
        return m;
    }

    // feq_q = 3 w_q rho [ 1/3 - u.u/2 + c_q.u + 3/2 (c_q.u)^2 ]; the same
    // arithmetic as feq_kernel in d2q9_kernels.cu
    LBM_HD static void equilibrium(T rho, T ux, T uy, T (&feq)[Q]) {
        constexpr T three_w0 = T(3) * w0, three_ws = T(3) * ws, three_wd = T(3) * wd;
        const T uxsq = ux * ux, uysq = uy * uy;
        const T indp = cs2 - T(0.5) * (uxsq + uysq);

        feq[0] = three_w0 * rho * indp;

        const T t13 = indp + T(1.5) * uxsq;
        feq[1] = three_ws * rho * (t13 + ux);
        feq[3] = three_ws * rho * (t13 - ux);

        const T t24 = indp + T(1.5) * uysq;
        feq[2] = three_ws * rho * (t24 + uy);
        feq[4] = three_ws * rho * (t24 - uy);

        const T upv = ux + uy;
        const T t57 = indp + T(1.5) * upv * upv;
        feq[5] = three_wd * rho * (t57 + upv);
        feq[7] = three_wd * rho * (t57 - upv);

        const T umv = ux - uy;
        const T t68 = indp + T(1.5) * umv * umv;
        feq[6] = three_wd * rho * (t68 - umv);
        feq[8] = three_wd * rho * (t68 + umv);
    }
};

} // namespace lbm

#endif // LBM_LATTICE_H
