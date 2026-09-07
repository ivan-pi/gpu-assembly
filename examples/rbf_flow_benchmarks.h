#ifndef RBF_FLOW_BENCHMARKS_H
#define RBF_FLOW_BENCHMARKS_H

/*
 * Analytic flow fields for verifying periodic (geometry-free) solvers.
 *
 * Every case here is defined on a doubly periodic box, so the point
 * cloud can be generated on the fly and no external geometry is
 * needed. Benchmarks that require a genuine point cloud (lid-driven
 * cavity, channels, cylinder arrays, flat plate, Turek cylinder, ...)
 * are deliberately not part of this header.
 *
 * Conventions:
 *   - a functor call returns the primitive fields at a point (and time)
 *   - `stress_tensor` returns the strain rate S = (grad u + grad u^T)/2
 *     packed as {sxx, sxy, syy}
 *   - for a box of side L the wave numbers must be integer multiples
 *     of 2*pi/L for the field to be periodic
 */

#include <array>
#include <cmath>
#include <tuple>

namespace flow_benchmarks {

/*
 * SHEAR WAVE
 *
 * A transverse plane wave with wave vector k = (kx, ky):
 *
 *     u(x, t) = u0 * sin(k.x + phase) * exp(-nu |k|^2 t) * e_perp
 *
 * where e_perp = (-ky, kx)/|k| is the unit vector perpendicular to k,
 * so the field is divergence free and the pressure stays uniform.
 *
 * The Navier-Stokes equations reduce to the diffusion equation for this
 * field, so the amplitude decays exponentially with the time constant
 * 1/(nu |k|^2). Measuring the amplitude along k at two instants gives
 * the effective viscosity of a scheme; see `viscosity`.
 */
template<typename T>
struct shear_wave {

    T kx, ky, nu, u0;
    T phase{};

    T ksqr() const { return kx*kx + ky*ky; }
    T time_constant() const { return T(1.0)/(nu*ksqr()); }
    T decay_time(T frac) const { return -time_constant()*std::log(frac); }

    // Amplitude of the velocity at time t
    T amplitude(T time) const { return u0*std::exp(-time/time_constant()); }

    // Viscosity recovered from the ratio of amplitudes measured at the
    // times t0 and t1: nu = -ln(A1/A0) / (|k|^2 (t1 - t0))
    T viscosity(T a0, T a1, T t0, T t1) const {
        return -std::log(a1/a0)/(ksqr()*(t1 - t0));
    }

    std::tuple<T,T,T> operator()(std::array<T,2> xy, T time = {}) const {

        const auto& [x,y] = xy;

        const T knorm = std::sqrt(ksqr());
        const T ex = -ky/knorm;
        const T ey =  kx/knorm;

        const T a = amplitude(time)*std::sin(kx*x + ky*y + phase);

        T ux = a*ex;
        T uy = a*ey;
        T pressure{};

        return std::make_tuple(pressure,ux,uy);
    }

    std::array<T,3> stress_tensor(std::array<T,2> xy, T time = {}) const {

        const auto& [x,y] = xy;

        const T knorm = std::sqrt(ksqr());
        const T half{0.5};

        // grad u = u0 cos(k.x) e_perp (x) k
        const T c = amplitude(time)*std::cos(kx*x + ky*y + phase)/knorm;

        T sxx = -c*kx*ky;
        T sxy = half*c*(kx*kx - ky*ky);
        T syy = -sxx;

        return {sxx,sxy,syy};
    }

}; // struct shear_wave


/*
 * TAYLOR-GREEN VORTEX
 *
 * Doubly periodic array of counter-rotating vortices with wave numbers
 * kx and ky. The amplitudes are chosen so the field is divergence free
 * for any kx, ky; the kinetic energy decays with the time constant
 * 1/(nu (kx^2 + ky^2)).
 */
template<typename T>
struct taylor_green {

    T kx, ky, nu, u0;

    T time_constant() const { return T(1.0)/(nu*(kx*kx + ky*ky)); }
    T decay_time(T frac) const { return -time_constant()*std::log(frac); }

    std::tuple<T,T,T> operator()(std::array<T,2> xy, T time = {}) const {

        const auto& [x,y] = xy;

        const T kykx = ky/kx;
        const T kxky = kx/ky;

        const T pfx = -u0*std::sqrt(kykx);
        const T pfy =  u0*std::sqrt(kxky);

        const T tfrac = -time/time_constant();

        T ux = pfx*std::cos(kx * x)*std::sin(ky * y)*std::exp(tfrac);
        T uy = pfy*std::sin(kx * x)*std::cos(ky * y)*std::exp(tfrac);
        T pressure = ((T) -0.25)*u0*u0*(kykx*std::cos(2*kx*x) +
                                        kxky*std::cos(2*ky*y))*std::exp(2*tfrac);

        return std::make_tuple(pressure,ux,uy);
    }

    std::array<T,3> stress_tensor(std::array<T,2> xy, T time = {}) const {

        const auto& [x,y] = xy;

        const T tfrac = -time/time_constant();
        const T half{0.5};

        T sxx = u0*std::sqrt(kx*ky)*std::sin(kx*x)*std::sin(ky*y)*std::exp(tfrac);
        T sxy = half*u0*(std::sqrt(kx*kx*kx/ky) - std::sqrt(ky*ky*ky/kx)) *
            std::cos(kx*x) * std::cos(ky*y) * std::exp(tfrac);
        T syy = -sxx; // follows from div u = 0

        // If kx = ky, the shear component sxy will be zero

        return {sxx,sxy,syy};
    }

}; // struct taylor_green


/*
 * DOUBLY PERIODIC SHEAR LAYER
 *
 * Two tanh shear layers of thickness ~1/k at y/L = 1/4 and 3/4, with
 * a small sinusoidal perturbation of amplitude delta in uy that rolls
 * the layers up into vortices (Minion & Brown, 1997).
 */
template<typename T>
struct shear_layer {

    T u0, k, delta, L;

    std::tuple<T,T> operator()(std::array<T,2> xy) const {

        const auto& [x,y] = xy;

        const T pi = 3.141592653589793238462643383279502884L;

        const T xd = x/L;
        const T yd = y/L;

        T ux, uy;
        uy = u0*delta*std::sin(2*pi*(xd + T(0.25)));

        if (yd <= T(0.5)) {
            ux = u0*std::tanh(k*(yd - T(0.25)));
        } else {
            ux = u0*std::tanh(k*(T(0.75) - yd));
        }

        return std::make_tuple(ux,uy);
    }

}; // struct shear_layer


/*
 * BAROTROPIC VORTEX
 *
 * For more details refer to the work:
 *    Wissocq, G., Boussuge, J. F., & Sagaut, P. (2020).
 *    Consistent vortex initialization for the athermal lattice Boltzmann
 *    method. Physical Review E, 101(4), 043306.
 */
template<typename T>
struct barotropic_vortex {

    T U0;
    std::array<T,2> center;
    T Rc;
    T eps;
    T rho0{1.0};

    std::tuple<T,T,T> operator()(std::array<T,2> xy, T csqr) const {

        const T Rc_sqr = Rc*Rc;
        const T vMa_sqr = eps*eps/csqr;

        const T xr = xy[0] - center[0];
        const T yr = xy[1] - center[1];

        const T r_sqr = xr*xr + yr*yr;

        // Convected vortex, Eqs. (3) and (4)
        T ux = U0 - eps * (yr/Rc) * std::exp(-r_sqr/(2*Rc_sqr));
        T uy =      eps * (xr/Rc) * std::exp(-r_sqr/(2*Rc_sqr));

        // Barotropic density initialization, Eq. (20)
        T rho = rho0 * std::exp(-vMa_sqr * std::exp(-r_sqr/Rc_sqr) / 2);

        return std::make_tuple(rho,ux,uy);
    }

}; // struct barotropic_vortex

} // namespace flow_benchmarks

#endif /* RBF_FLOW_BENCHMARKS_H */
