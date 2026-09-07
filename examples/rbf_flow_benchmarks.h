#ifndef RBF_FLOW_BENCHMARKS_H
#define RBF_FLOW_BENCHMARKS_H

/*
 * Analytic flow fields for verifying periodic (geometry-free) solvers.
 *
 * Every case here is defined on a doubly periodic box, so the point
 * cloud can be generated on the fly and no external geometry is
 * needed. Benchmarks that require a genuine point cloud (lid-driven
 * cavity, channels, cylinder arrays, flat plate, Turek cylinder, ...)
 * are deliberately not part of this header. The Fortran module
 * rbf_benchmarks offers the same cases with the same names.
 *
 * Common interface:
 *
 *   auto [scalar, ux, uy] = case({x, y});         // any case
 *   auto [scalar, ux, uy] = case({x, y}, time);   // decaying cases
 *
 * The scalar is the pressure, except for the barotropic vortex where
 * it is the density (the case is defined that way, p = cs^2 rho).
 *
 * The decaying cases (shear_wave, taylor_green) are exact solutions of
 * the incompressible Navier-Stokes equations and additionally offer
 *
 *   ksqr(), time_constant(), decay_time(frac),
 *   amplitude(time), viscosity(a0, a1, t0, t1)
 *   stress_tensor({x, y}, time)   -> {sxx, sxy, syy}
 *
 * where stress_tensor is the strain rate S = (grad u + grad u^T)/2.
 * The other two (shear_layer, barotropic_vortex) are initial conditions
 * only and take no time argument.
 *
 * Every constructor asserts the constraints its formulas rely on (a
 * positive viscosity, wave numbers that are not both zero, ...), and the
 * decaying cases assert a non-negative time, since a benchmark starts
 * from the initial field. The checks follow the rest of the library and
 * compile out with NDEBUG.
 *
 * For a box of side L the wave numbers must be integer multiples of
 * 2*pi/L for the field to be periodic. The cases do not know the box,
 * so `wavenumber(n, L)` builds such a wave number from a mode number,
 * and the two decaying cases answer `periodic(Lx, Ly)` for the caller to
 * assert.
 */

#include <array>
#include <cassert>
#include <cmath>
#include <tuple>

namespace flow_benchmarks {

template<typename T>
inline constexpr T pi = T(3.141592653589793238462643383279502884L);

// Wave number of the n-th mode on a box of side L
template<typename T>
T wavenumber(int n, T L) {
    assert(L > 0 && "box side must be positive");
    return 2*pi<T>*n/L;
}

// Whether the wave number k fits an integer number of periods in L
template<typename T>
bool periodic(T k, T L, T tol = T(1e-10)) {
    assert(L > 0 && "box side must be positive");
    const T n = k*L/(2*pi<T>);
    return std::abs(n - std::round(n)) <= tol*std::max(T(1), std::abs(n));
}


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
    T phase;

    shear_wave(T kx, T ky, T nu, T u0, T phase = {})
        : kx(kx), ky(ky), nu(nu), u0(u0), phase(phase)
    {
        assert(kx*kx + ky*ky > 0 && "wave vector must not be zero");
        assert(nu > 0 && "viscosity must be positive");
    }

    T ksqr() const { return kx*kx + ky*ky; }
    T time_constant() const { return T(1.0)/(nu*ksqr()); }
    bool periodic(T Lx, T Ly) const {
        return flow_benchmarks::periodic(kx, Lx) && flow_benchmarks::periodic(ky, Ly);
    }

    // Time at which the amplitude has dropped to the fraction frac of u0
    T decay_time(T frac) const {
        assert(frac > 0 && frac <= 1 && "fraction must be in (0, 1]");
        return -time_constant()*std::log(frac);
    }

    // Velocity amplitude at time t
    T amplitude(T time) const {
        assert(time >= 0 && "time must be non-negative");
        return u0*std::exp(-time/time_constant());
    }

    // Viscosity recovered from the amplitudes a0 and a1 measured at the
    // times t0 and t1: nu = -ln(a1/a0) / (|k|^2 (t1 - t0))
    T viscosity(T a0, T a1, T t0, T t1) const {
        assert(a1/a0 > 0 && "amplitudes must have the same sign");
        assert(t1 != t0 && "the two instants must differ");
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
 * for any kx, ky; the velocity decays with the time constant
 * 1/(nu (kx^2 + ky^2)) and the pressure with half of it.
 */
template<typename T>
struct taylor_green {

    T kx, ky, nu, u0;

    taylor_green(T kx, T ky, T nu, T u0)
        : kx(kx), ky(ky), nu(nu), u0(u0)
    {
        // the amplitudes take sqrt(ky/kx), sqrt(kx/ky) and sqrt(kx*ky)
        assert(kx*ky > 0 && "kx and ky must be nonzero and of the same sign");
        assert(nu > 0 && "viscosity must be positive");
    }

    T ksqr() const { return kx*kx + ky*ky; }
    T time_constant() const { return T(1.0)/(nu*ksqr()); }
    bool periodic(T Lx, T Ly) const {
        return flow_benchmarks::periodic(kx, Lx) && flow_benchmarks::periodic(ky, Ly);
    }

    // Time at which the amplitude has dropped to the fraction frac of u0
    T decay_time(T frac) const {
        assert(frac > 0 && frac <= 1 && "fraction must be in (0, 1]");
        return -time_constant()*std::log(frac);
    }

    // Velocity amplitude at time t
    T amplitude(T time) const {
        assert(time >= 0 && "time must be non-negative");
        return u0*std::exp(-time/time_constant());
    }

    // Viscosity recovered from the amplitudes a0 and a1 measured at the
    // times t0 and t1: nu = -ln(a1/a0) / (|k|^2 (t1 - t0))
    T viscosity(T a0, T a1, T t0, T t1) const {
        assert(a1/a0 > 0 && "amplitudes must have the same sign");
        assert(t1 != t0 && "the two instants must differ");
        return -std::log(a1/a0)/(ksqr()*(t1 - t0));
    }

    std::tuple<T,T,T> operator()(std::array<T,2> xy, T time = {}) const {

        const auto& [x,y] = xy;

        const T kykx = ky/kx;
        const T kxky = kx/ky;

        const T a = amplitude(time);

        T ux = -a*std::sqrt(kykx)*std::cos(kx * x)*std::sin(ky * y);
        T uy =  a*std::sqrt(kxky)*std::sin(kx * x)*std::cos(ky * y);
        T pressure = ((T) -0.25)*a*a*(kykx*std::cos(2*kx*x) +
                                      kxky*std::cos(2*ky*y));

        return std::make_tuple(pressure,ux,uy);
    }

    std::array<T,3> stress_tensor(std::array<T,2> xy, T time = {}) const {

        const auto& [x,y] = xy;

        const T a = amplitude(time);
        const T half{0.5};

        T sxx = a*std::sqrt(kx*ky)*std::sin(kx*x)*std::sin(ky*y);
        T sxy = half*a*(std::sqrt(kx*kx*kx/ky) - std::sqrt(ky*ky*ky/kx)) *
            std::cos(kx*x) * std::cos(ky*y);
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
 * the layers up into vortices (Minion & Brown, 1997). Initial condition
 * only; the pressure is returned as zero.
 */
template<typename T>
struct shear_layer {

    T u0, L, k, delta;

    shear_layer(T u0, T L, T k = T(80.0), T delta = T(0.05))
        : u0(u0), L(L), k(k), delta(delta)
    {
        assert(L > 0 && "box side must be positive");
        assert(k > 0 && "layer steepness must be positive");
    }

    std::tuple<T,T,T> operator()(std::array<T,2> xy) const {

        const auto& [x,y] = xy;

        const T xd = x/L;
        const T yd = y/L;

        T ux, uy;
        uy = u0*delta*std::sin(2*pi<T>*(xd + T(0.25)));

        if (yd <= T(0.5)) {
            ux = u0*std::tanh(k*(yd - T(0.25)));
        } else {
            ux = u0*std::tanh(k*(T(0.75) - yd));
        }

        T pressure{};

        return std::make_tuple(pressure,ux,uy);
    }

}; // struct shear_layer


/*
 * BAROTROPIC VORTEX
 *
 * Initial condition only. Returns the density instead of the pressure,
 * as the case is defined for the athermal lattice Boltzmann method
 * where p = cs^2 rho; csqr is the squared sound speed of the lattice.
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
    T rho0;
    T csqr;

    barotropic_vortex(T U0, std::array<T,2> center, T Rc, T eps,
                      T rho0 = T(1.0), T csqr = T(1.0)/3 /* D2Q9, lattice units */)
        : U0(U0), center(center), Rc(Rc), eps(eps), rho0(rho0), csqr(csqr)
    {
        assert(Rc > 0 && "vortex radius must be positive");
        assert(rho0 > 0 && "reference density must be positive");
        assert(csqr > 0 && "squared sound speed must be positive");
    }

    std::tuple<T,T,T> operator()(std::array<T,2> xy) const {

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
