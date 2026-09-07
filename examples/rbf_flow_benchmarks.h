#ifndef RBF_FLOW_BENCHMARKS_H
#define RBF_FLOW_BENCHMARKS_H

/*
 * Analytic flow fields on a doubly periodic box, for verifying a lattice
 * Boltzmann implementation without boundary conditions. The Fortran
 * module rbf_benchmarks offers the same cases with the same names.
 *
 * The box comes first: `periodic_box` holds the side lengths and turns
 * integer mode numbers into wave numbers, so every field built on it
 * is periodic by construction.
 *
 * Common interface:
 *
 *   auto [scalar, ux, uy] = case({x, y});         // any case
 *   auto [scalar, ux, uy] = case({x, y}, time);   // time-dependent cases
 *
 * The scalar is the pressure, or the density for the two cases defined
 * through the density (acoustic wave, barotropic vortex).
 *
 * Cases:
 *   shear_modes       superposition of shear waves with one |k|: exact
 *                     decaying solution of the incompressible NS
 *                     equations; shear_wave() and taylor_green() build
 *                     the two classic instances; body_force() holds it
 *                     steady (Kolmogorov flow, four-roll mill)
 *   acoustic_wave     standing sound wave, linear isothermal acoustics
 *   shear_layer       initial condition, roll-up of two tanh layers
 *   barotropic_vortex initial condition, convected Gaussian vortex
 *
 * Constructors assert the constraints the formulas rely on, with the
 * library's convention (assert with a message, compiled out by NDEBUG).
 */

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <vector>

namespace flow_benchmarks {

template<typename T>
inline constexpr T pi = T(3.141592653589793238462643383279502884L);


/*
 * PERIODIC BOX
 *
 * [0, Lx) x [0, Ly) with periodic images. The wave number of the mode
 * (nx, ny) is (2 pi nx/Lx, 2 pi ny/Ly). In lattice units Lx = nx cells.
 */
template<typename T>
struct periodic_box {

    T Lx, Ly;

    periodic_box(T Lx, T Ly) : Lx(Lx), Ly(Ly) {
        assert(Lx > 0 && Ly > 0 && "box sides must be positive");
    }

    std::array<T,2> wavenumber(int nx, int ny) const {
        return {2*pi<T>*nx/Lx, 2*pi<T>*ny/Ly};
    }

    // The point mapped into the box
    std::array<T,2> wrap(std::array<T,2> xy) const {
        return {xy[0] - Lx*std::floor(xy[0]/Lx), xy[1] - Ly*std::floor(xy[1]/Ly)};
    }

    // The shortest of the displacement and its periodic images
    std::array<T,2> minimum_image(std::array<T,2> d) const {
        return {d[0] - Lx*std::round(d[0]/Lx), d[1] - Ly*std::round(d[1]/Ly)};
    }

}; // struct periodic_box


/*
 * SHEAR MODES
 *
 * A superposition of transverse plane waves on the box,
 *
 *     u(x, t) = U + exp(-nu |k|^2 t) sum_m a_m sin(k_m.(x - U t) + phi_m) e_m
 *
 * with e_m = (-ky, kx)/|k| perpendicular to k_m, so every mode is
 * divergence free, and all modes sharing the same |k|^2. Then the
 * vorticity is |k|^2 times the stream function, the nonlinear term is a
 * gradient absorbed by the pressure, and the field is an exact solution
 * of the incompressible Navier-Stokes equations that decays with the
 * time constant 1/(nu |k|^2). The pressure is
 *
 *     p = -(|u - U|^2 + (sum_m a_m cos(k_m.x + phi_m))^2)/2 + mean
 *
 * decaying at twice the rate, normalized to zero mean, which assumes
 * the modes have distinct wave vectors (k_m != +-k_n, asserted). The
 * uniform drift U (zero by default) moves the pattern without changing
 * it, which a Galilean invariant scheme must reproduce.
 *
 * One mode is the shear wave, the classic viscosity measurement; two
 * modes (nx, ny) and (nx, -ny) are the Taylor-Green vortex. Applying
 * body_force() holds the time-zero field steady: with one mode along y
 * that is Kolmogorov flow, with the Taylor-Green pair the four-roll mill.
 */
template<typename T>
struct mode {
    T amplitude;   // velocity amplitude
    int nx, ny;    // mode numbers on the box
    T phase{};
};

template<typename T>
struct shear_modes {

    periodic_box<T> box;
    std::vector<mode<T>> modes;
    T nu;
    std::array<T,2> drift;

    shear_modes(periodic_box<T> box, std::vector<mode<T>> modes, T nu,
                std::array<T,2> drift = {})
        : box(box), modes(std::move(modes)), nu(nu), drift(drift)
    {
        assert(nu > 0 && "viscosity must be positive");
        assert(!this->modes.empty() && "at least one mode");
#ifndef NDEBUG
        for (std::size_t m = 0; m < this->modes.size(); ++m) {
            const auto& a = this->modes[m];
            assert((a.nx != 0 || a.ny != 0) && "mode numbers must not both be zero");
            for (std::size_t n = 0; n < m; ++n) {
                const auto& b = this->modes[n];
                assert(!((a.nx == b.nx && a.ny == b.ny) || (a.nx == -b.nx && a.ny == -b.ny))
                       && "wave vectors must be distinct up to sign");
            }
            assert(std::abs(ksqr_of(a) - ksqr()) <= T(1e-12)*ksqr()
                   && "all modes must share the same |k|^2");
        }
#endif
    }

    T ksqr() const { return ksqr_of(modes[0]); }
    T time_constant() const { return T(1.0)/(nu*ksqr()); }

    // Decay factor of the velocity at time t
    T decay(T time) const {
        assert(time >= 0 && "time must be non-negative");
        return std::exp(-time/time_constant());
    }

    // Time at which the velocity has dropped to the fraction frac
    T decay_time(T frac) const {
        assert(frac > 0 && frac <= 1 && "fraction must be in (0, 1]");
        return -time_constant()*std::log(frac);
    }

    // Viscosity recovered from the velocity amplitudes a0 and a1 measured
    // at the times t0 and t1: nu = -ln(a1/a0) / (|k|^2 (t1 - t0))
    T viscosity(T a0, T a1, T t0, T t1) const {
        assert(a1/a0 > 0 && "amplitudes must have the same sign");
        assert(t1 != t0 && "the two instants must differ");
        return -std::log(a1/a0)/(ksqr()*(t1 - t0));
    }

    std::tuple<T,T,T> operator()(std::array<T,2> xy, T time = {}) const {
        const T d = decay(time);
        auto [ux, uy, c] = pattern(xy, time);
        const T p = d*d*(-(ux*ux + uy*uy + c*c)/2 + mean_sqr());
        return std::make_tuple(p, drift[0] + d*ux, drift[1] + d*uy);
    }

    // Strain rate S = (grad u + grad u^T)/2 as {sxx, sxy, syy}
    std::array<T,3> stress_tensor(std::array<T,2> xy, T time = {}) const {
        const T d = decay(time);
        const T knorm = std::sqrt(ksqr());
        T sxx{}, sxy{};
        for (const auto& m : modes) {
            const auto [kx, ky] = box.wavenumber(m.nx, m.ny);
            const T c = m.amplitude*std::cos(theta(m, xy, time))/knorm;
            sxx -= c*kx*ky;
            sxy += c*(kx*kx - ky*ky)/2;
        }
        return {d*sxx, d*sxy, -d*sxx};
    }

    // Body force that holds the time-zero field steady (in the frame
    // drifting with U): f = nu |k|^2 (u(x, 0) - U) followed along the drift
    std::array<T,2> body_force(std::array<T,2> xy, T time = {}) const {
        auto [ux, uy, c] = pattern(xy, time);
        (void)c;
        return {nu*ksqr()*ux, nu*ksqr()*uy};
    }

private:

    T ksqr_of(const mode<T>& m) const {
        const auto [kx, ky] = box.wavenumber(m.nx, m.ny);
        return kx*kx + ky*ky;
    }

    // Phase of a mode at the point, followed along the drift
    T theta(const mode<T>& m, std::array<T,2> xy, T time) const {
        const auto [kx, ky] = box.wavenumber(m.nx, m.ny);
        return kx*(xy[0] - drift[0]*time) + ky*(xy[1] - drift[1]*time) + m.phase;
    }

    // Undecayed velocity of the modes and sum_m a_m cos(theta_m)
    std::tuple<T,T,T> pattern(std::array<T,2> xy, T time) const {
        const T knorm = std::sqrt(ksqr());
        T ux{}, uy{}, c{};
        for (const auto& m : modes) {
            const auto [kx, ky] = box.wavenumber(m.nx, m.ny);
            const T th = theta(m, xy, time);
            const T s = m.amplitude*std::sin(th);
            ux -= s*ky/knorm;
            uy += s*kx/knorm;
            c += m.amplitude*std::cos(th);
        }
        return std::make_tuple(ux, uy, c);
    }

    // Mean over the box of (|u|^2 + c^2)/2 at time zero
    T mean_sqr() const {
        T s{};
        for (const auto& m : modes) s += m.amplitude*m.amplitude;
        return s/2;
    }

}; // struct shear_modes

// One shear wave along the mode (nx, ny) with velocity amplitude u0
template<typename T>
shear_modes<T> shear_wave(periodic_box<T> box, int nx, int ny, T u0, T nu,
                          T phase = {}, std::array<T,2> drift = {}) {
    return shear_modes<T>(box, {{u0, nx, ny, phase}}, nu, drift);
}

// Taylor-Green vortex with the mode numbers (nx, ny) on the box:
//   ux = -u0 sqrt(ky/kx) cos(kx x) sin(ky y)
//   uy =  u0 sqrt(kx/ky) sin(kx x) cos(ky y)
//   p  = -u0^2/4 (ky/kx cos(2 kx x) + kx/ky cos(2 ky y))
template<typename T>
shear_modes<T> taylor_green(periodic_box<T> box, int nx, int ny, T u0, T nu,
                            std::array<T,2> drift = {}) {
    // the amplitudes take sqrt(ky/kx), sqrt(kx/ky) and sqrt(kx*ky)
    assert(nx*ny > 0 && "mode numbers must be nonzero and of the same sign");
    const auto [kx, ky] = box.wavenumber(nx, ny);
    // cos(kx x) cos(ky y) = [cos(kx x + ky y) + cos(kx x - ky y)]/2; the
    // sign carries the convention above, which flips with (kx, ky)
    const T a = (nx > 0 ? u0 : -u0)*std::sqrt((kx*kx + ky*ky)/(4*kx*ky));
    return shear_modes<T>(box, {{a, nx, ny, {}}, {a, nx, -ny, {}}}, nu, drift);
}


/*
 * ACOUSTIC WAVE
 *
 * A standing sound wave released from rest, in linear isothermal
 * acoustics with the equation of state p = cs^2 rho:
 *
 *     rho = rho0 (1 + delta cos(k.x + phi) exp(-gamma t)
 *                          (cos(W t) + gamma/W sin(W t)))
 *     u   = delta cs^2 |k|/W exp(-gamma t) sin(W t) sin(k.x + phi) k/|k|
 *
 * with the damping rate gamma = (nu + nu_bulk) |k|^2/2 (two dimensions)
 * and the frequency W = sqrt(cs^2 |k|^2 - gamma^2). The period gives the
 * sound speed, the damping the sum of shear and bulk viscosity, so with
 * the shear viscosity from a shear wave it measures the bulk viscosity.
 * The BGK collision on D2Q9 has nu_bulk = nu, the default. Valid for
 * delta << 1; the velocity amplitude is about delta cs.
 */
template<typename T>
struct acoustic_wave {

    periodic_box<T> box;
    int nx, ny;
    T delta, nu, nu_bulk, csqr, rho0, phase;

    acoustic_wave(periodic_box<T> box, int nx, int ny, T delta, T nu,
                  T nu_bulk = T(-1), T csqr = T(1.0)/3, T rho0 = T(1.0), T phase = {})
        : box(box), nx(nx), ny(ny), delta(delta), nu(nu),
          nu_bulk(nu_bulk < 0 ? nu : nu_bulk), csqr(csqr), rho0(rho0), phase(phase)
    {
        assert((nx != 0 || ny != 0) && "mode numbers must not both be zero");
        assert(delta > 0 && "relative density amplitude must be positive");
        assert(nu > 0 && "viscosity must be positive");
        assert(csqr > 0 && "squared sound speed must be positive");
        assert(rho0 > 0 && "reference density must be positive");
        assert(csqr*ksqr() > damping_rate()*damping_rate() && "wave must be underdamped");
    }

    T ksqr() const {
        const auto [kx, ky] = box.wavenumber(nx, ny);
        return kx*kx + ky*ky;
    }
    T sound_speed() const { return std::sqrt(csqr); }
    T damping_rate() const { return (nu + nu_bulk)*ksqr()/2; }
    T frequency() const { return std::sqrt(csqr*ksqr() - damping_rate()*damping_rate()); }
    T period() const { return 2*pi<T>/frequency(); }

    // Sum of shear and bulk viscosity from a measured damping rate
    T longitudinal_viscosity(T gamma) const { return 2*gamma/ksqr(); }

    std::tuple<T,T,T> operator()(std::array<T,2> xy, T time = {}) const {
        assert(time >= 0 && "time must be non-negative");
        const auto [kx, ky] = box.wavenumber(nx, ny);
        const T knorm = std::sqrt(ksqr());
        const T g = damping_rate(), w = frequency();
        const T th = kx*xy[0] + ky*xy[1] + phase;
        const T e = std::exp(-g*time);
        const T rho = rho0*(1 + delta*std::cos(th)*e*(std::cos(w*time) + g/w*std::sin(w*time)));
        const T u = delta*csqr*knorm/w*e*std::sin(w*time)*std::sin(th);
        return std::make_tuple(rho, u*kx/knorm, u*ky/knorm);
    }

}; // struct acoustic_wave


/*
 * DOUBLY PERIODIC SHEAR LAYER
 *
 * Two tanh shear layers of thickness ~1/k at y/Ly = 1/4 and 3/4, with
 * a small sinusoidal perturbation of amplitude delta in uy that rolls
 * the layers up into vortices (Minion & Brown, 1997). Initial condition
 * only; the pressure is returned as zero.
 */
template<typename T>
struct shear_layer {

    periodic_box<T> box;
    T u0, k, delta;

    shear_layer(periodic_box<T> box, T u0, T k = T(80.0), T delta = T(0.05))
        : box(box), u0(u0), k(k), delta(delta)
    {
        assert(k > 0 && "layer steepness must be positive");
    }

    std::tuple<T,T,T> operator()(std::array<T,2> xy) const {

        const T xd = xy[0]/box.Lx;
        const T yd = xy[1]/box.Ly;

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
