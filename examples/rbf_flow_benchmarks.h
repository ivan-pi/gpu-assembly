#ifndef RBF_FLOW_BENCHMARKS_H
#define RBF_FLOW_BENCHMARKS_H

/*
 * Analytic flow fields on a doubly periodic box, for verifying a lattice
 * Boltzmann implementation without boundary conditions. The physics and
 * the measurement recipes are in docs/periodic_benchmarks.md; the
 * Fortran module rbf_benchmarks offers the same cases.
 *
 *   auto [scalar, ux, uy] = case({x, y});         // any case
 *   auto [scalar, ux, uy] = case({x, y}, time);   // time-dependent cases
 *
 * The scalar is the pressure, or the density for acoustic_wave and
 * barotropic_vortex. Constructors assert their preconditions (compiled
 * out by NDEBUG, like the rest of the library).
 */

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <tuple>
#include <vector>

namespace flow_benchmarks {

/*
 * [0, Lx) x [0, Ly) with periodic images. The mode (nx, ny) has the wave
 * number (2 pi nx/Lx, 2 pi ny/Ly); in lattice units Lx = nx cells.
 */
template<typename T>
struct periodic_box {

    T Lx, Ly;

    periodic_box(T Lx, T Ly) : Lx(Lx), Ly(Ly) {
        assert(Lx > 0 && Ly > 0 && "box sides must be positive");
    }

    std::array<T,2> wavenumber(int nx, int ny) const {
        return {2*std::numbers::pi_v<T>*nx/Lx, 2*std::numbers::pi_v<T>*ny/Ly};
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
 * SHEAR MODES: shear waves sharing one |k|, an exact decaying solution
 * of the incompressible Navier-Stokes equations,
 *
 *     u = U + exp(-nu |k|^2 t) sum_m a_m sin(k_m.(x - U t) + phi_m) e_m,
 *     e_m = (-ky, kx)/|k|,
 *     p = -(|u - U|^2 + (sum_m a_m cos(...))^2)/2 + mean.
 *
 * One mode is the shear wave, the pair (nx, ny), (nx, -ny) the
 * Taylor-Green vortex; see shear_wave() and taylor_green() below.
 * body_force() gives the force that holds the time-zero field steady
 * (Kolmogorov flow, four-roll mill).
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
    std::array<T,2> drift;   // uniform velocity carrying the pattern

    shear_modes(periodic_box<T> box, std::vector<mode<T>> modes, T nu,
                std::array<T,2> drift = {})
        : box(box), modes(std::move(modes)), nu(nu), drift(drift)
    {
        assert(nu > 0 && "viscosity must be positive");
        assert(!this->modes.empty() && "at least one mode");
        ksqr_ = ksqr_of(this->modes[0]);
        const T knorm = std::sqrt(ksqr_);
        for (std::size_t m = 0; m < this->modes.size(); ++m) {
            assert((this->modes[m].nx != 0 || this->modes[m].ny != 0) && "mode numbers must not both be zero");
            for (std::size_t n = 0; n < m; ++n)
                assert(!same_line(this->modes[m], this->modes[n]) && "wave vectors must be distinct up to sign");
            assert(std::abs(ksqr_of(this->modes[m]) - ksqr_) <= 8*std::numeric_limits<T>::epsilon()*ksqr_
                   && "all modes must share the same |k|^2");
            const auto [kx, ky] = box.wavenumber(this->modes[m].nx, this->modes[m].ny);
            const T a = this->modes[m].amplitude;
            terms_.push_back({kx, ky, -a*ky/knorm, a*kx/knorm, a, this->modes[m].phase});
            mean_ += a*a/2;
        }
    }

    T ksqr() const { return ksqr_; }
    T time_constant() const { return T(1.0)/(nu*ksqr_); }

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

    std::tuple<T,T,T> operator()(std::array<T,2> xy, T time = {}) const {
        const T d = decay(time);
        const sums s = evaluate(xy, time);
        const T p = d*d*(-(s.ux*s.ux + s.uy*s.uy + s.c*s.c)/2 + mean_);
        return {p, drift[0] + d*s.ux, drift[1] + d*s.uy};
    }

    // Strain rate S = (grad u + grad u^T)/2 as {sxx, sxy, syy}
    std::array<T,3> stress_tensor(std::array<T,2> xy, T time = {}) const {
        const T d = decay(time);
        const sums s = evaluate(xy, time);
        return {d*s.sxx, d*s.sxy, -d*s.sxx};
    }

    // Body force holding the time-zero field steady in the drifting
    // frame: f = nu |k|^2 (u(x, 0) - U), followed along the drift
    std::array<T,2> body_force(std::array<T,2> xy, T time = {}) const {
        const sums s = evaluate(xy, time);
        return {nu*ksqr_*s.ux, nu*ksqr_*s.uy};
    }

private:

    struct term { T kx, ky, ex, ey, a, phase; };   // ex, ey = a e_m
    struct sums { T ux, uy, c, sxx, sxy; };        // undecayed, without drift

    std::vector<term> terms_;
    T ksqr_{}, mean_{};

    T ksqr_of(const mode<T>& m) const {
        const auto [kx, ky] = box.wavenumber(m.nx, m.ny);
        return kx*kx + ky*ky;
    }

    static bool same_line(const mode<T>& a, const mode<T>& b) {
        return (a.nx == b.nx && a.ny == b.ny) || (a.nx == -b.nx && a.ny == -b.ny);
    }

    sums evaluate(std::array<T,2> xy, T time) const {
        const T x = xy[0] - drift[0]*time, y = xy[1] - drift[1]*time;
        const T knorm = std::sqrt(ksqr_);
        sums s{};
        for (const term& t : terms_) {
            const T th = t.kx*x + t.ky*y + t.phase;
            const T sn = std::sin(th), cs = std::cos(th);
            s.ux += sn*t.ex;
            s.uy += sn*t.ey;
            s.c  += cs*t.a;
            s.sxx -= cs*t.a*t.kx*t.ky/knorm;
            s.sxy += cs*t.a*(t.kx*t.kx - t.ky*t.ky)/(2*knorm);
        }
        return s;
    }

}; // struct shear_modes

// One shear wave along the mode (nx, ny) with velocity amplitude u0
template<typename T>
shear_modes<T> shear_wave(periodic_box<T> box, int nx, int ny, T u0, T nu,
                          T phase = {}, std::array<T,2> drift = {}) {
    return shear_modes<T>(box, {{u0, nx, ny, phase}}, nu, drift);
}

// Taylor-Green vortex, the mode pair (nx, ny), (nx, -ny), which is
//   ux = -u0 sqrt(ky/kx) cos(kx x) sin(ky y)
//   uy =  u0 sqrt(kx/ky) sin(kx x) cos(ky y)
//   p  = -u0^2/4 (ky/kx cos(2 kx x) + kx/ky cos(2 ky y))
// with kx, ky the wave numbers of |nx|, |ny|; the signs of the mode
// numbers do not matter.
template<typename T>
shear_modes<T> taylor_green(periodic_box<T> box, int nx, int ny, T u0, T nu,
                            std::array<T,2> drift = {}) {
    assert(nx != 0 && ny != 0 && "Taylor-Green needs both mode numbers");
    nx = std::abs(nx); ny = std::abs(ny);
    const auto [kx, ky] = box.wavenumber(nx, ny);
    const T a = u0*std::sqrt((kx*kx + ky*ky)/(4*kx*ky));
    return shear_modes<T>(box, {{a, nx, ny, {}}, {a, nx, -ny, {}}}, nu, drift);
}


/*
 * ACOUSTIC WAVE: standing sound wave released from rest, in linear
 * isothermal acoustics (p = cs^2 rho, delta << 1),
 *
 *     rho = rho0 (1 + delta cos(k.x + phi) e^{-gamma t} (cos Wt + gamma/W sin Wt))
 *     u   = delta cs^2 |k|/W e^{-gamma t} sin Wt sin(k.x + phi) k/|k|
 *     gamma = (nu + nu_bulk) |k|^2/2,   W = sqrt(cs^2 |k|^2 - gamma^2)
 *
 * nu_bulk defaults to nu, the value of the BGK collision on D2Q9.
 */
template<typename T>
struct acoustic_wave {

    periodic_box<T> box;
    int nx, ny;
    T delta, nu, nu_bulk, csqr, rho0, phase;

    acoustic_wave(periodic_box<T> box, int nx, int ny, T delta, T nu,
                  std::optional<T> nu_bulk = {}, T csqr = T(1.0)/3, T rho0 = T(1.0), T phase = {})
        : box(box), nx(nx), ny(ny), delta(delta), nu(nu),
          nu_bulk(nu_bulk.value_or(nu)), csqr(csqr), rho0(rho0), phase(phase)
    {
        assert((nx != 0 || ny != 0) && "mode numbers must not both be zero");
        assert(delta > 0 && "relative density amplitude must be positive");
        assert(nu > 0 && this->nu_bulk >= 0 && "viscosities must be positive");
        assert(csqr > 0 && rho0 > 0 && "sound speed and density must be positive");
        const auto [kx, ky] = box.wavenumber(nx, ny);
        kx_ = kx; ky_ = ky;
        gamma_ = (nu + this->nu_bulk)*ksqr()/2;
        assert(csqr*ksqr() > gamma_*gamma_ && "wave must be underdamped");
        omega_ = std::sqrt(csqr*ksqr() - gamma_*gamma_);
    }

    T ksqr() const { return kx_*kx_ + ky_*ky_; }
    T damping_rate() const { return gamma_; }
    T frequency() const { return omega_; }
    T period() const { return 2*std::numbers::pi_v<T>/omega_; }

    std::tuple<T,T,T> operator()(std::array<T,2> xy, T time = {}) const {
        assert(time >= 0 && "time must be non-negative");
        const T th = kx_*xy[0] + ky_*xy[1] + phase;
        const T e = std::exp(-gamma_*time);
        const T rho = rho0*(1 + delta*std::cos(th)*e*(std::cos(omega_*time) + gamma_/omega_*std::sin(omega_*time)));
        const T u = delta*csqr/omega_*e*std::sin(omega_*time)*std::sin(th);   // |k| cancels against k/|k|
        return {rho, u*kx_, u*ky_};
    }

private:
    T kx_, ky_, gamma_, omega_;

}; // struct acoustic_wave


/*
 * SHEAR LAYER (Minion & Brown, 1997): two tanh layers at y/Ly = 1/4 and
 * 3/4 of thickness ~1/k, perturbed by uy = u0 delta sin(2 pi (x/Lx + 1/4)).
 * Initial condition only; the pressure is returned as zero.
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
        const T yd = xy[1]/box.Ly;
        const T ux = u0*std::tanh(k*(yd <= T(0.5) ? yd - T(0.25) : T(0.75) - yd));
        const T uy = u0*delta*std::sin(2*std::numbers::pi_v<T>*(xy[0]/box.Lx + T(0.25)));
        return {T{}, ux, uy};
    }

}; // struct shear_layer


/*
 * BAROTROPIC VORTEX (Wissocq, Boussuge & Sagaut, Phys. Rev. E 101,
 * 043306 (2020), Eqs. 3, 4, 20): Gaussian vortex of radius Rc and
 * strength eps convected at U0, with the density in balance with the
 * athermal equation of state p = cs^2 rho. Initial condition only; the
 * scalar returned is the density. Distances to the centre are taken
 * through the periodic images.
 */
template<typename T>
struct barotropic_vortex {

    periodic_box<T> box;
    T U0;
    std::array<T,2> center;
    T Rc, eps, rho0, csqr;

    barotropic_vortex(periodic_box<T> box, T U0, std::array<T,2> center, T Rc, T eps,
                      T rho0 = T(1.0), T csqr = T(1.0)/3 /* D2Q9, lattice units */)
        : box(box), U0(U0), center(center), Rc(Rc), eps(eps), rho0(rho0), csqr(csqr)
    {
        assert(Rc > 0 && "vortex radius must be positive");
        assert(rho0 > 0 && csqr > 0 && "density and sound speed must be positive");
    }

    std::tuple<T,T,T> operator()(std::array<T,2> xy) const {
        const auto [xr, yr] = box.minimum_image({xy[0] - center[0], xy[1] - center[1]});
        const T g = std::exp(-(xr*xr + yr*yr)/(2*Rc*Rc));
        const T ux = U0 - eps*(yr/Rc)*g;
        const T uy =      eps*(xr/Rc)*g;
        const T rho = rho0*std::exp(-eps*eps/csqr*g*g/2);
        return {rho, ux, uy};
    }

}; // struct barotropic_vortex

} // namespace flow_benchmarks

#endif /* RBF_FLOW_BENCHMARKS_H */
