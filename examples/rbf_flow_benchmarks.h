#ifndef RBF_FLOW_BENCHMARKS_H
#define RBF_FLOW_BENCHMARKS_H

#include <array>
#include <cmath>
#include <tuple>

namespace flow_benchmarks {

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

    std::array<T,3> stress_tensor(std::array<T,2> xy, T time = {}) {

        auto [x,y] = xy;

        const T tfrac = -time/time_constant();
        const T half{0.5};

        T sxx = u0*std::sqrt(kx*ky)*std::sin(kx*x)*std::sin(ky*y)*std::exp(tfrac);
        T sxy = half*u0*(std::sqrt(kx*kx*kx/ky) - std::sqrt(ky*ky*ky/kx)) * 
            std::cos(kx*x) * std::cos(ky*y) * std::exp(tfrac);
        T syy = -sxx;

        // If kx = ky, the shear component sxy will be zero
        // TODO: verify the syy term

        return {sxx,sxy,syy};
    }

}; // struct taylor_green


template<typename T>
struct shear_layer {

    T u0, k, delta, L;

    auto operator()(std::array<T,2> xy) const {

        auto [x,y] = xy;

        //constexpr T pi = 4*std::atan(T(1.0));
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

    auto operator()(std::array<T,2> xy, T csqr) const {

        const T Rc_sqr = Rc*Rc;
        const T vMa_sqr = eps*eps/csqr;

        const T xr = xy[1] - center[1];
        const T yr = xy[2] - center[2];

        const T r_sqr = xr*xr + yr*yr;

        // Convected vortex, Eqs. (3) and (4)
        T ux = U0 - eps * (yr/Rc) * exp(-r_sqr/(2*Rc_sqr));
        T uy =      eps * (xr/Rc) * exp(-r_sqr/(2*Rc_sqr));

        // Barotropic density initialization, Eq. (20)
        T rho = rho0 * std::exp(-vMa_sqr * std::exp(-r_sqr/Rc_sqr) / 2);

        return std::make_tuple(rho,ux,uy);
    }
    
}; // struct barotropic_vortex


/*
template<typename T>
struct lid_driven_cavity {
    
};*/

/*
template<typename T>
struct regularized_cavity {
    
};*/

/*
template<typename T>
struct cylinder_array {
    
};*/

/*
template<typename T>
struct regular_channel {
    
};*/

/*
template<typename T>
struct wavy_channel {
    
};*/

/*
template<typename T>
struct isotropic_turbulence {
    
};*/

/*
template<typename T>
struct flat_plate {
    
};*/

/*
template<typename T>
struct cylinder_turek {
    
};*/

} // namespace flow_benchmarks

#endif /* RBF_FLOW_BENCMARKS_H */

