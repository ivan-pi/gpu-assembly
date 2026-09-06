#ifndef RBF_IO_VTK_H
#define RBF_IO_VTK_H

// Legacy (ASCII, non-XML) VTK output for point clouds.
//
//   write_vtk_polydata       points + any number of named scalar and vector fields
//   write_lbm_vtk_polydata   the lattice-Boltzmann special case: Density, Velocity
//
// A node is written as one vertex, so the cloud renders as points, and every
// field lands in POINT_DATA under the name given. Values are written at full
// precision for T. Files should be given the .vtk extension.

#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include "rbf_io.h"

namespace rbf::io {

namespace detail {

// The scalar type label the legacy VTK format expects for T.
template <class T> constexpr const char* vtk_type_name();
template <> constexpr const char* vtk_type_name<float>()  { return "float"; }
template <> constexpr const char* vtk_type_name<double>() { return "double"; }

} // namespace detail

// A named per-node scalar: v[i * stride] is the value at node i.
template <class T>
struct VtkScalar {
    std::string_view name;
    const T* v;
    std::size_t stride = 1;
};

// A named per-node 2-d vector: (x[i * stride], y[i * stride]) at node i; the
// z component is written as 0. Interleaved {ux0, uy0, ux1, ...} storage is
// {name, vel, vel + 1, 2}.
template <class T>
struct VtkVector {
    std::string_view name;
    const T* x;
    const T* y;
    std::size_t stride = 1;
};

// Point cloud with fields as legacy VTK POLYDATA. Coordinates are
// (x[i * point_stride], y[i * point_stride]) with z = 0, so interleaved
// {x0, y0, x1, y1, ...} storage is written with x = p, y = p + 1,
// point_stride = 2. Fields are the same length as the points.
//
// T is deduced from the coordinates only (type_identity), so a vector of
// fields converts to the span and a braced list converts to the
// initializer_list below without either having to name T.
template <class T>
void write_vtk_polydata(const std::string& fname, std::size_t n,
                        const T* x, const T* y,
                        std::span<const VtkScalar<std::type_identity_t<T>>> scalars,
                        std::span<const VtkVector<std::type_identity_t<T>>> vectors = {},
                        std::size_t point_stride = 1)
{
    auto out = detail::open_out(fname);
    detail::full_precision<T>(out);
    const char* tn = detail::vtk_type_name<T>();

    out << "# vtk DataFile Version 2.0\n"
           "rbf point cloud\n"
           "ASCII\n"
           "DATASET POLYDATA\n";

    out << "POINTS " << n << ' ' << tn << '\n';
    for (std::size_t i = 0; i < n; ++i)
        out << x[i * point_stride] << ' ' << y[i * point_stride] << " 0\n";

    // one vertex cell per point, else the cloud has no renderable geometry
    out << "VERTICES " << n << ' ' << 2 * n << '\n';
    for (std::size_t i = 0; i < n; ++i)
        out << "1 " << i << '\n';

    if (scalars.empty() && vectors.empty()) return;
    out << "POINT_DATA " << n << '\n';

    for (const auto& s : scalars) {
        out << "SCALARS " << s.name << ' ' << tn << " 1\n"
               "LOOKUP_TABLE default\n";
        for (std::size_t i = 0; i < n; ++i)
            out << s.v[i * s.stride] << '\n';
    }
    for (const auto& v : vectors) {
        out << "VECTORS " << v.name << ' ' << tn << '\n';
        for (std::size_t i = 0; i < n; ++i)
            out << v.x[i * v.stride] << ' ' << v.y[i * v.stride] << " 0\n";
    }
}

// Same, with the fields listed in place:
//
//     write_vtk_polydata("u.vtk", n, x, y, {{"u", u}, {"residual", r}});
//     write_vtk_polydata("flow.vtk", n, x, y, {{"p", p}}, {{"U", ux, uy}});
//
template <class T>
void write_vtk_polydata(const std::string& fname, std::size_t n,
                        const T* x, const T* y,
                        std::initializer_list<VtkScalar<std::type_identity_t<T>>> scalars,
                        std::initializer_list<VtkVector<std::type_identity_t<T>>> vectors = {},
                        std::size_t point_stride = 1)
{
    write_vtk_polydata(fname, n, x, y,
                       std::span<const VtkScalar<T>>{scalars.begin(), scalars.size()},
                       std::span<const VtkVector<T>>{vectors.begin(), vectors.size()},
                       point_stride);
}

// Lattice-Boltzmann output: density as "Density", velocity as "Velocity".
//
// SoA form: x, y, rho, ux, uy are arrays of n.
template <class T>
void write_lbm_vtk_polydata(const std::string& fname, std::size_t n,
                            const T* x, const T* y,
                            const T* rho, const T* ux, const T* uy)
{
    write_vtk_polydata(fname, n, x, y, {{"Density", rho}}, {{"Velocity", ux, uy}});
}

// Interleaved form: p and vel are arrays of 2n laid out as
// {x0, y0, x1, y1, ...} and {ux0, uy0, ux1, uy1, ...}; rho is n.
template <class T>
void write_lbm_vtk_polydata(const std::string& fname, std::size_t n,
                            const T* p, const T* rho, const T* vel)
{
    write_vtk_polydata(fname, n, p, p + 1,
                       {{"Density", rho}}, {{"Velocity", vel, vel + 1, 2}}, 2);
}

} // namespace rbf::io

#endif // RBF_IO_VTK_H
