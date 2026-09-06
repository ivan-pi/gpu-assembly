#ifndef RBF_IO_VTK_H
#define RBF_IO_VTK_H

// Legacy (ASCII, non-XML) VTK output for point clouds.
//
//   write_lbm_vtk_polydata   points + velocity + density as POLYDATA

#include <cstddef>
#include <fstream>
#include <string>

#include "rbf_io.h"

namespace rbf::io {

namespace detail {

// The scalar type label the legacy VTK format expects for T.
template <class T> constexpr const char* vtk_type_name();
template <> constexpr const char* vtk_type_name<float>()  { return "float"; }
template <> constexpr const char* vtk_type_name<double>() { return "double"; }

} // namespace detail

// LBM output as legacy VTK POLYDATA, one vertex per node, with the
// velocity as a point vector field and the density as a point scalar.
// The z coordinate and z velocity are written as 0. Values are written
// at full precision for T; the file should be given the .vtk extension.
//
// SoA form: x, y, rho, ux, uy are arrays of n.
template <class T>
void write_lbm_vtk_polydata(const std::string& fname, std::size_t n,
                            const T* x, const T* y,
                            const T* rho, const T* ux, const T* uy)
{
    auto out = detail::open_out(fname);
    detail::full_precision<T>(out);
    const char* tn = detail::vtk_type_name<T>();

    out << "# vtk DataFile Version 2.0\n"
           "lbm pointcloud output\n"
           "ASCII\n"
           "DATASET POLYDATA\n";

    out << "POINTS " << n << ' ' << tn << '\n';
    for (std::size_t i = 0; i < n; ++i)
        out << x[i] << ' ' << y[i] << " 0\n";

    out << "POINT_DATA " << n << '\n';

    out << "VECTORS Velocity " << tn << '\n';
    for (std::size_t i = 0; i < n; ++i)
        out << ux[i] << ' ' << uy[i] << " 0\n";

    out << "SCALARS Density " << tn << " 1\n"
           "LOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < n; ++i)
        out << rho[i] << '\n';
}

// Interleaved form: p and vel are arrays of 2n laid out as
// {x0, y0, x1, y1, ...} and {ux0, uy0, ux1, uy1, ...}; rho is n.
template <class T>
void write_lbm_vtk_polydata(const std::string& fname, std::size_t n,
                            const T* p, const T* rho, const T* vel)
{
    auto out = detail::open_out(fname);
    detail::full_precision<T>(out);
    const char* tn = detail::vtk_type_name<T>();

    out << "# vtk DataFile Version 2.0\n"
           "lbm pointcloud output\n"
           "ASCII\n"
           "DATASET POLYDATA\n";

    out << "POINTS " << n << ' ' << tn << '\n';
    for (std::size_t i = 0; i < n; ++i)
        out << p[2*i] << ' ' << p[2*i + 1] << " 0\n";

    out << "POINT_DATA " << n << '\n';

    out << "VECTORS Velocity " << tn << '\n';
    for (std::size_t i = 0; i < n; ++i)
        out << vel[2*i] << ' ' << vel[2*i + 1] << " 0\n";

    out << "SCALARS Density " << tn << " 1\n"
           "LOOKUP_TABLE default\n";
    for (std::size_t i = 0; i < n; ++i)
        out << rho[i] << '\n';
}

} // namespace rbf::io

#endif // RBF_IO_VTK_H
