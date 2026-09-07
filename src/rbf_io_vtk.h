#ifndef RBF_IO_VTK_H
#define RBF_IO_VTK_H

// Legacy (ASCII, non-XML) VTK output for point clouds.
//
//   write_vtk_polydata   points + any number of named scalar and vector fields
//
// A node is written as one vertex, so the cloud renders as points, and every
// field lands in POINT_DATA under the name given. Files should be given the
// .vtk extension.
//
// Assisted-by: Claude Fable 5.1

#include <cassert>
#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

#include "rbf_io.h"

namespace rbf::io {

namespace detail {

// Legacy VTK reads array names as single whitespace-delimited tokens; a
// name with a space, or none at all, gives a file ParaView rejects.
inline bool vtk_name_ok(std::string_view name) {
    return !name.empty() && name.find_first_of(" \t\r\n") == std::string_view::npos;
}

// The scalar type label the legacy VTK format expects for T.
template <class T>
constexpr const char* vtk_type_name();
template <>
constexpr const char* vtk_type_name<float>() {
    return "float";
}
template <>
constexpr const char* vtk_type_name<double>() {
    return "double";
}

}  // namespace detail

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
// (x[i * xy_stride], y[i * xy_stride]) with z = 0: xy_stride is the
// spacing of consecutive coordinates, so interleaved {x0, y0, x1, y1, ...}
// storage is written with x = p, y = p + 1, xy_stride = 2. Fields are the
// same length as the points and are given as braced lists or containers of
// Column<T> and VtkVector<T>:
//
//     write_vtk_polydata("u.vtk", n, x, y, {{"u", u}, {"residual", r}});
//     write_vtk_polydata("flow.vtk", n, x, y, {{"p", p}}, {{"U", ux, uy}});
//
// T is deduced from the coordinates only (type_identity), so the lists need
// not name it. title is the file's second line, the free-text header the
// legacy format requires to be present and terminated by a newline; it may be
// empty. The format caps it at 256 characters including the newline, so at
// most 255 here, and it must not contain a line break.
template <class T>
void write_vtk_polydata(const std::string& fname,
                        std::size_t n,
                        const T* x,
                        const T* y,
                        List<Column<std::type_identity_t<T>>> scalars,
                        List<VtkVector<std::type_identity_t<T>>> vectors = {},
                        std::size_t xy_stride = 1,
                        std::string_view title = "rbf point cloud") {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "legacy VTK arrays are float or double");
    using detail::num;
    assert(xy_stride >= 1);
    assert(title.size() < 256 && title.find_first_of("\r\n") == std::string_view::npos);
    for ([[maybe_unused]] const auto& s : scalars) {
        assert(detail::vtk_name_ok(s.name) && "scalar name empty or contains whitespace");
        assert(s.v && s.stride >= 1);
    }
    for ([[maybe_unused]] const auto& v : vectors) {
        assert(detail::vtk_name_ok(v.name) && "vector name empty or contains whitespace");
        assert(v.x && v.y && v.stride >= 1);
    }
    auto out = detail::open_out(fname);
    const char* tn = detail::vtk_type_name<T>();

    out << "# vtk DataFile Version 2.0\n"
        << title << '\n'
        << "ASCII\n"
           "DATASET POLYDATA\n";

    out << "POINTS " << n << ' ' << tn << '\n';
    for (std::size_t i = 0; i < n; ++i)
        out << num(x[i * xy_stride]) << ' ' << num(y[i * xy_stride]) << " 0\n";

    // one vertex cell per point, else the cloud has no renderable geometry
    out << "VERTICES " << n << ' ' << 2 * n << '\n';
    for (std::size_t i = 0; i < n; ++i) out << "1 " << i << '\n';

    if (scalars.empty() && vectors.empty()) return;
    out << "POINT_DATA " << n << '\n';

    for (const auto& s : scalars) {
        out << "SCALARS " << s.name << ' ' << tn
            << " 1\n"
               "LOOKUP_TABLE default\n";
        for (std::size_t i = 0; i < n; ++i) out << num(s.v[i * s.stride]) << '\n';
    }
    for (const auto& v : vectors) {
        out << "VECTORS " << v.name << ' ' << tn << '\n';
        for (std::size_t i = 0; i < n; ++i)
            out << num(v.x[i * v.stride]) << ' ' << num(v.y[i * v.stride]) << " 0\n";
    }
}

}  // namespace rbf::io

#endif  // RBF_IO_VTK_H
