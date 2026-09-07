#ifndef RBF_IO_GNUPLOT_H
#define RBF_IO_GNUPLOT_H

// Whitespace-separated columns, the form gnuplot plots directly (and
// numpy.loadtxt, pandas and the like read as well).
//
//   write_columns   x y and any number of named per-node columns, one node per line
//
// The first line is a '#' comment naming the columns, which gnuplot skips:
//
//     # x y rho ux uy
//     0.5 0.5 1 0 0
//     ...
//
//     plot 'macros.dat' using 1:2:3 with points palette        # rho over (x, y)
//     plot 'macros.dat' using 1:2:4:5 with vectors             # velocity
//
// Assisted-by: Claude Fable 5.1

#include <cassert>
#include <cstddef>
#include <string>
#include <type_traits>

#include "rbf_io.h"

namespace rbf::io {

// Coordinates are (x[i * xy_stride], y[i * xy_stride]): xy_stride is the
// spacing of consecutive coordinates, so interleaved {x0, y0, x1, y1, ...}
// storage is written with x = p, y = p + 1, xy_stride = 2. Columns are the same length as the points, as a braced
// list or a container of Column<T>:
//
//     write_columns("macros.dat", n, x, y, {{"rho", rho}, {"ux", ux}, {"uy", uy}});
//     write_columns("rcond.dat", n, x, y, {{"rcond", rc}});
//
template <class T>
void write_columns(const std::string& fname, std::size_t n,
                   const T* x, const T* y,
                   List<Column<std::type_identity_t<T>>> columns,
                   std::size_t xy_stride = 1)
{
    using detail::num;
    assert(xy_stride >= 1);
    for ([[maybe_unused]] const auto& c : columns) assert(c.v && c.stride >= 1);

    auto out = detail::open_out(fname);

    out << "# x y";
    for (const auto& c : columns) out << ' ' << c.name;
    out << '\n';

    for (std::size_t i = 0; i < n; ++i) {
        out << num(x[i * xy_stride]) << ' ' << num(y[i * xy_stride]);
        for (const auto& c : columns) out << ' ' << num(c.v[i * c.stride]);
        out << '\n';
    }
}

} // namespace rbf::io

#endif // RBF_IO_GNUPLOT_H
