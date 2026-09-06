#ifndef RBF_IO_VTK_H
#define RBF_IO_VTK_H

#include <cstddef>
#include <fstream>

namespace rbf::io {

/**
 * @brief Write LBM output as VTK POLYDATA (sequential storage).
 * 
 * @param filepath The filename; if not ending with the extension .vtp,
 *                 it will be appended automatically.
 * @param n Number of points.
 * @param x Nodal coordinates, x-component.
 * @param y Nodal coordinates, y-component.
 * @param p Pointer to an array of size 2*n, with the coordinates
 *          layed out as `{x[0], y[0], x[1], y[1], ...}`
 * @param rho Density values
 * @param ux  Velocity values, x-component.
 * @param uy  Velocity values, y-component.
 */ 
template<typename T>
void writeLbmVtkPolydata(
    const char* filepath, 
    int n, 
    const T x[], const T y[],
    const T rho[], 
    const T ux[], const T uy[]) {

  using std::size_t;

  std::ofstream vtkfile(filepath);

  vtkfile << "# vtk DataFile Version 2.0\n";
  vtkfile << "lbm pointcloud output\n";
  vtkfile << "ASCII\n";
  vtkfile << "DATASET POLYDATA\n";

  vtkfile << "POINTS " << n << " float\n";
  
  for (int i = 0; i < n; ++i) {
    vtkfile << x[i] << " " << y[i] << " 0\n";
  }

  vtkfile << "POINT_DATA " << n << '\n';

  vtkfile << "VECTORS Velocity float\n";

  for (int i = 0; i < n; ++i) {
    vtkfile << ux[i] << " " << uy[i] << " 0\n";
  }

  vtkfile << "SCALARS Density float 1\n";
  vtkfile << "LOOKUP_TABLE default\n";

  for (int i = 0; i < n; ++i) {
    vtkfile << rho[i] << '\n';
  }

  vtkfile.close();
}

/**
 * @brief Write LBM output as VTK POLYDATA (inter-leaved storage).
 * 
 * @param filepath The filename; if not ending with the extension .vtp,
 *                 it will be appended automatically.
 * @param n Number of points.
 * @param p Nodal coordinates (interleaved storage), i.e. an array of size `2*n` with
 *          the coordinates laid out in memory as `{x[0], y[0], x[1], y[1], ...}`.
 * @param rho Density values, array of size n.
 * @param ux  Velocity values (interleaved storage), i.e. an array of size `2*n` with
 *            the velocity component laid out in memory as `{ux[0], uy[0], ux[1], uy[1], ...}`.
 */  
template<typename T>
void writeLbmVtkPolydata(
    const char* filepath, 
    int n, 
    const T p[],
    const T rho[], 
    const T vel[]) {

  std::ofstream vtkfile(filepath);

  vtkfile << "# vtk DataFile Version 2.0\n";
  vtkfile << "lbm pointcloud output\n";
  vtkfile << "ASCII\n";
  vtkfile << "DATASET POLYDATA\n";

  vtkfile << "POINTS " << n << " float\n";
  
  for (int i = 0; i < n; ++i) {
    vtkfile << p[2*i] << " " << p[2*i+1] << " 0\n";
  }

  vtkfile << "POINT_DATA " << n << '\n';

  vtkfile << "VECTORS Velocity float\n";

  for (int i = 0; i < n; ++i) {
    vtkfile << vel[2*i] << " " << vel[2*i+1] << " 0\n";
  }

  vtkfile << "SCALARS Density float 1\n";
  vtkfile << "LOOKUP_TABLE default\n";

  for (int i = 0; i < n; ++i) {
    vtkfile << rho[i] << '\n';
  }

  vtkfile.close();
}

} // namespace rbf::io

