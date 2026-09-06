#ifndef RBF_IO_H
#define RBF_IO_H

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <array>

namespace rbf::io {

/**
 * \brief Read nodal coordinates
 *
 * Read the node coordinates defining a point cloud
 * from an ASCII text file.
 *
 * @param filename Name of the input file.
 * @param       xy Container for the point cloud coordinates. This
 *                 could be something like std::vector<std::array<T,2>>.
 *                 In case C compatibility is important consider storing
 *                 the points as a struct, e.g. 
 
 *                   Point = struct{ double x, y; };
 * 
 *                 The type used for the items, must support default
 *                 construction from an initializer list, e.g. {x,y}.
 */
template<typename ArrayOfStructs>
void readNodeCoordinates(const char *filename, ArrayOfStructs &xy) {

  using Struct = typename ArrayOfStructs::value_type;
  using      T = typename Struct::value_type;

  std::ifstream infile(filename);

  size_t n;
  infile >> n;

  std::cout << n << '\n';

  xy.clear();
  xy.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    T x, y;
    infile >> x >> y;
    xy.push_back({x,y});
  }

}

/**
 * \brief Read nodal connectivity graph
 *
 * Read the graph defining the connectivity between nodes
 * into compressed sparse row storage. The row and column
 * pointer containers should be dynamically resizable.
 * Use of an STL container like std::vector<IndexType> with
 * a chosen index type, is likely to be the easiest.
 * 
 * @param filename Name of the input file.
 * @param   rowPtr Container for the row pointers.
 * @param   colPtr Container for the column pointers.
 *                 
 */
template<typename List_t>
void readNodeGraphCsr(const char *filename, List_t &rowPtr, List_t &colPtr) {
    
  using index_t = typename List_t::value_type;
  
  std::ifstream infile(filename);

  std::string header;
  std::getline(infile,header);

  std::istringstream hstream(header);

  size_t n, nnz;
  hstream >> n >> nnz;

  rowPtr.clear();
  rowPtr.reserve(n+1);

  colPtr.clear();
  colPtr.reserve(nnz);

  index_t ptr = 0;
  rowPtr.push_back(ptr);

  std::string temp;
  while (std::getline(infile, temp)) {

      std::istringstream rowstream(temp);
      std::vector<index_t> row{std::istream_iterator<index_t>(rowstream),{}};

      std::copy(row.begin(),row.end(),std::back_inserter(colPtr));
      ptr += row.size();

      rowPtr.push_back(ptr);
  }
}


template<typename T, typename I>
void write_csr_to_matrix_market(int n, int nnz, const I *ia, const I *ja, const T *a, const char *filename)
{
  std::ofstream file(filename);

  file << "%%MatrixMarket matrix coordinate real general\n";

  file << n << ' ' << n << ' ' << nnz << '\n';
  for (int i = 0; i < n; i++) {
    for (int j = ia[i]; j < ia[i+1]; j++) {
      file << i+1 << ' ' << ja[j] + 1 << ' ' << a[j] << '\n'; 
    }
  }

}

} // namespace rbf::io


#endif // RBF_IO_H

