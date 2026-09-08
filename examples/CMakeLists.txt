# Analytic flow fields on a doubly periodic box (docs/periodic_benchmarks.md).
# One target serves both languages: the Fortran module rbf_benchmarks is
# compiled into the library, and the include path exposes the header-only
# C++ twin rbf_flow_benchmarks.h alongside the Fortran .mod files.
add_library(rbf_benchmarks STATIC rbf_benchmarks.f90)
target_link_libraries(rbf_benchmarks PUBLIC rbf)
set_target_properties(rbf_benchmarks PROPERTIES
    Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/mod)
target_include_directories(rbf_benchmarks PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# bench_spmv times csr_mv and ellpack_mv on a matrix from
# data/tools/knn_stream.py (docs/solver.md). It wants OpenMP for its
# timer and thread count; without OpenMP it is not built.
if(OpenMP_Fortran_FOUND)
    add_executable(bench_spmv bench_spmv.f90)
    target_link_libraries(bench_spmv PRIVATE rbf OpenMP::OpenMP_Fortran)
endif()
