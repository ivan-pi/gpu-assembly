add_executable(test_reorder test_reorder.cpp)
target_link_libraries(test_reorder PRIVATE rbf)
add_test(NAME reorder COMMAND test_reorder)

add_executable(test_io test_io.cpp)
target_link_libraries(test_io PRIVATE rbf)
add_test(NAME io COMMAND test_io)

add_executable(test_spatial test_spatial.cpp)
target_link_libraries(test_spatial PRIVATE rbf)
add_test(NAME spatial COMMAND test_spatial)

add_executable(test_csr test_csr.f90)
target_link_libraries(test_csr PRIVATE rbf)
add_test(NAME csr COMMAND test_csr)

add_executable(test_ellpack test_ellpack.f90)
target_link_libraries(test_ellpack PRIVATE rbf)
add_test(NAME ellpack COMMAND test_ellpack)

add_executable(test_sell test_sell.f90)
target_link_libraries(test_sell PRIVATE rbf)
add_test(NAME sell COMMAND test_sell)

# The solver wrapper, from both sides of its interface: the C header
# and the Fortran module.
if(GPU_ASSEMBLY_ENABLE_SOLVER)
    add_executable(test_solver test_solver.cpp)
    target_link_libraries(test_solver PRIVATE rbf_solver)
    add_test(NAME solver COMMAND test_solver)

    add_executable(test_solver_fortran test_solver.f90)
    target_link_libraries(test_solver_fortran PRIVATE rbf_solver rbf)
    add_test(NAME solver_fortran COMMAND test_solver_fortran)
endif()
