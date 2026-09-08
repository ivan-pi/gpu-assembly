add_executable(test_reorder test_reorder.cpp)
target_link_libraries(test_reorder PRIVATE rbf)
add_test(NAME reorder COMMAND test_reorder)

add_executable(test_io test_io.cpp)
target_link_libraries(test_io PRIVATE rbf)
add_test(NAME io COMMAND test_io)

add_executable(test_spatial test_spatial.cpp)
target_link_libraries(test_spatial PRIVATE rbf)
add_test(NAME spatial COMMAND test_spatial)

# rbf_ops.h, the operators of the device kernels prototyped on the host
add_executable(test_rbf_ops test_rbf_ops.cpp)
target_link_libraries(test_rbf_ops PRIVATE rbf)
add_test(NAME rbf_ops COMMAND test_rbf_ops)

# The RBF-FD weights need LAPACK; the target exists only when it was found
if(TARGET rbf_fd)
    add_executable(test_rbf_fd test_rbf_fd.f90)
    target_link_libraries(test_rbf_fd PRIVATE rbf_fd)
    # a k-nearest-neighbour case on the periodic box [0, 32)^2 (data/README.md)
    add_test(NAME rbf_fd
        COMMAND test_rbf_fd ${CMAKE_SOURCE_DIR}/data/poisson_32_21 32 32)
endif()

# reclu's own tests (third_party/reclu/README.md): the optimized
# kernels against their plain-loop references, and those against the
# netlib LAPACK found by dlopen, which the test skips if there is none
foreach(t test_lu test_solve test_solve_adversarial test_sy test_sy_naive_vs_lapack)
    add_executable(reclu_${t} ${CMAKE_SOURCE_DIR}/third_party/reclu/tests/${t}.cpp)
    target_link_libraries(reclu_${t} PRIVATE reclu ${CMAKE_DL_LIBS})
    target_compile_options(reclu_${t} PRIVATE -fopenmp-simd -fno-math-errno)
    add_test(NAME reclu_${t} COMMAND reclu_${t})
endforeach()
set_tests_properties(reclu_test_sy_naive_vs_lapack PROPERTIES SKIP_RETURN_CODE 77)
