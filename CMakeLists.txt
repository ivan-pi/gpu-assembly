cmake_minimum_required(VERSION 3.23)

project(gpu-assembly LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The CUDA parts (cuSolverDx assembly kernels, CUDA Fortran demo) are
# isolated behind this option so the host library and its tests build
# with any C++/Fortran toolchain.
option(GPU_ASSEMBLY_ENABLE_CUDA
    "Build the CUDA assembly kernels (requires the CUDA toolkit and MathDx)" OFF)

include(CTest)

find_package(OpenMP COMPONENTS CXX Fortran)

# Nanoflann (vendored)
add_library(nanoflann INTERFACE)
target_include_directories(nanoflann INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nanoflann)

# Host library: the Fortran curve-key module plus the header-only C++
# parts (rbf_reorder.h, rbf_nodeset.h, rbf_io.h). Consumers get the
# include paths and the Fortran runtime by linking `rbf`.
add_library(rbf STATIC src/rbf_ordering.F90)
target_include_directories(rbf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(rbf PUBLIC nanoflann)
if(OpenMP_Fortran_FOUND)
    target_link_libraries(rbf PRIVATE OpenMP::OpenMP_Fortran)
endif()
if(OpenMP_CXX_FOUND)
    # rbf_nodeset.h carries omp pragmas; consumers inherit the flags
    target_link_libraries(rbf PUBLIC OpenMP::OpenMP_CXX)
endif()

if(GPU_ASSEMBLY_ENABLE_CUDA)
    enable_language(CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 90) # rbf_operators.cpp instantiates SM<900>
    endif()

    # find cuSolverDx; honours MATHDX_ROOT / mathdx_ROOT / CMAKE_PREFIX_PATH
    find_package(mathdx REQUIRED COMPONENTS cusolverdx CONFIG)

    # rbf_operators.cpp is a CUDA translation unit despite its extension
    set_source_files_properties(src/rbf_operators.cpp PROPERTIES LANGUAGE CUDA)

    add_library(rbf_operators STATIC src/rbf_operators.cpp)
    target_include_directories(rbf_operators PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(rbf_operators PUBLIC mathdx::cusolverdx)
    set_target_properties(rbf_operators
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            INTERPROCEDURAL_OPTIMIZATION ON)

    # CUDA Fortran demo (one-stencil assembly + solve_dp kernel).
    # CMake has no CUDA Fortran language, so this is driven through the
    # Fortran compiler with -cuda and only works with nvfortran.
    if(CMAKE_Fortran_COMPILER_ID STREQUAL "NVHPC")
        add_executable(assemble_demo src/main.f90 src/rbf_cuda.f90)
        target_compile_options(assemble_demo PRIVATE
            $<$<COMPILE_LANGUAGE:Fortran>:-cuda>)
        target_link_options(assemble_demo PRIVATE -cuda)
        target_link_libraries(assemble_demo PRIVATE rbf_operators)
    else()
        message(WARNING
            "CUDA Fortran demo (main.f90, rbf_cuda.f90) skipped: "
            "requires the NVHPC Fortran compiler")
    endif()

    # NOTE: src/main.cpp is an API sketch, not yet a compiling
    # translation unit; add it here once completed.
endif()

if(BUILD_TESTING)
    add_subdirectory(test)
endif()
