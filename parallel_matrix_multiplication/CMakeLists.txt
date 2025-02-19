cmake_minimum_required(VERSION 3.10)

project(parallel_matrix_multiplication)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(parallel_matrix_multiplication main.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(parallel_matrix_multiplication PRIVATE -pthread)
    target_link_libraries(parallel_matrix_multiplication PRIVATE pthread)
endif()
