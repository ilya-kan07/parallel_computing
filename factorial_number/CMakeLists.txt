cmake_minimum_required(VERSION 3.10)

project(FactorialThreads)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(FactorialThreads main.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(FactorialThreads PRIVATE -pthread)
    target_link_libraries(FactorialThreads PRIVATE pthread)
endif()
