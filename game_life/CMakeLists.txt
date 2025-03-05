cmake_minimum_required(VERSION 3.10)

project(GameLife)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(GameLife main.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(GameLife PRIVATE -pthread)
    target_link_libraries(GameLife PRIVATE pthread)
endif()
