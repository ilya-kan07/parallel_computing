cmake_minimum_required(VERSION 3.10)

project(RaceCondition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(RaceCondition main.cpp)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(RaceCondition PRIVATE -pthread)
    target_link_libraries(RaceCondition PRIVATE pthread)
endif()
