cmake_minimum_required(VERSION 3.10)

project(my_prog)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)  # Находим OpenMP
find_package(OpenCV REQUIRED)

add_executable(my_prog main.cpp)

if (MSVC)
    target_compile_options(my_prog PRIVATE /openmp)  # Включаем OpenMP для MSVC
    target_link_libraries(my_prog PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS})
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(my_prog PRIVATE -pthread -fopenmp)
    target_link_libraries(my_prog PRIVATE pthread OpenMP::OpenMP_CXX)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
