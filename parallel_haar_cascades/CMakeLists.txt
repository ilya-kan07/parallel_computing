cmake_minimum_required(VERSION 3.10)
project(MyProject)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(my_prog main.cpp)
target_link_libraries(my_prog PRIVATE ${OpenCV_LIBS}) # Используем PRIVATE
target_include_directories(my_prog PRIVATE ${OpenCV_INCLUDE_DIRS}) # Используем target_include_directories

if (MSVC)
    target_compile_options(my_prog PRIVATE /openmp)  # Включаем OpenMP для MSVC
    target_link_libraries(my_prog PRIVATE OpenMP::OpenMP_CXX)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(my_prog PRIVATE -pthread -fopenmp)
    target_link_libraries(my_prog PRIVATE pthread OpenMP::OpenMP_CXX)
endif()
