cmake_minimum_required(VERSION 3.10)

project(my_prog)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(MPI_C_COMPILER "C:/Program Files/Microsoft MPI/Bin/mpicc.bat")
set(MPI_CXX_COMPILER "C:/Program Files/Microsoft MPI/Bin/mpicxx.bat")
set(MPIEXEC "C:/Program Files/Microsoft MPI/Bin/mpiexec.exe")
set(MPI_CXX_INCLUDE_DIRS "C:/Program Files (x86)/Microsoft SDKs/MPI/Include")
set(MPI_CXX_LIBRARIES "C:/Program Files (x86)/Microsoft SDKs/MPI/Lib/x64/msmpi.lib")

set(MPI_C_FOUND TRUE)
set(MPI_CXX_FOUND TRUE)

add_executable(my_prog main.cpp)

target_include_directories(my_prog PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(my_prog PRIVATE ${MPI_CXX_LIBRARIES} ${OpenCV_LIBS})

include_directories(${OpenCV_INCLUDE_DIRS})

target_compile_options(my_prog PRIVATE ${MPI_CXX_COMPILE_OPTIONS})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(my_prog PRIVATE OpenMP::OpenMP_CXX)
endif()

if (MSVC)
    target_compile_options(my_prog PRIVATE /EHsc)
endif()
