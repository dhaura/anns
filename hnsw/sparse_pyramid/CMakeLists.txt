cmake_minimum_required(VERSION 3.12)
project(SparsePyramid)

# Use MPI
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include GrassRMA HNSW library
include_directories(${CMAKE_SOURCE_DIR}/grassRMA)
link_directories(${CMAKE_SOURCE_DIR}/grassRMA)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -ffast-math -fstrict-aliasing -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=native -funroll-loops -ffast-math -fstrict-aliasing -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -Wpedantic")

# Add the executable
add_executable(sparse_pyramid sparse_pyramid.cpp)
add_executable(sparse_pyramid_v2 sparse_pyramid_v2.cpp)
add_executable(sparse_pyramid_v3 sparse_pyramid_v3.cpp)
add_executable(sparse_pyramid_v4 sparse_pyramid_v4.cpp)

# Link against OpenMP and MPI
target_link_libraries(sparse_pyramid
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

target_link_libraries(sparse_pyramid_v2
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

target_link_libraries(sparse_pyramid_v3
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)

target_link_libraries(sparse_pyramid_v4
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
)
