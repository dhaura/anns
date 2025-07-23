cmake_minimum_required(VERSION 3.12)
project(SparsePyramid)

# Use MPI
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use pkg-config to find OpenCV4
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenCV REQUIRED opencv4)

# Include OpenCV headers and link dirs
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_CFLAGS_OTHER})

# Include GrassRMA HNSW library
include_directories(${CMAKE_SOURCE_DIR}/grassRMA)
link_directories(${CMAKE_SOURCE_DIR}/grassRMA)

# Add the executable
add_executable(sparse_pyramid sparse_pyramid.cpp)

# Link against OpenCV and MPI
target_link_libraries(sparse_pyramid
    ${OpenCV_LIBRARIES}
    MPI::MPI_CXX
)
