# mp1 consumer check (built out of tree against an installed prefix).
#
# The package job installs mp1 to a fresh prefix and configures this project
# with -DCMAKE_PREFIX_PATH=<prefix>.  It is deliberately NOT added with
# add_subdirectory() in test/CMakeLists.txt: the packaging policy forbids a
# build-tree export and requires find_package(mp1) against the installed prefix.
cmake_minimum_required(VERSION 3.21)

project(mp1_consumer LANGUAGES CXX)

# mp1::mp1 must upgrade this to C++20 through INTERFACE_COMPILE_FEATURES.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(mp1 1.0 CONFIG REQUIRED)

add_executable(consumer main.cc)
target_link_libraries(consumer PRIVATE mp1::mp1)
