# Child probe for ci/check-mpi-version.cmake: exercise the configure-time
# assertion with a synthetic MPI_CXX_VERSION.
cmake_minimum_required(VERSION 3.21)

if(NOT DEFINED MP1_FAKE_MPI_CXX_VERSION)
  message(FATAL_ERROR "MP1_FAKE_MPI_CXX_VERSION is required")
endif()

set(MPI_CXX_VERSION "${MP1_FAKE_MPI_CXX_VERSION}")
set(MPI_CXX_COMPILER "fake-mpicxx")
set(MPI_CXX_LIBRARY_VERSION_STRING "fake MPI library")
include("${CMAKE_CURRENT_LIST_DIR}/../cmake/MP1MpiVersion.cmake")
mp1_assert_mpi_version()
message(STATUS "mpi version self-check: accepted ${MPI_CXX_VERSION}")
