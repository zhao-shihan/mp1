# mp1 is a header-only C++20 library.  CMake 3.21 provides PROJECT_IS_TOP_LEVEL.
cmake_minimum_required(VERSION 3.21)

# Version single source.  The top-level VERSION file is the only version
# literal in the build metadata, headers and tests; CMake derives the project
# and package version from it and generates mp1/version.hpp from
# mp1/version.hpp.in.  Edit VERSION, never this file or a header, to change the
# version.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" MP1_VERSION LIMIT_COUNT 1)
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" MP1_VERSION_FILE_CONTENT)
string(STRIP "${MP1_VERSION}" MP1_VERSION)
if(NOT MP1_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR
    "mp1: VERSION must contain a major.minor.patch version; got "
    "'${MP1_VERSION}'")
endif()
string(STRIP "${MP1_VERSION_FILE_CONTENT}" MP1_VERSION_FILE_CONTENT)
if(NOT MP1_VERSION_FILE_CONTENT STREQUAL MP1_VERSION)
  message(FATAL_ERROR
    "mp1: VERSION must contain exactly one version line; got "
    "'${MP1_VERSION_FILE_CONTENT}'")
endif()
project(mp1 VERSION ${MP1_VERSION} LANGUAGES CXX)

# A release edits VERSION; make the build system rerun the configure step so
# the generated header and the package version cannot go stale.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/VERSION")

# Generated public header.  configure_file runs before any target compiles;
# the generated include directory is on the interface target ahead of the
# source directory (see target_include_directories below), so the generated
# header is the only mp1/version.hpp visible in the build tree.
set(MP1_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${MP1_GENERATED_INCLUDE_DIR}/mp1")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/mp1/version.hpp.in"
    "${MP1_GENERATED_INCLUDE_DIR}/mp1/version.hpp"
    @ONLY)

# C++20 baseline.  The target raises consumers through cxx_std_20; the checks
# below fail closed at configure time instead of producing thousands of
# diagnostics from an unsupported compiler.  GCC 13 and Clang 16 are the
# minimums for the C++20 library facilities adopted by the 1.0 toolchain
# (source_location plus the libstdc++ 13 / conforming libc++ <format> floor);
# MSVC 19.32 is Visual Studio 2022 17.2.
if(NOT "cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  message(FATAL_ERROR
    "mp1 requires a C++20 compiler, but ${CMAKE_CXX_COMPILER_ID} "
    "${CMAKE_CXX_COMPILER_VERSION} does not advertise CMAKE_CXX_COMPILE_FEATURES "
    "cxx_std_20")
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
    message(FATAL_ERROR
      "mp1 requires GCC 13 or newer for C++20 (found "
      "${CMAKE_CXX_COMPILER_VERSION}); the error layer uses <format>")
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 16)
    message(FATAL_ERROR
      "mp1 requires Clang 16 or newer for C++20 (found "
      "${CMAKE_CXX_COMPILER_VERSION}); pair Clang with libstdc++ 13+ or a "
      "conforming libc++")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.32)
    message(FATAL_ERROR
      "mp1 requires MSVC 19.32 (Visual Studio 2022 17.2) or newer for C++20 "
      "(found ${CMAKE_CXX_COMPILER_VERSION})")
  endif()
endif()

option(MP1_BUILD_EXAMPLES "Build the mp1 examples" ${PROJECT_IS_TOP_LEVEL})
option(MP1_INSTALL "Install the mp1 headers and CMake package" ${PROJECT_IS_TOP_LEVEL})

if(NOT DEFINED CACHE{BUILD_TESTING})
  set(BUILD_TESTING OFF CACHE BOOL "")
endif()
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Public-header manifest and its fail-closed checks.  This runs for every
# configure, including a parent project that adds mp1: a missing, stale or
# unlisted public header is a configuration error, not a late test failure.
include(cmake/MP1Headers.cmake)
include(cmake/MP1MpiVersion.cmake)
set(MP1_HEADER_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/mp1/headers.txt")
mp1_validate_header_manifest("${CMAKE_CURRENT_SOURCE_DIR}")

# The library requires C++20; consumers get the requirement from the target.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MP1_BUILD_EXAMPLES)
  # The two C examples link MPI::MPI_C, so the C language and component are
  # required when examples are enabled.  The library itself is C++-only.
  enable_language(C)
  find_package(MPI ${MP1_MIN_MPI_VERSION} REQUIRED COMPONENTS C CXX)
else()
  find_package(MPI ${MP1_MIN_MPI_VERSION} REQUIRED COMPONENTS CXX)
endif()

# Fail closed with the detected provider/version string, not only the generic
# find_package diagnostic (MPI 4.1 §9.1.1 version inquiries).
mp1_assert_mpi_version()

# The single public header-only target.  Every source header lives under mp1/;
# the umbrella is mp1/mp1.hpp and the implementation headers are mp1/impl/**.
# mp1/version.hpp is generated into MP1_GENERATED_INCLUDE_DIR.
add_library(mp1 INTERFACE)
add_library(mp1::mp1 ALIAS mp1)
target_compile_features(mp1 INTERFACE cxx_std_20)
target_include_directories(mp1
    INTERFACE
    $<BUILD_INTERFACE:${MP1_GENERATED_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(mp1 INTERFACE MPI::MPI_CXX)

if(MP1_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(BUILD_TESTING AND PROJECT_IS_TOP_LEVEL)
  add_subdirectory(test)
endif()

if(MP1_INSTALL)
  install(TARGETS mp1 EXPORT mp1Targets)

  # Install the source public header tree: mp1/mp1.hpp plus mp1/impl/**.
  # mp1/version.hpp is generated and installed explicitly below; the
  # mp1/version.hpp.in template is not installed.
  install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mp1
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      FILES_MATCHING PATTERN "*.hpp")

  install(FILES ${MP1_GENERATED_INCLUDE_DIR}/mp1/version.hpp
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mp1)

  install(EXPORT mp1Targets
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mp1
      NAMESPACE mp1::
      FILE mp1Targets.cmake)

  configure_package_config_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mp1Config.cmake.in
      ${CMAKE_CURRENT_BINARY_DIR}/mp1Config.cmake
      INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mp1)
  write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/mp1ConfigVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY SameMinorVersion
      ARCH_INDEPENDENT)
  install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/mp1Config.cmake
      ${CMAKE_CURRENT_BINARY_DIR}/mp1ConfigVersion.cmake
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mp1)

  install(FILES
      ${CMAKE_CURRENT_SOURCE_DIR}/COPYING
      ${CMAKE_CURRENT_SOURCE_DIR}/COPYING.BSD-3-Clause
      ${CMAKE_CURRENT_SOURCE_DIR}/LICENSING.md
      DESTINATION ${CMAKE_INSTALL_DATADIR}/doc/mp1)
endif()
