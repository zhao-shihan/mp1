# MPI floor, single source.
#
# MPI 4.1 §9.1.1 (Version Inquiries) defines the compile-time
# MPI_VERSION/MPI_SUBVERSION macros and the runtime MPI_Get_version pair, and
# lists the valid pairs through (4,1).  The project floor is "MPI 4.1 or newer":
# a build is acceptable when the implementation advertises (4,1) or a later
# major version.  A 4.0 (or 3.x) implementation is below the floor and must
# fail configure, never silently produce a build.
#
# `MP1_MIN_MPI_VERSION` is used by find_package(MPI ...) and by the explicit
# assertion below.  ci/mpi-matrix.json records the implementation that is
# required to advertise exactly 4.1 (the floor leg); this module only enforces
# the lower bound, because a 5.0 provider is a valid newer-than-floor build.

set(MP1_MIN_MPI_VERSION "4.1")

# Non-fatal predicate used by the matrix/version self-check.
function(mp1_mpi_version_meets_floor version out_var)
  if(version STREQUAL "" OR NOT version MATCHES "^[0-9]+(\\.[0-9]+)+")
    set("${out_var}" FALSE PARENT_SCOPE)
    return()
  endif()
  if(version VERSION_LESS "${MP1_MIN_MPI_VERSION}")
    set("${out_var}" FALSE PARENT_SCOPE)
  else()
    set("${out_var}" TRUE PARENT_SCOPE)
  endif()
endfunction()

function(mp1_assert_mpi_version)
  if(NOT MPI_CXX_VERSION)
    message(FATAL_ERROR
      "mp1: find_package(MPI) did not report MPI_CXX_VERSION; this build "
      "requires MPI ${MP1_MIN_MPI_VERSION} or newer (MPI 4.1 §9.1.1). "
      "Detected library: '${MPI_CXX_LIBRARY_VERSION_STRING}' "
      "from '${MPI_CXX_COMPILER}'.")
  endif()
  mp1_mpi_version_meets_floor("${MPI_CXX_VERSION}" meets_floor)
  if(NOT meets_floor)
    message(FATAL_ERROR
      "mp1: this build requires MPI ${MP1_MIN_MPI_VERSION} or newer "
      "(MPI 4.1 §9.1.1 version inquiries); detected MPI_CXX_VERSION "
      "'${MPI_CXX_VERSION}' from '${MPI_CXX_COMPILER}' "
      "(library: '${MPI_CXX_LIBRARY_VERSION_STRING}'). "
      "Select a conforming mpicxx/mpiexec pair; do not lower the floor or "
      "skip the job.")
  endif()
  if(MPI_C_VERSION AND MPI_C_COMPILER)
    mp1_mpi_version_meets_floor("${MPI_C_VERSION}" c_meets_floor)
    if(NOT c_meets_floor)
      message(FATAL_ERROR
        "mp1: the C MPI compiler '${MPI_C_COMPILER}' reports "
        "MPI_C_VERSION '${MPI_C_VERSION}', below the C++ floor "
        "${MP1_MIN_MPI_VERSION}; the C and C++ compilers must come from the "
        "same conforming implementation.")
    endif()
  endif()
endfunction()
