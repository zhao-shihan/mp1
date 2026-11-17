# Self-check for the configure-time MPI minimum-version gate (cmake/MP1MpiVersion.cmake).
#
# MPI 4.1 §9.1.1 defines the compile-time MPI_VERSION/MPI_SUBVERSION macros and
# the runtime MPI_Get_version pair.  The project minimum is 4.1: 3.1 and 4.0 must
# be rejected by configure, while 4.1 and newer majors are
# accepted.  This check runs the real assertion in a child CMake process so a
# missing `message(FATAL_ERROR)` cannot silently turn the minimum into a warning.

cmake_minimum_required(VERSION 3.21)

set(probe "${CMAKE_CURRENT_LIST_DIR}/check-mpi-version-assert.cmake")

set(failures 0)

function(expect_assert version expected_result)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DMP1_FAKE_MPI_CXX_VERSION=${version}"
      -P "${probe}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  set(text "${stdout}\n${stderr}")
  if(expected_result STREQUAL "pass" AND NOT result STREQUAL "0")
    message(SEND_ERROR
      "mpi version self-check: ${version} should be accepted, got exit ${result}:\n${text}")
    math(EXPR failures "${failures} + 1")
    set(failures "${failures}" PARENT_SCOPE)
  elseif(expected_result STREQUAL "fail")
    if(result STREQUAL "0")
      message(SEND_ERROR
        "mpi version self-check: ${version} should be rejected, but configure accepted it")
      math(EXPR failures "${failures} + 1")
      set(failures "${failures}" PARENT_SCOPE)
    elseif(NOT text MATCHES "below|requires MPI|MPI_CXX_VERSION")
      message(SEND_ERROR
        "mpi version self-check: ${version} failed for the wrong reason:\n${text}")
      math(EXPR failures "${failures} + 1")
      set(failures "${failures}" PARENT_SCOPE)
    endif()
  endif()
endfunction()

# Below the minimum.
expect_assert("3.1" fail)
expect_assert("4.0" fail)
expect_assert("4.0.9" fail)
expect_assert("" fail)
expect_assert("unknown" fail)
# At or above the minimum, including the exact 4.1 leg.
expect_assert("4.1" pass)
expect_assert("4.1.0" pass)
expect_assert("4.3" pass)
expect_assert("5.0" pass)

if(failures GREATER 0)
  message(FATAL_ERROR "mpi version self-check failed (${failures} case(s))")
endif()
message(STATUS "mpi version self-check passed")
