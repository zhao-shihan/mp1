# Self-check for cmake/CompileFail.cmake.
#
# Invoked by `harness_compile_fail_runner_selfcheck`:
#
#   cmake -DMP1_CF_RUNNER=... -DMP1_CF_CXX=... \
#         -DMP1_CF_INCLUDE_DIRS=...|... -DMP1_CF_SELFCHECK_DIR=... \
#         -DMP1_CF_WORK_DIR=... -P ci/check-compile-fail-selfcheck.cmake
#
# The runner is only trustworthy if it accepts the intended diagnostic and
# rejects every nearby wrong reason.  This matrix is the inverse control for
# the compile-fail probes: intended -> pass, wrong reason -> fail, successful
# compile -> fail, missing header -> fail, broken positive control -> fail,
# missing expectation -> fail.

cmake_minimum_required(VERSION 3.21)

foreach(required IN ITEMS
    MP1_CF_RUNNER
    MP1_CF_CXX
    MP1_CF_INCLUDE_DIRS
    MP1_CF_SELFCHECK_DIR
    MP1_CF_WORK_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR
      "check-compile-fail-selfcheck.cmake: ${required} is required")
  endif()
endforeach()

file(MAKE_DIRECTORY "${MP1_CF_WORK_DIR}")

function(run_runner name source expect control out_result out_text)
  set(command
    "${CMAKE_COMMAND}"
    "-DMP1_CF_SOURCE=${source}"
    "-DMP1_CF_CXX=${MP1_CF_CXX}"
    "-DMP1_CF_INCLUDE_DIRS=${MP1_CF_INCLUDE_DIRS}"
    "-DMP1_CF_LOG=${MP1_CF_WORK_DIR}/${name}.log")
  if(NOT expect STREQUAL "")
    list(APPEND command "-DMP1_CF_EXPECT=${expect}")
  endif()
  if(NOT control STREQUAL "")
    list(APPEND command "-DMP1_CF_CONTROL=${control}")
  endif()
  list(APPEND command -P "${MP1_CF_RUNNER}")

  execute_process(COMMAND ${command}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)

  set("${out_result}" "${result}" PARENT_SCOPE)
  set("${out_text}" "${stdout}\n${stderr}" PARENT_SCOPE)
endfunction()

set(failures 0)

function(expect_pass name)
  if(NOT result STREQUAL "0")
    message(SEND_ERROR
      "compile-fail self-check ${name}: expected PASS, got exit ${result}:\n${text}")
    math(EXPR failures "${failures} + 1")
    set(failures "${failures}" PARENT_SCOPE)
  else()
    message(STATUS "compile-fail self-check ${name}: PASS")
  endif()
endfunction()

function(expect_fail name pattern)
  if(result STREQUAL "0")
    message(SEND_ERROR
      "compile-fail self-check ${name}: expected FAIL, but the runner passed:\n${text}")
    math(EXPR failures "${failures} + 1")
    set(failures "${failures}" PARENT_SCOPE)
  elseif(NOT text MATCHES "${pattern}")
    message(SEND_ERROR
      "compile-fail self-check ${name}: failed for the wrong reason; "
      "expected /${pattern}/ in:\n${text}")
    math(EXPR failures "${failures} + 1")
    set(failures "${failures}" PARENT_SCOPE)
  else()
    message(STATUS "compile-fail self-check ${name}: PASS (rejected)")
  endif()
endfunction()

set(intended "${MP1_CF_SELFCHECK_DIR}/intended.cc")
set(wrong_reason "${MP1_CF_SELFCHECK_DIR}/wrong_reason.cc")
set(wrong_assert "${MP1_CF_SELFCHECK_DIR}/wrong_assert.cc")
set(ok_source "${MP1_CF_SELFCHECK_DIR}/ok.cc")
set(bad_include "${MP1_CF_SELFCHECK_DIR}/bad_include.cc")
set(control "${MP1_CF_SELFCHECK_DIR}/control.cc")
set(bad_control "${MP1_CF_SELFCHECK_DIR}/bad_control.cc")

set(intended_expect "mp1: SELF-CHECK-1 intended failure")

# 1. The intended diagnostic plus a valid positive control passes.
run_runner(intended_ok "${intended}" "${intended_expect}" "${control}"
  result text)
expect_pass(intended_ok)

# 2. A failure whose diagnostic does not match the expectation is rejected.
run_runner(wrong_reason "${wrong_reason}" "${intended_expect}" "${control}"
  result text)
expect_fail(wrong_reason "did not match expected regex")

# 3. A typo (wrong API name) is not the intended diagnostic.
run_runner(wrong_assert "${wrong_assert}" "${intended_expect}" "${control}"
  result text)
expect_fail(wrong_assert "did not match expected regex")

# 4. A successful compilation is rejected.
run_runner(compiles "${ok_source}" "${intended_expect}" ""
  result text)
expect_fail(compiles "compiled successfully")

# 5. A missing header is rejected as a fatal error, even before matching.
run_runner(bad_include "${bad_include}" "${intended_expect}" ""
  result text)
expect_fail(bad_include "fatal error")

# 6. A positive control that does not compile fails the probe.
run_runner(bad_control "${intended}" "${intended_expect}" "${bad_control}"
  result text)
expect_fail(bad_control "positive control failed to compile")

# 7. A probe cannot be registered without an expectation.
run_runner(missing_expect "${intended}" "" "${control}"
  result text)
expect_fail(missing_expect "MP1_CF_EXPECT is required")

if(failures GREATER 0)
  message(FATAL_ERROR
    "compile-fail runner self-check failed (${failures} case(s)); logs: "
    "${MP1_CF_WORK_DIR}")
endif()

message(STATUS "compile-fail runner self-check passed")
