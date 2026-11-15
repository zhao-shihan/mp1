# Compile every positive control in test/compile_pass/.
#
# Run as a CTest check:
#   cmake -DMP1_CP_CXX=... -DMP1_CP_INCLUDE_DIRS=... \
#         -DMP1_CP_SOURCE_DIR=... -DMP1_CP_WORK_DIR=... \
#         -P test/support/compile_pass_controls.cmake
#
# A probe-bound control is compiled by cmake/CompileFail.cmake when its probe
# runs.  This check compiles the whole directory instead, so a control whose
# compile-fail probe was removed (or renamed) cannot silently become inert: every
# file here is a positive compile check of a supported API shape.  It uses the
# same command line as the probe controls (`-std=c++20 -fsyntax-only
# -Werror=unused-result` plus the mp1 and MPI include directories).
#
# MP1_CP_INCLUDE_DIRS uses '|' as the list separator because it arrives through
# scalar -D values.

cmake_minimum_required(VERSION 3.21)

foreach(required IN ITEMS
    MP1_CP_CXX
    MP1_CP_INCLUDE_DIRS
    MP1_CP_SOURCE_DIR
    MP1_CP_WORK_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "compile_pass_controls.cmake: ${required} is required")
  endif()
endforeach()

file(GLOB controls "${MP1_CP_SOURCE_DIR}/*.cc")
list(SORT controls)
list(LENGTH controls control_count)
if(control_count EQUAL 0)
  message(FATAL_ERROR
    "compile_pass_controls.cmake: no controls in ${MP1_CP_SOURCE_DIR}")
endif()

string(REPLACE "|" ";" include_dirs "${MP1_CP_INCLUDE_DIRS}")
set(compile_flags -std=c++20 -fsyntax-only -Werror=unused-result)

file(MAKE_DIRECTORY "${MP1_CP_WORK_DIR}")

set(failures)
foreach(control IN LISTS controls)
  get_filename_component(name "${control}" NAME_WE)
  set(command "${MP1_CP_CXX}")
  list(APPEND command ${compile_flags})
  foreach(dir IN LISTS include_dirs)
    if(NOT dir STREQUAL "")
      list(APPEND command "-I${dir}")
    endif()
  endforeach()
  list(APPEND command "${control}")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${command}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)
  set(log "${MP1_CP_WORK_DIR}/${name}.log")
  file(WRITE "${log}"
    "command: ${command}\n"
    "exit: ${result}\n"
    "--- stdout ---\n${stdout}\n"
    "--- stderr ---\n${stderr}\n")
  if(NOT result STREQUAL "0")
    list(APPEND failures "${name}")
    message(STATUS "compile-pass control failed: ${control}")
    message(STATUS "${stderr}")
  endif()
endforeach()

if(failures)
  list(JOIN failures ", " failed_names)
  message(FATAL_ERROR
    "compile_pass_controls.cmake: ${failed_names} failed to compile; "
    "logs in ${MP1_CP_WORK_DIR}")
endif()

message(STATUS
  "compile-pass controls: ${control_count}/${control_count} compiled")
