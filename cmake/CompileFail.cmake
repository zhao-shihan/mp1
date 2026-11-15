# Internal compile-fail runner, invoked by `mp1_add_compile_fail` as:
#
#   cmake -DMP1_CF_SOURCE=... -DMP1_CF_CXX=... \
#         -DMP1_CF_INCLUDE_DIRS=...|... -DMP1_CF_EXPECT=... \
#         -DMP1_CF_LOG=... [-DMP1_CF_CONTROL=...] [-DMP1_CF_FLAGS=...|...] \
#         -P cmake/CompileFail.cmake
#
# A probe passes only when all of the following hold:
#
#   1. the compiler runs and exits nonzero;
#   2. the diagnostic does not contain `fatal error:` (a missing header or a
#      broken include path is not the property under test);
#   3. at least one diagnostic is attributed to the probe source or an mp1
#      header;
#   4. the full diagnostic matches MP1_CF_EXPECT, a CMake regular expression
#      written against the *library-owned* stable part of the message;
#   5. the optional MP1_CF_CONTROL source compiles successfully with exactly the
#      same command line.
#
# A successful compilation, a compiler crash without a diagnostic, a missing
# input, or a control that does not compile is a test failure.  `LC_ALL=C`
# keeps compiler diagnostics free of translated prose and locale-specific
# quoting; the expected fragments are deliberately compiler-independent.
#
# `MP1_CF_INCLUDE_DIRS` and `MP1_CF_FLAGS` use '|' as the list separator
# because they arrive through scalar -D values.

cmake_minimum_required(VERSION 3.21)

foreach(required IN ITEMS
    MP1_CF_SOURCE
    MP1_CF_CXX
    MP1_CF_INCLUDE_DIRS
    MP1_CF_EXPECT
    MP1_CF_LOG)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "CompileFail.cmake: ${required} is required")
  endif()
endforeach()

string(REPLACE "|" ";" include_dirs "${MP1_CF_INCLUDE_DIRS}")
set(extra_flags)
if(MP1_CF_FLAGS)
  string(REPLACE "|" ";" extra_flags "${MP1_CF_FLAGS}")
endif()

function(mp1_cf_command out)
  set(command "${MP1_CF_CXX}"
    -std=c++20
    -fsyntax-only
    -Werror=unused-result)
  foreach(dir IN LISTS include_dirs)
    if(NOT dir STREQUAL "")
      list(APPEND command "-I${dir}")
    endif()
  endforeach()
  list(APPEND command ${extra_flags} "${MP1_CF_SOURCE}")
  set("${out}" "${command}" PARENT_SCOPE)
endfunction()

function(mp1_cf_run label source log)
  mp1_cf_command(command)
  list(REMOVE_AT command -1)
  list(APPEND command "${source}")

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env LC_ALL=C ${command}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr)

  get_filename_component(log_dir "${log}" DIRECTORY)
  file(MAKE_DIRECTORY "${log_dir}")
  file(WRITE "${log}"
    "command: ${command}\n"
    "exit: ${result}\n"
    "--- stdout ---\n${stdout}\n"
    "--- stderr ---\n${stderr}\n")

  set("${label}_result" "${result}" PARENT_SCOPE)
  set("${label}_diagnostics" "${stdout}\n${stderr}" PARENT_SCOPE)
endfunction()

# ---------------------------------------------------------------------------
# 1. The probe must fail to compile.
# ---------------------------------------------------------------------------
mp1_cf_run(probe "${MP1_CF_SOURCE}" "${MP1_CF_LOG}")

if(probe_result STREQUAL "0")
  message(FATAL_ERROR
    "CompileFail.cmake: probe compiled successfully (expected failure): "
    "${MP1_CF_SOURCE}; log: ${MP1_CF_LOG}")
endif()
if(NOT probe_result MATCHES "^-?[0-9]+$")
  message(FATAL_ERROR
    "CompileFail.cmake: could not run compiler '${MP1_CF_CXX}': ${probe_result}; "
    "log: ${MP1_CF_LOG}")
endif()

set(diagnostics "${probe_diagnostics}")

# ---------------------------------------------------------------------------
# 2. Reject a wrong reason: a missing include path is not an intended
#    diagnostic, even when the compiler exits nonzero.
# ---------------------------------------------------------------------------
if(diagnostics MATCHES "fatal error:")
  message(FATAL_ERROR
    "CompileFail.cmake: probe failed with a fatal error (missing header or "
    "broken include path), not the intended diagnostic: ${MP1_CF_SOURCE}; "
    "expected '${MP1_CF_EXPECT}'; log: ${MP1_CF_LOG}")
endif()

# ---------------------------------------------------------------------------
# 3. Require attribution to the probe or an mp1 header.  A failure wholly
#    inside a system header (or an unparsed command-line error) is not
#    evidence about the library contract.
# ---------------------------------------------------------------------------
set(attributed FALSE)
if(diagnostics MATCHES "mp1/")
  set(attributed TRUE)
endif()
if(NOT attributed)
  string(FIND "${diagnostics}" "${MP1_CF_SOURCE}" position)
  if(position GREATER -1)
    set(attributed TRUE)
  endif()
endif()
if(NOT attributed)
  message(FATAL_ERROR
    "CompileFail.cmake: probe failed without a diagnostic attributed to the "
    "probe source or an mp1 header: ${MP1_CF_SOURCE}; log: ${MP1_CF_LOG}")
endif()

# ---------------------------------------------------------------------------
# 4. Require the expected diagnostic.
# ---------------------------------------------------------------------------
if(NOT diagnostics MATCHES "${MP1_CF_EXPECT}")
  message(FATAL_ERROR
    "CompileFail.cmake: diagnostic did not match expected regex "
    "'${MP1_CF_EXPECT}': ${MP1_CF_SOURCE}; log: ${MP1_CF_LOG}")
endif()

# ---------------------------------------------------------------------------
# 5. Positive control: the same command line must compile a valid source.
# ---------------------------------------------------------------------------
set(MP1_CF_CONTROL_SUFFIX "")
if(MP1_CF_CONTROL)
  set(MP1_CF_CONTROL_SUFFIX ", control ${MP1_CF_CONTROL}")
  mp1_cf_run(control "${MP1_CF_CONTROL}" "${MP1_CF_LOG}.control")
  if(NOT control_result STREQUAL "0")
    message(FATAL_ERROR
      "CompileFail.cmake: positive control failed to compile: "
      "${MP1_CF_CONTROL}; exit ${control_result}; log: ${MP1_CF_LOG}.control")
  endif()
  if(control_diagnostics MATCHES "fatal error:")
    message(FATAL_ERROR
      "CompileFail.cmake: positive control reported a fatal error: "
      "${MP1_CF_CONTROL}; log: ${MP1_CF_LOG}.control")
  endif()
endif()

message(STATUS
  "compile-fail probe passed: ${MP1_CF_SOURCE} "
  "(exit ${probe_result}, expected '${MP1_CF_EXPECT}'${MP1_CF_CONTROL_SUFFIX})")
