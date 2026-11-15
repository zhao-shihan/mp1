# mp1 performance suite.
#
# Built only when the test tree defines MP1_BUILD_PERF=ON (default OFF).  Two kinds of
# artifacts live here:
#
#   * `perf_calls`: deterministic call-count gates using the wrap shim.  It is registered
#     as exactly ONE CTest entry (`perf_calls`) that runs the executable at ranks 2 and 4
#     through run_mpi_ranks.cmake.  It is not registered at all when the shim is unavailable:
#     a skipped call-count gate is not a passing one.
#   * `perf_collective`, `perf_reduce`, `perf_p2p`: nightly microbenchmarks.  They are built
#     with MP1_BUILD_PERF but are NOT registered in CTest unless the separate
#     MP1_BUILD_PERF_RATIOS option (default OFF) is set.
#
# Nightly ratio gate:
#
#   cmake -DMP1_BUILD_PERF=ON -DMP1_BUILD_PERF_RATIOS=ON ...
#   ctest -L nightly
#
# `perf_ratios` drives all three benchmarks at ranks 2 and 4 with `--gate`; every row whose
# ratio threshold is set must pass, and a sample spread above 20 % on either side is reported
# as `skip` (flaky), not `fail`.  The nightly driver uses nine samples per row: a five-sample
# median was too sensitive to scheduler noise at 1 KiB.  CSV/JSON artifacts land in
# `<build>/test/perf/perf-results/`.  A bare run without `--gate` only records numbers.
#
# Thresholds (recorded in every CSV/JSON row):
#   dense variable collectives  <= 1.30x raw native
#   holey/padded variable       <= 2.50x raw typed-derived
#   flat reduction              <= 1.15x raw same-functor op
#   p2p pointer isend/irecv     <= 1.15x raw native
#   peak RSS                    recorded, no absolute gate
# Call-count gates are exact and deterministic; see perf_calls.cc.

function(mp1_perf_add_executable name source)
  add_executable("mp1_${name}" "${source}")
  set_target_properties("mp1_${name}" PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)
  target_include_directories("mp1_${name}" PRIVATE
    "${PROJECT_SOURCE_DIR}/test/support")
  target_compile_options("mp1_${name}" PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
      -Wall -Wextra -Wpedantic -Wno-c++98-compat>
    $<$<CXX_COMPILER_ID:MSVC>:/permissive- /W4 /WX>)
  target_link_libraries("mp1_${name}" PRIVATE mp1_test_target)

  if(ARGC GREATER 2 AND ARGV2 STREQUAL "WITH_WRAP")
    set(wrap_supported FALSE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
      set(wrap_supported TRUE)
    endif()
    target_sources("mp1_${name}" PRIVATE
      "${PROJECT_SOURCE_DIR}/test/support/mpi_wrap.cc")
    if(wrap_supported)
      set(symbols_header "${PROJECT_SOURCE_DIR}/test/support/mpi_wrap_symbols.hpp")
      if(NOT EXISTS "${symbols_header}")
        message(FATAL_ERROR "mp1_perf_add_executable: missing ${symbols_header}")
      endif()
      file(STRINGS "${symbols_header}" symbol_lines
        REGEX "^MP1_WRAP_SYMBOL\\([A-Za-z0-9_]+\\)$")
      if(NOT symbol_lines)
        message(FATAL_ERROR
          "mp1_perf_add_executable: no MP1_WRAP_SYMBOL entries in ${symbols_header}")
      endif()
      set(wrap_options)
      foreach(line IN LISTS symbol_lines)
        string(REGEX REPLACE
          "^MP1_WRAP_SYMBOL\\(([A-Za-z0-9_]+)\\)$" "\\1" symbol "${line}")
        list(APPEND wrap_options "-Wl,--wrap=${symbol}")
      endforeach()
      target_compile_definitions("mp1_${name}" PRIVATE MP1_TEST_WRAP=1)
      target_link_options("mp1_${name}" PRIVATE ${wrap_options})
    else()
      target_compile_definitions("mp1_${name}" PRIVATE MP1_TEST_WRAP=0)
    endif()
  endif()
endfunction()

mp1_perf_add_executable(perf_calls perf_calls.cc WITH_WRAP)
mp1_perf_add_executable(perf_collective perf_collective.cc)
mp1_perf_add_executable(perf_reduce perf_reduce.cc)
mp1_perf_add_executable(perf_p2p perf_p2p.cc)

# Encode list-valued MPIEXEC flags with '|' because `cmake -P` receives scalar -D values.
string(REPLACE ";" "|" mp1_perf_preflags "${MPIEXEC_PREFLAGS}")
string(REPLACE ";" "|" mp1_perf_postflags "${MPIEXEC_POSTFLAGS}")

# One CTest entry for the deterministic gates, at ranks 2 and 4.  Register only when the
# wrap shim is genuinely active; otherwise the gate would silently skip.
if(MP1_TEST_WRAP_SUPPORTED)
add_test(NAME perf_calls
  COMMAND ${CMAKE_COMMAND}
    "-DMP1_PERF_MPIEXEC=${MPIEXEC_EXECUTABLE}"
    "-DMP1_PERF_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
    "-DMP1_PERF_PREFLAGS=${mp1_perf_preflags}"
    "-DMP1_PERF_POSTFLAGS=${mp1_perf_postflags}"
    "-DMP1_PERF_EXECUTABLE=$<TARGET_FILE:mp1_perf_calls>"
    "-DMP1_PERF_RANKS=2,4"
    "-DMP1_PERF_TIMEOUT=300"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/run_mpi_ranks.cmake")
set_tests_properties(perf_calls PROPERTIES
  PROCESSORS 4
  TIMEOUT 600
  RUN_SERIAL TRUE
  LABELS "perf-calls")
# The registration invariant lives in the parent directory; a subdirectory's
# tests are not in the parent's TESTS property, so publish the row explicitly.
set_property(GLOBAL APPEND PROPERTY MP1_REGISTERED_TEST_ROWS "perf_calls|perf-calls")
else()
  message(STATUS "perf_calls: not registered (wrap shim unavailable)")
endif()

# Nightly ratio suite.  A separate opt-in keeps timing out of every default build; the benchmarks
# are built by MP1_BUILD_PERF but only registered here.
option(MP1_BUILD_PERF_RATIOS "Register the nightly perf ratio CTest entry" OFF)
if(MP1_BUILD_PERF_RATIOS)
  add_test(NAME perf_ratios
    COMMAND ${CMAKE_COMMAND}
      "-DMP1_PERF_MPIEXEC=${MPIEXEC_EXECUTABLE}"
      "-DMP1_PERF_NUMPROC_FLAG=${MPIEXEC_NUMPROC_FLAG}"
      "-DMP1_PERF_PREFLAGS=${mp1_perf_preflags}"
      "-DMP1_PERF_POSTFLAGS=${mp1_perf_postflags}"
      "-DMP1_PERF_COLLECTIVE=$<TARGET_FILE:mp1_perf_collective>"
      "-DMP1_PERF_REDUCE=$<TARGET_FILE:mp1_perf_reduce>"
      "-DMP1_PERF_P2P=$<TARGET_FILE:mp1_perf_p2p>"
      "-DMP1_PERF_RESULTS_DIR=${CMAKE_CURRENT_BINARY_DIR}/perf-results"
      "-DMP1_PERF_RANKS=2,4"
      "-DMP1_PERF_REPEATS=9"
      "-DMP1_PERF_TIMEOUT=600"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/run_perf_ratios.cmake")
  set_tests_properties(perf_ratios PROPERTIES
    PROCESSORS 4
    TIMEOUT 1800
    RUN_SERIAL TRUE
    LABELS "perf-ratio;nightly")
  set_property(GLOBAL APPEND PROPERTY MP1_REGISTERED_TEST_ROWS
    "perf_ratios|perf-ratio,nightly")
endif()
