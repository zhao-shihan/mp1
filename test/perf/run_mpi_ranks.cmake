# Run the deterministic perf_calls executable at several ranks inside one CTest entry.
#
# This driver keeps one CTest entry while still requiring every rank to pass at each size.
# The caller registers the entry only when the wrap shim is active, so exit code 77 is a
# failure here: a skipped call-count gate is not a passing one.

if(NOT DEFINED MP1_PERF_MPIEXEC OR NOT DEFINED MP1_PERF_EXECUTABLE)
  message(FATAL_ERROR "run_mpi_ranks: MPIEXEC/EXECUTABLE not provided")
endif()

string(REPLACE "|" ";" mp1_preflags "${MP1_PERF_PREFLAGS}")
string(REPLACE "|" ";" mp1_postflags "${MP1_PERF_POSTFLAGS}")
string(REPLACE "," ";" mp1_ranks "${MP1_PERF_RANKS}")
if(NOT mp1_ranks)
  set(mp1_ranks 2 4)
endif()

if(NOT DEFINED MP1_PERF_TIMEOUT OR MP1_PERF_TIMEOUT STREQUAL "")
  set(MP1_PERF_TIMEOUT 300)
endif()

foreach(rank IN LISTS mp1_ranks)
  message(STATUS "run_mpi_ranks: ${MP1_PERF_EXECUTABLE} at rank ${rank}")
  execute_process(
    COMMAND "${MP1_PERF_MPIEXEC}" "${MP1_PERF_NUMPROC_FLAG}" "${rank}"
            ${mp1_preflags} "${MP1_PERF_EXECUTABLE}" ${mp1_postflags}
    RESULT_VARIABLE mp1_result
    TIMEOUT "${MP1_PERF_TIMEOUT}"
    ECHO_OUTPUT_VARIABLE
    ECHO_ERROR_VARIABLE)
  if(mp1_result EQUAL 77)
    message(FATAL_ERROR
      "run_mpi_ranks: rank ${rank} reported a skip (exit 77) but the gate is "
      "registered only when the wrap shim is active")
  endif()
  if(NOT mp1_result EQUAL 0)
    message(FATAL_ERROR
      "run_mpi_ranks: rank ${rank} failed with exit code ${mp1_result}")
  endif()
endforeach()
