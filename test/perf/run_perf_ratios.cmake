# Nightly ratio driver: run every benchmark at ranks 2 and 4 with --gate, appending CSV rows
# and JSON lines under MP1_PERF_RESULTS_DIR.  Any threshold failure makes this CTest entry
# fail; a raw-baseline spread above 20 % is reported as `skip` (flaky) by the benchmark.
#
# Invoked by the `perf_ratios` CTest entry (label `nightly`) when MP1_BUILD_PERF_RATIOS=ON.
# It can also be called by hand:
#
#   cmake -DMP1_PERF_MPIEXEC=mpiexec -DMP1_PERF_NUMPROC_FLAG=-n \
#         -DMP1_PERF_COLLECTIVE=... -DMP1_PERF_REDUCE=... -DMP1_PERF_P2P=... \
#         -DMP1_PERF_RESULTS_DIR=... -P test/perf/run_perf_ratios.cmake

if(NOT DEFINED MP1_PERF_MPIEXEC
   OR NOT DEFINED MP1_PERF_COLLECTIVE
   OR NOT DEFINED MP1_PERF_REDUCE
   OR NOT DEFINED MP1_PERF_P2P)
  message(FATAL_ERROR "run_perf_ratios: benchmark/MPI paths not provided")
endif()

string(REPLACE "|" ";" mp1_preflags "${MP1_PERF_PREFLAGS}")
string(REPLACE "|" ";" mp1_postflags "${MP1_PERF_POSTFLAGS}")
string(REPLACE "," ";" mp1_ranks "${MP1_PERF_RANKS}")
if(NOT mp1_ranks)
  set(mp1_ranks 2 4)
endif()
if(NOT DEFINED MP1_PERF_REPEATS OR MP1_PERF_REPEATS STREQUAL "")
  set(MP1_PERF_REPEATS 5)
endif()
if(NOT DEFINED MP1_PERF_TIMEOUT OR MP1_PERF_TIMEOUT STREQUAL "")
  set(MP1_PERF_TIMEOUT 600)
endif()
if(NOT DEFINED MP1_PERF_RESULTS_DIR OR MP1_PERF_RESULTS_DIR STREQUAL "")
  set(MP1_PERF_RESULTS_DIR "${CMAKE_CURRENT_BINARY_DIR}/perf-results")
endif()
file(REMOVE_RECURSE "${MP1_PERF_RESULTS_DIR}")
file(MAKE_DIRECTORY "${MP1_PERF_RESULTS_DIR}")
message(STATUS "run_perf_ratios: artifacts in ${MP1_PERF_RESULTS_DIR}")

set(mp1_failed FALSE)

foreach(benchmark IN ITEMS collective reduce p2p)
  if(benchmark STREQUAL "collective")
    set(mp1_exe "${MP1_PERF_COLLECTIVE}")
  elseif(benchmark STREQUAL "reduce")
    set(mp1_exe "${MP1_PERF_REDUCE}")
  else()
    set(mp1_exe "${MP1_PERF_P2P}")
  endif()
  foreach(rank IN LISTS mp1_ranks)
    set(mp1_csv "${MP1_PERF_RESULTS_DIR}/perf_${benchmark}.csv")
    set(mp1_json "${MP1_PERF_RESULTS_DIR}/perf_${benchmark}.jsonl")
    message(STATUS
      "run_perf_ratios: ${benchmark} rank ${rank} -> ${mp1_csv}")
    execute_process(
      COMMAND "${MP1_PERF_MPIEXEC}" "${MP1_PERF_NUMPROC_FLAG}" "${rank}"
              ${mp1_preflags} "${mp1_exe}"
              --suite "perf_${benchmark}"
              --repeats "${MP1_PERF_REPEATS}"
              --csv "${mp1_csv}"
              --json "${mp1_json}"
              --gate
              ${mp1_postflags}
      RESULT_VARIABLE mp1_result
      TIMEOUT "${MP1_PERF_TIMEOUT}"
      ECHO_OUTPUT_VARIABLE
      ECHO_ERROR_VARIABLE)
    if(NOT mp1_result EQUAL 0)
      message(SEND_ERROR
        "run_perf_ratios: ${benchmark} at rank ${rank} failed (exit ${mp1_result})")
      set(mp1_failed TRUE)
    endif()
  endforeach()
endforeach()

if(mp1_failed)
  message(FATAL_ERROR
    "run_perf_ratios: at least one ratio threshold failed; inspect ${MP1_PERF_RESULTS_DIR}")
endif()
message(STATUS "run_perf_ratios: all ratio gates passed")
