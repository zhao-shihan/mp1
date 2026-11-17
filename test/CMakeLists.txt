# mp1 test registration (single glob + shared unit main).
#
# Every test/test_*.cc source is registered through the glob below and by
# default goes through `mp1_add_test(NAME <file> SOURCE <source>)`, which links
# the shared `mp1_test_main` (Boost.Test main defined in unit_test_main.cc) and
# registers CTest entries at ranks 2 and 4.  test_environment is the one source
# registered as `KIND scenario` in seven CTest entries (owned 2, external 1,
# external_single 1, concurrent_init 1, concurrent_finalize 1, none 1);
# `test_error_handler` (predefined error-handler control) and
# `file_external_finalize` (external-finalize close/release fallbacks) are one
# process-wide scenario at rank 1.  `test_spawn_contract` builds a second child
# executable (CHILD_SOURCE), passes its path as the parent's first argument and
# runs at ranks 1 and 2.  `test_version_contract` is a rank-1 scenario for the
# generated version macros; it makes no MPI call.  Every other test uses the
# shared unit main.
#
# CTest registration is not counted in prose.  The enforced invariant is the
# `harness_test_matrix` entry at the end of this file: it compares CTest's own
# --show-only=json-v1 name/label set with the set CMake registered here, so a
# registration that silently disappears (or a label that replaces the defaults)
# fails.  The generated set is the minimal fail-closed gate.
#
# `test/consumer` is deliberately not added here: it is built out of tree
# against an installed prefix.  `test_rma_shared` is a scenario because a
# provider or launcher may not expose a multi-process shared-memory domain; it
# returns 77 (CTest skip) in that case.  Only that entry declares the `shared`
# requirement, and a job that opts into `MP1_TEST_REQUIRE_FEATURES=shared`
# exports `MP1_REQUIRE_SHARED=1` for it so the skip becomes a failure.  The
# nightly `MP1_TEST_EXTRA_RANKS=ON` configuration additionally registers a
# 5-rank run of test_communicator_persistent_collective (asymmetric 3+2
# intercommunicator).

if(POLICY CMP0167)
  # The FindBoost module is removed. Boost 1.70.0 and later provide a
  # BoostConfig.cmake package configuration file.
  find_package(Boost 1.70.0 COMPONENTS unit_test_framework CONFIG REQUIRED)
else()
  # Use build-in FindBoost module.
  find_package(Boost COMPONENTS unit_test_framework REQUIRED)
endif()

# Extra rank counts for long-running configurations (nightly).  The default
# suite keeps the fixed 2/4-rank unit entries.
option(MP1_TEST_EXTRA_RANKS
  "Register extra rank counts for the nightly test configuration" OFF)

include(${PROJECT_SOURCE_DIR}/cmake/MP1Test.cmake)

# ---------------------------------------------------------------------------
# CTest execution parallelism (how many CTest entries run at once).
#
# This is independent of the MPI rank count of every entry.  Unit tests keep
# their fixed 2/4 rank coverage because those rank counts are protocol
# coverage, not a measurement of the host's CPU capacity.  mp1_add_test sets
# the PROCESSORS test property to the rank count instead, so `ctest -j N`
# accounts for the process slots a running test actually consumes.
#
# CTest has no cache variable for its default -j, so the configured value is
# applied at every layer that launches it:
#   * CMAKE_CTEST_ARGUMENTS makes `cmake --build <build> --target test`
#     (ninja/make test) run at MP1_TEST_JOBS.
#   * ci/run-ctest.sh is the CI and direct-run wrapper; it reads the
#     MP1_TEST_JOBS cache entry from the build tree.
# A bare `ctest` can only be made parallel through CTest's own interface:
# pass -j N or export CTEST_PARALLEL_LEVEL=N.  The generated `test` target
# passes -j explicitly, so CTEST_PARALLEL_LEVEL does not override it there.
# A -j/--parallel already present in CMAKE_CTEST_ARGUMENTS is treated as an
# explicit caller choice and suppresses the generated argument.
#
# Precedence: -DMP1_TEST_JOBS=N, then $ENV{MP1_TEST_JOBS}, then the portable
# CPU count.  ProcessorCount is implemented for Linux (nproc), macOS (sysctl)
# and Windows (NUMBER_OF_PROCESSORS); 0 (undetectable) falls back to 1.
# ---------------------------------------------------------------------------
include(ProcessorCount)
ProcessorCount(mp1_detected_jobs)
if(mp1_detected_jobs EQUAL 0)
  set(mp1_detected_jobs 1)
endif()
set(mp1_default_jobs "${mp1_detected_jobs}")
if(NOT "$ENV{MP1_TEST_JOBS}" STREQUAL "")
  set(mp1_default_jobs "$ENV{MP1_TEST_JOBS}")
endif()
set(MP1_TEST_JOBS "${mp1_default_jobs}" CACHE STRING
  "Concurrent CTest jobs (default: all detected logical CPUs)")
if(NOT MP1_TEST_JOBS MATCHES "^[1-9][0-9]*$")
  # Do not leave an invalid value in the cache.  A cached value wins over the
  # environment, so a failed configure would otherwise make every later
  # configure fail even after the user supplies a valid environment value.
  # Discarding it lets the next configure recover from -D, the environment or
  # the detected CPU count.
  set(mp1_invalid_jobs "${MP1_TEST_JOBS}")
  unset(MP1_TEST_JOBS CACHE)
  message(FATAL_ERROR
    "mp1: MP1_TEST_JOBS must be a positive integer, got '${mp1_invalid_jobs}'. "
    "The invalid value was discarded from the cache; reconfigure with "
    "-DMP1_TEST_JOBS=<n> or MP1_TEST_JOBS=<n> in the environment.")
endif()

# The generated `test` target is created in the top-level scope, so pass the
# argument list there.  Append rather than replace so a caller that already
# set CMAKE_CTEST_ARGUMENTS keeps its arguments, but never append a second job
# count: CTest uses the last -j/--parallel it receives, so appending ours
# would silently override an explicit caller -j.  Detect both the separate
# (-j 1, --parallel 1) and joined (-j1, --parallel=1) forms.
set(mp1_caller_sets_jobs FALSE)
foreach(mp1_ctest_argument IN LISTS CMAKE_CTEST_ARGUMENTS)
  if(mp1_ctest_argument MATCHES "^(--parallel|-j)([0-9]+|=.*)?$")
    set(mp1_caller_sets_jobs TRUE)
    break()
  endif()
endforeach()

if(mp1_caller_sets_jobs)
  message(STATUS
    "mp1: CTest jobs: CMAKE_CTEST_ARGUMENTS already sets -j; leaving it "
    "unchanged instead of appending -j ${MP1_TEST_JOBS}")
else()
  set(mp1_ctest_arguments ${CMAKE_CTEST_ARGUMENTS})
  list(APPEND mp1_ctest_arguments -j "${MP1_TEST_JOBS}")
  set(CMAKE_CTEST_ARGUMENTS "${mp1_ctest_arguments}" PARENT_SCOPE)
  message(STATUS "mp1: CTest jobs: ${MP1_TEST_JOBS}")
endif()

# Platform features that must be available for this configuration.  The list is
# empty by default: requiring a runtime capability is an opt-in per job.  A
# REQUIRES feature that is missing at configure time fails rather than skipping
# (do not silently skip the floor harness); a runtime-only feature is exported
# as MP1_REQUIRE_<FEATURE>=1 for the declaring entries so a 77 becomes a
# failure.
set(MP1_TEST_REQUIRE_FEATURES "" CACHE STRING
  "Semicolon-separated test platform requirements that must not skip")

# --wrap is available on Linux with GNU/Clang; the shim is used by the harness
# and perf call-count gates.
set(MP1_TEST_WRAP_SUPPORTED FALSE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
  set(MP1_TEST_WRAP_SUPPORTED TRUE)
endif()
if("wrap" IN_LIST MP1_TEST_REQUIRE_FEATURES AND NOT MP1_TEST_WRAP_SUPPORTED)
  message(FATAL_ERROR
    "mp1: MP1_TEST_REQUIRE_FEATURES lists 'wrap' but the --wrap shim is "
    "unavailable on this platform; the call-count harness would silently skip")
endif()

# Interface target for the tests.  It links the single public mp1::mp1 target,
# so every test builds against the same headers and requirements as a consumer.
add_library(mp1_test_target INTERFACE)
target_link_libraries(mp1_test_target INTERFACE mp1::mp1)
target_compile_features(mp1_test_target INTERFACE cxx_std_20)

# ---------------------------------------------------------------------------
# Compile-only gates (build-only; not CTest entries).
#
# 1. Every manifest header is compiled standalone, so a missing include in a
#    public header fails the build even if no consumer happens to include the
#    umbrella in that order.
# 2. Representative public-API TUs include the umbrella and name every public
#    family, so an advertised function that is not declared fails the build.
#    A failing target fails `cmake --build`, which is the gate.
# The manifest and the source list are post-surface: the topology/neighbor/grid
# headers and api_topology.cc are not part of this tree.
# ---------------------------------------------------------------------------
include(${PROJECT_SOURCE_DIR}/cmake/MP1Headers.cmake)
mp1_read_header_manifest(MP1_PUBLIC_HEADERS)
set(header_self_containment_sources)
foreach(header IN LISTS MP1_PUBLIC_HEADERS)
  string(MAKE_C_IDENTIFIER "${header}" header_id)
  set(generated_source "${CMAKE_CURRENT_BINARY_DIR}/header_check/${header_id}.cc")
  file(WRITE "${generated_source}"
    "// Generated by test/CMakeLists.txt: standalone include check.\n"
    "#include \"${header}\"\n")
  list(APPEND header_self_containment_sources "${generated_source}")
endforeach()

# mp1/version.hpp is generated by configure_file, not a source manifest entry,
# but it is part of the installed surface: compile it standalone exactly like a
# manifest header so it cannot start depending on another header's includes.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/header_check/generated_version.cpp"
  "// Generated by test/CMakeLists.txt: standalone include check.\n"
  "#include \"mp1/version.hpp\"\n")
list(APPEND header_self_containment_sources
  "${CMAKE_CURRENT_BINARY_DIR}/header_check/generated_version.cpp")

add_library(mp1_header_self_containment OBJECT ${header_self_containment_sources})
target_link_libraries(mp1_header_self_containment PRIVATE mp1_test_target)
set_target_properties(mp1_header_self_containment PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO)
target_compile_options(mp1_header_self_containment PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
    -Wall -Wextra -Wpedantic -Wno-c++98-compat>)

set(MP1_COMPILE_ALL_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_all/api_core.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_all/api_communicator.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_all/api_concepts.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_all/api_collectives.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_all/api_layout.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_all/api_layout_surface.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_all/api_vocabulary_surface.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_all/api_rma.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/compile_all/api_file_spawn.cc")
add_library(mp1_compile_all_api OBJECT ${MP1_COMPILE_ALL_SOURCES})
target_link_libraries(mp1_compile_all_api PRIVATE mp1_test_target)
set_target_properties(mp1_compile_all_api PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO)
target_compile_options(mp1_compile_all_api PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
    -Wall -Wextra -Wpedantic -Wno-c++98-compat>)

# The single Boost.Test main.  `unit_test_main.cc` is deliberately not named
# test_*.cc, so the glob below can never register it as a test or add a second
# main to a unit-test target.
add_library(mp1_test_main STATIC unit_test_main.cc)
target_compile_features(mp1_test_main PUBLIC cxx_std_20)
target_link_libraries(mp1_test_main PUBLIC Boost::unit_test_framework mp1::mp1)
target_compile_options(mp1_test_main PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
    -Wall -Wextra -Wpedantic -Wno-c++98-compat>
  $<$<CXX_COMPILER_ID:MSVC>:/permissive- /W4 /WX>)

# Unit tests.  Registration is automatic: a test exists iff its test_*.cc
# source exists.  test_environment is the one source registered four times (to
# exercise MPI environment modes without extra files).
file(GLOB MP1_TEST_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cc")
foreach(source IN LISTS MP1_TEST_SOURCES)
  get_filename_component(name "${source}" NAME_WE)

  if(name STREQUAL "test_environment")
    mp1_add_test(NAME test_environment_owned SOURCE "${source}"
      KIND scenario RANKS "2;4" LABELS environment
      ENVIRONMENT MP1_TEST_SCENARIO=owned)
    mp1_add_test(NAME test_environment_external SOURCE "${source}"
      KIND scenario RANKS "2" LABELS environment
      ENVIRONMENT MP1_TEST_SCENARIO=external)
    mp1_add_test(NAME test_environment_external_single SOURCE "${source}"
      KIND scenario RANKS "1" LABELS environment
      ENVIRONMENT MP1_TEST_SCENARIO=external_single)
    mp1_add_test(NAME test_environment_concurrent_init SOURCE "${source}"
      KIND scenario RANKS "1" LABELS environment
      ENVIRONMENT MP1_TEST_SCENARIO=concurrent_init)
    mp1_add_test(NAME test_environment_concurrent_finalize SOURCE "${source}"
      KIND scenario RANKS "1" LABELS environment
      ENVIRONMENT MP1_TEST_SCENARIO=concurrent_finalize)
    mp1_add_test(NAME test_environment_none SOURCE "${source}"
      KIND scenario RANKS "1" LABELS environment
      ENVIRONMENT MP1_TEST_SCENARIO=none)
  elseif(name STREQUAL "test_environment_owned_multiple")
    # The MPI 4.1 section 11.2.1 MULTIPLE-only MPI_Init configuration is reproduced by a
    # process-wide MPI_Init override in this executable, so it needs its own main and a
    # single rank.
    mp1_add_test(NAME "${name}" SOURCE "${source}"
      KIND scenario RANKS "1" LABELS environment)
  elseif(name STREQUAL "test_error_handler")
    # One process: probes before init and after finalize, so a separate main is required.
    mp1_add_test(NAME "${name}" SOURCE "${source}"
      KIND scenario RANKS "1" LABELS environment)
  elseif(name STREQUAL "test_rma_shared")
    # Shared-memory RMA needs one multi-process shared-memory domain.  The
    # scenario returns 77 (CTest skip) when the provider or launcher does not
    # expose one.  This is the only entry that declares `shared`; a job that
    # lists shared in MP1_TEST_REQUIRE_FEATURES exports MP1_REQUIRE_SHARED=1
    # here, and the test turns the skip into a failure.
    mp1_add_test(NAME "${name}" SOURCE "${source}"
      KIND scenario RANKS "2" LABELS "rma" REQUIRES shared)
  elseif(name STREQUAL "test_rma_contract")
    mp1_add_test(NAME "${name}" SOURCE "${source}" LABELS "rma")
  elseif(name STREQUAL "test_rma_shapes_contract")
    mp1_add_test(NAME "${name}" SOURCE "${source}" LABELS "rma")
  elseif(name STREQUAL "test_rma_homogeneous_contract")
    mp1_add_test(NAME "${name}" SOURCE "${source}" LABELS "rma")
  elseif(name STREQUAL "test_rma_count_contract")
    mp1_add_test(NAME "${name}" SOURCE "${source}" LABELS "rma")
  elseif(name STREQUAL "test_file")
    mp1_add_test(NAME "${name}" SOURCE "${source}" LABELS "file")
  elseif(name STREQUAL "test_file_dense_contract")
    mp1_add_test(NAME "${name}" SOURCE "${source}" LABELS "file")
  elseif(name STREQUAL "test_file_nonblocking_dense_contract")
    mp1_add_test(NAME "${name}" SOURCE "${source}" LABELS "file")
  elseif(name STREQUAL "test_file_typed_view_contract")
    mp1_add_test(NAME "${name}" SOURCE "${source}" LABELS "file")
  elseif(name STREQUAL "test_file_collapsed_surface")
    mp1_add_test(NAME "${name}" SOURCE "${source}" LABELS "file")
  elseif(name STREQUAL "test_file_spawn_contract")
    mp1_add_test(NAME "${name}" SOURCE "${source}" LABELS "file;spawn")
  elseif(name STREQUAL "test_spawn_contract")
    # Parent/child two-executable spawn suite (spawn contract): the child program is built by
    # CHILD_SOURCE and its path is passed as the parent test's first argument.  Rank 1 covers
    # the single-process parent; 2 and 4 are the gate ranks, and 4 makes the root-positional
    # and errcodes-on-every-rank properties observable at more than one parent rank.
    mp1_add_test(NAME "${name}" SOURCE "${source}"
      KIND scenario RANKS "1;2;4" LABELS "spawn"
      CHILD_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/spawn_child.cc" TIMEOUT 300)
  elseif(name STREQUAL "test_version_contract")
    # Version macros need no MPI rank or interoperability; one process proves
    # the generated string and the numeric macros agree.
    mp1_add_test(NAME "${name}" SOURCE "${source}"
      KIND scenario RANKS "1" LABELS "version;static")
  else()
    mp1_add_test(NAME "${name}" SOURCE "${source}")
  endif()
endforeach()

# Version consistency gate: the VERSION file, CMake's PROJECT_VERSION and the
# generated public macros must agree.  This is a configure-time artifact check,
# not a test of runtime behavior, so it runs once regardless of MPI ranks.
add_test(NAME version_matches_project
  COMMAND ${CMAKE_COMMAND}
    "-DMP1_VERSION_FILE=${PROJECT_SOURCE_DIR}/VERSION"
    "-DMP1_GENERATED_VERSION_HEADER=${MP1_GENERATED_INCLUDE_DIR}/mp1/version.hpp"
    "-DMP1_EXPECTED_PROJECT_VERSION=${PROJECT_VERSION}"
    -P "${PROJECT_SOURCE_DIR}/test/check_version.cmake")
set_tests_properties(version_matches_project PROPERTIES
  PROCESSORS 1
  TIMEOUT 60
  LABELS "version;static")

# Persistent collectives: the nightly extra-ranks configuration adds a 5-rank run of the
# intra/inter unit so the intercommunicator half is exercised with an asymmetric
# 3+2 split.  The default configuration keeps the 2/4-rank entries above.
if(MP1_TEST_EXTRA_RANKS AND TARGET mp1_test_communicator_persistent_collective)
  add_test(NAME "test_communicator_persistent_collective.5"
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 5
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:mp1_test_communicator_persistent_collective>
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties("test_communicator_persistent_collective.5" PROPERTIES
    PROCESSORS 5
    SKIP_RETURN_CODE 77
    TIMEOUT 180
    LABELS "unit;mpi")
endif()

# External-finalize file scenario: its own main finalizes MPI directly and checks the
# documented close/release/destructor fallbacks. One rank, no Boost.Test shared main.
mp1_add_test(NAME file_external_finalize
  SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/file_external_finalize.cc"
  KIND scenario RANKS "1" LABELS "file;scenario" TIMEOUT 60)

# Both spawn suites start extra processes with MPI_Comm_spawn.  PROCESSORS
# budgets only the parent ranks and RUN_SERIAL only keeps another CTest entry
# from running concurrently; neither reserves the slots consumed by the
# spawned children.  A launcher that enforces a slot/core limit can therefore
# still fail a spawn on a small host without oversubscription (Open MPI needs
# --oversubscribe, which the CI matrix supplies; MPICH/Hydra allows it by
# default).  Serializing these entries is still worthwhile: they are a small
# fraction of the suite, and it keeps their parent+child processes from
# competing with unrelated rank-4 entries.
get_property(mp1_spawn_tests DIRECTORY PROPERTY TESTS)
foreach(test_name IN LISTS mp1_spawn_tests)
  if(test_name MATCHES "^(test_spawn_contract|test_file_spawn_contract)\\.")
    set_tests_properties("${test_name}" PROPERTIES RUN_SERIAL TRUE)
  endif()
endforeach()

# Compile-fail probes.  Every probe must name the *library-owned* part of its
# intended diagnostic (EXPECT) and a positive control that compiles with the
# same flags (CONTROL).  cmake/CompileFail.cmake rejects a success, a missing
# header, an unattributed diagnostic, or an unmatched expectation.  The probe
# set lives in test/compile_fail/README.md; the controls in test/compile_pass/.
set(MP1_COMPILE_FAIL_PROBES
  "invalid_send_mode@test/compile_fail/invalid_send_mode.cc@mp1: send_mode_name: send_mode must be standard@test/compile_pass/send_mode_valid.cc"
  "nodiscard_request@test/compile_fail/nodiscard_request.cc@ignoring return value@test/compile_pass/request_consumed.cc"
  "no_inter_communicator@test/compile_fail/no_inter_communicator.cc@(is not a member of|no type named).*(mp1|inter.communicator)@test/compile_pass/communicator_names.cc"
  "no_member_collectives@test/compile_fail/no_member_collectives.cc@no member named .gather.@test/compile_pass/gather_free_function.cc"
  "rma_custom_functor@test/compile_fail/rma_custom_functor.cc@mp1: RMA only accepts a predefined MPI operation@test/compile_pass/rma_predefined_functor.cc"
  "rma_cas_double@test/compile_fail/rma_cas_double.cc@mp1: MPI_Compare_and_swap requires a predefined C[+][+] datatype@test/compile_pass/rma_predefined_atomic_families.cc"
  "rma_cas_character@test/compile_fail/rma_cas_character.cc@mp1: MPI_Compare_and_swap requires a predefined C[+][+] datatype@test/compile_pass/rma_predefined_atomic_families.cc"
  "rma_fop_derived@test/compile_fail/rma_fop_derived.cc@(mp1: MPI_Fetch_and_op requires a predefined datatype|mp1::predefined_op<T> requires a predefined operand type)@test/compile_pass/rma_predefined_atomic_families.cc"
  "rma_accumulate_mixed@test/compile_fail/rma_accumulate_mixed.cc@T has mixed basic components@test/compile_pass/rma_accumulate_homogeneous.cc"
  "rma_accumulate_mixed_pair@test/compile_fail/rma_accumulate_mixed_pair.cc@T has mixed basic components@test/compile_pass/rma_accumulate_homogeneous.cc"
  "rma_accumulate_mixed_replace@test/compile_fail/rma_accumulate_mixed_replace.cc@T has mixed basic components@test/compile_pass/rma_accumulate_homogeneous.cc"
  "rma_accumulate_mixed_nested@test/compile_fail/rma_accumulate_mixed_nested.cc@T has mixed basic components@test/compile_pass/rma_accumulate_homogeneous.cc"
  "rma_accumulate_mixed_widths@test/compile_fail/rma_accumulate_mixed_widths.cc@T has mixed basic components@test/compile_pass/rma_accumulate_homogeneous.cc"
  "rma_accumulate_empty@test/compile_fail/rma_accumulate_empty.cc@T has no basic components@test/compile_pass/rma_accumulate_homogeneous.cc"
  "rma_accumulate_opaque@test/compile_fail/rma_accumulate_opaque.cc@cannot determine T's basic component@test/compile_pass/rma_accumulate_custom_basic.cc"
  "rma_accumulate_custom_predefined_mixed@test/compile_fail/rma_accumulate_custom_predefined_mixed.cc@cannot determine T's basic component@test/compile_pass/rma_accumulate_homogeneous.cc"
  "rma_accumulate_predefined_op_aggregate@test/compile_fail/rma_accumulate_predefined_op_aggregate.cc@requires a predefined operand type T@test/compile_pass/rma_predefined_atomic_families.cc"
  "rma_accumulate_functor_operand_mismatch@test/compile_fail/rma_accumulate_functor_operand_mismatch.cc@mp1: RMA only accepts a predefined MPI operation@test/compile_pass/rma_accumulate_homogeneous.cc"
  "rma_accumulate_c_array_toplevel@test/compile_fail/rma_accumulate_c_array_toplevel.cc@a C array is not supported as the accumulating element type T@test/compile_pass/rma_accumulate_homogeneous.cc"
  "rma_pointer_origin_removed@test/compile_fail/rma_pointer_origin_removed.cc@no matching function for call to .*put@test/compile_pass/rma_span_region_forms.cc"
  "rma_pointer_layout_removed@test/compile_fail/rma_pointer_layout_removed.cc@no matching function for call to .*put@test/compile_pass/rma_span_region_forms.cc"
  "rma_const_result_removed@test/compile_fail/rma_const_result_removed.cc@no matching function for call to .*get@test/compile_pass/rma_span_region_forms.cc"
  "rma_container_origin_removed@test/compile_fail/rma_container_origin_removed.cc@no matching function for call to .*put@test/compile_pass/rma_span_region_forms.cc"
  "rma_origin_result_type_mismatch@test/compile_fail/rma_origin_result_type_mismatch.cc@no matching function for call to .*get_accumulate@test/compile_pass/rma_span_region_forms.cc"
  "rma_reflection_120_members@test/compile_fail/rma_reflection_120_members.cc@supports at most 119 members@test/compile_pass/rma_reflection_119_members.cc"
  "scalar_rvalue_isend@test/compile_fail/scalar_rvalue_isend.cc@no matching (member )?function for call to .*isend.*constraints not satisfied.*borrowable@test/compile_pass/scalar_lvalue_isend.cc"
  "counts_removed@test/compile_fail/counts_removed.cc@(is not a member of|no type named|no member named|no template named|does not name a (template )?type).*(mp1|counts)@test/compile_pass/vocabulary_removed_controls.cc"
  "scatter_raw_count@test/compile_fail/scatter_raw_count.cc@(no matching function|cannot convert|no known conversion).*(scatter|block_count)@test/compile_pass/scatter_block_count.cc"
  "capacity_raw_count@test/compile_fail/capacity_raw_count.cc@(no matching function|cannot convert|no known conversion).*(recv|capacity)@test/compile_pass/capacity_recv.cc"
  "capacity_count_t_4arg@test/compile_fail/capacity_count_t_4arg.cc@(deleted function|deleted).*recv@test/compile_pass/capacity_recv.cc"
  "owner_view_base_assignment@test/compile_fail/owner_view_base_assignment.cc@(cannot bind|discards qualifiers|drops .const. qualifier|no viable conversion|invalid initialization of non-const reference)@test/compile_pass/owner_view_borrow.cc"
  "owner_view_pointer@test/compile_fail/owner_view_pointer.cc@(cannot convert|invalid conversion|cannot initialize)@test/compile_pass/owner_view_borrow.cc"
  "file_nonblocking_write_rvalue@test/compile_fail/file_nonblocking_write_rvalue.cc@no matching (member )?function for call to .*iwrite.*constraints not satisfied.*borrowable@test/compile_pass/file_lvalue_iwrite.cc"
  "file_split_write_begin_rvalue@test/compile_fail/file_split_write_begin_rvalue.cc@no matching (member )?function for call to .*write_all_begin.*constraints not satisfied.*borrowable@test/compile_pass/file_lvalue_iwrite.cc"
  "igatherv_scalar_rvalue@test/compile_fail/igatherv_scalar_rvalue.cc@no matching (member )?function for call to .*igatherv.*constraints not satisfied.*borrowable@test/compile_pass/collectives_lvalue_gatherv.cc"
  "iallgatherv_scalar_rvalue@test/compile_fail/iallgatherv_scalar_rvalue.cc@no matching (member )?function for call to .*iallgatherv.*constraints not satisfied.*borrowable@test/compile_pass/collectives_lvalue_gatherv.cc"
  "isendrecv_scalar_rvalue@test/compile_fail/isendrecv_scalar_rvalue.cc@no matching (member )?function for call to .*isendrecv.*constraints not satisfied.*borrowable@test/compile_pass/isendrecv_lvalues.cc"
  "isendrecv_container_rvalue@test/compile_fail/isendrecv_container_rvalue.cc@no matching (member )?function for call to .*isendrecv.*constraints not satisfied.*borrowable@test/compile_pass/isendrecv_lvalues.cc"
  "recv_init_raw_count@test/compile_fail/recv_init_raw_count.cc@(use of deleted function|call to deleted function).*recv_init@test/compile_pass/capacity_recv_init.cc"
  "dense_igatherv_scalar_rvalue@test/compile_fail/dense_igatherv_scalar_rvalue.cc@no matching (member )?function for call to .*igatherv.*constraints not satisfied.*borrowable@test/compile_pass/dense_collectives_lvalues.cc"
  "dense_gatherv_init_scalar_rvalue@test/compile_fail/dense_gatherv_init_scalar_rvalue.cc@no matching (member )?function for call to .*gatherv_init.*constraints not satisfied.*borrowable@test/compile_pass/dense_collectives_lvalues.cc"
  "displacements_removed@test/compile_fail/displacements_removed.cc@(is not a member of|no type named|no member named|no template named|does not name a (template )?type).*(mp1|displacements)@test/compile_pass/vocabulary_removed_controls.cc"
  "layouts_removed@test/compile_fail/layouts_removed.cc@(is not a member of|no type named|no member named|no template named|does not name a (template )?type).*(mp1|layouts)@test/compile_pass/vocabulary_removed_controls.cc"
  "layout_view_removed@test/compile_fail/layout_view_removed.cc@(is not a member of|no type named|no member named|no template named|does not name a (template )?type).*(mp1|layout_view)@test/compile_pass/vocabulary_removed_controls.cc"
  "dense_iallgatherv_scalar_rvalue@test/compile_fail/dense_iallgatherv_scalar_rvalue.cc@no matching (member )?function for call to .*iallgatherv.*constraints not satisfied.*borrowable@test/compile_pass/dense_collectives_lvalues.cc"
  "dense_allgatherv_init_scalar_rvalue@test/compile_fail/dense_allgatherv_init_scalar_rvalue.cc@no matching (member )?function for call to .*allgatherv_init.*constraints not satisfied.*borrowable@test/compile_pass/dense_collectives_lvalues.cc"
  "typed_igatherv_scalar_rvalue@test/compile_fail/typed_igatherv_scalar_rvalue.cc@no matching (member )?function for call to .*igatherv.*constraints not satisfied.*borrowable@test/compile_pass/layouts_lvalue_iallgatherv.cc"
  "typed_allgatherv_init_scalar_rvalue@test/compile_fail/typed_allgatherv_init_scalar_rvalue.cc@no matching (member )?function for call to .*allgatherv_init.*constraints not satisfied.*borrowable@test/compile_pass/layouts_lvalue_iallgatherv.cc"
  "iscatter_container_in_place_rvalue@test/compile_fail/iscatter_container_in_place_rvalue.cc@no matching (member )?function for call to .*iscatter.*constraints not satisfied.*borrowable@test/compile_pass/iscatter_container_in_place_lvalue.cc"
  "persistent_collective_container@test/compile_fail/persistent_collective_container.cc@(no matching function|cannot convert|no known conversion).*reduce_init@test/compile_pass/persistent_collective_pointers.cc"
  "collective_gather_recv_in_place@test/compile_fail/collective_gather_recv_in_place.cc@no matching function for call to .*gather.*constraints not satisfied@test/compile_pass/collective_in_place_sides.cc"
  "collective_scatter_send_in_place@test/compile_fail/collective_scatter_send_in_place.cc@no matching function for call to .*scatter.*constraints not satisfied@test/compile_pass/collective_in_place_sides.cc"
  "collective_bcast_in_place@test/compile_fail/collective_bcast_in_place.cc@no matching function for call to .*bcast.*constraints not satisfied@test/compile_pass/collective_in_place_sides.cc"
  "collective_alltoall_recv_in_place@test/compile_fail/collective_alltoall_recv_in_place.cc@no matching function for call to .*alltoall.*constraints not satisfied@test/compile_pass/collective_in_place_sides.cc"
  "collective_reduce_local_in_place@test/compile_fail/collective_reduce_local_in_place.cc@no matching function for call to .*reduce_local.*constraints not satisfied@test/compile_pass/collective_in_place_sides.cc"
  "variable_gatherv_recv_in_place@test/compile_fail/variable_gatherv_recv_in_place.cc@no matching function for call to .*gatherv.*constraints not satisfied@test/compile_pass/collectives_lvalue_gatherv.cc"
  "variable_scatterv_send_in_place@test/compile_fail/variable_scatterv_send_in_place.cc@no matching function for call to .*scatterv.*constraints not satisfied@test/compile_pass/layouts_lvalue_iallgatherv.cc"
  "collective_igather_rvalue_send@test/compile_fail/collective_igather_rvalue_send.cc@no matching function for call to .*igather.*constraints not satisfied@test/compile_pass/collective_lvalue_init_forms.cc"
  "collective_init_container_send@test/compile_fail/collective_init_container_send.cc@no matching function for call to .*gather_init.*constraints not satisfied@test/compile_pass/collective_lvalue_init_forms.cc"
  "block_count_removed@test/compile_fail/block_count_removed.cc@(is not a member of|no member named|no type named).*(mp1|block_count)@test/compile_pass/scatter_block_count.cc"
  "container_element_datatype@test/compile_fail/container_element_datatype.cc@no matching function for call to .*send.*constraints not satisfied@test/compile_pass/container_element_datatype.cc"
  "make_layout_span_removed@test/compile_fail/make_layout_span_removed.cc@(is not a member of|no member named|no matching function).*(mp1|make_layout_span)@test/compile_pass/vocabulary_removed_controls.cc"
  "file_dense_vector_bool@test/compile_fail/file_dense_vector_bool.cc@(no matching function|constraints not satisfied).*iread_at@test/compile_pass/file_dense_nonblocking_lvalues.cc"
  "file_nonblocking_container_rvalue@test/compile_fail/file_nonblocking_container_rvalue.cc@no matching (member )?function for call to .*iwrite.*constraints not satisfied.*borrowable@test/compile_pass/file_dense_nonblocking_lvalues.cc"
  "file_write_sequence@test/compile_fail/file_write_sequence.cc@no matching (member )?function for call to .*write.*constraints not satisfied@test/compile_pass/file_dense_nonblocking_lvalues.cc"
  "file_non_element_type@test/compile_fail/file_non_element_type.cc@no matching (member )?function for call to .*get_type_extent.*constraints not satisfied.*element@test/compile_pass/file_element_type.cc"
  "request_span_removed@test/compile_fail/request_span_removed.cc@(is not a member of|no type named).*mp1@test/compile_pass/request_span_replacement.cc"
  "region_rvalue_layout@test/compile_fail/region_rvalue_layout.cc@(use of deleted function|call to deleted constructor).*region@test/compile_pass/region_lvalue_layout.cc"
  "region_ctad_rvalue_layout@test/compile_fail/region_ctad_rvalue_layout.cc@(use of deleted function|call to deleted constructor).*region@test/compile_pass/region_lvalue_layout.cc"
  "region_ctad_nullptr@test/compile_fail/region_ctad_nullptr.cc@(no matching function for call to .*region|no viable constructor or deduction guide.*region)@test/compile_pass/region_ctad_nullptr_explicit.cc"
  "region_array_temporary_layout@test/compile_fail/region_array_temporary_layout.cc@(use of deleted function|call to deleted constructor).*region@test/compile_pass/region_array_named_layout.cc"
  "isend_container_rvalue@test/compile_fail/isend_container_rvalue.cc@no matching (member )?function for call to .*isend.*constraints not satisfied.*borrowable@test/compile_pass/isend_container_lvalue.cc"
  "ibsend_scalar_rvalue@test/compile_fail/ibsend_scalar_rvalue.cc@no matching (member )?function for call to .*ibsend.*constraints not satisfied.*borrowable@test/compile_pass/ibsend_scalar_lvalue.cc"
  "issend_scalar_rvalue@test/compile_fail/issend_scalar_rvalue.cc@no matching (member )?function for call to .*issend.*constraints not satisfied.*borrowable@test/compile_pass/issend_scalar_lvalue.cc"
  "irsend_scalar_rvalue@test/compile_fail/irsend_scalar_rvalue.cc@no matching (member )?function for call to .*irsend.*constraints not satisfied.*borrowable@test/compile_pass/irsend_scalar_lvalue.cc"
  "send_init_container_rvalue@test/compile_fail/send_init_container_rvalue.cc@no matching (member )?function for call to .*send_init.*constraints not satisfied.*borrowable@test/compile_pass/send_init_container_lvalue.cc"
  "bsend_init_scalar_rvalue@test/compile_fail/bsend_init_scalar_rvalue.cc@no matching (member )?function for call to .*bsend_init.*constraints not satisfied.*borrowable@test/compile_pass/bsend_init_scalar_lvalue.cc"
  "ssend_init_scalar_rvalue@test/compile_fail/ssend_init_scalar_rvalue.cc@no matching (member )?function for call to .*ssend_init.*constraints not satisfied.*borrowable@test/compile_pass/ssend_init_scalar_lvalue.cc"
  "rsend_init_scalar_rvalue@test/compile_fail/rsend_init_scalar_rvalue.cc@no matching (member )?function for call to .*rsend_init.*constraints not satisfied.*borrowable@test/compile_pass/rsend_init_scalar_lvalue.cc"
  "p2p_char_literal_send@test/compile_fail/p2p_char_literal_send.cc@no matching function for call to .*send.*constraints not satisfied@test/compile_pass/p2p_send_string_view.cc"
  "p2p_char_array_recv@test/compile_fail/p2p_char_array_recv.cc@no matching function for call to .*recv.*constraints not satisfied@test/compile_pass/p2p_span_char_recv.cc"
  "p2p_sequence_recv@test/compile_fail/p2p_sequence_recv.cc@no matching function for call to .*recv.*constraints not satisfied@test/compile_pass/p2p_vector_recv.cc"
  "p2p_map_send@test/compile_fail/p2p_map_send.cc@no matching function for call to .*send.*constraints not satisfied@test/compile_pass/p2p_vector_pair_send.cc"
  "p2p_set_recv@test/compile_fail/p2p_set_recv.cc@no matching function for call to .*recv.*constraints not satisfied@test/compile_pass/p2p_vector_recv.cc"
  "p2p_vector_bool_send@test/compile_fail/p2p_vector_bool_send.cc@no matching function for call to .*send.*constraints not satisfied@test/compile_pass/p2p_bool_array_send.cc"
  "p2p_forward_list_send@test/compile_fail/p2p_forward_list_send.cc@no matching function for call to .*send.*constraints not satisfied@test/compile_pass/p2p_list_send.cc"
  "p2p_span_const_recv@test/compile_fail/p2p_span_const_recv.cc@no matching function for call to .*recv.*constraints not satisfied@test/compile_pass/p2p_span_recv.cc"
  "p2p_region_const_recv@test/compile_fail/p2p_region_const_recv.cc@no matching function for call to .*recv.*constraints not satisfied@test/compile_pass/p2p_region_recv.cc"
  "p2p_mrecv_rvalue_message@test/compile_fail/p2p_mrecv_rvalue_message.cc@no matching function for call to .*mrecv.*(cannot convert .*mp1::message.*to type .*mp1::message&|expects an lvalue for 2nd argument)@test/compile_pass/p2p_mrecv_lvalue_message.cc"
  "p2p_pointer_count_send@test/compile_fail/p2p_pointer_count_send.cc@no matching function for call to .*send.*constraints not satisfied@test/compile_pass/p2p_span_from_send.cc"
  "p2p_recv_rvalue_container@test/compile_fail/p2p_recv_rvalue_container.cc@no matching function for call to .*recv.*constraints not satisfied@test/compile_pass/p2p_vector_recv.cc"
  "layout_builder_braced_init@test/compile_fail/layout_builder_braced_init.cc@(no matching (function|constructor)|no viable constructor).*(indexed_parameter|hindexed_parameter|indexed_block_parameter|hindexed_block_parameter|subarray_parameter)@test/compile_pass/layout_builder_named_ranges.cc"
  "layout_empty_removed@test/compile_fail/layout_empty_removed.cc@(empty.*is not a member of.*mp1::layout|no member named .empty. in .*mp1::layout)@test/compile_pass/layout_zero_control.cc"
  "throwing_reduction_functor@test/compile_fail/throwing_reduction_functor.cc@mp1: a reduction functor used on the callback path must be nothrow-invocable@test/compile_pass/reduction_functor_noexcept.cc")
foreach(probe IN LISTS MP1_COMPILE_FAIL_PROBES)
  string(REPLACE "@" ";" fields "${probe}")
  list(GET fields 0 name)
  list(GET fields 1 source)
  list(GET fields 2 expect)
  list(GET fields 3 control)
  mp1_add_compile_fail(NAME "${name}" SOURCE "${PROJECT_SOURCE_DIR}/${source}"
    EXPECT "${expect}" CONTROL "${PROJECT_SOURCE_DIR}/${control}")
endforeach()

mp1_add_compile_fail(NAME harness_compile_fail_self_check
  SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/harness/compile_fail_self_check.cc"
  EXPECT "mp1 compile-fail runner self-check"
  CONTROL "${PROJECT_SOURCE_DIR}/test/compile_pass/self_check_control.cc")
# The runner self-check is a harness gate, not one of the contract
# probes.  Label it `harness` so `ctest -N -LE harness` counts contract entries
# only; the probe itself remains in the compile-fail label for CI.
set_tests_properties(harness_compile_fail_self_check PROPERTIES
  LABELS "compile-fail;harness")

# Inverse control for the runner: intended -> pass, wrong reason / successful
# compile / missing header / broken control / missing EXPECT -> fail.  This is
# a CTest scenario rather than a compile-fail probe so it can invoke the runner
# matrix itself.
# The compile-fail runner invokes the compiler directly, not through the mp1
# target, so it needs the generated include directory explicitly: every probe
# includes the umbrella, which includes the generated mp1/version.hpp.
set(compile_fail_include_dirs
  "${MP1_GENERATED_INCLUDE_DIR}"
  "${PROJECT_SOURCE_DIR}")
if(MPI_CXX_INCLUDE_DIRS)
  list(APPEND compile_fail_include_dirs ${MPI_CXX_INCLUDE_DIRS})
endif()
string(REPLACE ";" "|" compile_fail_include_dirs "${compile_fail_include_dirs}")
add_test(NAME harness_compile_fail_runner_selfcheck
  COMMAND ${CMAKE_COMMAND}
    "-DMP1_CF_RUNNER=${PROJECT_SOURCE_DIR}/cmake/CompileFail.cmake"
    "-DMP1_CF_CXX=${CMAKE_CXX_COMPILER}"
    "-DMP1_CF_INCLUDE_DIRS=${compile_fail_include_dirs}"
    "-DMP1_CF_SELFCHECK_DIR=${PROJECT_SOURCE_DIR}/test/compile_fail/self_check"
    "-DMP1_CF_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/compile_fail/selfcheck"
    -P "${PROJECT_SOURCE_DIR}/ci/check-compile-fail-selfcheck.cmake")
set_tests_properties(harness_compile_fail_runner_selfcheck PROPERTIES
  PROCESSORS 1
  TIMEOUT 120
  LABELS "compile-fail;harness;static")

# Positive controls: compile the whole test/compile_pass/ directory, not only the
# controls reached through a probe.  A control whose probe is removed (or renamed)
# must not become an inert source file; every control here is a positive compile
# check of a supported API shape, with the same command line as a probe control.
add_test(NAME harness_compile_pass_controls
  COMMAND ${CMAKE_COMMAND}
    "-DMP1_CP_CXX=${CMAKE_CXX_COMPILER}"
    "-DMP1_CP_INCLUDE_DIRS=${compile_fail_include_dirs}"
    "-DMP1_CP_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/compile_pass"
    "-DMP1_CP_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/compile_pass"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/support/compile_pass_controls.cmake")
set_tests_properties(harness_compile_pass_controls PROPERTIES
  PROCESSORS 1
  TIMEOUT 600
  LABELS "compile-pass;harness;static")

# Harness self-checks: unit + scenario kinds, ranks 2/4 (and rank 1 for the
# scenario path), the ENVIRONMENT property, and the wrap/allocation counters.
mp1_add_test(NAME harness_unit_smoke
  SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/harness/harness_smoke.cc"
  KIND unit RANKS "2;4" LABELS "harness;unit" REQUIRES wrap TIMEOUT 60)
mp1_add_test(NAME harness_scenario_smoke
  SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/harness/harness_scenario_smoke.cc"
  KIND scenario RANKS "1;2;4" LABELS "harness;scenario" REQUIRES wrap
  ENVIRONMENT MP1_TEST_SCENARIO=smoke TIMEOUT 60)

# MPI identity/version gate: header macros, MPI_Get_version, the CMake-detected
# version and the provider family must agree (MPI 4.1 §9.1.1, §11.4.1).
string(REGEX MATCH "^[0-9]+\\.[0-9]+" MP1_TEST_MPI_CMAKE_MAJOR_MINOR "${MPI_CXX_VERSION}")
if(NOT MP1_TEST_MPI_CMAKE_MAJOR_MINOR)
  message(FATAL_ERROR
    "mp1: could not derive the MPI major.minor version from MPI_CXX_VERSION "
    "'${MPI_CXX_VERSION}' for the identity gate")
endif()
set(MP1_TEST_MPI_EXPECTED_FAMILY "")
if(MPI_CXX_LIBRARY_VERSION_STRING MATCHES "Open MPI")
  set(MP1_TEST_MPI_EXPECTED_FAMILY "Open MPI")
elseif(MPI_CXX_LIBRARY_VERSION_STRING MATCHES "MPICH|mpich|Hydra")
  set(MP1_TEST_MPI_EXPECTED_FAMILY "MPICH")
endif()
mp1_add_test(NAME harness_mpi_identity
  SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/harness/mpi_identity.cc"
  KIND scenario RANKS "1" LABELS "harness;mpi;static" TIMEOUT 60)
target_compile_definitions(mp1_harness_mpi_identity PRIVATE
  "MP1_TEST_MPI_CMAKE_MAJOR_MINOR=\"${MP1_TEST_MPI_CMAKE_MAJOR_MINOR}\"")
if(MP1_TEST_MPI_EXPECTED_FAMILY)
  target_compile_definitions(mp1_harness_mpi_identity PRIVATE
    "MP1_TEST_MPI_EXPECTED_FAMILY=\"${MP1_TEST_MPI_EXPECTED_FAMILY}\"")
endif()

# Configure-time floor assertion self-check: 3.1/4.0 must fail, 4.1+ must pass.
add_test(NAME harness_mpi_version_selfcheck
  COMMAND ${CMAKE_COMMAND} -P "${PROJECT_SOURCE_DIR}/ci/check-mpi-version.cmake")
set_tests_properties(harness_mpi_version_selfcheck PROPERTIES
  PROCESSORS 1
  TIMEOUT 60
  LABELS "harness;static")

# Wrap-symbol coverage check (canonical list <-> wrappers <-> mp1/impl calls).
add_test(NAME harness_wrap_coverage
  COMMAND ${CMAKE_COMMAND}
    "-DMP1_REPO_ROOT=${PROJECT_SOURCE_DIR}"
    "-DMP1_WRAP_SYMBOLS_HEADER=${PROJECT_SOURCE_DIR}/test/support/mpi_wrap_symbols.hpp"
    "-DMP1_WRAP_SOURCE=${PROJECT_SOURCE_DIR}/test/support/mpi_wrap.cc"
    "-DMP1_WRAP_EXEMPTIONS=${PROJECT_SOURCE_DIR}/test/support/mpi_wrap_exemptions.txt"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/support/wrap_coverage.cmake")
set_tests_properties(harness_wrap_coverage PROPERTIES
  PROCESSORS 1
  TIMEOUT 60
  LABELS "harness;static")

# Perf is a test-tree option and is not part of the default CTest set.  The suite owns
# test/perf/: MP1_BUILD_PERF builds the deterministic `perf_calls` gate (one CTest entry
# when the wrap shim is available) and the nightly microbenchmarks.  The timing-ratio
# entries need the separate MP1_BUILD_PERF_RATIOS opt-in and are never registered in the
# default suite.
option(MP1_BUILD_PERF "Build the mp1 performance suite" OFF)
if(MP1_BUILD_PERF)
  add_subdirectory(perf)
endif()

# ---------------------------------------------------------------------------
# Example smoke as a CTest entry when the examples are built.  matrix_gather is the
# real-client check for self-placing subarray layouts at the CI rank count:
# its gatherv/alltoallv receive blocks are non-dense tiles whose conservative true
# extents overlap.  The default job skips the `example` label and runs the dedicated
# example-smoke step; the nightly job runs every entry, so a green unit suite can no
# longer hide a broken example.
# ---------------------------------------------------------------------------
if(TARGET matrix_gather)
  add_test(NAME example_matrix_gather
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:matrix_gather> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(example_matrix_gather PROPERTIES
    PROCESSORS 4
    TIMEOUT 120
    LABELS "example;mpi")
endif()

# ---------------------------------------------------------------------------
# Registration invariant: CMake's directory TESTS property must match what
# CTest actually exposes (names and labels).  The expected set is generated
# here, after every other registration, and the check is itself an entry, so
# the counts never live in prose.
# ---------------------------------------------------------------------------
add_test(NAME harness_test_matrix
  COMMAND ${CMAKE_COMMAND}
    "-DMP1_EXPECTED_TESTS=${CMAKE_CURRENT_BINARY_DIR}/registered-tests.txt"
    "-DMP1_CTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
    "-DMP1_BUILD_DIR=${CMAKE_BINARY_DIR}"
    -P "${PROJECT_SOURCE_DIR}/ci/check-test-matrix.cmake")
set_tests_properties(harness_test_matrix PROPERTIES
  PROCESSORS 1
  TIMEOUT 120
  LABELS "harness;static")

get_property(mp1_registered_tests DIRECTORY PROPERTY TESTS)
set(mp1_registration_lines)
foreach(test_name IN LISTS mp1_registered_tests)
  get_test_property("${test_name}" LABELS test_labels)
  if(NOT test_labels)
    set(test_labels "")
  endif()
  string(REPLACE ";" "," test_labels "${test_labels}")
  list(APPEND mp1_registration_lines "${test_name}|${test_labels}")
endforeach()
# Rows registered in test/perf/ (a subdirectory) are published explicitly.
get_property(mp1_subdirectory_test_rows GLOBAL PROPERTY MP1_REGISTERED_TEST_ROWS)
if(mp1_subdirectory_test_rows)
  list(APPEND mp1_registration_lines ${mp1_subdirectory_test_rows})
endif()
list(SORT mp1_registration_lines)
string(REPLACE ";" "\n" mp1_registration_text "${mp1_registration_lines}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/registered-tests.txt"
  "${mp1_registration_text}\n")

# Runtime-feature propagation: mp1_add_test exports
# MP1_REQUIRE_<FEATURE>=1 for every entry that declares the feature in REQUIRES
# when the job lists it in MP1_TEST_REQUIRE_FEATURES.  Verify the declaration
# table and the CTest ENVIRONMENT properties agree, and require every listed
# feature to be declared by at least one registered entry; a typo in the
# feature list must not turn the opt-in gate into a no-op.
get_property(mp1_runtime_feature_decls GLOBAL
  PROPERTY MP1_TEST_RUNTIME_FEATURE_DECLS)
set(mp1_required_runtime_features)
foreach(feature IN LISTS MP1_TEST_REQUIRE_FEATURES)
  string(TOUPPER "${feature}" feature_upper)
  list(APPEND mp1_required_runtime_features "${feature_upper}")
endforeach()
set(mp1_declared_runtime_features)
foreach(declaration IN LISTS mp1_runtime_feature_decls)
  string(REPLACE "|" ";" declaration_fields "${declaration}")
  list(GET declaration_fields 0 declared_test)
  list(GET declaration_fields 1 declared_feature)
  list(APPEND mp1_declared_runtime_features "${declared_feature}")
  get_test_property("${declared_test}" ENVIRONMENT declared_environment)
  if(declared_feature IN_LIST mp1_required_runtime_features)
    if(NOT declared_environment
       MATCHES "(^|;)MP1_REQUIRE_${declared_feature}=1(;|$)")
      message(FATAL_ERROR
        "mp1: ${declared_test} declares the required runtime feature "
        "'${declared_feature}' but does not export "
        "MP1_REQUIRE_${declared_feature}=1 (ENVIRONMENT is "
        "'${declared_environment}')")
    endif()
  elseif(declared_environment
         MATCHES "(^|;)MP1_REQUIRE_${declared_feature}=")
    message(FATAL_ERROR
      "mp1: ${declared_test} exports MP1_REQUIRE_${declared_feature} but "
      "MP1_TEST_REQUIRE_FEATURES does not list '${declared_feature}'")
  endif()
endforeach()
foreach(feature IN LISTS mp1_required_runtime_features)
  list(FIND mp1_declared_runtime_features "${feature}" declared_feature_index)
  if(declared_feature_index EQUAL -1)
    message(FATAL_ERROR
      "mp1: MP1_TEST_REQUIRE_FEATURES lists '${feature}' but no registered "
      "test declares it; the requirement would be a no-op")
  endif()
endforeach()
foreach(test_name IN LISTS mp1_registered_tests)
  get_test_property("${test_name}" ENVIRONMENT test_environment)
  foreach(feature IN LISTS mp1_required_runtime_features)
    if(test_environment MATCHES "(^|;)MP1_REQUIRE_${feature}=")
      list(FIND mp1_runtime_feature_decls "${test_name}|${feature}"
        feature_declaration_index)
      if(feature_declaration_index EQUAL -1)
        message(FATAL_ERROR
          "mp1: ${test_name} exports MP1_REQUIRE_${feature} without "
          "declaring '${feature}' in REQUIRES")
      endif()
    endif()
  endforeach()
endforeach()

# A compile-fail entry can never legitimately return 77: its runner
# exits 0/1.  Enforce that at configure time instead of trusting the helper.
foreach(test_name IN LISTS mp1_registered_tests)
  get_test_property("${test_name}" LABELS test_labels)
  if("compile-fail" IN_LIST test_labels)
    get_test_property("${test_name}" SKIP_RETURN_CODE test_skip_code)
    if(DEFINED test_skip_code AND NOT test_skip_code STREQUAL "NOTFOUND")
      message(FATAL_ERROR
        "mp1: compile-fail entry '${test_name}' must not carry "
        "SKIP_RETURN_CODE (found '${test_skip_code}')")
    endif()
  endif()
endforeach()
