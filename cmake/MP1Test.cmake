# MP1 test harness.
#
# This module defines exactly two public functions:
#
#   mp1_add_test(NAME <name> SOURCE <file>
#                [KIND unit|scenario] [RANKS "2;4"] [LABELS <label>...]
#                [REQUIRES <feature>...]
#                [ENVIRONMENT <var=value>...] [TIMEOUT <seconds>]
#                [CHILD_SOURCE <file>])
#   mp1_add_compile_fail(NAME <name> SOURCE <file> EXPECT <regex>
#                        [CONTROL <file>] [FLAGS <flag>...])
#
# CHILD_SOURCE builds a second executable (same include paths, warning flags and MPI wrap
# shim) from <file> and passes its path as the test program's first argument.  It is used by
# the spawn suite, where the parent re-executes a child program; the child target is not
# registered as a test itself.
#
# Labels append to the kind defaults (`unit;mpi` or `scenario;mpi`); they never
# replace them.  SKIP_RETURN_CODE 77 is attached when REQUIRES is non-empty;
# a caller that needs a provider-conditional skip may also set the property
# explicitly (the spawn suite does so).  A feature listed in
# MP1_TEST_REQUIRE_FEATURES must be available at configure time (wrap) or is
# exported as MP1_REQUIRE_<FEATURE>=1 for the declaring tests, so a runtime
# skip becomes a failure.
#
# `KIND unit` links the shared `mp1_test_main` static library and
# `Boost::unit_test_framework`; the test source must not provide main.  `KIND
# scenario` links no Boost and the source provides main.  Both kinds link the
# `mp1_test_target` interface (mp1::mp1) and, on GNU/Clang/ELF, the `--wrap`
# based MPI counter shim.

# Guard against accidental inclusion before Boost is available.  The include
# site is responsible for find_package(Boost ...); the check is intentionally
# deferred to mp1_add_test so the compile-fail-only path does not need Boost.
include_guard(GLOBAL)

function(mp1_add_test)
  cmake_parse_arguments(PARSE_ARGV 0 arg
    ""
    "NAME;SOURCE;KIND;TIMEOUT;CHILD_SOURCE"
    "RANKS;LABELS;REQUIRES;ENVIRONMENT")

  if(arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "mp1_add_test: unknown arguments: ${arg_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT arg_NAME)
    message(FATAL_ERROR "mp1_add_test: NAME is required")
  endif()
  if(NOT arg_SOURCE)
    message(FATAL_ERROR "mp1_add_test: SOURCE is required")
  endif()

  if(NOT arg_KIND)
    set(arg_KIND unit)
  endif()
  if(NOT arg_KIND STREQUAL "unit" AND NOT arg_KIND STREQUAL "scenario")
    message(FATAL_ERROR
      "mp1_add_test(${arg_NAME}): KIND must be unit or scenario, got '${arg_KIND}'")
  endif()

  get_filename_component(source "${arg_SOURCE}" ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR
      "mp1_add_test(${arg_NAME}): source does not exist: ${source}")
  endif()

  # Ranks are fixed at 2 and 4 for unit tests.  Scenario tests may select
  # another set (the environment `none` scenario is the rank-1 exception).
  if(arg_KIND STREQUAL "unit")
    # Unit ranks are fixed: ignore any RANKS value and register 2 and 4.
    set(ranks "2;4")
  else()
    if(NOT arg_RANKS)
      set(ranks "2;4")
    else()
      # cmake_parse_arguments stores a single "1;2;4" argument escaped as one
      # list element; expand unquoted to recover the intended rank list.
      set(ranks ${arg_RANKS})
    endif()
  endif()
  if(NOT ranks)
    message(FATAL_ERROR "mp1_add_test(${arg_NAME}): RANKS is empty")
  endif()
  foreach(rank IN LISTS ranks)
    if(NOT rank MATCHES "^[1-9][0-9]*$")
      message(FATAL_ERROR
        "mp1_add_test(${arg_NAME}): invalid rank '${rank}'")
    endif()
  endforeach()

  if(arg_TIMEOUT)
    if(NOT arg_TIMEOUT MATCHES "^[1-9][0-9]*$")
      message(FATAL_ERROR
        "mp1_add_test(${arg_NAME}): invalid TIMEOUT '${arg_TIMEOUT}'")
    endif()
    set(timeout "${arg_TIMEOUT}")
  elseif(arg_KIND STREQUAL "scenario")
    set(timeout 60)
  else()
    set(timeout 120)
  endif()

  if(arg_KIND STREQUAL "unit")
    if(NOT TARGET mp1_test_main)
      message(FATAL_ERROR
        "mp1_add_test(${arg_NAME}): mp1_test_main has not been created")
    endif()
    set(labels "unit;mpi")
  else()
    set(labels "scenario;mpi")
  endif()
  if(arg_LABELS)
    list(APPEND labels ${arg_LABELS})
  endif()

  set(target "mp1_${arg_NAME}")
  if(TARGET "${target}")
    message(FATAL_ERROR "mp1_add_test: target already exists: ${target}")
  endif()

  add_executable("${target}" "${source}")
  set_target_properties("${target}" PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO)

  target_include_directories("${target}" PRIVATE
    "${PROJECT_SOURCE_DIR}/test/support")

  target_compile_options("${target}" PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
      -Wall -Wextra -Wpedantic -Wno-c++98-compat>
    $<$<CXX_COMPILER_ID:MSVC>:/permissive- /W4 /WX>)

  target_link_libraries("${target}" PRIVATE mp1_test_target)
  if(arg_KIND STREQUAL "unit")
    target_link_libraries("${target}" PRIVATE
      mp1_test_main
      Boost::unit_test_framework)
  endif()

  # Wrap the MPI entry points on GNU/Clang/ELF.  Elsewhere the shim compiles to
  # stubs, `mpi_wrap::supported()` is false, and a wrap-dependent test is not
  # registered unless the platform requirement was declared mandatory.
  if(DEFINED MP1_TEST_WRAP_SUPPORTED)
    set(wrap_supported "${MP1_TEST_WRAP_SUPPORTED}")
  else()
    set(wrap_supported FALSE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
       AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
      set(wrap_supported TRUE)
    endif()
  endif()

  set(skip_environment)
  set(runtime_features)
  foreach(feature IN LISTS arg_REQUIRES)
    string(TOUPPER "${feature}" feature_upper)
    if(feature STREQUAL "wrap" AND NOT wrap_supported)
      if("wrap" IN_LIST MP1_TEST_REQUIRE_FEATURES)
        message(FATAL_ERROR
          "mp1_add_test(${arg_NAME}): REQUIRES wrap but --wrap is unavailable "
          "on this platform and MP1_TEST_REQUIRE_FEATURES lists wrap")
      endif()
      message(STATUS
        "mp1_add_test(${arg_NAME}): not registered (REQUIRES wrap, unavailable)")
      return()
    endif()
    list(APPEND runtime_features "${feature_upper}")
    if(feature IN_LIST MP1_TEST_REQUIRE_FEATURES)
      list(APPEND skip_environment "MP1_REQUIRE_${feature_upper}=1")
    endif()
  endforeach()
  if(runtime_features)
    list(REMOVE_DUPLICATES runtime_features)
  endif()

  target_sources("${target}" PRIVATE
    "${PROJECT_SOURCE_DIR}/test/support/mpi_wrap.cc")
  target_include_directories("${target}" PRIVATE
    "${PROJECT_SOURCE_DIR}/test/support")

  if(wrap_supported)
    # The canonical symbol list is the single source of truth for both the
    # counter table in mpi_wrap.cc and the linker --wrap options.
    set(symbols_header
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../test/support/mpi_wrap_symbols.hpp")
    if(NOT EXISTS "${symbols_header}")
      message(FATAL_ERROR "mp1_add_test: missing ${symbols_header}")
    endif()
    file(STRINGS "${symbols_header}" symbol_lines
      REGEX "^MP1_WRAP_SYMBOL\\([A-Za-z0-9_]+\\)$")
    if(NOT symbol_lines)
      message(FATAL_ERROR
        "mp1_add_test: no MP1_WRAP_SYMBOL entries in ${symbols_header}")
    endif()
    set(wrap_options)
    foreach(line IN LISTS symbol_lines)
      string(REGEX REPLACE
        "^MP1_WRAP_SYMBOL\\(([A-Za-z0-9_]+)\\)$" "\\1" symbol "${line}")
      list(APPEND wrap_options "-Wl,--wrap=${symbol}")
    endforeach()
    target_compile_definitions("${target}" PRIVATE MP1_TEST_WRAP=1)
    target_link_options("${target}" PRIVATE ${wrap_options})
  else()
    target_compile_definitions("${target}" PRIVATE MP1_TEST_WRAP=0)
  endif()

  # Optional second executable for parent/child scenario tests (spawn).  It is built with the
  # same surface as the test target, and its path is passed as the test program's first
  # argument; it is not registered as a CTest entry.
  set(child_target)
  if(arg_CHILD_SOURCE)
    get_filename_component(child_source "${arg_CHILD_SOURCE}" ABSOLUTE
      BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${child_source}")
      message(FATAL_ERROR
        "mp1_add_test(${arg_NAME}): child source does not exist: ${child_source}")
    endif()
    set(child_target "${target}_child")
    if(TARGET "${child_target}")
      message(FATAL_ERROR "mp1_add_test: target already exists: ${child_target}")
    endif()
    add_executable("${child_target}" "${child_source}")
    set_target_properties("${child_target}" PROPERTIES
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO)
    target_include_directories("${child_target}" PRIVATE
      "${PROJECT_SOURCE_DIR}/test/support")
    target_compile_options("${child_target}" PRIVATE
      $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
      $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
        -Wall -Wextra -Wpedantic -Wno-c++98-compat>)
    target_link_libraries("${child_target}" PRIVATE mp1_test_target)
    target_sources("${child_target}" PRIVATE
      "${PROJECT_SOURCE_DIR}/test/support/mpi_wrap.cc")
    if(wrap_supported)
      target_compile_definitions("${child_target}" PRIVATE MP1_TEST_WRAP=1)
      target_link_options("${child_target}" PRIVATE ${wrap_options})
    else()
      target_compile_definitions("${child_target}" PRIVATE MP1_TEST_WRAP=0)
    endif()
  endif()

  foreach(rank IN LISTS ranks)
    set(test_command
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${rank}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:${target}>)
    if(child_target)
      list(APPEND test_command $<TARGET_FILE:${child_target}>)
    endif()
    list(APPEND test_command ${MPIEXEC_POSTFLAGS})
    add_test(NAME "${arg_NAME}.${rank}" COMMAND ${test_command})
    set_tests_properties("${arg_NAME}.${rank}" PROPERTIES
      PROCESSORS "${rank}"
      TIMEOUT "${timeout}"
      LABELS "${labels}")
    if(arg_REQUIRES)
      set_tests_properties("${arg_NAME}.${rank}" PROPERTIES
        SKIP_RETURN_CODE 77)
    endif()
    set(environment ${arg_ENVIRONMENT} ${skip_environment})
    if(environment)
      set_tests_properties("${arg_NAME}.${rank}" PROPERTIES
        ENVIRONMENT "${environment}")
    endif()
    # Publish the runtime-feature declarations per registered entry.  The
    # include site verifies that MP1_TEST_REQUIRE_FEATURES exports each
    # required feature to exactly the entries that declare it.
    foreach(feature IN LISTS runtime_features)
      set_property(GLOBAL APPEND PROPERTY MP1_TEST_RUNTIME_FEATURE_DECLS
        "${arg_NAME}.${rank}|${feature}")
    endforeach()
  endforeach()
endfunction()


function(mp1_add_compile_fail)
  cmake_parse_arguments(PARSE_ARGV 0 arg ""
    "NAME;SOURCE;EXPECT;CONTROL"
    "FLAGS")

  if(arg_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "mp1_add_compile_fail: unknown arguments: ${arg_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT arg_NAME)
    message(FATAL_ERROR "mp1_add_compile_fail: NAME is required")
  endif()
  if(NOT arg_SOURCE)
    message(FATAL_ERROR "mp1_add_compile_fail: SOURCE is required")
  endif()
  if(NOT arg_EXPECT)
    message(FATAL_ERROR
      "mp1_add_compile_fail(${arg_NAME}): EXPECT is required; a probe without "
      "an expected diagnostic can pass for a wrong reason")
  endif()

  get_filename_component(source "${arg_SOURCE}" ABSOLUTE
    BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR
      "mp1_add_compile_fail(${arg_NAME}): source does not exist: ${source}")
  endif()

  set(control "")
  if(arg_CONTROL)
    get_filename_component(control "${arg_CONTROL}" ABSOLUTE
      BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${control}")
      message(FATAL_ERROR
        "mp1_add_compile_fail(${arg_NAME}): control does not exist: ${control}")
    endif()
  endif()

  # `cmake -P` receives scalar -D values only; encode the include and flag
  # lists with '|' so paths/arguments containing semicolons are not split by
  # the command line.  The expectation is passed through unmodified.  The
  # generated include directory comes first: every probe includes the umbrella,
  # which includes the generated mp1/version.hpp.
  set(include_dirs
    "${MP1_GENERATED_INCLUDE_DIR}"
    "${PROJECT_SOURCE_DIR}")
  if(MPI_CXX_INCLUDE_DIRS)
    list(APPEND include_dirs ${MPI_CXX_INCLUDE_DIRS})
  endif()
  string(REPLACE ";" "|" include_dirs "${include_dirs}")

  set(flags)
  if(arg_FLAGS)
    set(flags ${arg_FLAGS})
  endif()
  string(REPLACE ";" "|" flags "${flags}")

  set(log_file "${CMAKE_CURRENT_BINARY_DIR}/compile_fail/${arg_NAME}.log")
  add_test(NAME "${arg_NAME}"
    COMMAND ${CMAKE_COMMAND}
      "-DMP1_CF_SOURCE=${source}"
      "-DMP1_CF_CXX=${CMAKE_CXX_COMPILER}"
      "-DMP1_CF_INCLUDE_DIRS=${include_dirs}"
      "-DMP1_CF_EXPECT=${arg_EXPECT}"
      "-DMP1_CF_CONTROL=${control}"
      "-DMP1_CF_FLAGS=${flags}"
      "-DMP1_CF_LOG=${log_file}"
      -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CompileFail.cmake")

  # A compile-fail entry can only pass or fail; it never legitimately returns
  # 77.  Do not attach SKIP_RETURN_CODE here.
  set_tests_properties("${arg_NAME}" PROPERTIES
    PROCESSORS 1
    TIMEOUT 60
    LABELS "compile-fail")
endfunction()
