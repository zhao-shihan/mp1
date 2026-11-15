# CTest registration/label invariant.
#
# `test/CMakeLists.txt` writes registered-tests.txt from CMake's directory
# TESTS property immediately after registering every entry (including this
# check).  This script compares that generated set with what `ctest
# --show-only=json-v1` actually reports, and fails on any name or label
# difference.  It catches a registration that silently disappears, a test added
# outside the manifest-writing point, and a label that replaces the defaults.
#
# This is the minimal enforced set-equality invariant.

cmake_minimum_required(VERSION 3.21)

foreach(required IN ITEMS
    MP1_EXPECTED_TESTS
    MP1_CTEST_COMMAND
    MP1_BUILD_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR
      "check-test-matrix.cmake: ${required} is required")
  endif()
endforeach()

if(NOT EXISTS "${MP1_EXPECTED_TESTS}")
  message(FATAL_ERROR
    "check-test-matrix: expected registration file is missing: "
    "${MP1_EXPECTED_TESTS}")
endif()

file(STRINGS "${MP1_EXPECTED_TESTS}" expected_lines)
set(expected_names)
set(expected_labels_)
foreach(line IN LISTS expected_lines)
  string(STRIP "${line}" line)
  if(line STREQUAL "")
    continue()
  endif()
  string(FIND "${line}" "|" separator)
  if(separator EQUAL -1)
    message(FATAL_ERROR
      "check-test-matrix: malformed expected line '${line}'")
  endif()
  string(SUBSTRING "${line}" 0 ${separator} name)
  math(EXPR after "${separator} + 1")
  string(SUBSTRING "${line}" ${after} -1 labels)
  list(APPEND expected_names "${name}")
  string(REPLACE "," ";" labels "${labels}")
  list(SORT labels)
  list(REMOVE_DUPLICATES labels)
  string(REPLACE ";" "," labels "${labels}")
  set("expected_labels_${name}" "${labels}")
endforeach()
list(SORT expected_names)

execute_process(
  COMMAND "${MP1_CTEST_COMMAND}" --test-dir "${MP1_BUILD_DIR}"
          --show-only=json-v1
  RESULT_VARIABLE result
  OUTPUT_VARIABLE json
  ERROR_VARIABLE error)
if(NOT result STREQUAL "0")
  message(FATAL_ERROR
    "check-test-matrix: ctest --show-only=json-v1 failed: ${result}\n${error}")
endif()

string(JSON actual_count LENGTH "${json}" tests)
set(actual_names)
set(actual_labels_)
if(actual_count GREATER 0)
  math(EXPR last "${actual_count} - 1")
  foreach(index RANGE 0 ${last})
    string(JSON name GET "${json}" tests ${index} name)
    # CTest --show-only writes labels inside the `properties` array as
    # {"name":"LABELS","value":[...]}, not as a top-level `labels` member.
    string(JSON property_count LENGTH "${json}" tests ${index} properties)
    set(labels "")
    if(property_count GREATER 0)
      math(EXPR last_property "${property_count} - 1")
      foreach(property_index RANGE 0 ${last_property})
        string(JSON property_name GET
          "${json}" tests ${index} properties ${property_index} name)
        if(property_name STREQUAL "LABELS")
          string(JSON label_count LENGTH
            "${json}" tests ${index} properties ${property_index} value)
          if(label_count GREATER 0)
            math(EXPR last_label "${label_count} - 1")
            foreach(label_index RANGE 0 ${last_label})
              string(JSON label GET
                "${json}" tests ${index} properties ${property_index} value
                ${label_index})
              list(APPEND labels "${label}")
            endforeach()
          endif()
        endif()
      endforeach()
    endif()
    list(SORT labels)
    list(REMOVE_DUPLICATES labels)
    string(REPLACE ";" "," labels "${labels}")
    list(APPEND actual_names "${name}")
    set("actual_labels_${name}" "${labels}")
  endforeach()
endif()
list(SORT actual_names)

set(failures 0)
if(NOT expected_names STREQUAL actual_names)
  message(SEND_ERROR
    "check-test-matrix: registered/test-listed set mismatch.\n"
    "expected (${MP1_EXPECTED_TESTS}): ${expected_names}\n"
    "actual (ctest --show-only): ${actual_names}")
  math(EXPR failures "${failures} + 1")
endif()

foreach(name IN LISTS expected_names)
  if(NOT name IN_LIST actual_names)
    continue()
  endif()
  set(expected_key "expected_labels_${name}")
  set(actual_key "actual_labels_${name}")
  set(expected_labels "${${expected_key}}")
  set(actual_labels "${${actual_key}}")
  if(NOT expected_labels STREQUAL actual_labels)
    message(SEND_ERROR
      "check-test-matrix: label mismatch for '${name}': "
      "expected '${expected_labels}', actual '${actual_labels}'")
    math(EXPR failures "${failures} + 1")
  endif()
endforeach()

if(failures GREATER 0)
  message(FATAL_ERROR
    "check-test-matrix: ${failures} registration mismatch(es)")
endif()

message(STATUS
  "check-test-matrix: PASS (${actual_count} CTest entries, names and labels match)")
