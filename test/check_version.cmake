# Cross-check the single version source, the generated public header and the
# CMake project version.
#
# Usage:
#   cmake -DMP1_VERSION_FILE=<VERSION> \
#         -DMP1_GENERATED_VERSION_HEADER=<generated/mp1/version.hpp> \
#         -DMP1_EXPECTED_PROJECT_VERSION=<PROJECT_VERSION> \
#         -P test/check_version.cmake
#
# Fails with a specific message when any derived value drifts from VERSION.

foreach(required IN ITEMS MP1_VERSION_FILE MP1_GENERATED_VERSION_HEADER
    MP1_EXPECTED_PROJECT_VERSION)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "check_version: ${required} is not set")
  endif()
endforeach()

if(NOT EXISTS "${MP1_VERSION_FILE}")
  message(FATAL_ERROR "check_version: VERSION file not found: ${MP1_VERSION_FILE}")
endif()
if(NOT EXISTS "${MP1_GENERATED_VERSION_HEADER}")
  message(FATAL_ERROR
    "check_version: generated header not found: ${MP1_GENERATED_VERSION_HEADER}")
endif()

file(STRINGS "${MP1_VERSION_FILE}" version_lines)
list(LENGTH version_lines version_line_count)
if(NOT version_line_count EQUAL 1)
  message(FATAL_ERROR
    "check_version: ${MP1_VERSION_FILE} must contain exactly one line, found "
    "${version_line_count}")
endif()
list(GET version_lines 0 version)
string(STRIP "${version}" version)
if(NOT version MATCHES "^[0-9]+[.][0-9]+[.][0-9]+$")
  message(FATAL_ERROR "check_version: VERSION is not major.minor.patch: '${version}'")
endif()

if(NOT MP1_EXPECTED_PROJECT_VERSION MATCHES "^([0-9]+)[.]([0-9]+)[.]([0-9]+)$")
  message(FATAL_ERROR
    "check_version: PROJECT_VERSION is not major.minor.patch: "
    "'${MP1_EXPECTED_PROJECT_VERSION}'")
endif()
set(expected_major "${CMAKE_MATCH_1}")
set(expected_minor "${CMAKE_MATCH_2}")
set(expected_patch "${CMAKE_MATCH_3}")
if(NOT version STREQUAL MP1_EXPECTED_PROJECT_VERSION)
  message(FATAL_ERROR
    "check_version: VERSION '${version}' != PROJECT_VERSION "
    "'${MP1_EXPECTED_PROJECT_VERSION}'")
endif()

file(READ "${MP1_GENERATED_VERSION_HEADER}" header)

string(REGEX MATCH "#define[ \t]+MP1_VERSION_STRING[ \t]+\"([0-9]+[.][0-9]+[.][0-9]+)\""
  header_string_match "${header}")
if(CMAKE_MATCH_1 STREQUAL "")
  message(FATAL_ERROR
    "check_version: no MP1_VERSION_STRING in ${MP1_GENERATED_VERSION_HEADER}")
endif()
set(header_string "${CMAKE_MATCH_1}")
if(NOT header_string STREQUAL version)
  message(FATAL_ERROR
    "check_version: MP1_VERSION_STRING '${header_string}' != VERSION '${version}'")
endif()

foreach(component IN ITEMS MAJOR MINOR PATCH)
  string(REGEX MATCH "#define[ \t]+MP1_VERSION_${component}[ \t]+([0-9]+)"
    header_component_match "${header}")
  if(CMAKE_MATCH_1 STREQUAL "")
    message(FATAL_ERROR
      "check_version: no MP1_VERSION_${component} in ${MP1_GENERATED_VERSION_HEADER}")
  endif()
  if(component STREQUAL "MAJOR")
    set(expected_component "${expected_major}")
  elseif(component STREQUAL "MINOR")
    set(expected_component "${expected_minor}")
  else()
    set(expected_component "${expected_patch}")
  endif()
  if(NOT CMAKE_MATCH_1 STREQUAL expected_component)
    message(FATAL_ERROR
      "check_version: MP1_VERSION_${component} '${CMAKE_MATCH_1}' != "
      "PROJECT_VERSION_${component} '${expected_component}'")
  endif()
endforeach()

# An installed consumer includes the header directly; it must be self-contained
# and carry the same guard name the source template declares.
string(FIND "${header}" "#define MP1_VERSION_HPP" guard_position)
if(guard_position EQUAL -1)
  message(FATAL_ERROR
    "check_version: generated header is missing the MP1_VERSION_HPP guard")
endif()

message(STATUS
  "check_version: PASS (VERSION, PROJECT_VERSION and generated macros report ${version})")
