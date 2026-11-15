# Public-header manifest (gate helper).
#
# `mp1/headers.txt` is the authoritative list of headers that are part of the
# public/installed tree.  The generated `mp1/version.hpp` is not a source file
# and is therefore not listed; `test/CMakeLists.txt` appends it to the
# self-containment target separately.  The manifest is deliberately fail-closed:
#
#   * every *.hpp under mp1/ must have exactly one manifest line;
#   * every manifest line must name an existing file;
#   * every non-detail header must be included by the umbrella mp1/mp1.hpp.
#
# Adding a header without listing it fails configure, and deleting a header
# without dropping its line fails configure.  The topology/cartesian/graph/
# distributed-graph/neighbor/grid headers are removed from the 1.0 surface;
# the manifest and the umbrella already match.

function(mp1_read_header_manifest out_var)
  if(NOT DEFINED MP1_HEADER_MANIFEST)
    message(FATAL_ERROR "mp1_read_header_manifest: MP1_HEADER_MANIFEST is not set")
  endif()
  if(NOT EXISTS "${MP1_HEADER_MANIFEST}")
    message(FATAL_ERROR
      "mp1: header manifest is missing: ${MP1_HEADER_MANIFEST}")
  endif()

  file(STRINGS "${MP1_HEADER_MANIFEST}" lines)
  set(headers)
  foreach(line IN LISTS lines)
    string(STRIP "${line}" line)
    if(line STREQUAL "" OR line MATCHES "^#")
      continue()
    endif()
    if(line MATCHES " " OR line MATCHES "\t")
      message(FATAL_ERROR
        "mp1: header manifest entries must be single paths: '${line}'")
    endif()
    list(APPEND headers "${line}")
  endforeach()
  if(NOT headers)
    message(FATAL_ERROR "mp1: header manifest is empty: ${MP1_HEADER_MANIFEST}")
  endif()
  list(REMOVE_DUPLICATES headers)
  set("${out_var}" "${headers}" PARENT_SCOPE)
endfunction()

function(mp1_validate_header_manifest source_dir)
  set(MP1_HEADER_MANIFEST "${source_dir}/mp1/headers.txt")
  mp1_read_header_manifest(manifest_headers)

  file(GLOB_RECURSE discovered RELATIVE "${source_dir}" "${source_dir}/mp1/*.hpp")
  list(SORT discovered)
  list(SORT manifest_headers)

  foreach(header IN LISTS discovered)
    if(NOT header IN_LIST manifest_headers)
      message(FATAL_ERROR
        "mp1: header '${header}' exists but is not listed in mp1/headers.txt; "
        "add it (and include it from mp1/mp1.hpp if it is a family header)")
    endif()
  endforeach()
  foreach(header IN LISTS manifest_headers)
    if(NOT EXISTS "${source_dir}/${header}")
      message(FATAL_ERROR
        "mp1: header '${header}' is listed in mp1/headers.txt but does not exist; "
        "remove the stale manifest line")
    endif()
  endforeach()

  # Family headers must be part of the umbrella.  Detail headers are reached
  # through their family header, so they are intentionally not required there.
  file(READ "${source_dir}/mp1/mp1.hpp" umbrella)
  foreach(header IN LISTS manifest_headers)
    if(header STREQUAL "mp1/mp1.hpp")
      continue()
    endif()
    if(header MATCHES "^mp1/impl/detail/")
      continue()
    endif()
    string(FIND "${umbrella}" "#include \"${header}\"" position)
    if(position EQUAL -1)
      message(FATAL_ERROR
        "mp1: header '${header}' is public but is not included by mp1/mp1.hpp; "
        "add it to the umbrella or remove it from mp1/headers.txt")
    endif()
  endforeach()
endfunction()
