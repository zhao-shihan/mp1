# Wrap-symbol coverage check.
#
# Run as a CTest check:
#   cmake -DMP1_REPO_ROOT=... -DMP1_WRAP_SYMBOLS_HEADER=... \
#         -DMP1_WRAP_SOURCE=... -DMP1_WRAP_EXEMPTIONS=... \
#         -P test/support/wrap_coverage.cmake
#
# It enforces three properties:
#   1. the canonical symbol list and the wrapper definitions agree;
#   2. every call-count gate symbol is in the canonical list;
#   3. every `MPI_*(...)` call in mp1/impl/** is either wrapped or explicitly
#      exempted.
# Non-gate native calls are allowed only through the explicit exemption list.

cmake_minimum_required(VERSION 3.21)

foreach(required IN ITEMS
    MP1_REPO_ROOT
    MP1_WRAP_SYMBOLS_HEADER
    MP1_WRAP_SOURCE
    MP1_WRAP_EXEMPTIONS)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "wrap_coverage.cmake: ${required} is required")
  endif()
  if(NOT EXISTS "${${required}}")
    message(FATAL_ERROR "wrap_coverage.cmake: missing ${${required}}")
  endif()
endforeach()

# --- canonical list ---------------------------------------------------------
file(STRINGS "${MP1_WRAP_SYMBOLS_HEADER}" symbol_lines
  REGEX "^MP1_WRAP_SYMBOL\\([A-Za-z0-9_]+\\)$")
if(NOT symbol_lines)
  message(FATAL_ERROR "wrap_coverage.cmake: canonical symbol list is empty")
endif()

set(symbols)
foreach(line IN LISTS symbol_lines)
  string(REGEX REPLACE
    "^MP1_WRAP_SYMBOL\\(([A-Za-z0-9_]+)\\)$" "\\1" symbol "${line}")
  list(APPEND symbols "${symbol}")
endforeach()

set(unique_symbols ${symbols})
list(REMOVE_DUPLICATES unique_symbols)
list(LENGTH symbols symbol_count)
list(LENGTH unique_symbols unique_count)
if(NOT symbol_count EQUAL unique_count)
  message(FATAL_ERROR
    "wrap_coverage.cmake: duplicate entries in mpi_wrap_symbols.hpp")
endif()

# --- wrapper definitions ----------------------------------------------------
file(STRINGS "${MP1_WRAP_SOURCE}" wrapper_lines
  REGEX "^MP1_WRAP_INT(_POST)?\\([A-Za-z0-9_]+,")
if(NOT wrapper_lines)
  message(FATAL_ERROR "wrap_coverage.cmake: no wrapper definitions found")
endif()

set(wrappers)
foreach(line IN LISTS wrapper_lines)
  if(line MATCHES "^MP1_WRAP_INT(_POST)?\\(([A-Za-z0-9_]+),")
    list(APPEND wrappers "${CMAKE_MATCH_2}")
  endif()
endforeach()
set(unique_wrappers ${wrappers})
list(REMOVE_DUPLICATES unique_wrappers)
list(LENGTH wrappers wrapper_count)
list(LENGTH unique_wrappers unique_wrapper_count)
if(NOT wrapper_count EQUAL unique_wrapper_count)
  message(FATAL_ERROR "wrap_coverage.cmake: duplicate wrapper definitions")
endif()

set(missing_wrappers)
foreach(symbol IN LISTS symbols)
  if(NOT symbol IN_LIST wrappers)
    list(APPEND missing_wrappers "${symbol}")
  endif()
endforeach()
if(missing_wrappers)
  message(FATAL_ERROR
    "wrap_coverage.cmake: canonical symbols without a wrapper definition: "
    "${missing_wrappers}")
endif()

set(unknown_wrappers)
foreach(wrapper IN LISTS wrappers)
  if(NOT wrapper IN_LIST symbols)
    list(APPEND unknown_wrappers "${wrapper}")
  endif()
endforeach()
if(unknown_wrappers)
  message(FATAL_ERROR
    "wrap_coverage.cmake: wrapper definitions absent from the canonical list: "
    "${unknown_wrappers}")
endif()

# --- required call-count symbols -------------------------------------------------
set(required_gate_symbols
  MPI_Send MPI_Recv MPI_Isend MPI_Irecv MPI_Send_init MPI_Recv_init
  MPI_Isendrecv_c MPI_Isendrecv_replace_c
  MPI_Gatherv MPI_Gatherv_c MPI_Igatherv MPI_Igatherv_c
  MPI_Scatterv MPI_Scatterv_c MPI_Iscatterv MPI_Iscatterv_c
  MPI_Allgatherv MPI_Allgatherv_c MPI_Iallgatherv MPI_Iallgatherv_c
  MPI_Alltoallv MPI_Alltoallv_c MPI_Ialltoallv MPI_Ialltoallv_c
  MPI_Reduce MPI_Reduce_c MPI_Reduce_local_c MPI_Ireduce MPI_Ireduce_c
  MPI_Allreduce MPI_Allreduce_c MPI_Iallreduce MPI_Iallreduce_c
  MPI_Reduce_scatter MPI_Reduce_scatter_c
  MPI_Ireduce_scatter MPI_Ireduce_scatter_c
  MPI_Reduce_scatter_block MPI_Reduce_scatter_block_c
  MPI_Ireduce_scatter_block MPI_Ireduce_scatter_block_c
  MPI_Type_dup
  MPI_Pack MPI_Pack_c MPI_Pack_size MPI_Pack_size_c
  MPI_Unpack MPI_Unpack_c
  MPI_Buffer_attach_c MPI_Buffer_detach_c
  MPI_File_read_at_c MPI_File_write_at_c
  MPI_File_read_c MPI_File_write_c
  MPI_File_read_shared_c MPI_File_write_shared_c
  MPI_File_read_at_all_c MPI_File_write_at_all_c
  MPI_File_read_all_c MPI_File_write_all_c
  MPI_File_read_ordered_c MPI_File_write_ordered_c
  MPI_File_read_at_all_begin_c MPI_File_write_at_all_begin_c
  MPI_File_read_all_begin_c MPI_File_write_all_begin_c
  MPI_File_read_ordered_begin_c MPI_File_write_ordered_begin_c
  MPI_File_iread_at_c MPI_File_iwrite_at_c
  MPI_File_iread_c MPI_File_iwrite_c
  MPI_File_iread_shared_c MPI_File_iwrite_shared_c
  MPI_File_iread_at_all_c MPI_File_iwrite_at_all_c
  MPI_File_iread_all_c MPI_File_iwrite_all_c
  MPI_Comm_free MPI_Group_free MPI_File_close MPI_Request_free
  MPI_Initialized MPI_Finalized MPI_Finalize)

set(missing_required)
foreach(symbol IN LISTS required_gate_symbols)
  if(NOT symbol IN_LIST symbols)
    list(APPEND missing_required "${symbol}")
  endif()
endforeach()
if(missing_required)
  message(FATAL_ERROR
    "wrap_coverage.cmake: missing required call-count symbols: "
    "${missing_required}")
endif()

# --- exemptions -------------------------------------------------------------
file(STRINGS "${MP1_WRAP_EXEMPTIONS}" exemption_lines)
set(exemptions)
foreach(line IN LISTS exemption_lines)
  string(REGEX REPLACE "#.*$" "" line "${line}")
  string(STRIP "${line}" line)
  if(NOT line STREQUAL "")
    list(APPEND exemptions "${line}")
  endif()
endforeach()

# --- implementation call-site scan -----------------------------------------
file(GLOB_RECURSE impl_sources
  "${MP1_REPO_ROOT}/mp1/impl/*.hpp"
  "${MP1_REPO_ROOT}/mp1/impl/*.ipp"
  "${MP1_REPO_ROOT}/mp1/impl/*.cc")
list(LENGTH impl_sources impl_source_count)

set(uncovered)
foreach(source IN LISTS impl_sources)
  file(READ "${source}" contents)
  string(REGEX MATCHALL "MPI_[A-Za-z0-9_]+[ \t]*\\(" calls "${contents}")
  foreach(call IN LISTS calls)
    string(REGEX REPLACE "[ \t]*\\($" "" symbol "${call}")
    if(symbol IN_LIST symbols)
      continue()
    endif()
    if(symbol IN_LIST exemptions)
      continue()
    endif()
    list(APPEND uncovered "${symbol} (${source})")
  endforeach()
endforeach()
list(REMOVE_DUPLICATES uncovered)
if(uncovered)
  message(FATAL_ERROR
    "wrap_coverage.cmake: MPI calls in mp1/impl/** are neither wrapped nor "
    "exempt: ${uncovered}")
endif()

message(STATUS
  "wrap coverage ok: ${symbol_count} wrapped symbols, "
  "${wrapper_count} wrapper definitions, ${impl_source_count} impl files")
