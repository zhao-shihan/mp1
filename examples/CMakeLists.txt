function(add_mp1_executable TARGET_NAME SOURCES)
  add_executable(${TARGET_NAME} ${SOURCES})
  target_compile_options(${TARGET_NAME} PRIVATE
     $<$<CXX_COMPILER_ID:GNU>:
          -Wall -Wextra -Wpedantic>
     $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:
          -Wall -Wextra -Wpedantic -Wno-c++98-compat>
     $<$<CXX_COMPILER_ID:Intel>:
          -Wall>
     $<$<CXX_COMPILER_ID:MSVC>:
          /permissive- /W4 /WX>)
  target_link_libraries(${TARGET_NAME} PRIVATE mp1::mp1)
endfunction()


add_mp1_executable(arrays arrays.cc)
add_mp1_executable(blocking blocking.cc)
add_mp1_executable(blocking_vector blocking_vector.cc)
add_mp1_executable(collective collective.cc)
add_mp1_executable(communicator communicator.cc)
add_mp1_executable(gather gather.cc)
add_mp1_executable(gatherv gatherv.cc)
add_mp1_executable(hello_world hello_world.cc)
add_mp1_executable(iterators iterators.cc)
add_mp1_executable(layouts layouts.cc)
add_mp1_executable(matrix_gather matrix_gather.cc)
add_mp1_executable(nonblocking nonblocking.cc)
add_mp1_executable(nonblocking_mult nonblocking_mult.cc)
add_mp1_executable(probe probe.cc)
add_mp1_executable(reduce_lcm reduce_lcm.cc)
add_mp1_executable(reduce_min_loc reduce_min_loc.cc)
add_mp1_executable(stl_container stl_container.cc)
add_mp1_executable(struct struct.cc)
add_mp1_executable(subarray subarray.cc)
add_mp1_executable(vibrating_string_mp1 vibrating_string_mp1.cc)
add_mp1_executable(standard_types standard_types.cc)
add_mp1_executable(parallel_sort_mp1 parallel_sort_mp1.cc)
add_mp1_executable(intercommunicator intercommunicator.cc)
add_mp1_executable(process_creation process_creation.cc)
add_mp1_executable(process_creation_multiple process_creation_multiple.cc)
add_mp1_executable(process_creation_client process_creation_client.cc)
add_mp1_executable(file file.cc)
add_mp1_executable(rma rma.cc)

# c only examples
add_executable(vibrating_string_mpi vibrating_string_mpi.c)
target_link_libraries(vibrating_string_mpi PUBLIC m MPI::MPI_C)
add_executable(parallel_sort_mpi parallel_sort_mpi.c)
target_link_libraries(parallel_sort_mpi PUBLIC MPI::MPI_C)

foreach(c_target IN ITEMS vibrating_string_mpi parallel_sort_mpi)
  target_compile_options(${c_target} PRIVATE
    $<$<C_COMPILER_ID:GNU>:-Wall -Wextra -Wpedantic>
    $<$<C_COMPILER_ID:Clang>:-Wall -Wextra -Wpedantic>)
endforeach()
