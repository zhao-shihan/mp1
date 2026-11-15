// Canonical list of MPI entry points wrapped by test/support/mpi_wrap.cc.
//
// This file is intentionally macro-included: define MP1_WRAP_SYMBOL(name) and
// include it.  cmake/MP1Test.cmake parses the same lines (one per line, at
// column 1) to emit `-Wl,--wrap=` options, so the counter table and the linker
// interposition list cannot drift.
//
// Every symbol here must also have a wrapper definition in mpi_wrap.cc.  The
// `harness_wrap_coverage` CTest check enforces this and checks the new
// mp1/impl tree against this list plus mpi_wrap_exemptions.txt.

#if !defined(MP1_WRAP_SYMBOL)
#error "mpi_wrap_symbols.hpp must be included with MP1_WRAP_SYMBOL defined"
#endif

// lifecycle / after-finalize detector.  MPI_Initialized/MPI_Finalized are counted too:
// they are the release-path lifecycle guard, and the deterministic call-count gates assert
// that a settled phase does not issue the redundant pair.
MP1_WRAP_SYMBOL(MPI_Init)
MP1_WRAP_SYMBOL(MPI_Init_thread)
MP1_WRAP_SYMBOL(MPI_Finalize)
MP1_WRAP_SYMBOL(MPI_Initialized)
MP1_WRAP_SYMBOL(MPI_Finalized)

// p2p, standard modes and their _c spellings
MP1_WRAP_SYMBOL(MPI_Send)
MP1_WRAP_SYMBOL(MPI_Send_c)
MP1_WRAP_SYMBOL(MPI_Recv)
MP1_WRAP_SYMBOL(MPI_Recv_c)
MP1_WRAP_SYMBOL(MPI_Isend)
MP1_WRAP_SYMBOL(MPI_Isend_c)
MP1_WRAP_SYMBOL(MPI_Irecv)
MP1_WRAP_SYMBOL(MPI_Irecv_c)
MP1_WRAP_SYMBOL(MPI_Send_init)
MP1_WRAP_SYMBOL(MPI_Send_init_c)
MP1_WRAP_SYMBOL(MPI_Recv_init)
MP1_WRAP_SYMBOL(MPI_Recv_init_c)
MP1_WRAP_SYMBOL(MPI_Sendrecv)
MP1_WRAP_SYMBOL(MPI_Sendrecv_c)
MP1_WRAP_SYMBOL(MPI_Sendrecv_replace)
MP1_WRAP_SYMBOL(MPI_Sendrecv_replace_c)
MP1_WRAP_SYMBOL(MPI_Isendrecv_c)
MP1_WRAP_SYMBOL(MPI_Isendrecv_replace_c)

// p2p, buffered / synchronous / ready modes
MP1_WRAP_SYMBOL(MPI_Bsend)
MP1_WRAP_SYMBOL(MPI_Bsend_c)
MP1_WRAP_SYMBOL(MPI_Ibsend)
MP1_WRAP_SYMBOL(MPI_Ibsend_c)
MP1_WRAP_SYMBOL(MPI_Bsend_init)
MP1_WRAP_SYMBOL(MPI_Bsend_init_c)
MP1_WRAP_SYMBOL(MPI_Ssend)
MP1_WRAP_SYMBOL(MPI_Ssend_c)
MP1_WRAP_SYMBOL(MPI_Issend)
MP1_WRAP_SYMBOL(MPI_Issend_c)
MP1_WRAP_SYMBOL(MPI_Ssend_init)
MP1_WRAP_SYMBOL(MPI_Ssend_init_c)
MP1_WRAP_SYMBOL(MPI_Rsend)
MP1_WRAP_SYMBOL(MPI_Rsend_c)
MP1_WRAP_SYMBOL(MPI_Irsend)
MP1_WRAP_SYMBOL(MPI_Irsend_c)
MP1_WRAP_SYMBOL(MPI_Rsend_init)
MP1_WRAP_SYMBOL(MPI_Rsend_init_c)

// p2p, matching and message probes
MP1_WRAP_SYMBOL(MPI_Probe)
MP1_WRAP_SYMBOL(MPI_Iprobe)
MP1_WRAP_SYMBOL(MPI_Mprobe)
MP1_WRAP_SYMBOL(MPI_Improbe)
MP1_WRAP_SYMBOL(MPI_Mrecv)
MP1_WRAP_SYMBOL(MPI_Imrecv)

// request lifecycle
MP1_WRAP_SYMBOL(MPI_Cancel)
MP1_WRAP_SYMBOL(MPI_Request_free)
MP1_WRAP_SYMBOL(MPI_Wait)
MP1_WRAP_SYMBOL(MPI_Test)
MP1_WRAP_SYMBOL(MPI_Waitall)
MP1_WRAP_SYMBOL(MPI_Testall)
MP1_WRAP_SYMBOL(MPI_Waitany)
MP1_WRAP_SYMBOL(MPI_Testany)
MP1_WRAP_SYMBOL(MPI_Waitsome)
MP1_WRAP_SYMBOL(MPI_Testsome)
MP1_WRAP_SYMBOL(MPI_Request_get_status)
MP1_WRAP_SYMBOL(MPI_Request_get_status_all)
MP1_WRAP_SYMBOL(MPI_Request_get_status_any)
MP1_WRAP_SYMBOL(MPI_Request_get_status_some)
MP1_WRAP_SYMBOL(MPI_Start)
MP1_WRAP_SYMBOL(MPI_Startall)

// fixed collectives (the perf_calls gate asserts one native _c spelling per flat call)
MP1_WRAP_SYMBOL(MPI_Barrier)
MP1_WRAP_SYMBOL(MPI_Gather_c)
MP1_WRAP_SYMBOL(MPI_Scatter_c)
MP1_WRAP_SYMBOL(MPI_Allgather_c)
MP1_WRAP_SYMBOL(MPI_Alltoall_c)

// variable collectives
MP1_WRAP_SYMBOL(MPI_Gatherv)
MP1_WRAP_SYMBOL(MPI_Gatherv_c)
MP1_WRAP_SYMBOL(MPI_Igatherv)
MP1_WRAP_SYMBOL(MPI_Igatherv_c)
MP1_WRAP_SYMBOL(MPI_Scatterv)
MP1_WRAP_SYMBOL(MPI_Scatterv_c)
MP1_WRAP_SYMBOL(MPI_Iscatterv)
MP1_WRAP_SYMBOL(MPI_Iscatterv_c)
MP1_WRAP_SYMBOL(MPI_Allgatherv)
MP1_WRAP_SYMBOL(MPI_Allgatherv_c)
MP1_WRAP_SYMBOL(MPI_Iallgatherv)
MP1_WRAP_SYMBOL(MPI_Iallgatherv_c)
MP1_WRAP_SYMBOL(MPI_Alltoallv)
MP1_WRAP_SYMBOL(MPI_Alltoallv_c)
MP1_WRAP_SYMBOL(MPI_Ialltoallv)
MP1_WRAP_SYMBOL(MPI_Ialltoallv_c)

// forbidden Alltoallw engine: wrapped so a call-count gate can assert exactly zero
MP1_WRAP_SYMBOL(MPI_Alltoallw)
MP1_WRAP_SYMBOL(MPI_Alltoallw_c)
MP1_WRAP_SYMBOL(MPI_Ialltoallw)
MP1_WRAP_SYMBOL(MPI_Ialltoallw_c)

// reductions
MP1_WRAP_SYMBOL(MPI_Reduce)
MP1_WRAP_SYMBOL(MPI_Reduce_c)
MP1_WRAP_SYMBOL(MPI_Reduce_local_c)
MP1_WRAP_SYMBOL(MPI_Ireduce)
MP1_WRAP_SYMBOL(MPI_Ireduce_c)
MP1_WRAP_SYMBOL(MPI_Allreduce)
MP1_WRAP_SYMBOL(MPI_Allreduce_c)
MP1_WRAP_SYMBOL(MPI_Iallreduce)
MP1_WRAP_SYMBOL(MPI_Iallreduce_c)
MP1_WRAP_SYMBOL(MPI_Reduce_scatter)
MP1_WRAP_SYMBOL(MPI_Reduce_scatter_c)
MP1_WRAP_SYMBOL(MPI_Ireduce_scatter)
MP1_WRAP_SYMBOL(MPI_Ireduce_scatter_c)
MP1_WRAP_SYMBOL(MPI_Reduce_scatter_block)
MP1_WRAP_SYMBOL(MPI_Reduce_scatter_block_c)
MP1_WRAP_SYMBOL(MPI_Ireduce_scatter_block)
MP1_WRAP_SYMBOL(MPI_Ireduce_scatter_block_c)
MP1_WRAP_SYMBOL(MPI_Scan)
MP1_WRAP_SYMBOL(MPI_Scan_c)
MP1_WRAP_SYMBOL(MPI_Iscan)
MP1_WRAP_SYMBOL(MPI_Iscan_c)
MP1_WRAP_SYMBOL(MPI_Exscan)
MP1_WRAP_SYMBOL(MPI_Exscan_c)
MP1_WRAP_SYMBOL(MPI_Iexscan)
MP1_WRAP_SYMBOL(MPI_Iexscan_c)

// layout / staging
MP1_WRAP_SYMBOL(MPI_Type_dup)
MP1_WRAP_SYMBOL(MPI_Type_commit)
MP1_WRAP_SYMBOL(MPI_Type_free)
MP1_WRAP_SYMBOL(MPI_Get_address)
MP1_WRAP_SYMBOL(MPI_Get_count_c)
MP1_WRAP_SYMBOL(MPI_Type_create_hindexed)
MP1_WRAP_SYMBOL(MPI_Pack)
MP1_WRAP_SYMBOL(MPI_Pack_c)
MP1_WRAP_SYMBOL(MPI_Pack_size)
MP1_WRAP_SYMBOL(MPI_Pack_size_c)
MP1_WRAP_SYMBOL(MPI_Unpack)
MP1_WRAP_SYMBOL(MPI_Unpack_c)

// RMA data movement counted by the homogeneous-contract test (a rejected untrusted annotation
// must not reach a data-movement call) and the datatype-walk metadata queries (the one-time
// cache must not re-walk).
MP1_WRAP_SYMBOL(MPI_Accumulate_c)
MP1_WRAP_SYMBOL(MPI_Get_accumulate_c)
MP1_WRAP_SYMBOL(MPI_Raccumulate_c)
MP1_WRAP_SYMBOL(MPI_Rget_accumulate_c)
MP1_WRAP_SYMBOL(MPI_Type_get_envelope_c)
MP1_WRAP_SYMBOL(MPI_Type_get_contents_c)

// communicator metadata (comm_world/comm_self cache gate)
MP1_WRAP_SYMBOL(MPI_Comm_size)
MP1_WRAP_SYMBOL(MPI_Comm_rank)
MP1_WRAP_SYMBOL(MPI_Comm_test_inter)

// process buffer (dense MPI_Count forms; the int forms stay exempt)
MP1_WRAP_SYMBOL(MPI_Buffer_attach_c)
MP1_WRAP_SYMBOL(MPI_Buffer_detach_c)

// file data access (dense pointer+count/container forms; the int forms stay exempt)
MP1_WRAP_SYMBOL(MPI_File_read_at_c)
MP1_WRAP_SYMBOL(MPI_File_write_at_c)
MP1_WRAP_SYMBOL(MPI_File_read_c)
MP1_WRAP_SYMBOL(MPI_File_write_c)
MP1_WRAP_SYMBOL(MPI_File_read_shared_c)
MP1_WRAP_SYMBOL(MPI_File_write_shared_c)
MP1_WRAP_SYMBOL(MPI_File_read_at_all_c)
MP1_WRAP_SYMBOL(MPI_File_write_at_all_c)
MP1_WRAP_SYMBOL(MPI_File_read_all_c)
MP1_WRAP_SYMBOL(MPI_File_write_all_c)
MP1_WRAP_SYMBOL(MPI_File_read_ordered_c)
MP1_WRAP_SYMBOL(MPI_File_write_ordered_c)
MP1_WRAP_SYMBOL(MPI_File_read_at_all_begin_c)
MP1_WRAP_SYMBOL(MPI_File_write_at_all_begin_c)
MP1_WRAP_SYMBOL(MPI_File_read_all_begin_c)
MP1_WRAP_SYMBOL(MPI_File_write_all_begin_c)
MP1_WRAP_SYMBOL(MPI_File_read_ordered_begin_c)
MP1_WRAP_SYMBOL(MPI_File_write_ordered_begin_c)

// file nonblocking data access (dense pointer+count/container forms)
MP1_WRAP_SYMBOL(MPI_File_iread_at_c)
MP1_WRAP_SYMBOL(MPI_File_iwrite_at_c)
MP1_WRAP_SYMBOL(MPI_File_iread_c)
MP1_WRAP_SYMBOL(MPI_File_iwrite_c)
MP1_WRAP_SYMBOL(MPI_File_iread_shared_c)
MP1_WRAP_SYMBOL(MPI_File_iwrite_shared_c)
MP1_WRAP_SYMBOL(MPI_File_iread_at_all_c)
MP1_WRAP_SYMBOL(MPI_File_iwrite_at_all_c)
MP1_WRAP_SYMBOL(MPI_File_iread_all_c)
MP1_WRAP_SYMBOL(MPI_File_iwrite_all_c)

// ownership
MP1_WRAP_SYMBOL(MPI_Bcast)
MP1_WRAP_SYMBOL(MPI_Comm_free)
MP1_WRAP_SYMBOL(MPI_Comm_disconnect)
MP1_WRAP_SYMBOL(MPI_Comm_spawn)
MP1_WRAP_SYMBOL(MPI_Comm_spawn_multiple)
MP1_WRAP_SYMBOL(MPI_Comm_get_parent)
MP1_WRAP_SYMBOL(MPI_Win_free)
MP1_WRAP_SYMBOL(MPI_Group_free)
MP1_WRAP_SYMBOL(MPI_File_close)
MP1_WRAP_SYMBOL(MPI_Info_free)
MP1_WRAP_SYMBOL(MPI_Errhandler_free)
MP1_WRAP_SYMBOL(MPI_Op_create)
MP1_WRAP_SYMBOL(MPI_Op_free)

#undef MP1_WRAP_SYMBOL
