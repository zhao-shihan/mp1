// Negative probe: a nonblocking variable collective keeps the send buffer address until
// completion (MPI 4.1 §6.12), so a scalar send must be a stable lvalue.
//
// The positive control is test/compile_pass/collectives_lvalue_gatherv.cc.

#include "mp1/mp1.hpp"

void allgather_temporary(const mp1::mpi_communicator& c, int* recv) {
  const std::span<int> recv_span{recv, 2};
  (void)mp1::iallgatherv(c, 42, recv_span, mp1::blocks{});
}
