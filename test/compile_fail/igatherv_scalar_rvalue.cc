// Negative probe: a nonblocking variable collective keeps the send buffer address until
// completion (MPI 4.1 §6.12), so a scalar send must be a stable lvalue.  The single-block
// position is constrained by borrowable<S&&> in the nonblocking declarations.
//
// The positive control is test/compile_pass/collectives_lvalue_gatherv.cc.

#include "mp1/mp1.hpp"

void gather_temporary(const mp1::mpi_communicator& c, int* recv) {
  const std::span<int> recv_span{recv, 2};
  (void)mp1::igatherv(c, 0, 42, recv_span, mp1::blocks{});
}
