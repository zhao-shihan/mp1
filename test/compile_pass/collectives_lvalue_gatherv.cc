// Positive control for the variable-collective rvalue probes: the same nonblocking calls
// accept a named lvalue scalar send, whose address stays stable until the request completes.

#include "mp1/mp1.hpp"

void gather_lvalues(const mp1::mpi_communicator& c, int send, int* recv) {
  const std::span<int> recv_span{recv, 2};
  (void)mp1::igatherv(c, 0, send, recv_span, mp1::blocks{});
  (void)mp1::iallgatherv(c, send, recv_span, mp1::blocks{});
}
