// Positive control for the dense scalar-rvalue compile-fail probes: a named lvalue scalar send
// is the supported spelling for the nonblocking and persistent dense variable collectives.
#include "mp1/impl/collective.hpp"

void control(const mp1::mpi_communicator& c) {
  int send = 42;
  int recv[2] = {0, 0};
  const std::span<int> recv_span{recv, 2};
  const std::vector<mp1::count_t> counts{1, 1};
  const std::vector<mp1::count_t> displs{0, 1};
  const mp1::blocks dense{counts, displs};
  mp1::request r1 = mp1::igatherv(c, 0, send, recv_span, dense);
  mp1::request r2 = mp1::gatherv_init(c, 0, send, recv_span, dense);
  mp1::request r3 = mp1::iallgatherv(c, send, recv_span, dense);
  mp1::request r4 = mp1::allgatherv_init(c, send, recv_span, dense);
  (void)r1;
  (void)r2;
  (void)r3;
  (void)r4;
}
