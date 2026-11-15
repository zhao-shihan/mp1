// Compile-fail: a combined nonblocking send-receive keeps the send buffer address until
// completion (MPI 4.1 §3.7.2), so a temporary scalar send has no stable address.
#include "mp1/impl/p2p.hpp"

void probe(const mp1::mpi_communicator& c) {
  int recv = 0;
  (void)mp1::isendrecv(c, 42, 0, mp1::tag{0}, recv, 1, mp1::tag{0});
}
