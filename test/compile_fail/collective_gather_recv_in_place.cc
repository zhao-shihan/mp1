// Compile-fail: gather's receive position is `recv_sink`, which excludes the in-place
// sentinel.  The sentinel is legal only in the send slot.
#include "mp1/mp1.hpp"

#include <span>

void probe(const mp1::mpi_communicator& c) {
  int recv = 0;
  int send = 1;
  mp1::gather(c, 0, send, mp1::in_place);
  (void)recv;
}
