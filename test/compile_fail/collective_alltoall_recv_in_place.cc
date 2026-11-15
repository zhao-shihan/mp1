// Compile-fail: alltoall's receive position is `recv_sink`; the in-place sentinel occupies
// the send position only.
#include "mp1/mp1.hpp"

void probe(const mp1::mpi_communicator& c) {
  int send = 1;
  mp1::alltoall(c, send, mp1::in_place);
}
