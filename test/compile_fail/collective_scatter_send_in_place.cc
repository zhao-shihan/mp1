// Compile-fail: scatter's send position is `send_source`, which excludes the in-place
// sentinel.  The sentinel is legal only in the receive slot.
#include "mp1/mp1.hpp"

void probe(const mp1::mpi_communicator& c) {
  int recv = 0;
  mp1::scatter(c, 0, mp1::in_place, recv);
}
