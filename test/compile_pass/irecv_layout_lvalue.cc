// Positive control for the typed nonblocking receive: a named layout in a region.
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  int data = 0;
  mp1::layout<int> operation_layout = mp1::layout<int>::contiguous(1);
  mp1::request r = mp1::irecv(c, mp1::region{&data, operation_layout}, 0);
  (void)r;
  return 0;
}
