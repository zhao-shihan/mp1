// Positive control for the p2p three-form probes: a named layout in a region is the
// supported typed spelling (the old pointer+layout overloads are deleted).
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  int value = 0;
  mp1::layout<int> operation_layout = mp1::layout<int>::contiguous(1);
  mp1::request r = mp1::isend(c, mp1::region{&value, operation_layout}, 1);
  (void)r;
  return 0;
}
