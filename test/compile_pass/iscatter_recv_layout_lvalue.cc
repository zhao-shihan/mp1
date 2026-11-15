// Positive control for the deleted rvalue-layout scatter probes: the layout form is now a
// region; a named region/layout is the supported spelling, and the nonblocking initiation
// duplicates the layout into the request payload.
#include "mp1/impl/collective.hpp"

#include <span>

void control(const mp1::mpi_communicator& c) {
  int recv = 0;
  mp1::layout<int> recvl = mp1::layout<int>::contiguous(1);
  mp1::region<int> recv_region{&recv, recvl};
  (void)mp1::iscatter(c, 0, std::span<const int>{}, recv_region);
}
