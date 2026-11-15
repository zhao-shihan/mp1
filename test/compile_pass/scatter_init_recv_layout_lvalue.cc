// Positive control for the deleted rvalue-layout scatter probes: the persistent scatter takes
// a region on the receive side and the layout is duplicated at initiation.
#include "mp1/impl/collective.hpp"

#include <span>

void control(const mp1::mpi_communicator& c) {
  int recv = 0;
  mp1::layout<int> recvl = mp1::layout<int>::contiguous(1);
  mp1::region<int> recv_region{&recv, recvl};
  (void)mp1::scatter_init(c, 0, std::span<const int>{}, recv_region);

  int send[2] = {0, 0};
  mp1::layout<int> sendl = mp1::layout<int>::contiguous(1);
  mp1::region<int> send_region{send, sendl};
  (void)mp1::scatter_init(c, 0, send_region, recv_region);
}
