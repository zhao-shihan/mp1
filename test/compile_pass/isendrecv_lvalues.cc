// Positive control for the isendrecv compile-fail probes: named lvalue scalar, container and
// region arguments are the supported forms.
#include "mp1/impl/p2p.hpp"

#include <vector>

void control(const mp1::mpi_communicator& c) {
  int send_scalar = 0;
  int recv_scalar = 0;
  mp1::request r1 = mp1::isendrecv(c, send_scalar, 0, mp1::tag{0}, recv_scalar, 1, mp1::tag{0});

  std::vector<int> send_container(1, 0);
  std::vector<int> recv_container(1, 0);
  mp1::request r2 = mp1::isendrecv(c, send_container, 0, mp1::tag{0}, recv_container, 1, mp1::tag{0});

  int send_buffer[1] = {0};
  int recv_buffer[1] = {0};
  mp1::layout<int> sendl = mp1::layout<int>::contiguous(1);
  mp1::layout<int> recvl = mp1::layout<int>::contiguous(1);
  mp1::request r3 = mp1::isendrecv(c, mp1::region{send_buffer, sendl}, 0, mp1::tag{0},
                                   mp1::region{recv_buffer, recvl}, 1, mp1::tag{0});
  (void)r1;
  (void)r2;
  (void)r3;
}
