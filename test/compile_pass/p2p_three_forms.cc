// Three-form positive control: value / range / region for the send and receive families.
#include "mp1/impl/p2p.hpp"

#include <array>
#include <span>
#include <vector>

void control(const mp1::mpi_communicator& c) {
  int value = 0;
  int data[4] = {0, 0, 0, 0};
  std::vector<int> range(4, 0);
  std::array<int, 4> fixed{};
  mp1::layout<int> map = mp1::layout<int>::contiguous(4);

  mp1::send(c, value, 0);
  mp1::send(c, range, 0);
  mp1::send(c, std::span(data), 0);
  mp1::send(c, mp1::region{data, map}, 0);

  (void)mp1::recv(c, value, 0);
  (void)mp1::recv(c, std::span(fixed), 0);
  (void)mp1::recv(c, range, 0);
  (void)mp1::recv(c, mp1::region{data, map}, 0);
}
