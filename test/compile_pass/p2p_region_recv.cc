#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

void control(const mp1::mpi_communicator& c) {
  int buffer[4] = {0, 0, 0, 0};
  mp1::layout<int> map = mp1::layout<int>::contiguous(4);
  (void)mp1::recv(c, mp1::region{buffer, map}, 0);
}
