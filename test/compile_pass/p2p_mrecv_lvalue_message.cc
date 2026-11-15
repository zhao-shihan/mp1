#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

void control(const mp1::mpi_communicator& c) {
  int value = 0;
  mp1::message message = mp1::mprobe(c);
  (void)mp1::mrecv(value, message);
}
