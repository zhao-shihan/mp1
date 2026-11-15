#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

void control(const mp1::mpi_communicator& c) {
  std::array<bool, 3> flags{{true, false, true}};
  mp1::send(c, flags, 0);
}
