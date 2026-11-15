#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

void control(const mp1::mpi_communicator& c) {
  std::vector<std::pair<int, int>> entries{{1, 2}, {3, 4}};
  mp1::send(c, entries, 0);
}
