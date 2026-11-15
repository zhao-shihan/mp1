#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: std::forward_list is unsized; an unsized send range is excluded (§2.5).
#include <forward_list>

void probe(const mp1::mpi_communicator& c) {
  std::forward_list<int> values{1, 2};
  mp1::send(c, values, 0);
}
