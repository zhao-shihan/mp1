#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: an associative container is not a receive target (§2.5).
void probe(const mp1::mpi_communicator& c) {
  std::set<int> values;
  (void)mp1::recv(c, values, 0);
}
