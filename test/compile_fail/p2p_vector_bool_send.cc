#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: std::vector<bool> is bit-packed with proxy references; it is not a
// contiguous range of bool objects.
void probe(const mp1::mpi_communicator& c) {
  std::vector<bool> flags{true, false};
  mp1::send(c, flags, 0);
}
