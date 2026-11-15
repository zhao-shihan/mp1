#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: a built-in character array is a byte buffer, not a string, and is excluded
// on receive.  Use std::span<char> or std::array<char,N>.
void probe(const mp1::mpi_communicator& c) {
  char buffer[4] = {};
  (void)mp1::recv(c, buffer, 0);
}
