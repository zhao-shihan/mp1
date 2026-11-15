#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: a const buffer is not writable and cannot be a receive target (§2.5).
void probe(const mp1::mpi_communicator& c) {
  const int data[4] = {0, 0, 0, 0};
  std::span<const int> buffer(data, 4);
  (void)mp1::recv(c, buffer, 0);
}
