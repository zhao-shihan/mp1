#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: a receive into a temporary container is useless (the resize is lost) and is
// excluded (§2.5).  Pass an lvalue.
void probe(const mp1::mpi_communicator& c) {
  (void)mp1::recv(c, std::vector<int>{}, 0);
}
