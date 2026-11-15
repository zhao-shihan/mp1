#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: a sequence container is not a receive target (order/duplicate semantics,
// §2.5).  Use std::vector/std::array/std::span.
void probe(const mp1::mpi_communicator& c) {
  std::list<int> buffer;
  (void)mp1::recv(c, buffer, 0);
}
