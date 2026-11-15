// Compile-fail: the persistent capacity receive spells its bound with mp1::capacity; a raw
// integer in the capacity position must not select the pointer form or the auto-size form.
#include "mp1/impl/p2p.hpp"

#include <vector>

void probe(const mp1::mpi_communicator& c) {
  std::vector<int> data;
  (void)mp1::recv_init(c, data, 4, mp1::tag{0});
}
