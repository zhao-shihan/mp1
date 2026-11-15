// Positive control for compile_fail/recv_init_raw_count.cc: the persistent receive capacity is
// spelled with mp1::capacity, which selects the fixed-capacity container overload.
#include "mp1/impl/p2p.hpp"

#include <vector>

void control(const mp1::mpi_communicator& c) {
  std::vector<int> data;
  mp1::request r = mp1::recv_init(c, data, mp1::capacity{4}, 0);
  (void)r;
}
