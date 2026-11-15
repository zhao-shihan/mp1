// Positive control for compile_fail/send_init_container_rvalue.cc.
#include "mp1/impl/p2p.hpp"

#include <vector>

int main() {
  const mp1::mpi_communicator c;
  std::vector<int> data(4);
  mp1::request r = mp1::send_init(c, data, 1);
  (void)r;
  return 0;
}
