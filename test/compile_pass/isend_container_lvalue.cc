// Positive control for compile_fail/isend_container_rvalue.cc: a named
// container is borrowed by the nonblocking send.
#include "mp1/impl/p2p.hpp"

#include <vector>

int main() {
  const mp1::mpi_communicator c;
  std::vector<int> data(4);
  mp1::request r = mp1::isend(c, data, 1);
  (void)r;
  return 0;
}
