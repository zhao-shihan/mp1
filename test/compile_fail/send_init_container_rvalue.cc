// Compile-fail: a persistent send borrows the container storage for every
// restart; an rvalue container has no stable address.
#include "mp1/impl/p2p.hpp"

#include <vector>

int main() {
  const mp1::mpi_communicator c;
  (void)mp1::send_init(c, std::vector<int>(4), 1);
}
