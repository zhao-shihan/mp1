// Compile-fail: a nonblocking send borrows the container storage until the
// request completes; an rvalue container has no stable address.
#include "mp1/impl/p2p.hpp"

#include <vector>

int main() {
  const mp1::mpi_communicator c;
  (void)mp1::isend(c, std::vector<int>(4), 1);
}
