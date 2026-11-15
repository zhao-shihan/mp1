// Compile-fail: a scalar rvalue must not bind to an asynchronous send.
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  (void)mp1::isend(c, 42, 1);
}
