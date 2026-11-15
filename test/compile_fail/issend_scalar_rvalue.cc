// Compile-fail: a synchronous nonblocking send keeps the scalar address until
// completion; a scalar rvalue has no stable address.
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  (void)mp1::issend(c, 42, 1);
}
