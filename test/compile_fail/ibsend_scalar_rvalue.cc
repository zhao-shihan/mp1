// Compile-fail: a buffered nonblocking send keeps the scalar address until
// completion; a scalar rvalue has no stable address.
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  (void)mp1::ibsend(c, 42, 1);
}
