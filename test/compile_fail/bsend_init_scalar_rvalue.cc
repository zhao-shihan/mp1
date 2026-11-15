// Compile-fail: a persistent buffered send keeps the scalar address until the
// request is freed; a scalar rvalue has no stable address.
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  (void)mp1::bsend_init(c, 42, 1);
}
