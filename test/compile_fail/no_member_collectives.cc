// Compile-fail: p2p/collective operations are free functions; no member collectives exist.
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  int value = 0;
  c.gather(&value, 1, 0);
}
