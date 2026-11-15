// Compile-fail: request-returning operations are [[nodiscard]].
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  int value = 0;
  mp1::isend(c, value, 1);
}
