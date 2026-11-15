// Positive control for compile_fail/nodiscard_request.cc: consuming the
// returned request satisfies the [[nodiscard]] contract.
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  int value = 0;
  mp1::request r = mp1::isend(c, value, 1);
  (void)r;
  return 0;
}
