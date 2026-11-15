// Positive control for compile_fail/issend_scalar_rvalue.cc.
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  int value = 0;
  mp1::request r = mp1::issend(c, value, 1);
  (void)r;
  return 0;
}
