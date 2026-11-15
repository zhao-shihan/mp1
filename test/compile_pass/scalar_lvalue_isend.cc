// Positive control for compile_fail/scalar_rvalue_isend.cc: an lvalue scalar
// may bind to the asynchronous send.
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  int value = 0;
  mp1::request r = mp1::isend(c, value, 1);
  (void)r;
  return 0;
}
