// Positive control for compile_fail/no_inter_communicator.cc: the supported
// communicator and integer tag form remain available.
#include "mp1/impl/communicator.hpp"

int main() {
  const mp1::mpi_communicator c;
  int tag = 0;
  (void)c;
  (void)tag;
  return 0;
}
