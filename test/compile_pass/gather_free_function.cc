// Positive control for compile_fail/no_member_collectives.cc: gather is a free
// function and the supported span form resolves.
#include "mp1/impl/collective.hpp"

#include <span>

int main() {
  const mp1::mpi_communicator c;
  int send[4] = {0, 0, 0, 0};
  int recv[4] = {0, 0, 0, 0};
  mp1::gather(c, 0, std::span<const int>{send, 1}, std::span<int>{recv});
  return 0;
}
