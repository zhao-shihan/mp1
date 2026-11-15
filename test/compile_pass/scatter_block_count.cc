// Positive control for compile_fail/scatter_raw_count.cc: the per-peer scatter block is the
// receive span itself (the former `block_count` value type is deleted).  A non-root rank uses
// the documented empty send span; a raw integer is not a form in that arity.
#include "mp1/mp1.hpp"

#include <span>

int main() {
  const mp1::mpi_communicator c;
  int send[2] = {1, 2};
  int scalar = 0;
  mp1::scatter(c, 0, std::span<const int>{send}, scalar);
  int recv[2] = {0, 0};
  mp1::scatter(c, 0, std::span<const int>{}, std::span<int>{recv});
  return 0;
}
