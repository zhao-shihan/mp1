// Compile-fail: the non-root scatter receive position is a span (the former `block_count`
// value type is deleted), so a raw integer no longer binds in the receive slot.
//
// Registered through mp1_add_compile_fail; the positive control is
// test/compile_pass/scatter_block_count.cc.
#include "mp1/mp1.hpp"

int main() {
  const mp1::mpi_communicator c;
  int recv[2] = {0, 0};
  mp1::scatter(c, 0, recv, 2);
  return 0;
}
