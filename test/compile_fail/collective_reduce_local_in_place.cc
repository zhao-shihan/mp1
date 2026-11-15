// Compile-fail: reduce_local has no in-place form ("not allowed", MPI 4.1 §6.9.7); both
// positions are plain send/receive.
#include "mp1/mp1.hpp"

#include <functional>

void probe(const mp1::mpi_communicator& c) {
  int recv = 0;
  (void)c;
  mp1::reduce_local(std::plus<int>{}, mp1::in_place, recv);
}
