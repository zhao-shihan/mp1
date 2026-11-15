// Negative probe: the typed persistent allgatherv captures the scalar send address at init;
// a prvalue scalar has no stable address through the request lifetime (MPI 4.1 §6.13).

#include "mp1/mp1.hpp"

void probe(const mp1::mpi_communicator& c, int* recv) {
  const std::span<int> recv_span{recv, 2};
  std::vector<mp1::layout<int>> layout_owner(2, mp1::layout<int>::contiguous(1));
  const std::span<const mp1::layout<int>> layouts{layout_owner};
  const std::vector<mp1::count_t> displs(2, 0);
  (void)mp1::allgatherv_init(c, 42, recv_span, layouts, displs);
}
