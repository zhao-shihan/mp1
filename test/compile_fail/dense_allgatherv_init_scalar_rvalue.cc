// Negative probe: a persistent allgatherv captures the scalar send address at init; an
// unnamed temporary has no stable address through the request lifetime (MPI 4.1 §6.13).

#include "mp1/mp1.hpp"

void probe(const mp1::mpi_communicator& c, int* recv) {
  const std::span<int> recv_span{recv, 2};
  const std::vector<mp1::count_t> counts{1, 1};
  const std::vector<mp1::count_t> displs{0, 1};
  (void)mp1::allgatherv_init(c, 42, recv_span, mp1::blocks{counts, displs});
}
