// Negative probe: the dense `blocks` nonblocking allgatherv keeps the scalar send address
// until completion (MPI 4.1 §6.12); the single-block position is constrained by
// borrowable<S&&>.

#include "mp1/mp1.hpp"

void probe(const mp1::mpi_communicator& c, int* recv) {
  const std::span<int> recv_span{recv, 2};
  const std::vector<mp1::count_t> counts{1, 1};
  const std::vector<mp1::count_t> displs{0, 1};
  (void)mp1::iallgatherv(c, 42, recv_span, mp1::blocks{counts, displs});
}
