// Compile-fail: the in-place sentinel belongs to scatterv's **receive** position;
// the aggregate send position is `send_source`, which excludes `in_place`.
#include "mp1/mp1.hpp"

void probe(const mp1::mpi_communicator& c, int* recv) {
  const std::span<int> recv_span{recv, 2};
  (void)mp1::scatterv(c, 0, mp1::in_place, mp1::blocks{}, recv_span);
}
