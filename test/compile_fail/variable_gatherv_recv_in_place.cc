// Compile-fail: the in-place sentinel belongs to gatherv's **send** position; the aggregate
// receive position is `recv_sink`, which excludes `in_place`.
#include "mp1/mp1.hpp"

void probe(const mp1::mpi_communicator& c, int* recv) {
  (void)mp1::gatherv(c, 0, 7, mp1::in_place, mp1::blocks{});
}
