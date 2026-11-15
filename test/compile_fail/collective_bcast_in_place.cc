// Compile-fail: bcast has one in/out buffer and no in-place form ("not meaningful",
// MPI 4.1 §6.4); the buffer position is `recv_sink`, which excludes the sentinel.
#include "mp1/mp1.hpp"

void probe(const mp1::mpi_communicator& c) {
  mp1::bcast(c, 0, mp1::in_place);
}
