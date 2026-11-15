// Negative probe: a C array is classified only as a member type.  The public accumulating
// surface takes a pointer to one element type, so a top-level C array T (here int[3] via a
// pointer to array) is not supported; use std::array<int,3> or a reflected struct with a C
// array member.  This encodes the frozen 1.0 decision.

#include "mp1/mp1.hpp"

#include <span>

int main() {
  const mp1::mpi_window w;
  int storage[2][3] = {};
  int (*origin)[3] = storage;
  std::span<int[3]> origin_span{origin, 1};
  mp1::accumulate(w, origin_span, 0, 0, 1, MPI_SUM);
  return 0;
}
