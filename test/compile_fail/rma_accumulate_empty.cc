// Negative probe: an aggregate with no basic components has no reduction category, so the
// accumulating families must reject it at compile time.  layout<T>::contiguous(0) for a non-empty T
// remains a valid zero-element call (see test_rma_homogeneous_contract).

#include "mp1/mp1.hpp"

struct empty_value {};

MP1_REFLECTION_EMPTY(empty_value)

int main() {
  const mp1::mpi_window w;
  empty_value origin{};
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_SUM);
  return 0;
}
