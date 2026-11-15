// Positive control for compile_fail/rma_custom_functor.cc: a built-in
// reduction functor with a predefined MPI_Op mapping is accepted by RMA.
#include "mp1/mp1.hpp"

#include <functional>

int main() {
  const mp1::mpi_window w;
  const int origin = 0;
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, std::plus<int>{});
  return 0;
}
