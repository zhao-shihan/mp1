// Positive control for compile_fail/rma_logical_functor.cc: a raw predefined MPI_Op is
// accepted when it is defined for the element category (MPI_LAND is valid for C integers
// under MPI 4.1 §6.9.2), unlike the unmappable integer logical functor.
#include "mp1/mp1.hpp"

int main() {
  const mp1::mpi_window w;
  const int origin = 0;
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_LAND);
  return 0;
}
