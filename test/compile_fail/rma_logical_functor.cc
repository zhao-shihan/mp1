// Negative probe: a C-integer logical functor has no exact predefined RMA mapping.  MPI 4.1
// §6.9.2 (Predefined Reduction Operations) allows MPI_LAND/MPI_LOR/MPI_LXOR for C integers
// and Logical types, but it does not specify the numeric result for the C-integer case, and
// RMA has no callback fallback.  bool logical functors do map (MPI_CXX_BOOL is Logical).

#include "mp1/mp1.hpp"

#include <functional>

int main() {
  const mp1::mpi_window w;
  const int origin = 3;
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, std::logical_and<int>{});
  return 0;
}
