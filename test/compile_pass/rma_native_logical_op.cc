// Positive compile check of the predefined logical operations in RMA: MPI_LAND is accepted for
// a raw C-integer handle and for the typed C-integer/Logical functors (MPI 4.1 §6.9.2), and the
// transparent mp1::logical_xor<> maps for int and for bool (MPI_CXX_BOOL is Logical).
#include "mp1/mp1.hpp"

#include <functional>

int main() {
  const mp1::mpi_window w;
  const int origin = 0;
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_LAND);
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, std::logical_and<int>{});
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, mp1::logical_xor{});
  const bool flag = false;
  mp1::accumulate(w, mp1::span_from(&flag, 1), 0, 0, 1, mp1::logical_xor{});
  return 0;
}
