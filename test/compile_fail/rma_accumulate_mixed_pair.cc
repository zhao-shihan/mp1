// Negative probe: std::pair<int,double> has two distinct known basic components, so the
// accumulating families must reject it at compile time (MPI 4.1 §12.3.4).

#include "mp1/mp1.hpp"

#include <utility>

int main() {
  const mp1::mpi_window w;
  std::pair<int, double> origin{};
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_SUM);
  return 0;
}
