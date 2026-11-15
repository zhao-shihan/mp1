// Negative probe: a nested homogeneous aggregate combined with a different basic type is
// mixed at the outer level: pair<pair<int,int>,double> has int and double components
// (MPI 4.1 §12.3.4).

#include "mp1/mp1.hpp"

#include <tuple>
#include <utility>

int main() {
  const mp1::mpi_window w;
  std::pair<std::pair<int, int>, double> pair_origin{};
  mp1::accumulate(w, mp1::span_from(&pair_origin, 1), 0, 0, 1, MPI_SUM);
  return 0;
}
