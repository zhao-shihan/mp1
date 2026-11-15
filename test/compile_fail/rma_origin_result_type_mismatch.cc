// Negative probe: the origin and result of get_accumulate/rget_accumulate must have the same
// element type; MPI 4.1 §12.3.4 requires every datatype argument to be constructed from the
// same predefined basic datatype.  A span<int> origin with a span<double> result is rejected
// at compile time.

#include "mp1/mp1.hpp"

#include <span>

int main() {
  const mp1::mpi_window w;
  int origin[4] = {1, 1, 1, 1};
  double result[4] = {0., 0., 0., 0.};
  mp1::get_accumulate(w, std::span<const int>{origin, 4}, std::span<double>{result, 4}, 0, 0, 4,
                      MPI_SUM);
  return 0;
}
