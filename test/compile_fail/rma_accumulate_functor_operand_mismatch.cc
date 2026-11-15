// Negative probe: the selected predefined_op<U> requires the functor operand to match the
// basic component U.  std::pair<int,int> selects predefined_op<int>, so std::plus<double>
// (which would change the arithmetic) is rejected at compile time.

#include "mp1/mp1.hpp"

#include <functional>
#include <utility>

int main() {
  const mp1::mpi_window w;
  std::pair<int, int> origin{};
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, std::plus<double>{});
  return 0;
}
