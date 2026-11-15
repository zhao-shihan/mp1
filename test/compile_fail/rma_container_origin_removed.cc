// Negative probe: the RMA data-movement forms are span/region only; a container is not a
// data-movement form.  The caller materializes
// std::span{container} so the element count is explicit.

#include "mp1/mp1.hpp"

#include <vector>

int main() {
  const mp1::mpi_window w;
  std::vector<int> origin(4);
  mp1::put(w, origin, 0, 0, 4);
  return 0;
}
