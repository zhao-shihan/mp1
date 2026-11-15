// Positive control for compile_fail/persistent_collective_container.cc: a persistent
// collective accepts fixed buffers only (span/array/C array/region/scalar), not a resizable
// container.  The span spelling is the direct replacement for the deleted pointer+count form.
#include "mp1/impl/collective.hpp"

#include <functional>
#include <span>
#include <vector>

void control(const mp1::mpi_communicator& c) {
  std::vector<int> send(2, 0);
  std::vector<int> recv(2, 0);
  std::plus<int> plus;
  mp1::request r = mp1::reduce_init(c, plus, 0, std::span<int>{send}, std::span<int>{recv});
  (void)r;
}
