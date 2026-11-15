// Positive control for compile_fail/iscatter_container_in_place_rvalue.cc: a named (including
// const) container is borrowable for the in-place nonblocking scatter.
#include "mp1/impl/collective.hpp"

#include <vector>

void control(const mp1::mpi_communicator& c) {
  std::vector<int> send(4, 0);
  (void)mp1::iscatter(c, 0, send, mp1::in_place);
  const std::vector<int> const_send(4, 0);
  (void)mp1::iscatter(c, 0, const_send, mp1::in_place);
}
