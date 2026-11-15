// Positive control for compile_fail/container_element_datatype.cc: the same container shape
// with a mapped element (int) selects the element-buffer send.
#include "mp1/mp1.hpp"

#include <vector>

void control(const mp1::mpi_communicator& c, int dest) {
  std::vector<int> buffer(2);
  (void)mp1::send(c, buffer, dest, mp1::tag{0});
}
