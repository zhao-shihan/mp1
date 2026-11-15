// Compile-fail: the four-argument container receive with a `count_t` third argument is the
// removed untyped capacity spelling.  Without the deleted overload it would silently bind the
// auto-size form and read the capacity as a source rank; with it the move to
// `mp1::capacity` is a compile error.
//
// Registered through mp1_add_compile_fail; the positive control is
// test/compile_pass/capacity_recv.cc.
#include "mp1/mp1.hpp"

#include <vector>

int main() {
  const mp1::mpi_communicator c;
  std::vector<int> values;
  const mp1::count_t capacity = 4;
  (void)mp1::recv(c, values, capacity, mp1::tag{0});
  return 0;
}
