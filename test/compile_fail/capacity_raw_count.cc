// Compile-fail: the resizable-container receive upper bound is the strong
// `mp1::capacity` type.  A raw count in the third argument of the five-argument container
// form no longer binds -- before the strong type this spelling silently selected the
// untyped capacity overload, and the four-argument form additionally collided with the
// auto-size `(source, tag)` overload.
//
// Registered through mp1_add_compile_fail; the positive control is
// test/compile_pass/capacity_recv.cc.
#include "mp1/mp1.hpp"

#include <vector>

int main() {
  const mp1::mpi_communicator c;
  std::vector<int> values;
  (void)mp1::recv(c, values, static_cast<mp1::count_t>(4), 0, mp1::tag{0});
  return 0;
}
