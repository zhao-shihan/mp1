// Positive control for compile_fail/capacity_raw_count.cc: the resizable-container
// receive upper bound is the strong `mp1::capacity` type, the fixed-extent std::array receive
// needs no capacity at all, and the four-argument container form is the auto-size receive
// (third argument = source, fourth = tag).
#include "mp1/mp1.hpp"

#include <array>
#include <vector>

int main() {
  const mp1::mpi_communicator c;
  std::vector<int> dynamic;
  std::array<int, 3> fixed{};

  (void)mp1::recv(c, dynamic, mp1::capacity{4}, 0, mp1::tag{0});
  (void)mp1::irecv(c, dynamic, mp1::capacity{4}, 0, mp1::tag{0});
  (void)mp1::recv(c, fixed, 0, mp1::tag{0});
  (void)mp1::irecv(c, fixed, 0, mp1::tag{0});
  (void)mp1::recv_init(c, fixed, 0, mp1::tag{0});
  // Four-argument auto-size form: source 0, tag 0.  There is no overload that reads the raw
  // `0` as a capacity.
  (void)mp1::recv(c, dynamic, 0, mp1::tag{0});
  return 0;
}
