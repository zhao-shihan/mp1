// Compile-fail: the five layout parameter builders are
// constructed from a named contiguous range (std::array/std::vector/std::span) or with
// `.add()`; they have no initializer-list constructor, and a braced-init-list does not convert
// to std::span, so every braced spelling below must be rejected.
#include "mp1/mp1.hpp"

int main() {
  (void)mp1::indexed_parameter{{1, 2}};
  (void)mp1::hindexed_parameter{{1, 0}};
  (void)mp1::indexed_block_parameter{0, 3};
  (void)mp1::hindexed_block_parameter{0, 16};
  (void)mp1::subarray_parameter{{{4, 2, 1}, {5, 3, 2}}};
  return 0;
}
