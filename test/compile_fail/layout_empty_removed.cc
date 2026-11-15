// Compile-fail: `layout<T>::empty()` is removed.  `contiguous(0)` is the only zero-layout
// spelling; the explicit empty *message* is `mp1::empty_data`, not an empty datatype.
//
// Registered through mp1_add_compile_fail; the positive control is
// test/compile_pass/layout_zero_control.cc.
#include "mp1/mp1.hpp"

int main() {
  const mp1::layout<int> zero = mp1::layout<int>::empty();
  (void)zero;
  return 0;
}
