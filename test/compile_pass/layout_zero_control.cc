// Positive control for compile_fail/layout_empty_removed.cc: the zero layout is
// `contiguous(0)`; the removed `layout<T>::empty()` factory has no replacement spelling.
#include "mp1/mp1.hpp"

namespace {

  [[maybe_unused]] void zero_layout_spellings() {
    const mp1::layout<int> zero = mp1::layout<int>::contiguous(0);
    const mp1::layout<int> repeated = mp1::layout<int>::contiguous(3, zero);
    const mp1::layout<int> vector_block = mp1::layout<int>::vector(2, 1, 1, zero);
    const bool valid = zero.is_valid();
    const mp1::count_t size = zero.size();
    (void)repeated;
    (void)vector_block;
    (void)valid;
    (void)size;
  }

}  // namespace
