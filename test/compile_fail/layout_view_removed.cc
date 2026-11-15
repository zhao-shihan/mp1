// Compile-fail: `mp1::layout_view<T>` is not part of the API; a single-block position
// takes a `region<T>` or a contiguous element range instead.
#include "mp1/mp1.hpp"

void probe(const mp1::layout<int>& map) {
  mp1::layout_view<int> view{map};
  (void)view;
}
