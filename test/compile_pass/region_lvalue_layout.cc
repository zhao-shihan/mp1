// Positive control for compile_fail/region_rvalue_layout.cc and
// compile_fail/region_ctad_rvalue_layout.cc: an lvalue layout binds to a region, CTAD
// deduces the element type from the pointer, and copy/move construction are legal.
#include "mp1/impl/layout.hpp"
#include "mp1/impl/vocabulary.hpp"

#include <utility>

int main() {
  int value = 0;
  mp1::layout<int> map = mp1::layout<int>::contiguous(1);

  mp1::region deduced{&value, map};  // CTAD -> region<int>
  mp1::region<int> explicit_t{&value, map};
  mp1::region<const int> const_region{&value, map};
  mp1::region<int> null_data{nullptr, map};
  const mp1::region<int> copy{deduced};
  const mp1::region<int> moved{std::move(explicit_t)};

  (void)const_region;
  (void)null_data;
  (void)copy;
  (void)moved;
  return deduced.data() == &value and &deduced.map() == &map ? 0 : 1;
}
