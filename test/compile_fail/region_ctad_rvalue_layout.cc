// Compile-fail: CTAD deduces T from the pointer and then selects the deleted
// `region(T*, const layout<...>&&)` constructor for a prvalue layout.
#include "mp1/impl/layout.hpp"
#include "mp1/impl/vocabulary.hpp"

int main() {
  int value = 0;
  mp1::region rejected{&value, mp1::layout<int>::contiguous(1)};
  (void)rejected;
}
