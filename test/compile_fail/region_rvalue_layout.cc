// Compile-fail: a prvalue layout must not bind to a region.
// The deleted `region(T*, const layout<...>&&)` constructor is the local guard that makes
// `region{p, layout<T>{...}}` an error instead of a dangling reference member.
#include "mp1/impl/layout.hpp"
#include "mp1/impl/vocabulary.hpp"

int main() {
  int value = 0;
  mp1::region<int> rejected{&value, mp1::layout<int>::contiguous(1)};
  (void)rejected;
}
