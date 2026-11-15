// Compile-fail: the one-object std::array spelling with a prvalue layout.
// A named layout is required; `region{&array, layout<std::array<T,N>>::contiguous(1)}`
// intentionally selects the deleted rvalue-layout constructor.
#include "mp1/impl/layout.hpp"
#include "mp1/impl/vocabulary.hpp"

#include <array>

int main() {
  std::array<int, 3> object{};
  mp1::region rejected{&object, mp1::layout<std::array<int, 3>>::contiguous(1)};
  (void)rejected;
}
