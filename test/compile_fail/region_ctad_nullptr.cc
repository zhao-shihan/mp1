// Compile-fail: `region{nullptr, l}` is not a CTAD spelling.
// T cannot be deduced from nullptr; the supported null spelling is `region<int>{nullptr, l}`.
#include "mp1/impl/layout.hpp"
#include "mp1/impl/vocabulary.hpp"

int main() {
  mp1::layout<int> map = mp1::layout<int>::contiguous(1);
  mp1::region rejected{nullptr, map};
  (void)rejected;
}
