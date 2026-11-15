// Positive control for compile_fail/region_ctad_nullptr.cc: the explicit-T null spelling.
#include "mp1/impl/layout.hpp"
#include "mp1/impl/vocabulary.hpp"

int main() {
  mp1::layout<int> map = mp1::layout<int>::contiguous(1);
  mp1::region<int> accepted{nullptr, map};
  return accepted.data() == nullptr and &accepted.map() == &map ? 0 : 1;
}
