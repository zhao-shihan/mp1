// Positive control for compile_fail/region_array_temporary_layout.cc: one std::array object
// as one datatype with the required *named* layout.
#include "mp1/impl/layout.hpp"
#include "mp1/impl/vocabulary.hpp"

#include <array>

int main() {
  std::array<int, 3> object{};
  const mp1::layout<std::array<int, 3>> map = mp1::layout<std::array<int, 3>>::contiguous(1);
  mp1::region accepted{&object, map};
  return accepted.data() == &object and accepted.map().size() == 1 ? 0 : 1;
}
