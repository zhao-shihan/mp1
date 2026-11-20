// group_incl/group_excl take std::span<const int>: a braced initializer list does not convert.
// Spell the ranks as std::array{...} or std::vector{...}.

#include "mp1/mp1.hpp"


void include(const mp1::group& g) {
  (void)mp1::group_incl(g, {0});
}
