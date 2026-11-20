// group_range_incl takes std::span<const rank_range>; a positional std::array<int, 3> does not
// convert, so first/last/stride cannot be silently swapped.

#include "mp1/mp1.hpp"

#include <array>
#include <vector>


void range(const mp1::group& g) {
  const std::vector<std::array<int, 3>> ranges{{0, 1, 1}};
  (void)mp1::group_range_incl(g, ranges);
}
