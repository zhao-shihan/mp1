// Compile-pass control: group_incl/group_excl take std::span<const int>, and the range
// constructors take std::span<const rank_range>.  A std::array, std::vector, C array, span and
// a span slice all convert (a braced initializer list does not; see the compile-fail probe).

#include "mp1/mp1.hpp"

#include <array>
#include <span>
#include <vector>


void group_span_forms(const mp1::group& g) {
  std::array<int, 2> array_ranks{0, 1};
  std::vector<int> vector_ranks{0, 1};
  int c_ranks[2]{0, 1};
  std::span<const int> span_ranks{array_ranks};

  mp1::group included_array = mp1::group_incl(g, array_ranks);
  mp1::group excluded_vector = mp1::group_excl(g, vector_ranks);
  mp1::group included_c = mp1::group_incl(g, c_ranks);
  mp1::group excluded_span = mp1::group_excl(g, span_ranks);
  mp1::group excluded_slice = mp1::group_excl(g, span_ranks.first(1));
  (void)included_array;
  (void)excluded_vector;
  (void)included_c;
  (void)excluded_span;
  (void)excluded_slice;

  std::array<mp1::rank_range, 2> ranges{mp1::rank_range{0, 1, 1}, mp1::rank_range{1, 0, -1}};
  std::vector<mp1::rank_range> range_vector{mp1::rank_range{0, 0, 1}};
  mp1::group range_included = mp1::group_range_incl(g, ranges);
  mp1::group range_excluded = mp1::group_range_excl(g, range_vector);
  mp1::group range_slice =
      mp1::group_range_incl(g, std::span<const mp1::rank_range>{ranges}.first(1));
  (void)range_included;
  (void)range_excluded;
  (void)range_slice;
}
