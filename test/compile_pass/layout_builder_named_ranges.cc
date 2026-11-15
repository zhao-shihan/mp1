// Positive control for `compile_fail/layout_builder_braced_init.cc`:
// every layout parameter builder accepts a named contiguous range (std::array/std::vector/
// std::span) and the `.add()` spelling, and feeds the corresponding factory.  No
// initializer-list constructor is required for any of these forms.
#include "mp1/mp1.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

int main() {
  using indexed_entry = std::pair<mp1::count_t, mp1::count_t>;
  const std::array<indexed_entry, 2> indexed_entries{indexed_entry{2, 0}, indexed_entry{1, 6}};
  const mp1::indexed_parameter indexed_array{indexed_entries};
  const std::vector<indexed_entry> indexed_vector{indexed_entries.begin(),
                                                  indexed_entries.end()};
  const std::span<const indexed_entry> indexed_span{indexed_entries};
  const mp1::indexed_parameter indexed_from_vector{indexed_vector};
  const mp1::indexed_parameter indexed_from_span{indexed_span};
  mp1::indexed_parameter indexed_added;
  indexed_added.add(2, 0);
  indexed_added.add(1, 6);
  (void)mp1::layout<int>::indexed(indexed_array);
  (void)mp1::layout<int>::indexed(indexed_from_vector);
  (void)mp1::layout<int>::indexed(indexed_from_span);
  (void)mp1::layout<int>::indexed(indexed_added);

  using hindexed_entry = std::pair<mp1::count_t, MPI_Aint>;
  const std::array<hindexed_entry, 2> hindexed_entries{hindexed_entry{1, MPI_Aint{0}},
                                                       hindexed_entry{2, MPI_Aint{16}}};
  const mp1::hindexed_parameter hindexed_array{hindexed_entries};
  mp1::hindexed_parameter hindexed_added;
  hindexed_added.add(1, MPI_Aint{0});
  hindexed_added.add(2, MPI_Aint{16});
  (void)mp1::layout<int>::hindexed(hindexed_array);
  (void)mp1::layout<int>::hindexed(hindexed_added);

  const std::array<mp1::count_t, 2> offsets{mp1::count_t{0}, mp1::count_t{3}};
  const mp1::indexed_block_parameter indexed_block_array{offsets};
  const std::vector<mp1::count_t> offsets_vector{offsets.begin(), offsets.end()};
  const mp1::indexed_block_parameter indexed_block_vector{offsets_vector};
  mp1::indexed_block_parameter indexed_block_added;
  indexed_block_added.add(mp1::count_t{0});
  indexed_block_added.add(mp1::count_t{3});
  (void)mp1::layout<int>::indexed_block(2, indexed_block_array);
  (void)mp1::layout<int>::indexed_block(2, indexed_block_vector);
  (void)mp1::layout<int>::indexed_block(2, indexed_block_added);

  const std::array<MPI_Aint, 2> hoffsets{MPI_Aint{0}, MPI_Aint{16}};
  const mp1::hindexed_block_parameter hindexed_block_array{hoffsets};
  mp1::hindexed_block_parameter hindexed_block_added;
  hindexed_block_added.add(MPI_Aint{0});
  hindexed_block_added.add(MPI_Aint{16});
  (void)mp1::layout<int>::hindexed_block(2, hindexed_block_array);
  (void)mp1::layout<int>::hindexed_block(2, hindexed_block_added);

  using dimension = std::array<mp1::count_t, 3>;
  const std::array<dimension, 2> dimensions{
      dimension{mp1::count_t{4}, mp1::count_t{2}, mp1::count_t{1}},
      dimension{mp1::count_t{5}, mp1::count_t{3}, mp1::count_t{2}}};
  const mp1::subarray_parameter subarray_array{dimensions};
  const std::vector<dimension> dimensions_vector{dimensions.begin(), dimensions.end()};
  const mp1::subarray_parameter subarray_vector{dimensions_vector};
  const std::span<const dimension> dimensions_span{dimensions};
  const mp1::subarray_parameter subarray_span{dimensions_span};
  mp1::subarray_parameter subarray_added;
  subarray_added.add(mp1::count_t{4}, mp1::count_t{2}, mp1::count_t{1});
  subarray_added.add(mp1::count_t{5}, mp1::count_t{3}, mp1::count_t{2});
  (void)mp1::layout<int>::subarray(subarray_array);
  (void)mp1::layout<int>::subarray(subarray_vector);
  (void)mp1::layout<int>::subarray(subarray_span);
  (void)mp1::layout<int>::subarray(subarray_added);
  return 0;
}
