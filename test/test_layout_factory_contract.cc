// The layout factories, the block/resize/order signatures, and the named-range builder
// spellings.
//
// The differential runtime matrix for every factory lives in test_layout_contract.cc
// (`contiguous`, `vector`, `strided_vector`, `indexed`, `hindexed`, `indexed_block`,
// `hindexed_block`, `subarray`, the block overloads, byte_resize/resized, the column-major
// order and the 119-member reflection) and is not duplicated here.  This file pins what that
// matrix cannot: the exact member-function signatures with function-pointer types (the
// signature probe) and the named-range positive spellings (named std::array/std::vector/
// std::span and `.add()`); it also exercises the repaired factories once so the pinned
// declarations are the ones that actually build.
//
// Negative: `indexed_parameter{{1,2}}` and `subarray_parameter{{{4,2,1},{5,3,2}}}` must be
// rejected because a braced-init-list does not convert to `std::span` and the builders have
// no initializer-list constructor.  The negative is pinned by
// `test/compile_fail/layout_builder_braced_init.cc` (all five parameter builders) with
// `test/compile_pass/layout_builder_named_ranges.cc` as the positive control; the named
// ranges and `.add()` calls below are the supported spellings exercised at runtime.
//
// Runs at 2 and 4 ranks; layout factories create/commit MPI datatypes, so the shared fixture
// initializes MPI.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>


namespace {

  struct layout_factory_fixture {
    layout_factory_fixture() {
      mp1::init();
    }

    ~layout_factory_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a fixture destructor.
      }
    }
  };

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(layout_factory_fixture);

BOOST_AUTO_TEST_SUITE(layout_factory_contract)

// The five block-layout factory families have the exact count_t signatures, and the
// resize/order members are unchanged.
BOOST_AUTO_TEST_CASE(repaired_signatures_are_pinned) {
  using layout_type = mp1::layout<int>;
  using indexed_fn = layout_type (*)(const mp1::indexed_parameter&);
  using indexed_block_fn = layout_type (*)(const mp1::indexed_parameter&, const layout_type&);
  using hindexed_fn = layout_type (*)(const mp1::hindexed_parameter&);
  using hindexed_block_fn = layout_type (*)(const mp1::hindexed_parameter&, const layout_type&);
  using indexed_block_count_fn =
      layout_type (*)(mp1::count_t, const mp1::indexed_block_parameter&);
  using indexed_block_count_block_fn =
      layout_type (*)(mp1::count_t, const mp1::indexed_block_parameter&, const layout_type&);
  using hindexed_block_count_fn =
      layout_type (*)(mp1::count_t, const mp1::hindexed_block_parameter&);
  using hindexed_block_count_block_fn =
      layout_type (*)(mp1::count_t, const mp1::hindexed_block_parameter&, const layout_type&);
  using subarray_fn = layout_type (*)(const mp1::subarray_parameter&);
  using subarray_block_fn = layout_type (*)(const mp1::subarray_parameter&, const layout_type&);
  using byte_resize_fn = void (layout_type::*)(mp1::count_t, mp1::count_t);
  using resized_fn = layout_type (layout_type::*)(mp1::count_t, mp1::count_t) const;
  using order_set_fn = void (mp1::subarray_parameter::*)(mp1::array_order) noexcept;
  using order_get_fn = mp1::array_order (mp1::subarray_parameter::*)() const noexcept;

  static_assert(
      std::is_same_v<decltype(static_cast<indexed_fn>(&layout_type::indexed)), indexed_fn>);
  static_assert(std::is_same_v<decltype(static_cast<indexed_block_fn>(&layout_type::indexed)),
                               indexed_block_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<hindexed_fn>(&layout_type::hindexed)), hindexed_fn>);
  static_assert(std::is_same_v<decltype(static_cast<hindexed_block_fn>(&layout_type::hindexed)),
                               hindexed_block_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<indexed_block_count_fn>(&layout_type::indexed_block)),
                     indexed_block_count_fn>);
  static_assert(std::is_same_v<decltype(static_cast<indexed_block_count_block_fn>(
                                   &layout_type::indexed_block)),
                               indexed_block_count_block_fn>);
  static_assert(std::is_same_v<decltype(static_cast<hindexed_block_count_fn>(
                                   &layout_type::hindexed_block)),
                               hindexed_block_count_fn>);
  static_assert(std::is_same_v<decltype(static_cast<hindexed_block_count_block_fn>(
                                   &layout_type::hindexed_block)),
                               hindexed_block_count_block_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<subarray_fn>(&layout_type::subarray)), subarray_fn>);
  static_assert(std::is_same_v<decltype(static_cast<subarray_block_fn>(&layout_type::subarray)),
                               subarray_block_fn>);
  static_assert(std::is_same_v<decltype(static_cast<byte_resize_fn>(&layout_type::byte_resize)),
                               byte_resize_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<resized_fn>(&layout_type::resized)), resized_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<order_set_fn>(&mp1::subarray_parameter::order)),
                     order_set_fn>);
  static_assert(
      std::is_same_v<decltype(static_cast<order_get_fn>(&mp1::subarray_parameter::order)),
                     order_get_fn>);

  BOOST_TEST(true);  // compile-time case; keep the harness entry non-empty
}

// The named contiguous ranges and the `.add()` builder spellings construct every
// parameter type and feed the factories.
BOOST_AUTO_TEST_CASE(named_range_and_add_builders) {
  using pair_type = std::pair<mp1::count_t, mp1::count_t>;
  using hpair_type = std::pair<mp1::count_t, MPI_Aint>;
  using dimension_type = std::array<mp1::count_t, 3>;

  const std::array<pair_type, 2> indexed_entries{pair_type{mp1::count_t{2}, mp1::count_t{0}},
                                                 pair_type{mp1::count_t{1}, mp1::count_t{6}}};
  const mp1::indexed_parameter from_array{indexed_entries};
  const std::vector<pair_type> indexed_vector{indexed_entries.begin(), indexed_entries.end()};
  const mp1::indexed_parameter from_vector{indexed_vector};
  const std::span<const pair_type> indexed_span{indexed_entries};
  const mp1::indexed_parameter from_span{indexed_span};

  mp1::indexed_parameter added_indexed;
  added_indexed.add(mp1::count_t{2}, mp1::count_t{0});
  added_indexed.add(mp1::count_t{1}, mp1::count_t{6});

  const mp1::layout<int> block = mp1::layout<int>::contiguous(4);
  const std::array<const mp1::indexed_parameter*, 4> indexed_parameters{
      &from_array, &from_vector, &from_span, &added_indexed};
  for (const mp1::indexed_parameter* parameter : indexed_parameters) {
    const mp1::layout<int> layout = mp1::layout<int>::indexed(*parameter);
    BOOST_TEST(layout.size() == 3);
    const mp1::layout<int> composed = mp1::layout<int>::indexed(*parameter, block);
    BOOST_TEST(composed.size() == 12);
  }

  const std::array<hpair_type, 2> hindexed_entries{hpair_type{mp1::count_t{1}, MPI_Aint{0}},
                                                   hpair_type{mp1::count_t{2}, MPI_Aint{16}}};
  const mp1::hindexed_parameter hindexed_named{hindexed_entries};
  mp1::hindexed_parameter hindexed_added;
  hindexed_added.add(mp1::count_t{1}, MPI_Aint{0});
  hindexed_added.add(mp1::count_t{2}, MPI_Aint{16});
  BOOST_TEST(mp1::layout<int>::hindexed(hindexed_named).size() == 3);
  BOOST_TEST(mp1::layout<int>::hindexed(hindexed_added).size() == 3);
  BOOST_TEST(mp1::layout<int>::hindexed(hindexed_named, block).size() == 12);

  const std::array<mp1::count_t, 2> offsets{mp1::count_t{0}, mp1::count_t{3}};
  const mp1::indexed_block_parameter indexed_block_named{offsets};
  const std::vector<mp1::count_t> offsets_vector{offsets.begin(), offsets.end()};
  const mp1::indexed_block_parameter indexed_block_vector{offsets_vector};
  const std::span<const mp1::count_t> offsets_span{offsets};
  const mp1::indexed_block_parameter indexed_block_span{offsets_span};
  mp1::indexed_block_parameter indexed_block_added;
  indexed_block_added.add(mp1::count_t{0});
  indexed_block_added.add(mp1::count_t{3});

  const std::array<const mp1::indexed_block_parameter*, 4> indexed_block_parameters{
      &indexed_block_named, &indexed_block_vector, &indexed_block_span, &indexed_block_added};
  for (const mp1::indexed_block_parameter* parameter : indexed_block_parameters) {
    const mp1::layout<int> layout = mp1::layout<int>::indexed_block(2, *parameter);
    BOOST_TEST(layout.size() == 4);
    BOOST_TEST(mp1::layout<int>::indexed_block(2, *parameter, block).size() == 16);
  }

  const std::array<MPI_Aint, 2> hoffsets{MPI_Aint{0}, MPI_Aint{16}};
  const mp1::hindexed_block_parameter hindexed_block_named{hoffsets};
  mp1::hindexed_block_parameter hindexed_block_added;
  hindexed_block_added.add(MPI_Aint{0});
  hindexed_block_added.add(MPI_Aint{16});
  BOOST_TEST(mp1::layout<int>::hindexed_block(2, hindexed_block_named).size() == 4);
  BOOST_TEST(mp1::layout<int>::hindexed_block(2, hindexed_block_added).size() == 4);
  BOOST_TEST(mp1::layout<int>::hindexed_block(2, hindexed_block_named, block).size() == 16);

  const std::array<dimension_type, 2> dimensions{
      dimension_type{mp1::count_t{4}, mp1::count_t{2}, mp1::count_t{1}},
      dimension_type{mp1::count_t{5}, mp1::count_t{3}, mp1::count_t{2}}};
  const mp1::subarray_parameter subarray_named{dimensions};
  const std::vector<dimension_type> dimensions_vector{dimensions.begin(), dimensions.end()};
  const mp1::subarray_parameter subarray_vector{dimensions_vector};
  const std::span<const dimension_type> dimensions_span{dimensions};
  const mp1::subarray_parameter subarray_span{dimensions_span};
  mp1::subarray_parameter subarray_added;
  subarray_added.add(mp1::count_t{4}, mp1::count_t{2}, mp1::count_t{1});
  subarray_added.add(mp1::count_t{5}, mp1::count_t{3}, mp1::count_t{2});
  subarray_added.order(mp1::array_order::column_major);
  BOOST_TEST(static_cast<int>(subarray_added.order()) ==
             static_cast<int>(mp1::array_order::column_major));

  const std::array<const mp1::subarray_parameter*, 4> subarray_parameters{
      &subarray_named, &subarray_vector, &subarray_span, &subarray_added};
  for (const mp1::subarray_parameter* parameter : subarray_parameters) {
    const mp1::layout<int> layout = mp1::layout<int>::subarray(*parameter);
    BOOST_TEST(layout.size() == 6);
    BOOST_TEST(mp1::layout<int>::subarray(*parameter, block).size() == 24);
  }

  // A negative extent is rejected with the object unchanged; a negative lower bound is
  // legal.  The differential check against MPI_Type_create_resized is in the layout suite.
  mp1::layout<int> resized = mp1::layout<int>::contiguous(2);
  BOOST_CHECK_THROW(resized.byte_resize(8, -8), std::invalid_argument);
  BOOST_TEST(resized.byte_extent() == static_cast<mp1::count_t>(2 * sizeof(int)));
  const mp1::layout<int> padded = mp1::layout<int>::contiguous(2).resized(-4, 16);
  BOOST_TEST(padded.byte_lower_bound() == -4);
  BOOST_TEST(padded.byte_extent() == 16);
}

BOOST_AUTO_TEST_SUITE_END()
