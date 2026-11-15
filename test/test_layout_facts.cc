// Layout/datatype foundation regressions: construction-recorded facts, validated base
// datatypes, marker-only resize, borrowed views, and the named lowerings.

#include "mp1/impl/detail/layout_lowering.hpp"
#include "mp1/impl/detail/xfer_side.hpp"
#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>


namespace {

  struct mpi_fixture {
    mpi_fixture() {
      mp1::init();
    }

    ~mpi_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // Never throw from a fixture destructor.
      }
    }
  };

  template<class Call>
  [[nodiscard]] bool throws_invalid_argument(Call&& call) {
    try {
      call();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  }

  template<class Call>
  [[nodiscard]] bool throws_logic_error(Call&& call) {
    try {
      call();
    } catch (const std::logic_error&) {
      return true;
    }
    return false;
  }

}  // namespace

// A reflected type whose only listed member is at offset 4, so its base datatype has a
// non-zero true lower bound.
struct layout_facts_padded {
  int ignored{-777};
  int value{0};
};

MP1_REFLECTION(layout_facts_padded, value)

struct layout_facts_two_ints {
  int first{0};
  int second{0};
};

MP1_REFLECTION(layout_facts_two_ints, first, second)

// A deliberately invalid trait: MPI_INT describes four bytes, sizeof is eight.
struct layout_facts_bad_trait {
  long long value{0};
};

MP1_DATATYPE_TRAITS(layout_facts_bad_trait, MPI_INT)

struct layout_facts_custom_int {
  int value{0};
};

MP1_DATATYPE_TRAITS(layout_facts_custom_int, MPI_INT)


BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);

BOOST_AUTO_TEST_SUITE(layout_facts_test)

BOOST_AUTO_TEST_CASE(validated_base_datatype) {
  const mp1::detail::datatype_info& padded =
      mp1::detail::datatype_info_of<layout_facts_padded>();
  BOOST_TEST(padded.extent == static_cast<mp1::count_t>(sizeof(layout_facts_padded)));
  BOOST_TEST(padded.size == static_cast<mp1::count_t>(sizeof(int)));
  BOOST_TEST(padded.true_lb == 4);
  BOOST_TEST(padded.true_extent == static_cast<mp1::count_t>(sizeof(int)));
  BOOST_TEST(padded.packed);  // one listed entry: the map is a single ascending run
  BOOST_TEST(static_cast<int>(padded.overlap) !=
             static_cast<int>(mp1::detail::overlap_state::overlap));

  const mp1::detail::datatype_info& two_ints =
      mp1::detail::datatype_info_of<layout_facts_two_ints>();
  BOOST_TEST(two_ints.extent == static_cast<mp1::count_t>(sizeof(layout_facts_two_ints)));
  BOOST_TEST(two_ints.true_lb == 0);
  BOOST_TEST(two_ints.packed);

  BOOST_TEST(mp1::detail::datatype_info_of<layout_facts_custom_int>().packed);
  BOOST_TEST((mp1::detail::datatype_info_of<std::pair<int, int>>().packed));
  BOOST_TEST((not mp1::detail::datatype_info_of<std::pair<int, double>>().packed));

  // A trait with the correct extent but a map outside [0, sizeof(T)) is rejected.
  BOOST_TEST(throws_invalid_argument(
      [] { (void)mp1::layout<layout_facts_bad_trait>::contiguous(1); }));

  // A correct padded reflected type is legal: extent == sizeof(T), true_lb + span inside T.
  const mp1::layout<layout_facts_padded> padded_layout =
      mp1::layout<layout_facts_padded>::contiguous(2);
  BOOST_TEST(padded_layout.size() == 2);
  BOOST_TEST(padded_layout.byte_extent() ==
             static_cast<mp1::count_t>(2 * sizeof(layout_facts_padded)));
  BOOST_TEST(padded_layout.true_byte_lower_bound() == 4);
  BOOST_TEST(padded_layout.true_byte_extent() ==
             static_cast<mp1::count_t>(sizeof(layout_facts_padded) + sizeof(int)));
}

BOOST_AUTO_TEST_CASE(factory_recorded_facts) {
  using mp1::layout;
  using mp1::indexed_parameter;
  using mp1::hindexed_parameter;

  const layout<int> contiguous = layout<int>::contiguous(3);
  BOOST_TEST(contiguous.dense());
  BOOST_TEST(contiguous.is_base_repetition());
  BOOST_TEST(not contiguous.has_overlapping_entries());

  BOOST_TEST(layout<int>::vector(3, 2, 2).dense());
  BOOST_TEST(not layout<int>::vector(3, 2, 5).dense());
  indexed_parameter dense_parameter;
  dense_parameter.add(2, 0);
  dense_parameter.add(3, 2);
  BOOST_TEST(layout<int>::indexed(dense_parameter).dense());
  BOOST_TEST(layout<int>::contiguous(0).dense());
  BOOST_TEST(layout<int>::contiguous(0).is_base_repetition());

  // Descending map order: legal displacements, dense scalar span, not packed.
  indexed_parameter descending_parameter;
  descending_parameter.add(1, 1);
  descending_parameter.add(1, 0);
  const layout<int> descending = layout<int>::indexed(descending_parameter);
  BOOST_TEST(descending.byte_size() == descending.true_byte_extent());
  BOOST_TEST(not descending.dense());
  BOOST_TEST(not descending.has_overlapping_entries());

  // Duplicate + hole: size == true_extent == 12 but the map does not tile the span.
  hindexed_parameter duplicate_parameter;
  duplicate_parameter.add(1, 0);
  duplicate_parameter.add(1, 0);
  duplicate_parameter.add(1, 8);
  const layout<int> duplicate_hole = layout<int>::hindexed(duplicate_parameter);
  BOOST_TEST(duplicate_hole.byte_size() == static_cast<mp1::count_t>(3 * sizeof(int)));
  BOOST_TEST(duplicate_hole.byte_size() == duplicate_hole.true_byte_extent());
  BOOST_TEST(not duplicate_hole.dense());
  BOOST_TEST(duplicate_hole.has_overlapping_entries());

  mp1::indexed_block_parameter block_duplicate_parameter;
  block_duplicate_parameter.add(0);
  block_duplicate_parameter.add(0);
  const layout<int> block_duplicate = layout<int>::indexed_block(2, block_duplicate_parameter);
  BOOST_TEST(not block_duplicate.dense());
  BOOST_TEST(block_duplicate.has_overlapping_entries());

  // A proven overlap is accepted as a send (staged), but rejected as a receive.
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::detail::require_receive_layout(duplicate_hole, "test"); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::detail::require_receive_layout(block_duplicate, "test"); }));
  BOOST_TEST(not throws_invalid_argument(
      [&] { mp1::detail::require_receive_layout(descending, "test"); }));
  const std::vector<mp1::layout<int>> receive_blocks{duplicate_hole};
  std::array<int, 16> receive_buffer{};
  BOOST_TEST(throws_invalid_argument([&] {
    // The typed plan takes public element displacements (nullptr means all-zero), so this
    // probe reaches the receive-layout overlap check with no displacement array at all.
    (void)mp1::detail::xfer_plan<int>::many_layouts(
        receive_buffer.data(), std::span<const mp1::layout<int>>{receive_blocks},
        std::span<const mp1::count_t>{}, 1, mp1::detail::transfer_direction::receive,
        mp1::detail::prime_policy::none);
  }));

  // A row-subarray with all inner dimensions full is one linear run; a partial inner
  // dimension is not.
  mp1::subarray_parameter full_rows_parameter;
  full_rows_parameter.add(4, 2, 1);
  full_rows_parameter.add(5, 5, 0);
  const layout<int> full_rows = layout<int>::subarray(full_rows_parameter);
  BOOST_TEST(full_rows.dense());
  mp1::subarray_parameter partial_inner_parameter;
  partial_inner_parameter.add(4, 2, 1);
  partial_inner_parameter.add(5, 3, 2);
  const layout<int> partial_inner = layout<int>::subarray(partial_inner_parameter);
  BOOST_TEST(not partial_inner.dense());

  // Padded base: the scalar base datatype is packed (one entry), but contiguous(2) is not
  // packed (base extent != base size) while it is a base repetition.
  const layout<layout_facts_padded> padded = layout<layout_facts_padded>::contiguous(2);
  BOOST_TEST(not padded.dense());
  BOOST_TEST(padded.is_base_repetition());

  // Marker-only resize preserves map facts; a negative extent is rejected.
  layout<int> resized = layout<int>::contiguous(2);
  BOOST_TEST(resized.dense());
  resized.byte_resize(-4, 16);
  BOOST_TEST(resized.dense());
  BOOST_TEST(resized.byte_lower_bound() == -4);
  BOOST_TEST(resized.byte_extent() == 16);
  BOOST_TEST(resized.true_byte_lower_bound() == 0);
  BOOST_TEST(throws_invalid_argument([&] { resized.byte_resize(0, -8); }));
  BOOST_TEST(resized.dense());
}

BOOST_AUTO_TEST_CASE(named_lowerings_match_mpi_pack) {
  using mp1::layout;
  std::array<int, 16> buffer{};
  for (std::size_t i = 0; i < buffer.size(); ++i)
    buffer[i] = static_cast<int>(i);

  const auto pack_with_handle = [&](const layout<int>& l, const int* ptr) {
    std::vector<std::byte> bytes(
        static_cast<std::size_t>(l.byte_size() == 0 ? 1 : l.byte_size()));
    MPI_Count position{0};
    mp1::detail::check_mpi(
        MPI_Pack_c(ptr, 1, l.native_handle(), bytes.data(),
                   static_cast<MPI_Count>(bytes.size()), &position, MPI_COMM_SELF),
        "MPI_Pack_c(layout)");
    BOOST_TEST(position == l.byte_size());
    return bytes;
  };

  const layout<int> packed = layout<int>::vector(3, 2, 2);
  const auto reference = pack_with_handle(packed, buffer.data());
  const mp1::detail::packed_view<int> view =
      mp1::detail::packed_view_of(buffer.data(), packed, "test");
  BOOST_TEST(view.ptr == buffer.data() + packed.true_byte_lower_bound());
  const std::vector<std::byte> linear(
      reinterpret_cast<const std::byte*>(view.ptr),
      reinterpret_cast<const std::byte*>(view.ptr) + static_cast<std::ptrdiff_t>(view.bytes));
  BOOST_TEST(linear == reference);

  BOOST_TEST(throws_logic_error([&] {
    (void)mp1::detail::packed_view_of(buffer.data(), layout<int>::vector(3, 2, 5), "test");
  }));

  // Repeated-base lowering uses p + (true_lb(layout) - true_lb(base)).
  mp1::indexed_parameter padded_parameter;
  padded_parameter.add(2, 2);
  const layout<layout_facts_padded> padded =
      layout<layout_facts_padded>::indexed(padded_parameter);
  std::array<layout_facts_padded, 8> padded_buffer{};
  for (std::size_t i = 0; i < padded_buffer.size(); ++i)
    padded_buffer[i] = layout_facts_padded{static_cast<int>(100 + i), static_cast<int>(i)};
  const mp1::detail::repetition_view<layout_facts_padded> repeated =
      mp1::detail::repetition_view_of(padded_buffer.data(), padded, "test");
  BOOST_TEST(repeated.ptr == padded_buffer.data() + 2);
  BOOST_TEST(repeated.count == padded.size());
  BOOST_TEST(repeated.base == mp1::native_datatype<layout_facts_padded>());
  BOOST_TEST(padded.is_base_repetition());
  BOOST_TEST(padded.true_byte_lower_bound() ==
             static_cast<mp1::count_t>(2 * sizeof(layout_facts_padded) + 4));

  const layout<int> identity = layout<int>::contiguous(4);
  BOOST_TEST(mp1::detail::is_identity_repetition(identity, "test"));
  mp1::indexed_parameter identity_parameter;
  identity_parameter.add(1, 1);
  identity_parameter.add(1, 0);
  const layout<int> descending = layout<int>::indexed(identity_parameter);
  BOOST_TEST(not mp1::detail::is_identity_repetition(descending, "test"));
  const mp1::detail::identity_view<int> identity_view =
      mp1::detail::identity_view_of(buffer.data(), identity, "test");
  BOOST_TEST(identity_view.ptr == buffer.data());
  BOOST_TEST(identity_view.count == 4);

  BOOST_TEST(mp1::detail::staged_ptr(buffer.data(), layout<int>::vector(3, 2, 5), "test") ==
             buffer.data());
}

BOOST_AUTO_TEST_CASE(borrowed_views) {
  const mp1::layout<int> block = mp1::layout<int>::contiguous(1);
  const std::vector<mp1::layout<int>> blocks(3, block);

  // The typed descriptor is a plain std::span over the caller's layout array; a blocking call
  // borrows it, and a nonblocking/persistent call copies what completion/restart reads.
  const std::span<const mp1::layout<int>> span = blocks;
  BOOST_TEST(span.size() == 3);
  BOOST_TEST(span.data() == blocks.data());
  BOOST_TEST(span[1].native_handle() == blocks[1].native_handle());

  // A region is the single-layout borrow; the deleted rvalue-layout constructor keeps a
  // temporary layout from being captured.
  const mp1::region<int> region{const_cast<int*>(reinterpret_cast<const int*>(blocks.data())),
                                block};
  BOOST_TEST(region.map().size() == 1);
  static_assert(not std::is_constructible_v<mp1::region<int>, int*, mp1::layout<int>&&>);
  static_assert(std::is_constructible_v<mp1::region<int>, int*, const mp1::layout<int>&>);
}

BOOST_AUTO_TEST_SUITE_END()
