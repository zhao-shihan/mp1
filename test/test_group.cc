// Group tests: default/null vs empty state, MPI-free equality, translate (including
// MPI_PROC_NULL), compare semantics, the free group set operations, duplicate-rank
// validation, and move-only ownership.
//
// MPI 4.1 basis: §7.3.1 (accessors; MPI_PROC_NULL is a valid translate input and returns
// MPI_PROC_NULL; compare returns only IDENT/SIMILAR/UNEQUAL), §7.3.2 (each incl/excl rank
// must be valid and distinct; n == 0 gives MPI_GROUP_EMPTY for incl and the identical group
// for excl), §7.3.3 (free marks the group for deallocation and sets the handle to
// MPI_GROUP_NULL).

#include "mp1/mp1.hpp"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace {

  template<class F>
  bool throws_invalid_argument(F&& f) {
    try {
      f();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  }

}  // namespace


BOOST_AUTO_TEST_CASE(group) {
  // Default state and MPI-free equality paths work before any initialization: two invalid
  // groups compare equal, and two MPI_GROUP_EMPTY handles compare equal without an MPI call.
  {
    mp1::group invalid_left;
    mp1::group invalid_right;
    BOOST_TEST(not invalid_left.is_valid());
    BOOST_TEST(not invalid_left.is_empty());
    BOOST_TEST(invalid_left.native_handle() == MPI_GROUP_NULL);
    BOOST_TEST((invalid_left == invalid_right));
    BOOST_TEST(not(invalid_left != invalid_right));

    const mp1::group empty_left = mp1::group::empty();
    const mp1::group empty_right = mp1::group::empty();
    BOOST_TEST(empty_left.is_valid());
    BOOST_TEST(empty_left.is_empty());
    BOOST_TEST(empty_left.native_handle() == MPI_GROUP_EMPTY);
    BOOST_TEST((empty_left == empty_right));
    BOOST_TEST((invalid_left != empty_left));
  }

  if (not mp1::initialized())
    mp1::init();

  const auto comm_world{mp1::comm_world()};
  const auto comm_self{mp1::comm_self()};
  const int world_size = comm_world.size();
  const int world_rank = comm_world.rank();

  auto group_world = mp1::get_group(comm_world);
  auto group_world_2 = mp1::get_group(comm_world);
  auto group_self = mp1::get_group(comm_self);

  BOOST_TEST(group_world.is_valid());
  BOOST_TEST(not group_world.is_empty());
  BOOST_TEST(group_world_2.is_valid());
  BOOST_TEST(group_self.is_valid());
  BOOST_TEST(group_world.size() == world_size);
  BOOST_TEST(group_world.rank() == world_rank);
  BOOST_TEST(group_self.size() == 1);
  BOOST_TEST(group_self.rank() == 0);

  // Two independent queries of the same communicator produce identical groups (MPI 4.1
  // §7.3.1: MPI_IDENT iff members and order are the same).
  BOOST_TEST((group_world == group_world_2));
  BOOST_TEST((group_world.compare(group_world_2) == mp1::equality_type::identical));

  if (world_size > 1)
    BOOST_TEST((group_world != group_self));
  else
    BOOST_TEST((group_world == group_self));
  if (world_size > 1)
    BOOST_TEST((group_world.compare(group_self) == mp1::equality_type::unequal));
  else
    BOOST_TEST((group_world.compare(group_self) == mp1::equality_type::identical));

  // Equality is MPI_IDENT: the same members in a different order are MPI_SIMILAR, not equal.
  if (world_size > 1) {
    std::vector<int> permutation(static_cast<std::size_t>(world_size));
    for (int i = 0; i < world_size; ++i)
      permutation[static_cast<std::size_t>(i)] = i;
    std::swap(permutation[0], permutation[1]);
    const mp1::group reordered = mp1::group_incl(group_world, permutation);
    BOOST_TEST((group_world.compare(reordered) == mp1::equality_type::similar));
    BOOST_TEST((group_world != reordered));
  }

  BOOST_TEST(group_self.translate(0, group_world) == group_world.rank());
  // MPI 4.1 §7.3.1: MPI_PROC_NULL is a valid input rank and translates to MPI_PROC_NULL.
  BOOST_TEST(group_world.translate(MPI_PROC_NULL, group_world) == MPI_PROC_NULL);
  BOOST_TEST(group_world.translate(MPI_PROC_NULL, group_self) == MPI_PROC_NULL);

  auto group_union = mp1::group_union(group_world, group_self);
  auto group_intersection = mp1::group_intersection(group_world, group_self);
  auto group_difference = mp1::group_difference(group_world, group_self);
  auto group_with_0 = mp1::group_incl(group_world, std::array{0});
  auto group_without_0 = mp1::group_excl(group_world, std::array{0});

  BOOST_TEST(group_union.size() == group_world.size());
  BOOST_TEST(group_intersection.size() == 1);
  BOOST_TEST(group_difference.size() == group_world.size() - 1);
  BOOST_TEST(group_with_0.size() == 1);
  BOOST_TEST(group_without_0.size() == group_world.size() - 1);

  // MPI 4.1 §7.3.2: n == 0 gives MPI_GROUP_EMPTY for incl and the identical group for excl.
  {
    const mp1::group none = mp1::group_incl(group_world, std::span<const int>{});
    BOOST_TEST(none.is_valid());
    BOOST_TEST(none.is_empty());
    BOOST_TEST(none.native_handle() == MPI_GROUP_EMPTY);
    BOOST_TEST(none.size() == 0);
    const mp1::group all = mp1::group_excl(group_world, std::span<const int>{});
    BOOST_TEST(all.size() == group_world.size());
    BOOST_TEST((all.compare(group_world) == mp1::equality_type::identical));
  }

  // MPI 4.1 §7.3.2 range constructors.  The wrapper validates stride, direction and the
  // computed endpoints locally without expanding the progression; the tests compare every
  // result with the equivalent group_incl/group_excl expansion (which is safe here because
  // the ranges are duplicate-free).
  {
    const int group_size = group_world.size();
    std::array<mp1::rank_range, 1> every_other{mp1::rank_range{0, group_size - 1, 2}};
    std::vector<int> every_other_ranks;
    for (int rank = 0; rank < group_size; rank += 2)
      every_other_ranks.push_back(rank);
    const mp1::group ranged_every_other = mp1::group_range_incl(group_world, every_other);
    const mp1::group expanded_every_other = mp1::group_incl(group_world, every_other_ranks);
    BOOST_TEST(
        (ranged_every_other.compare(expanded_every_other) == mp1::equality_type::identical));
    BOOST_TEST(ranged_every_other.size() == static_cast<int>(every_other_ranks.size()));

    // first == last emits exactly that one rank (stride is irrelevant).
    std::array<mp1::rank_range, 1> single{mp1::rank_range{group_size - 1, group_size - 1, 7}};
    const mp1::group ranged_single = mp1::group_range_incl(group_world, single);
    BOOST_TEST(ranged_single.size() == 1);
    BOOST_TEST(
        (ranged_single.compare(mp1::group_incl(group_world, std::array{group_size - 1})) ==
         mp1::equality_type::identical));

    // Negative stride with first > last reverses the group; excluding that range leaves the
    // empty group.
    std::array<mp1::rank_range, 1> reversed_range{mp1::rank_range{group_size - 1, 0, -1}};
    std::vector<int> reversed_ranks;
    for (int rank = group_size - 1; rank >= 0; --rank)
      reversed_ranks.push_back(rank);
    const mp1::group ranged_reversed = mp1::group_range_incl(group_world, reversed_range);
    BOOST_TEST((ranged_reversed.compare(mp1::group_incl(group_world, reversed_ranks)) ==
                mp1::equality_type::identical));
    const mp1::group excluded_all = mp1::group_range_excl(group_world, reversed_range);
    BOOST_TEST(excluded_all.is_empty());
    BOOST_TEST(excluded_all.size() == 0);

    // A range whose last value is not emitted may have an out-of-bounds last.
    std::array<mp1::rank_range, 1> coarse{
        mp1::rank_range{0, group_size + 99, group_size + 100}};
    const mp1::group ranged_coarse = mp1::group_range_incl(group_world, coarse);
    BOOST_TEST(ranged_coarse.size() == 1);
    BOOST_TEST(ranged_coarse.native_handle() != MPI_GROUP_EMPTY);
    BOOST_TEST((ranged_coarse.compare(mp1::group_incl(group_world, std::array{0})) ==
                mp1::equality_type::identical));

    // n == 0: incl yields MPI_GROUP_EMPTY, excl yields an identical group.
    const mp1::group no_ranges =
        mp1::group_range_incl(group_world, std::span<const mp1::rank_range>{});
    BOOST_TEST(no_ranges.is_valid());
    BOOST_TEST(no_ranges.is_empty());
    const mp1::group all_ranges =
        mp1::group_range_excl(group_world, std::span<const mp1::rank_range>{});
    BOOST_TEST((all_ranges.compare(group_world) == mp1::equality_type::identical));

    // An empty group with empty ranges follows MPI: incl empty, excl identical.
    const mp1::group empty_ranges_incl =
        mp1::group_range_incl(mp1::group::empty(), std::span<const mp1::rank_range>{});
    BOOST_TEST(empty_ranges_incl.is_empty());
    const mp1::group empty_ranges_excl =
        mp1::group_range_excl(mp1::group::empty(), std::span<const mp1::rank_range>{});
    BOOST_TEST(empty_ranges_excl.is_empty());
    BOOST_TEST(empty_ranges_excl.size() == 0);

    // Local validation, before any MPI call.
    const auto range_throws = [&](mp1::rank_range range) {
      const std::array<mp1::rank_range, 1> ranges{range};
      return throws_invalid_argument([&] { (void)mp1::group_range_incl(group_world, ranges); });
    };
    BOOST_TEST(range_throws(mp1::rank_range{0, 0, 0}));                    // zero stride
    BOOST_TEST(range_throws(mp1::rank_range{-1, -1, 1}));                  // first negative
    BOOST_TEST(range_throws(mp1::rank_range{group_size, group_size, 1}));  // first out of range
    BOOST_TEST(range_throws(mp1::rank_range{0, group_size, 1}));  // emitted last out of range
    if (group_size > 1) {
      BOOST_TEST(range_throws(mp1::rank_range{0, group_size - 1, -1}));  // direction mismatch
      BOOST_TEST(range_throws(mp1::rank_range{group_size - 1, 0, 1}));   // direction mismatch
    }
    const std::array<mp1::rank_range, 1> stride_zero{mp1::rank_range{0, 0, 0}};
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)mp1::group_range_excl(group_world, stride_zero); }));
    const std::array<mp1::rank_range, 1> first_out{mp1::rank_range{group_size, group_size, 1}};
    BOOST_TEST(
        throws_invalid_argument([&] { (void)mp1::group_range_excl(group_world, first_out); }));

    // An invalid group is rejected before the range checks or any MPI call.
    const std::array<mp1::rank_range, 1> valid_range{mp1::rank_range{0, 0, 1}};
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)mp1::group_range_incl(mp1::group{}, valid_range); }));
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)mp1::group_range_excl(mp1::group{}, valid_range); }));
  }

  // Move-only ownership: the moved-from group is invalid, the moved-to group is usable.
  auto moved = std::move(group_world);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not group_world.is_valid());
  BOOST_TEST(moved.is_valid());
  BOOST_TEST(moved.size() == world_size);
  BOOST_TEST(throws_invalid_argument([&] { (void)group_world.size(); }));
  BOOST_TEST((group_world == mp1::group{}));  // invalid == invalid, MPI-free

  // Validation is always on and local (no collective is entered).
  BOOST_TEST(throws_invalid_argument([&] { (void)mp1::group_incl(moved, std::array{-1}); }));
  BOOST_TEST(
      throws_invalid_argument([&] { (void)mp1::group_incl(moved, std::array{moved.size()}); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)mp1::group_excl(moved, std::array{-1}); }));
  // MPI_PROC_NULL is a legal translate input; use a value that is never MPI_PROC_NULL.
  BOOST_TEST(
      throws_invalid_argument([&] { (void)moved.translate(MPI_UNDEFINED, group_self); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)moved.translate(0, group_world); }));

  // MPI 4.1 §7.3.2: all incl/excl ranks must be distinct; the wrapper diagnoses the
  // erroneous program locally instead of entering MPI with it.
  BOOST_TEST(throws_invalid_argument([&] { (void)mp1::group_incl(moved, std::array{0, 0}); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)mp1::group_excl(moved, std::array{0, 0}); }));

  // The default group is invalid; group::empty() is the valid, empty MPI_GROUP_EMPTY group.
  mp1::group empty;
  BOOST_TEST(not empty.is_valid());
  BOOST_TEST(not empty.is_empty());
  const mp1::group valid_empty = mp1::group::empty();
  BOOST_TEST(valid_empty.is_valid());
  BOOST_TEST(valid_empty.size() == 0);
  BOOST_TEST(valid_empty.rank() == MPI_UNDEFINED);
  BOOST_TEST((moved != valid_empty));
}
