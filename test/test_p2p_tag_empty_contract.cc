// Contract test for the strong mp1::tag vocabulary, the cached MPI_TAG_UB bound, and the
// explicit empty-message span mp1::empty_data.
//
// MPI 4.1 basis: §3.2.3 defines tags, MPI_ANY_TAG and the inclusive MPI_TAG_UB attribute;
// §3.2.4 makes the type signature of a zero-count message empty, so mp1::empty_data matches
// any zero-count receive.  The tag range is validated locally against the communicator's
// cached MPI_TAG_UB, so no validation case enters MPI.
//
// The file is registered at 2 and 4 ranks; point-to-point work uses ranks 0/1 and the other
// ranks wait at a barrier, exactly like the other p2p contract tests.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <compare>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

  // Compile-time half of the tag contract: aggregate storage, tag 0 default, all six
  // comparisons, wildcard, guaranteed minimum bound, and the status accessor type.
  static_assert(std::is_aggregate_v<mp1::tag>);
  static_assert(std::is_trivially_copyable_v<mp1::tag>);
  static_assert(std::is_default_constructible_v<mp1::tag>);
  static_assert(std::is_same_v<decltype(mp1::tag::value), int>);
  static_assert(mp1::tag{}.value == 0);
  static_assert(mp1::tag{1} == mp1::tag{1});
  static_assert(mp1::tag{1} != mp1::tag{2});
  static_assert(mp1::tag{1} < mp1::tag{2});
  static_assert(mp1::tag{1} <= mp1::tag{1});
  static_assert(mp1::tag{2} > mp1::tag{1});
  static_assert(mp1::tag{2} >= mp1::tag{2});
  static_assert((mp1::tag{1} <=> mp1::tag{2}) == std::strong_ordering::less);
  static_assert(mp1::any_tag == mp1::tag{MPI_ANY_TAG});
  static_assert(mp1::tag_ub_min == mp1::tag{32767});
  static_assert(std::is_same_v<decltype(std::declval<const mp1::status&>().tag()), mp1::tag>);
  static_assert(std::is_same_v<decltype(std::declval<const mp1::status&>().source()), int>);

  int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  template<class F>
  bool throws_invalid_argument(F&& f) {
    try {
      f();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

}  // namespace

struct mp1_tag_empty_contract_fixture {
  mp1_tag_empty_contract_fixture() {
    mp1::init();
  }

  ~mp1_tag_empty_contract_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Already finalized through another path; a destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_tag_empty_contract_fixture);

BOOST_AUTO_TEST_CASE(tag_bound_and_status_vocabulary) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int upper = mp1::tag_ub(world);
  BOOST_TEST(upper >= mp1::tag_ub_min.value);
  BOOST_TEST(upper == world.tag_ub());

  // The bound is copied with the view/owner cache, not re-queried by the accessor.
  mp1::communicator duplicate = mp1::communicator::dup(world);
  BOOST_TEST(duplicate.tag_ub() == upper);


  // The wrapper's normalized/empty status uses the wildcard tag.
  const mp1::status empty;
  BOOST_TEST(empty.tag().value == MPI_ANY_TAG);
  BOOST_TEST(empty.tag().value == MPI_ANY_TAG);

  const mp1::mpi_communicator invalid;
  BOOST_TEST(throws_invalid_argument([&] { (void)invalid.tag_ub(); }));
}

BOOST_AUTO_TEST_CASE(tag_validation_is_local_and_range_checked) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int upper = mp1::tag_ub(world);
  int value = 0;

  // A send tag must be 0..MPI_TAG_UB and never the wildcard.
  BOOST_TEST(throws_invalid_argument([&] { mp1::send(world, value, 1, mp1::tag{-1}); }));
  BOOST_TEST(throws_invalid_argument([&] { mp1::send(world, value, 1, mp1::any_tag); }));
  if (upper < std::numeric_limits<int>::max()) {
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::send(world, value, MPI_PROC_NULL, mp1::tag{upper + 1}); }));
  }

  // Receive and probe tags also accept the wildcard, and reject other negatives.
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::recv(world, value, MPI_PROC_NULL, mp1::tag{-2}); }));
  BOOST_TEST(
      throws_invalid_argument([&] { (void)mp1::probe(world, MPI_PROC_NULL, mp1::tag{-3}); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::irecv(world, value, MPI_PROC_NULL, mp1::tag{-4}); }));
  BOOST_TEST(not throws_invalid_argument(
      [&] { (void)mp1::recv(world, value, MPI_PROC_NULL, mp1::any_tag); }));
  BOOST_TEST(not throws_invalid_argument(
      [&] { (void)mp1::probe(world, MPI_PROC_NULL, mp1::any_tag); }));

  // MPI_PROC_NULL probe: normalized status, wildcard tag, count 0 and no cancellation query.
  const mp1::status probed = mp1::probe(world, MPI_PROC_NULL, mp1::any_tag);
  BOOST_TEST(static_cast<int>(probed.kind()) == static_cast<int>(mp1::status_kind::normalized));
  BOOST_TEST(probed.source() == MPI_PROC_NULL);
  BOOST_TEST(probed.tag().value == MPI_ANY_TAG);
  BOOST_TEST(probed.error() == MPI_SUCCESS);
  BOOST_TEST(probed.get_count<int>() == 0);
  BOOST_TEST(probed.get_elements<int>() == 0);
  BOOST_TEST(not probed.is_cancelled());

  // The group-creation tag shares the range, uses the source communicator's cached bound,
  // and is not a wildcard.  A rejected tag throws before the collective is entered.
  const mp1::group all = mp1::get_group(world);
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::communicator::create_group(world, all, mp1::any_tag); }));
  if (upper < std::numeric_limits<int>::max()) {
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)mp1::communicator::create_group(world, all, mp1::tag{upper + 1}); }));
  }
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::communicator::intercomm_create(world, 0, world, 0, mp1::tag{-5}); }));
  world_barrier();
}

BOOST_AUTO_TEST_CASE(tag_upper_bound_round_trip) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int upper = mp1::tag_ub(world);

  // The attribute value itself is an accepted tag (the bound is inclusive).
  if (rank == 0) {
    mp1::send(world, 42, 1, mp1::tag{upper});
    mp1::send(world, 43, 1, mp1::tag_ub_min);
  } else if (rank == 1) {
    int received = 0;
    const mp1::status at_upper = mp1::recv(world, received, 0, mp1::tag{upper});
    BOOST_TEST(received == 42);
    BOOST_TEST(at_upper.tag().value == upper);
    const mp1::status at_min = mp1::recv(world, received, 0, mp1::tag_ub_min);
    BOOST_TEST(received == 43);
    BOOST_TEST(at_min.tag().value == mp1::tag_ub_min.value);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(empty_data_round_trip) {
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  if (rank == 0) {
    // mp1::empty_data is an ordinary zero-count byte span: no special send path and no MPI
    // call to construct it.
    mp1::send(world, mp1::empty_data, 1, mp1::tag{7100});
    mp1::send(world, mp1::empty_data, 1, mp1::tag{7101});
  } else if (rank == 1) {
    const mp1::status probed = mp1::probe(world, 0, mp1::tag{7100});
    BOOST_TEST(probed.get_count<std::byte>() == 0);
    BOOST_TEST(probed.tag().value == 7100);

    const mp1::status received = mp1::recv(world, mp1::empty_data, 0, mp1::tag{7100});
    BOOST_TEST(received.get_count<std::byte>() == 0);

    // The auto-sizing container receive learns a count of 0 from the matched message.
    std::vector<std::byte> buffer{std::byte{1}, std::byte{2}};
    const mp1::status received_into_container = mp1::recv(world, buffer, 0, mp1::tag{7101});
    BOOST_TEST(received_into_container.get_count<std::byte>() == 0);
    BOOST_TEST(buffer.empty());
  }
  world_barrier();
}
