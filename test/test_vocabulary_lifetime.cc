// Vocabulary lifetime/semantic tests.
//
// The vocabulary types are consumed by the public operation families.  These tests pin the
// types' object semantics and exercise the lifetime mechanism the request-owned plans use:
//
//   * `region<T>` is a two-word borrow; copy/move construction keeps the same pointer and
//     layout, and a blocking call made through the borrowed map is valid while the layout
//     lvalue outlives it;
//   * `detail::capture_region` is the async/persistent plan hook: it captures the data
//     pointer and duplicates the layout (MPI_Type_dup), so the initiating layout may die
//     before the nonblocking operation completes;
//   * `blocks` is a non-owning aggregate: designated initialization, an empty descriptor,
//     temporary count/displacement vectors within one full-expression, and the plan-side
//     copy that nonblocking/persistent initiation performs;
//   * `capacity` and the `in_place_t`/`in_place` sentinel keep their current shapes and
//     concept rows.

#include "mp1/mp1.hpp"
#include "mpi_wrap.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>


namespace {

  struct vocabulary_lifetime_fixture {
    vocabulary_lifetime_fixture() {
      mp1::init();
    }

    ~vocabulary_lifetime_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a fixture destructor.
      }
    }
  };

  // The exact accessor/element types of the region borrow and of the async capture hook.
  static_assert(std::is_same_v<mp1::region_element_t<mp1::region<int>>, int>);
  static_assert(std::is_same_v<mp1::region_element_t<mp1::region<const int>>, const int>);
  static_assert(std::is_same_v<mp1::buffer_element_t<mp1::region<int>>, int>);
  static_assert(std::is_same_v<mp1::buffer_element_t<mp1::region<const int>>, int>);
  static_assert(std::is_same_v<decltype(std::declval<mp1::region<int>>().data()), int*>);
  static_assert(
      std::is_same_v<decltype(std::declval<mp1::region<const int>>().data()), const int*>);
  static_assert(
      std::is_same_v<decltype(std::declval<mp1::detail::region_capture<int>>().data), int*>);
  static_assert(
      std::is_same_v<decltype(std::declval<mp1::detail::region_capture<const int>>().data),
                     const int*>);
  static_assert(
      std::is_same_v<decltype(std::declval<mp1::detail::region_capture<const int>>().map),
                     mp1::layout<int>>);

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(vocabulary_lifetime_fixture);

BOOST_AUTO_TEST_SUITE(vocabulary_lifetime)

BOOST_AUTO_TEST_CASE(region_two_word_borrow) {
  mp1::layout<int> map = mp1::layout<int>::contiguous(2);
  int data[2] = {1, 2};

  mp1::region r{&data[0], map};
  static_assert(std::is_same_v<decltype(r), mp1::region<int>>);
  BOOST_TEST(r.data() == &data[0]);
  BOOST_TEST(&r.map() == &map);
  BOOST_TEST(r.map().native_handle() == map.native_handle());
  BOOST_TEST(r.map().size() == 2);

  // Copying a borrow copies the two words: same data pointer, same layout referent.
  const mp1::region<int> copy{r};
  BOOST_TEST(copy.data() == r.data());
  BOOST_TEST(&copy.map() == &map);

  // Move construction is a reference copy; the moved-from object is not used afterwards.
  mp1::region<int> source{r};
  const mp1::region<int> moved{std::move(source)};
  BOOST_TEST(moved.data() == r.data());
  BOOST_TEST(&moved.map() == &map);

  // The borrow is writable through the data pointer for a non-const T.
  *r.data() = 7;
  BOOST_TEST(data[0] == 7);

  // A const region shares the unqualified layout and exposes a const data pointer.
  mp1::region<const int> const_region{&data[0], map};
  static_assert(std::is_same_v<decltype(const_region.data()), const int*>);
  BOOST_TEST(const_region.data() == &data[0]);
  BOOST_TEST(&const_region.map() == &map);

  // The explicit-T null spelling (CTAD cannot deduce T from nullptr).
  mp1::region<int> null_data{nullptr, map};
  BOOST_TEST(null_data.data() == nullptr);
  BOOST_TEST(&null_data.map() == &map);
}

BOOST_AUTO_TEST_CASE(region_one_object_named_layout) {
  // One std::array object as one datatype needs a *named* layout.  The
  // prvalue spelling is intentionally rejected by the deleted rvalue-layout constructor
  // (compile-fail probes region_array_temporary_layout.cc / region_rvalue_layout.cc).
  std::array<int, 3> object{4, 5, 6};
  const mp1::layout<std::array<int, 3>> map = mp1::layout<std::array<int, 3>>::contiguous(1);

  mp1::region r{&object, map};
  static_assert(std::is_same_v<decltype(r), mp1::region<std::array<int, 3>>>);
  BOOST_TEST(r.data() == &object);
  BOOST_TEST(r.map().size() == 1);
  BOOST_TEST(r.map().byte_size() == static_cast<mp1::count_t>(sizeof(object)));
  BOOST_TEST(r.map().native_handle() == map.native_handle());
}

BOOST_AUTO_TEST_CASE(region_blocking_borrow) {
  const mp1::mpi_communicator& world = mp1::comm_world();
  constexpr int tag = 2101;

  // The layout lvalue is in the caller's frame; the region borrows it for the blocking call.
  mp1::layout<int> map = mp1::layout<int>::contiguous(2);
  int data[2] = {0, 0};

  if (world.size() >= 2) {
    if (world.rank() == 0) {
      data[0] = 41;
      data[1] = 1;
      const mp1::region<const int> send_buffer{&data[0], map};
      mp1::send(world, send_buffer, 1, mp1::tag{tag});
    } else if (world.rank() == 1) {
      mp1::region<int> recv_buffer{&data[0], map};
      (void)mp1::recv(world, recv_buffer, 0, mp1::tag{tag});
      BOOST_TEST(data[0] == 41);
      BOOST_TEST(data[1] == 1);
    }
    mp1::barrier(world);
  }

  // The map survived every blocking call above and is still usable.
  BOOST_TEST(map.is_valid());
  BOOST_TEST(map.size() == 2);
}

BOOST_AUTO_TEST_CASE(region_capture_duplicates_layout) {
  const mp1::mpi_communicator& world = mp1::comm_world();
  constexpr int tag = 2102;

  int data[3] = {7, 8, 9};
  std::optional<mp1::request> request;

  if (world.rank() == 0) {
    // The initiating layout dies while the request owns a duplicate: the request still
    // reaches the datatype at wait().
    mp1::layout<int> scoped = mp1::layout<int>::contiguous(3);
    request.emplace(mp1::isend(world, mp1::region{&data[0], scoped}, 1, mp1::tag{tag}));
    // The initiating layout is destroyed here and the request is still valid.
    BOOST_TEST(request->is_valid());
  }

  if (world.size() >= 2) {
    if (world.rank() == 0) {
      request->wait();
    } else if (world.rank() == 1) {
      int received[3] = {0, 0, 0};
      mp1::layout<int> recv_map = mp1::layout<int>::contiguous(3);
      (void)mp1::recv(world, mp1::region{&received[0], recv_map}, 0, mp1::tag{tag});
      BOOST_TEST(received[0] == 7);
      BOOST_TEST(received[1] == 8);
      BOOST_TEST(received[2] == 9);
    }
    mp1::barrier(world);
  }

  // The vocabulary hook captures the same (data, owning duplicate) pair; a const region
  // captures a const data pointer and still owns an unqualified-element map.
  const int const_data[1] = {0};
  mp1::layout<int> const_map = mp1::layout<int>::contiguous(1);
  auto const_capture =
      mp1::detail::capture_region(mp1::region<const int>{&const_data[0], const_map});
  static_assert(std::is_same_v<decltype(const_capture.data), const int*>);
  static_assert(std::is_same_v<decltype(const_capture.map), mp1::layout<int>>);
  BOOST_TEST(const_capture.data == &const_data[0]);
  BOOST_TEST(const_capture.map.size() == 1);
}

BOOST_AUTO_TEST_CASE(span_from_checked_pointer_and_count) {
  // (nullptr, 0) is the empty span and stays null (std::span's empty-range spelling).
  int* null_buffer = nullptr;
  const std::span<int> empty = mp1::span_from(null_buffer, 0);
  BOOST_TEST(empty.empty());
  BOOST_TEST(empty.data() == nullptr);
  BOOST_TEST(empty.size() == 0);

  // A null pointer with a positive count would construct a non-empty span over null (a
  // std::span precondition violation) and would hand MPI a null buffer with a positive
  // count; the helper rejects it with std::invalid_argument instead.
  BOOST_CHECK_THROW((void)mp1::span_from(null_buffer, 1), std::invalid_argument);
  BOOST_CHECK_THROW((void)mp1::span_from(null_buffer, 2), std::invalid_argument);

  // Negative counts are rejected for every pointer.
  int data[3] = {1, 2, 3};
  BOOST_CHECK_THROW((void)mp1::span_from(&data[0], -1), std::invalid_argument);

  // A non-null pointer with a zero count is the empty span at that address.
  const std::span<int> zero = mp1::span_from(&data[0], 0);
  BOOST_TEST(zero.empty());
  BOOST_TEST(zero.data() == &data[0]);

  // The positive case is unchanged.
  const std::span<int> positive = mp1::span_from(&data[0], 3);
  BOOST_TEST(positive.size() == 3);
  BOOST_TEST(positive[2] == 3);

  // The template parameter keeps the element constness.
  const int const_data[3] = {4, 5, 6};
  const std::span<const int> const_span = mp1::span_from(&const_data[0], 3);
  static_assert(
      std::is_same_v<decltype(mp1::span_from(&const_data[0], 3)), std::span<const int>>);
  BOOST_TEST(const_span[0] == 4);
}

BOOST_AUTO_TEST_CASE(blocks_spans_and_empty) {
  const std::array<mp1::count_t, 3> counts{2, 1, 3};
  const std::array<mp1::count_t, 3> displacements{0, 2, 3};

  const mp1::blocks designated{.counts = counts, .displacements = displacements};
  static_assert(std::is_aggregate_v<decltype(designated)>);
  BOOST_TEST(designated.counts.size() == 3);
  BOOST_TEST(designated.displacements.size() == 3);
  BOOST_TEST(designated.counts[2] == 3);
  BOOST_TEST(designated.displacements[2] == 3);

  // Empty is the zero-block descriptor: two empty spans, no allocation.
  const mp1::blocks empty{};
  BOOST_TEST(empty.counts.empty());
  BOOST_TEST(empty.displacements.empty());
  BOOST_TEST(empty.counts.data() == nullptr);
  BOOST_TEST(empty.displacements.data() == nullptr);

  // A temporary vector bound by the spans is valid for the whole full-expression, which is
  // exactly the blocking-call lifetime.
  const auto read_back = [](mp1::blocks value) {
    mp1::count_t counts_total = 0;
    mp1::count_t displacement_last = 0;
    for (const mp1::count_t count : value.counts)
      counts_total += count;
    for (const mp1::count_t displacement : value.displacements)
      displacement_last = displacement;
    return std::pair{counts_total, displacement_last};
  };
  const auto temporary = read_back(mp1::blocks{
      .counts = std::vector<mp1::count_t>{4, 5},
      .displacements = std::vector<mp1::count_t>{0, 4},
  });
  BOOST_TEST(temporary.first == 9);
  BOOST_TEST(temporary.second == 4);
}

BOOST_AUTO_TEST_CASE(blocks_initiation_copies_arrays) {
  // The nonblocking/persistent plan copies both spans at initiation (§2.3): the caller's
  // count/displacement arrays may die after the call, as this scope change demonstrates.
  std::vector<mp1::count_t> plan_counts;
  std::vector<mp1::count_t> plan_displacements;

  {
    std::vector<mp1::count_t> counts{6, 7};
    std::vector<mp1::count_t> displacements{0, 6};
    const mp1::blocks descriptor{.counts = counts, .displacements = displacements};
    plan_counts.assign(descriptor.counts.begin(), descriptor.counts.end());
    plan_displacements.assign(descriptor.displacements.begin(), descriptor.displacements.end());
  }

  BOOST_TEST(plan_counts == (std::vector<mp1::count_t>{6, 7}));
  BOOST_TEST(plan_displacements == (std::vector<mp1::count_t>{0, 6}));
}

BOOST_AUTO_TEST_CASE(release_paths_use_single_lifecycle_query) {
  if (not mpi_wrap::supported())
    return;

  // Adopt the externally initialized environment first (the global fixture already did via
  // mp1::init, so this is the fast path); the gate measures only the settled phase.
  const mp1::mpi_communicator c = mp1::comm_world();
  const mp1::layout<int> base = mp1::layout<int>::vector(3, 1, 1);

  constexpr int copies = 16;
  mpi_wrap::reset_counts();
  for (int i = 0; i < copies; ++i) {
    const mp1::layout<int> duplicate{base};
    (void)duplicate;
  }
  BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == static_cast<std::size_t>(copies));
  BOOST_TEST(mpi_wrap::count("MPI_Type_free") == static_cast<std::size_t>(copies));
  // One MPI_Finalized per release; the redundant MPI_Initialized of the old two-query guard
  // is gone once the phase settled (the single-guard fast path).
  BOOST_TEST(mpi_wrap::count("MPI_Finalized") == static_cast<std::size_t>(copies));
  BOOST_TEST(mpi_wrap::count("MPI_Initialized") == 0);

  constexpr int duplications = 8;
  mpi_wrap::reset_counts();
  for (int i = 0; i < duplications; ++i) {
    const mp1::communicator duplicate = mp1::communicator::dup(c);
    (void)duplicate;
  }
  // A dup snapshots the metadata and the destructor releases: two settled-phase checks, each
  // with the single finalize query and no initialized query.
  BOOST_TEST(mpi_wrap::count("MPI_Initialized") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Finalized") == static_cast<std::size_t>(2 * duplications));

  // A settled hot operation still takes the phase-only fast path (no lifecycle query).
  mpi_wrap::reset_counts();
  for (int i = 0; i < 8; ++i)
    mp1::barrier(c);
  BOOST_TEST(mpi_wrap::count("MPI_Initialized") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Finalized") == 0);
}

BOOST_AUTO_TEST_CASE(capacity_and_in_place_compatibility) {
  const mp1::capacity bound{4};
  BOOST_TEST(bound.value() == 4);
  static_assert(!std::is_convertible_v<mp1::count_t, mp1::capacity>);

  static_assert(std::is_same_v<decltype(mp1::in_place), const mp1::in_place_t>);
  static_assert(mp1::borrowable<mp1::in_place_t&&>);
  static_assert(mp1::in_place_send_source<mp1::in_place_t>);
  static_assert(mp1::in_place_recv_sink<mp1::in_place_t>);
  static_assert(!mp1::send_source<mp1::in_place_t>);
  static_assert(!mp1::recv_sink<mp1::in_place_t>);
  const mp1::in_place_t sentinel{};
  (void)sentinel;
}

BOOST_AUTO_TEST_SUITE_END()
