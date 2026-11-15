// std::span totals, p2p subset.
//
// A span is the pointer+count spelling: its extent is the element count in both directions,
// the empty span is the zero-count message, and a larger fixed receive extent is legal while a
// smaller one is a truncation error (MPI 4.1 §3.2.4).  Spans are borrowed, not owning: a span
// over a temporary container compiles but dangles (documented in §2.5; the library cannot
// detect it), which this test pins as a compile-time fact rather than a runtime use.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>


namespace {

  [[nodiscard]] int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  [[nodiscard]] int world_size() {
    int size{1};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  class world_return_errors_scope {
  public:
    world_return_errors_scope() {
      (void)MPI_Comm_get_errhandler(MPI_COMM_WORLD, &previous_);
      (void)MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    }

    world_return_errors_scope(const world_return_errors_scope&) = delete;
    world_return_errors_scope& operator=(const world_return_errors_scope&) = delete;

    ~world_return_errors_scope() {
      (void)MPI_Comm_set_errhandler(MPI_COMM_WORLD, previous_);
      (void)MPI_Errhandler_free(&previous_);
    }

  private:
    MPI_Errhandler previous_{MPI_ERRORS_ARE_FATAL};
  };

  template<class Function>
  [[nodiscard]] bool throws_invalid(Function&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

}  // namespace


struct mp1_span_totals_fixture {
  mp1_span_totals_fixture() {
    mp1::init();
  }

  ~mp1_span_totals_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Already finalized through another path; a destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_span_totals_fixture);

BOOST_AUTO_TEST_SUITE(span_totals_contract)

BOOST_AUTO_TEST_CASE(span_extent_is_the_count) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  std::vector<int> values{4, 5, 6};
  if (rank == 0) {
    mp1::send(world, std::span(values), 1, mp1::tag{7200});
  } else if (rank == 1) {
    const mp1::status probed = mp1::probe(world, 0, mp1::tag{7200});
    BOOST_TEST(probed.get_count<int>() == 3);
    std::vector<int> received(3);
    const mp1::status status = mp1::recv(world, std::span(received), 0, mp1::tag{7200});
    BOOST_TEST(received == values);
    BOOST_TEST(status.get_count<int>() == 3);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(empty_span_is_the_zero_count_message) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  if (rank == 0) {
    mp1::send(world, std::span<int>{}, 1, mp1::tag{7201});
  } else if (rank == 1) {
    const mp1::status probed = mp1::probe(world, 0, mp1::tag{7201});
    BOOST_TEST(probed.get_count<int>() == 0);
    const mp1::status status = mp1::recv(world, std::span<int>{}, 0, mp1::tag{7201});
    BOOST_TEST(status.get_count<int>() == 0);
  }
  world_barrier();

  // A zero-count message matches any zero-count receive regardless of datatype: the type
  // signature of a zero-count message contributes no elements.
  if (rank == 0) {
    mp1::send(world, std::span<double>{}, 1, mp1::tag{7202});
  } else if (rank == 1) {
    (void)mp1::recv(world, std::span<int>{}, 0, mp1::tag{7202});
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(larger_and_smaller_fixed_extent) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  // Larger receive extent: legal, extra elements untouched.
  if (rank == 0) {
    const std::array<int, 2> sent{{7, 8}};
    mp1::send(world, std::span(sent), 1, mp1::tag{7210});
  } else if (rank == 1) {
    std::array<int, 4> received{{-1, -1, -1, -1}};
    const mp1::status status = mp1::recv(world, std::span(received), 0, mp1::tag{7210});
    BOOST_TEST(received[0] == 7);
    BOOST_TEST(received[1] == 8);
    BOOST_TEST(received[2] == -1);
    BOOST_TEST(received[3] == -1);
    BOOST_TEST(status.get_count<int>() == 2);
  }
  world_barrier();

  // Smaller receive extent: truncation error.
  if (rank == 0) {
    const std::array<int, 3> sent{{1, 2, 3}};
    mp1::ssend(world, sent, 1, mp1::tag{7211});
  } else if (rank == 1) {
    world_return_errors_scope return_errors;
    std::array<int, 2> received{{-1, -1}};
    bool threw{false};
    try {
      (void)mp1::recv(world, std::span(received), 0, mp1::tag{7211});
    } catch (const mp1::mpi_error&) {
      threw = true;
    }
    BOOST_TEST(threw);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(c_array_and_std_array_spans) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  int c_array[3] = {1, 2, 3};
  std::array<int, 3> std_array{{4, 5, 6}};
  if (rank == 0) {
    mp1::send(world, std::span(c_array), 1, mp1::tag{7220});
    mp1::send(world, std::span(std_array), 1, mp1::tag{7221});
  } else if (rank == 1) {
    int c_received[3] = {0, 0, 0};
    std::array<int, 3> std_received{};
    (void)mp1::recv(world, std::span(c_received), 0, mp1::tag{7220});
    (void)mp1::recv(world, std::span(std_received), 0, mp1::tag{7221});
    BOOST_TEST(c_received[2] == 3);
    BOOST_TEST(std_received[0] == 4);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(subrange_over_contiguous_iterators) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  std::vector<int> values{31, 32, 33, 34};
  auto middle = std::ranges::subrange(values.begin() + 1, values.end() - 1);
  if (rank == 0) {
    mp1::send(world, middle, 1, mp1::tag{7230});
  } else if (rank == 1) {
    std::array<int, 2> received{};
    (void)mp1::recv(world, std::span(received), 0, mp1::tag{7230});
    BOOST_TEST(received[0] == 32);
    BOOST_TEST(received[1] == 33);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(span_const_and_string_view_send_only) {
  BOOST_TEST(true);
  static_assert(mp1::send_source<std::span<const int>>);
  static_assert(not mp1::recv_sink<std::span<const int>>);
  static_assert(mp1::send_source<std::string_view>);
  static_assert(not mp1::recv_sink<std::string_view>);
  static_assert(mp1::borrowable<std::span<int>>,
                "a borrowed view is async-borrowable regardless of its pointee");
  // The documented caveat: a span over a temporary container is constructible (borrowed, not
  // owning) and therefore passes every compile-time gate; the caller must keep the pointee
  // alive.  The object is never used here, so no dangling access occurs.
  static_assert(std::is_constructible_v<std::span<const int>, std::vector<int>&&>);
  static_assert(mp1::borrowable<std::span<const int>>);

  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  std::string_view text{"hello"};
  if (rank == 0) {
    mp1::send(world, text, 1, mp1::tag{7240});
  } else if (rank == 1) {
    std::array<char, 5> received{};
    (void)mp1::recv(world, std::span(received), 0, mp1::tag{7240});
    BOOST_TEST(std::string_view(received.data(), received.size()) == text);
  }
  world_barrier();
}


// An aggregate span at the root/for a rootless family is the whole
// `send.size() * peers` vector; the per-peer block is `aggregate.size() / peers` and must be
// integral and equal to the single-block side before any MPI call.
BOOST_AUTO_TEST_CASE(aggregate_spans_use_the_peer_count) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const int root = size - 1;
  BOOST_TEST(size >= 2);

  // gather: send is one block of two elements, the root receive is 2 * peers.
  const std::array<int, 2> send{{rank, rank + 10}};
  std::vector<int> gathered(static_cast<std::size_t>(2 * size), -1);
  if (rank == root) {
    mp1::gather(world, root, std::span<const int>{send}, gathered);
    for (int peer = 0; peer < size; ++peer) {
      BOOST_TEST(gathered[static_cast<std::size_t>(2 * peer)] == peer);
      BOOST_TEST(gathered[static_cast<std::size_t>(2 * peer + 1)] == peer + 10);
    }
    // Neither a non-integral aggregate nor a block mismatch may reach MPI.
    std::vector<int> not_divisible(static_cast<std::size_t>(2 * size + 1), -1);
    BOOST_TEST(throws_invalid(
        [&] { mp1::gather(world, root, std::span<const int>{send}, not_divisible); }));
    std::vector<int> short_blocks(static_cast<std::size_t>(2 * (size - 1)), -1);
    BOOST_TEST(throws_invalid(
        [&] { mp1::gather(world, root, std::span<const int>{send}, short_blocks); }));
  } else {
    mp1::gather(world, root, std::span<const int>{send}, std::span<int>{});
  }
  world_barrier();

  // scatter: the root send aggregate is 2 * peers, every rank receives one two-element block.
  std::vector<int> scattered(static_cast<std::size_t>(2 * size), -1);
  for (int peer = 0; peer < size; ++peer) {
    scattered[static_cast<std::size_t>(2 * peer)] = 100 + peer;
    scattered[static_cast<std::size_t>(2 * peer + 1)] = 200 + peer;
  }
  std::array<int, 2> block{{-1, -1}};
  if (rank == root) {
    mp1::scatter(world, root, scattered, std::span<int>{block});
    BOOST_TEST(block[0] == 100 + root);
    BOOST_TEST(block[1] == 200 + root);
    std::vector<int> not_divisible(static_cast<std::size_t>(2 * size + 1), -1);
    BOOST_TEST(throws_invalid(
        [&] { mp1::scatter(world, root, not_divisible, std::span<int>{block}); }));
  } else {
    mp1::scatter(world, root, std::span<const int>{}, std::span<int>{block});
    BOOST_TEST(block[0] == 100 + rank);
    BOOST_TEST(block[1] == 200 + rank);
  }
  world_barrier();

  // Rootless aggregate families: every rank's receive is peers * send.size().
  const std::array<int, 1> one{{rank + 1}};
  std::vector<int> all_gathered(static_cast<std::size_t>(size), -1);
  mp1::allgather(world, std::span<const int>{one}, all_gathered);
  for (int peer = 0; peer < size; ++peer)
    BOOST_TEST(all_gathered[static_cast<std::size_t>(peer)] == peer + 1);

  std::vector<int> all_send(static_cast<std::size_t>(size), -1);
  for (int peer = 0; peer < size; ++peer)
    all_send[static_cast<std::size_t>(peer)] = rank * 100 + peer;
  std::vector<int> all_recv(static_cast<std::size_t>(size), -1);
  mp1::alltoall(world, all_send, all_recv);
  for (int peer = 0; peer < size; ++peer)
    BOOST_TEST(all_recv[static_cast<std::size_t>(peer)] == peer * 100 + rank);
  world_barrier();
}

// Intercommunicator directions use the *remote* group size for the aggregate block rule; the
// local group size is irrelevant (MPI 4.1 §6.7/§6.8).
BOOST_AUTO_TEST_CASE(inter_remote_group_sizes) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  // Asymmetric at 4 ranks (1 + 3), so remote_size != local_size and a buffer sized by the
  // local group is observably wrong.
  const int group = world_rank() < 1 ? 0 : 1;
  mp1::communicator local = mp1::communicator::split(world, group);
  mp1::communicator inter =
      mp1::communicator::intercomm_create(local, 0, world, group == 0 ? 1 : 0, mp1::tag{7301});
  const int local_rank = inter.rank();
  const int remote_size = inter.remote_size();
  const int root_arg = group == 0 ? (local_rank == 0 ? MPI_ROOT : MPI_PROC_NULL) : 0;

  // gather: the root group's root receives remote_size blocks of two elements each.
  const std::array<int, 2> send{{group, local_rank + 10}};
  if (group == 0 and local_rank == 0) {
    std::vector<int> gathered(static_cast<std::size_t>(2 * remote_size), -1);
    mp1::gather(inter, root_arg, std::span<const int>{send}, gathered);
    for (int peer = 0; peer < remote_size; ++peer) {
      BOOST_TEST(gathered[static_cast<std::size_t>(2 * peer)] == 1);
      BOOST_TEST(gathered[static_cast<std::size_t>(2 * peer + 1)] == peer + 10);
    }
    // The aggregate divisor is the remote group size.  With a single remote peer every
    // size is integral and the cross-group block equality is MPI's runtime signature check,
    // so the local divisibility rejection is observable only for remote_size > 1 (the 4-rank
    // configuration; at 2 ranks the split is 1+1).
    if (remote_size > 1) {
      std::vector<int> wrong_aggregate(static_cast<std::size_t>(2 * remote_size + 1), -1);
      BOOST_TEST(throws_invalid(
          [&] { mp1::gather(inter, root_arg, std::span<const int>{send}, wrong_aggregate); }));
    }
  } else if (group == 1) {
    mp1::gather(inter, root_arg, std::span<const int>{send}, std::span<int>{});
  } else {
    mp1::gather(inter, root_arg, std::span<const int>{}, std::span<int>{});
  }
  world_barrier();

  // allgather: each group receives exactly remote_size blocks.
  const std::array<int, 1> one{{group * 1000 + local_rank}};
  std::vector<int> gathered(static_cast<std::size_t>(remote_size), -1);
  mp1::allgather(inter, std::span<const int>{one}, gathered);
  for (int peer = 0; peer < remote_size; ++peer)
    BOOST_TEST(gathered[static_cast<std::size_t>(peer)] == (1 - group) * 1000 + peer);

  // scatter: the root group's root sends remote_size two-element blocks.
  std::array<int, 2> block{{-1, -1}};
  if (group == 0 and local_rank == 0) {
    std::vector<int> scattered(static_cast<std::size_t>(2 * remote_size), -1);
    for (int peer = 0; peer < remote_size; ++peer) {
      scattered[static_cast<std::size_t>(2 * peer)] = 300 + peer;
      scattered[static_cast<std::size_t>(2 * peer + 1)] = 400 + peer;
    }
    int ignored = 0;
    mp1::scatter(inter, root_arg, scattered, ignored);
    if (remote_size > 1) {
      std::vector<int> wrong_aggregate(static_cast<std::size_t>(2 * remote_size + 1), -1);
      BOOST_TEST(
          throws_invalid([&] { mp1::scatter(inter, root_arg, wrong_aggregate, ignored); }));
    }
  } else {
    mp1::scatter(inter, root_arg, std::span<const int>{}, std::span<int>{block});
    if (group == 1) {
      BOOST_TEST(block[0] == 300 + local_rank);
      BOOST_TEST(block[1] == 400 + local_rank);
    }
  }
  world_barrier();

  inter.disconnect();
}

BOOST_AUTO_TEST_SUITE_END()
