// Combined nonblocking send-receive contract: isendrecv / isendrecv_replace for every shape
// (scalar, pointer+count, layout, contiguous container, empty), the wait-only request kind,
// MPI_PROC_NULL handling, unequal receive capacity, and the send/receive alias rejection that
// points at the *_replace form.
//
// MPI 4.1 basis: §3.2.7 (combined send-receive) and §3.7.2 (nonblocking buffer lifetime).  The
// test runs at 2 and 4 ranks; every rank is both a sender and a receiver in a ring, so no rank
// can pass through a test case while a peer still waits for it.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_ISENDRECV_WRAP 1
#else
#define MP1_ISENDRECV_WRAP 0
#endif


namespace {

  [[nodiscard]] int next_rank(const mp1::mpi_communicator& c, int rank) {
    return (rank + 1) % c.size();
  }

  [[nodiscard]] int previous_rank(const mp1::mpi_communicator& c, int rank) {
    return (rank - 1 + c.size()) % c.size();
  }

  template<class Function>
  bool throws_invalid_argument(Function&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class Function>
  bool throws_logic_error(Function&& function) {
    try {
      function();
    } catch (const std::logic_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  // The receive side of a combined nonblocking send-receive with source = MPI_PROC_NULL
  // returns the MPI 4.1 §3.10 status: source MPI_PROC_NULL, tag MPI_ANY_TAG, count 0 and
  // success, synthesized with normalized provenance on every provider.
  void check_proc_null_receive_status(const mp1::status& status) {
    BOOST_TEST(static_cast<int>(status.kind()) ==
               static_cast<int>(mp1::status_kind::normalized));
    BOOST_TEST(status.source() == MPI_PROC_NULL);
    BOOST_TEST(status.tag().value == mp1::any_tag.value);
    BOOST_TEST(status.error() == MPI_SUCCESS);
    BOOST_TEST(status.get_count<int>() == 0);
    BOOST_TEST(status.get_elements<int>() == 0);
    BOOST_TEST(not status.is_cancelled());
#if MP1_ISENDRECV_WRAP
    if (mpi_wrap::supported()) {
      // The synthesized status answers the count/elements/cancellation queries locally.
      mpi_wrap::reset_counts();
      (void)status.get_count<int>();
      (void)status.get_elements<int>();
      (void)status.is_cancelled();
      BOOST_TEST(mpi_wrap::total() == 0);
    }
#endif
  }

  struct isendrecv_fixture {
    isendrecv_fixture() {
      mp1::init();
    }

    ~isendrecv_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture cleanup must not throw
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(isendrecv_fixture);

BOOST_AUTO_TEST_SUITE(p2p_isendrecv_contract)

BOOST_AUTO_TEST_CASE(scalar_empty_and_wait_only_request) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  int send = rank * 10;
  int recv = -1;
  mp1::request request =
      mp1::isendrecv(c, send, next_rank(c, rank), mp1::tag{11}, recv, previous_rank(c, rank), mp1::tag{11});
  BOOST_TEST(request.is_valid());
  BOOST_TEST(request.is_active());
  BOOST_TEST(not request.is_persistent());
  BOOST_TEST(not request.cancelable());
  BOOST_TEST(request.drop_would_block());
  BOOST_TEST(static_cast<int>(request.kind()) == static_cast<int>(mp1::request_kind::sendrecv));
  BOOST_TEST(throws_logic_error([&request] { request.cancel(); }));
  const mp1::status received = request.wait();
  BOOST_TEST(not request.is_valid());
  // A provider may report the MPI_Isendrecv status of the send side rather than the receive
  // status; the portable contract is the completed request and the received data.
  BOOST_TEST(not received.is_cancelled());
  BOOST_TEST(recv == previous_rank(c, rank) * 10);

  // Empty forms are the std::span<T>{} spelling and exchange a zero-length message.
  const mp1::status empty = mp1::isendrecv(c, std::span<int>{}, next_rank(c, rank), mp1::tag{12},
                                           std::span<int>{}, previous_rank(c, rank), mp1::tag{12})
                                .wait();
  BOOST_TEST(not empty.is_cancelled());
  int empty_value = 7;
  const mp1::status replaced_empty =
      mp1::isendrecv_replace(c, std::span(&empty_value, 0), next_rank(c, rank), mp1::tag{13},
                             previous_rank(c, rank), mp1::tag{13})
          .wait();
  BOOST_TEST(not replaced_empty.is_cancelled());

  const mp1::status blocking_empty =
      mp1::sendrecv(c, std::span<int>{}, next_rank(c, rank), mp1::tag{14}, std::span<int>{},
                    previous_rank(c, rank), mp1::tag{14});
  BOOST_TEST(blocking_empty.source() == previous_rank(c, rank));
  const mp1::status blocking_replace_empty = mp1::sendrecv_replace(
      c, std::span(&empty_value, 0), next_rank(c, rank), mp1::tag{15}, previous_rank(c, rank), mp1::tag{15});
  BOOST_TEST(blocking_replace_empty.source() == previous_rank(c, rank));

  // Each combined call issues exactly one native MPI_Isendrecv_c; no fallback pair.
#if MP1_ISENDRECV_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    mp1::request counted =
        mp1::isendrecv(c, send, next_rank(c, rank), mp1::tag{16}, recv, previous_rank(c, rank), mp1::tag{16});
    counted.wait();
    BOOST_TEST(mpi_wrap::count("MPI_Isendrecv_c") == 1u);
    BOOST_TEST(mpi_wrap::count("MPI_Isend") == 0u);
    BOOST_TEST(mpi_wrap::count("MPI_Irecv") == 0u);

    mpi_wrap::reset_counts();
    mp1::request counted_replace =
        mp1::isendrecv_replace(c, send, next_rank(c, rank), mp1::tag{17}, previous_rank(c, rank), mp1::tag{17});
    counted_replace.wait();
    BOOST_TEST(mpi_wrap::count("MPI_Isendrecv_replace_c") == 1u);
  }
#endif
}

BOOST_AUTO_TEST_CASE(pointer_counts_unequal_capacity) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  int send[2] = {rank, rank + 100};
  int recv[3] = {-7, -7, -7};
  mp1::isendrecv(c, std::span(send), next_rank(c, rank), mp1::tag{20}, std::span(recv),
                 previous_rank(c, rank), mp1::tag{20})
      .wait();
  const int source = previous_rank(c, rank);
  BOOST_TEST(recv[0] == source);
  BOOST_TEST(recv[1] == source + 100);
  // A larger receive capacity is legal and leaves the extra element untouched.
  BOOST_TEST(recv[2] == -7);

  // The pointer form takes independent send/receive counts; a smaller receive count is the
  // caller's truncation obligation (MPI 4.1 §3.2.7), while the container form carries the same
  // obligation as well.  The legal larger-capacity case above is the contract this test pins.
}

BOOST_AUTO_TEST_CASE(layout_and_container_forms) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  mp1::layout<int> even_slots = mp1::layout<int>::vector(2, 1, 2);
  int send[4] = {rank, -1, rank + 10, -1};
  int recv[4] = {-7, -7, -7, -7};
  mp1::isendrecv(c, mp1::region{send, even_slots}, next_rank(c, rank), mp1::tag{22},
                 mp1::region{recv, even_slots}, previous_rank(c, rank), mp1::tag{22})
      .wait();
  const int source = previous_rank(c, rank);
  BOOST_TEST(recv[0] == source);
  BOOST_TEST(recv[1] == -7);
  BOOST_TEST(recv[2] == source + 10);
  BOOST_TEST(recv[3] == -7);

  // Container send/receive with a larger receive capacity; the tail stays untouched.
  std::vector<int> send_container{rank, rank + 1};
  std::vector<int> recv_container(3, -7);
  mp1::isendrecv(c, send_container, next_rank(c, rank), mp1::tag{23}, recv_container,
                 previous_rank(c, rank), mp1::tag{23})
      .wait();
  BOOST_TEST(recv_container[0] == source);
  BOOST_TEST(recv_container[1] == source + 1);
  BOOST_TEST(recv_container[2] == -7);
}

BOOST_AUTO_TEST_CASE(proc_null_directions) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  // Both directions unused: every rank is a pure no-op whose completed receive status is
  // the synthesized MPI 4.1 §3.10 status (the providers leave it unspecified on their own).
  int send = rank;
  int recv = -1;
  const mp1::status both_null =
      mp1::isendrecv(c, send, MPI_PROC_NULL, mp1::tag{30}, recv, MPI_PROC_NULL, mp1::tag{30}).wait();
  check_proc_null_receive_status(both_null);
  BOOST_TEST(recv == -1);

  // A region send makes the combined request own a payload; the receive side is still
  // MPI_PROC_NULL, so the payload-carrying completion path must normalize the same way.
  {
    mp1::layout<int> layout = mp1::layout<int>::contiguous(2);
    std::array<int, 2> outgoing{rank * 10, rank * 10 + 1};
    int unused = -1;
    const mp1::status region_null =
        mp1::isendrecv(c, mp1::region{outgoing.data(), layout}, MPI_PROC_NULL, mp1::tag{31},
                       std::span(&unused, 0), MPI_PROC_NULL, mp1::tag{31})
            .wait();
    check_proc_null_receive_status(region_null);
    BOOST_TEST(unused == -1);
  }

  // Exactly one send-only and one receive-only direction: rank 0 sends to rank 1, which
  // receives with a PROC_NULL send direction, so the combined request completes on both
  // sides.  All other ranks stay pure no-ops.
  if (rank == 0) {
    int out = 42;
    int unused = -1;
    const mp1::status sent =
        mp1::isendrecv(c, std::span(&out, 1), 1, mp1::tag{32}, std::span(&unused, 0), MPI_PROC_NULL, mp1::tag{32})
            .wait();
    // The receive side was posted with MPI_PROC_NULL, so the completion status is synthesized
    // even though the send direction delivered a real message; the receive buffer is untouched.
    check_proc_null_receive_status(sent);
    BOOST_TEST(unused == -1);
  } else if (rank == 1) {
    int unused = 0;
    int received = -1;
    const mp1::status status = mp1::isendrecv(c, std::span(&unused, 0), MPI_PROC_NULL, mp1::tag{32},
                                              std::span(&received, 1), 0, mp1::tag{32})
                                   .wait();
    // A real receive source keeps the provider's native, count-defining status.
    BOOST_TEST(static_cast<int>(status.kind()) == static_cast<int>(mp1::status_kind::native));
    BOOST_TEST(status.source() == 0);
    BOOST_TEST(status.tag().value == mp1::tag{32}.value);
    BOOST_TEST(status.get_count<int>() == 1);
    BOOST_TEST(received == 42);
  } else {
    int unused = 0;
    mp1::isendrecv(c, std::span(&unused, 0), MPI_PROC_NULL, mp1::tag{32}, std::span(&unused, 0),
                   MPI_PROC_NULL, mp1::tag{32})
        .wait();
  }

  // The container alias check is skipped when one direction is MPI_PROC_NULL: rank 1 passes
  // the same container as send and receive, but only the receive direction is used.
  if (rank == 0) {
    std::vector<int> outgoing{7, 8};
    std::vector<int> unused(1, -1);
    mp1::isendrecv(c, outgoing, 1, mp1::tag{33}, unused, MPI_PROC_NULL, mp1::tag{33}).wait();
  } else if (rank == 1) {
    std::vector<int> same(2, -1);
    const mp1::status status = mp1::isendrecv(c, same, MPI_PROC_NULL, mp1::tag{33}, same, 0, mp1::tag{33}).wait();
    BOOST_TEST(static_cast<int>(status.kind()) == static_cast<int>(mp1::status_kind::native));
    BOOST_TEST(status.source() == 0);
    BOOST_TEST(status.get_count<int>() == 2);
    BOOST_TEST(same[0] == 7);
    BOOST_TEST(same[1] == 8);
  } else {
    std::vector<int> same(2, -1);
    mp1::isendrecv(c, same, MPI_PROC_NULL, mp1::tag{33}, same, MPI_PROC_NULL, mp1::tag{33}).wait();
  }
}

BOOST_AUTO_TEST_CASE(replace_proc_null_null_positive_span) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int next = next_rank(c, rank);
  const int previous = previous_rank(c, rank);

  // MPI 4.1 §3.10: with both directions MPI_PROC_NULL the combined call is a no-op and
  // never accesses the buffer, so a null data pointer with a positive extent is legal.  The
  // fixed-buffer *_replace forms used to reject it locally.
  const auto null_span = [] { return std::span<int>(static_cast<int*>(nullptr), 3); };

  bool blocking_null_ok = true;
  try {
    (void)mp1::sendrecv_replace(c, null_span(), MPI_PROC_NULL, mp1::tag{34}, MPI_PROC_NULL, mp1::tag{34});
  } catch (...) {
    blocking_null_ok = false;
  }
  BOOST_TEST(blocking_null_ok);

  bool async_null_ok = true;
  std::optional<mp1::status> async_null_status;
  try {
    mp1::request request =
        mp1::isendrecv_replace(c, null_span(), MPI_PROC_NULL, mp1::tag{35}, MPI_PROC_NULL, mp1::tag{35});
    async_null_status = request.wait();
  } catch (...) {
    async_null_ok = false;
  }
  BOOST_TEST(async_null_ok);
  if (async_null_status)
    check_proc_null_receive_status(*async_null_status);

  // One ignored direction leaves the other direction active, so the buffer is read or
  // written and a null positive-extent buffer stays rejected.  This is what makes the fix
  // exact: the ignored flag is the both-null conjunction, not either direction alone.
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::sendrecv_replace(c, null_span(), next, mp1::tag{36}, MPI_PROC_NULL, mp1::tag{36}); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::sendrecv_replace(c, null_span(), MPI_PROC_NULL, mp1::tag{37}, previous, mp1::tag{37}); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::isendrecv_replace(c, null_span(), next, mp1::tag{38}, MPI_PROC_NULL, mp1::tag{38}); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::isendrecv_replace(c, null_span(), MPI_PROC_NULL, mp1::tag{39}, previous, mp1::tag{39}); }));

  // Non-null control: both directions ignored, the buffer is untouched by both forms.
  int value[3] = {rank, rank + 1, rank + 2};
  (void)mp1::sendrecv_replace(c, std::span(value), MPI_PROC_NULL, mp1::tag{40}, MPI_PROC_NULL, mp1::tag{40});
  BOOST_TEST(value[0] == rank);
  BOOST_TEST(value[1] == rank + 1);
  BOOST_TEST(value[2] == rank + 2);
  mp1::request replace_request =
      mp1::isendrecv_replace(c, std::span(value), MPI_PROC_NULL, mp1::tag{41}, MPI_PROC_NULL, mp1::tag{41});
  check_proc_null_receive_status(replace_request.wait());
  BOOST_TEST(value[0] == rank);
  BOOST_TEST(value[1] == rank + 1);
  BOOST_TEST(value[2] == rank + 2);

  // The region form owns a duplicated receive layout, so this covers the payload-carrying
  // completion path of the in-place combined request with both directions PROC_NULL.
  {
    mp1::layout<int> layout = mp1::layout<int>::contiguous(3);
    std::array<int, 3> untouched{rank, rank + 1, rank + 2};
    mp1::request region_request =
        mp1::isendrecv_replace(c, mp1::region{untouched.data(), layout}, MPI_PROC_NULL,
                               mp1::tag{46}, MPI_PROC_NULL, mp1::tag{46});
    check_proc_null_receive_status(region_request.wait());
    BOOST_TEST(untouched[0] == rank);
    BOOST_TEST(untouched[2] == rank + 2);
  }

  // One ignored direction with a valid buffer is a real one-way exchange, not a blanket
  // no-op.  Even ranks send to the successor while odd ranks receive from the predecessor;
  // the matching calls are issued concurrently, so no eager-progress assumption is needed.
  // The second pass swaps the roles with a different tag.
  const auto pattern = [](int owner) {
    return std::vector<int>{owner * 10, owner * 10 + 1, owner * 10 + 2};
  };
  const bool even = rank % 2 == 0;

  if (even) {
    std::vector<int> outgoing = pattern(rank);
    (void)mp1::sendrecv_replace(c, outgoing, next, mp1::tag{42}, MPI_PROC_NULL, mp1::tag{42});
  } else {
    std::vector<int> incoming(3, -1);
    (void)mp1::sendrecv_replace(c, incoming, MPI_PROC_NULL, mp1::tag{42}, previous, mp1::tag{42});
    BOOST_TEST(incoming == pattern(previous));
  }
  if (even) {
    std::vector<int> incoming(3, -1);
    (void)mp1::sendrecv_replace(c, incoming, MPI_PROC_NULL, mp1::tag{43}, previous, mp1::tag{43});
    BOOST_TEST(incoming == pattern(previous));
  } else {
    std::vector<int> outgoing = pattern(rank);
    (void)mp1::sendrecv_replace(c, outgoing, next, mp1::tag{43}, MPI_PROC_NULL, mp1::tag{43});
  }

  if (even) {
    std::vector<int> outgoing = pattern(rank + 100);
    mp1::request send_only = mp1::isendrecv_replace(c, outgoing, next, mp1::tag{44}, MPI_PROC_NULL, mp1::tag{44});
    // The receive side is MPI_PROC_NULL even though the send direction is a real transfer.
    check_proc_null_receive_status(send_only.wait());
  } else {
    std::vector<int> incoming(3, -1);
    mp1::request recv_only =
        mp1::isendrecv_replace(c, incoming, MPI_PROC_NULL, mp1::tag{44}, previous, mp1::tag{44});
    const mp1::status status = recv_only.wait();
    // A real receive source keeps the provider's native, count-defining status.
    BOOST_TEST(static_cast<int>(status.kind()) == static_cast<int>(mp1::status_kind::native));
    BOOST_TEST(status.source() == previous);
    BOOST_TEST(status.get_count<int>() == 3);
    BOOST_TEST(incoming == pattern(previous + 100));
  }
  if (even) {
    std::vector<int> incoming(3, -1);
    mp1::request recv_only =
        mp1::isendrecv_replace(c, incoming, MPI_PROC_NULL, mp1::tag{45}, previous, mp1::tag{45});
    const mp1::status status = recv_only.wait();
    BOOST_TEST(static_cast<int>(status.kind()) == static_cast<int>(mp1::status_kind::native));
    BOOST_TEST(status.source() == previous);
    BOOST_TEST(status.get_count<int>() == 3);
    BOOST_TEST(incoming == pattern(previous + 100));
  } else {
    std::vector<int> outgoing = pattern(rank + 100);
    mp1::request send_only = mp1::isendrecv_replace(c, outgoing, next, mp1::tag{45}, MPI_PROC_NULL, mp1::tag{45});
    check_proc_null_receive_status(send_only.wait());
  }
}

BOOST_AUTO_TEST_CASE(replace_forms) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int source = previous_rank(c, rank);

  int scalar = rank;
  mp1::isendrecv_replace(c, scalar, next_rank(c, rank), mp1::tag{40}, source, mp1::tag{40}).wait();
  BOOST_TEST(scalar == source);

  int pointer[2] = {rank, rank + 1};
  mp1::isendrecv_replace(c, std::span(pointer), next_rank(c, rank), mp1::tag{41}, source, mp1::tag{41}).wait();
  BOOST_TEST(pointer[0] == source);
  BOOST_TEST(pointer[1] == source + 1);

  mp1::layout<int> layout = mp1::layout<int>::contiguous(2);
  int layout_data[2] = {rank + 2, rank + 3};
  mp1::isendrecv_replace(c, mp1::region{layout_data, layout}, next_rank(c, rank), mp1::tag{42}, source, mp1::tag{42})
      .wait();
  BOOST_TEST(layout_data[0] == source + 2);
  BOOST_TEST(layout_data[1] == source + 3);

  std::vector<int> container{rank + 4, rank + 5};
  mp1::isendrecv_replace(c, container, next_rank(c, rank), mp1::tag{43}, source, mp1::tag{43}).wait();
  BOOST_TEST(container[0] == source + 4);
  BOOST_TEST(container[1] == source + 5);

  // Restart cycles: the combined request is wait-only, never restarted.
  for (int cycle = 0; cycle < 8; ++cycle) {
    int value = rank + cycle;
    mp1::request request =
        mp1::isendrecv_replace(c, value, next_rank(c, rank), mp1::tag{44 + cycle}, source, mp1::tag{44 + cycle});
    request.wait();
    BOOST_TEST(value == source + cycle);
  }
}

BOOST_AUTO_TEST_CASE(alias_rejection_names_replace) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  std::vector<int> shared(2, rank);
  bool invalid = false;
  std::string diagnostic;
  try {
    (void)mp1::isendrecv(c, shared, next_rank(c, rank), mp1::tag{50}, shared, previous_rank(c, rank), mp1::tag{50});
  } catch (const std::invalid_argument& error) {
    invalid = true;
    diagnostic = error.what();
  }
  BOOST_TEST(invalid);
  BOOST_TEST(diagnostic.find("isendrecv_replace") != std::string::npos);

  invalid = false;
  diagnostic.clear();
  try {
    mp1::sendrecv(c, shared, next_rank(c, rank), mp1::tag{51}, shared, previous_rank(c, rank), mp1::tag{51});
  } catch (const std::invalid_argument& error) {
    invalid = true;
    diagnostic = error.what();
  }
  BOOST_TEST(invalid);
  BOOST_TEST(diagnostic.find("sendrecv_replace") != std::string::npos);

  // A pointer pair that overlaps but is not identical is also rejected.
  int overlap[4] = {rank, rank, rank, rank};
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::isendrecv(c, std::span(overlap, 2), next_rank(c, rank), mp1::tag{52},
                         std::span(overlap + 1, 2), previous_rank(c, rank), mp1::tag{52});
  }));

  // *_replace is exempt: aliasing is its purpose.
  int in_place_value = rank;
  mp1::isendrecv_replace(c, in_place_value, next_rank(c, rank), mp1::tag{53}, previous_rank(c, rank), mp1::tag{53})
      .wait();
  BOOST_TEST(in_place_value == previous_rank(c, rank));
}

BOOST_AUTO_TEST_CASE(lifetime_stress) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int source = previous_rank(c, rank);
  std::vector<int> send(4, rank);
  std::vector<int> recv(4, -1);
  for (int cycle = 0; cycle < 200; ++cycle) {
    mp1::request request = mp1::isendrecv(c, send, next_rank(c, rank), mp1::tag{60}, recv, source, mp1::tag{60});
    request.wait();
    BOOST_TEST(recv[0] == source);
  }
}

BOOST_AUTO_TEST_SUITE_END()
