// Contract test for the p2p receive-layout role gating: a proven-overlapping receive datatype
// is only erroneous in association with a receive buffer the operation actually updates
// (MPI 4.1 §5.1.11), so the wrapper applies the overlap rejection only to used receive
// directions.
//
// Covered roles:
//   * recv / irecv / recv_init with source = MPI_PROC_NULL: the receive is an explicit no-op
//     (§3.10), so an overlapping layout is accepted and the buffer stays untouched;
//   * mrecv / imrecv of MPI_MESSAGE_NO_PROC: same no-op role and the same buffer guarantee;
//   * recv / irecv / recv_init with a real source: the overlap is rejected locally with
//     std::invalid_argument before any MPI receive is posted;
//   * sendrecv / isendrecv: recvl is checked only when source != MPI_PROC_NULL;
//   * sendrecv_replace / isendrecv_replace: the shared buffer's layout is checked only when
//     source != MPI_PROC_NULL, so a real dest with a MPI_PROC_NULL source is accepted and both
//     directions MPI_PROC_NULL leaves the buffer untouched.
//
// Every rejected case asserts zero wrapped MPI calls where the entry point is in the wrap
// table (MPI_Recv_c, MPI_Irecv_c, MPI_Recv_init_c, MPI_Sendrecv_c, MPI_Isendrecv_c,
// MPI_Sendrecv_replace_c, MPI_Isendrecv_replace_c).  The test runs at 2 and 4 ranks; the
// accepted send/receive exchanges use ranks 0 and 1, and every other rank waits at a barrier.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <stdexcept>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_P2P_ROLE_WRAP 1
#else
#define MP1_P2P_ROLE_WRAP 0
#endif


namespace {

  int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  int world_size() {
    int size{1};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
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

  bool is_proc_null_status(const mp1::status& s) {
    return s.source() == MPI_PROC_NULL and s.tag() == mp1::any_tag and s.error() == MPI_SUCCESS;
  }

  // A proven-overlapping receive layout: blocklength 2 at element 0 and blocklength 2 at
  // element 1 cover elements {0, 1} and {1, 2}, so element 1 is written twice.  The layout is
  // constructed and committed before the assertion counters are reset, so a later rejection
  // cannot be confused with the layout's own datatype setup.
  [[nodiscard]] mp1::layout<int> overlap_layout() {
    mp1::indexed_parameter parameter;
    parameter.add(2, 0);
    parameter.add(2, 1);
    return mp1::layout<int>::indexed(parameter);
  }

  void reset_wrap_counts() {
#if MP1_P2P_ROLE_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
  }

  void expect_wrap_count(const char* symbol, std::size_t expected) {
#if MP1_P2P_ROLE_WRAP
    if (mpi_wrap::supported())
      BOOST_TEST(mpi_wrap::count(symbol) == expected);
#else
    (void)symbol;
    (void)expected;
#endif
  }

  struct p2p_receive_layout_role_fixture {
    p2p_receive_layout_role_fixture() {
      mp1::init();
    }

    ~p2p_receive_layout_role_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture cleanup must not throw
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(p2p_receive_layout_role_fixture);

BOOST_AUTO_TEST_SUITE(p2p_receive_layout_role_contract)

BOOST_AUTO_TEST_CASE(recv_family_accepts_overlap_only_for_proc_null) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const mp1::layout<int> overlap = overlap_layout();
  const int real_source = (rank + 1) % size;

  std::array<int, 4> buffer{{11, 22, 33, 44}};
  const std::array<int, 4> original = buffer;

  // Blocking recv from MPI_PROC_NULL: no message, no buffer update, and no local overlap
  // rejection.  The call still reaches MPI_Recv_c.
  reset_wrap_counts();
  const mp1::status blocking =
      mp1::recv(world, mp1::region{buffer.data(), overlap}, MPI_PROC_NULL, mp1::tag{9000});
  expect_wrap_count("MPI_Recv_c", 1);
  BOOST_TEST(is_proc_null_status(blocking));
  BOOST_TEST(buffer == original);
  world_barrier();

  // Nonblocking recv from MPI_PROC_NULL.
  reset_wrap_counts();
  mp1::request posted =
      mp1::irecv(world, mp1::region{buffer.data(), overlap}, MPI_PROC_NULL, mp1::tag{9001});
  expect_wrap_count("MPI_Irecv_c", 1);
  const mp1::status nonblocking = posted.wait();
  BOOST_TEST(is_proc_null_status(nonblocking));
  BOOST_TEST(buffer == original);
  world_barrier();

  // Persistent recv from MPI_PROC_NULL.
  reset_wrap_counts();
  mp1::request persistent =
      mp1::recv_init(world, mp1::region{buffer.data(), overlap}, MPI_PROC_NULL, mp1::tag{9002});
  expect_wrap_count("MPI_Recv_init_c", 1);
  persistent.start();
  const mp1::status repeated = persistent.wait();
  // The empty-status envelope for MPI_PROC_NULL is specified, but a provider can leave the
  // source field of a completed persistent request unspecified.  The portable contract here is
  // that the wait succeeds and applies no datatype: the buffer stays untouched.
  BOOST_TEST(repeated.error() == MPI_SUCCESS);
  BOOST_TEST(buffer == original);
  world_barrier();

  // An active receive with the same proven-overlapping layout is rejected before MPI.
  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::recv(world, mp1::region{buffer.data(), overlap}, real_source, mp1::tag{9003}); }));
  expect_wrap_count("MPI_Recv_c", 0);
  BOOST_TEST(buffer == original);

  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::irecv(world, mp1::region{buffer.data(), overlap}, real_source, mp1::tag{9004});
  }));
  expect_wrap_count("MPI_Irecv_c", 0);

  reset_wrap_counts();
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::recv_init(world, mp1::region{buffer.data(), overlap}, real_source, mp1::tag{9005});
  }));
  expect_wrap_count("MPI_Recv_init_c", 0);
  BOOST_TEST(buffer == original);
  world_barrier();
}

BOOST_AUTO_TEST_CASE(matched_receive_accepts_overlap_only_for_no_proc) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const mp1::layout<int> overlap = overlap_layout();

  std::array<int, 4> buffer{{51, 52, 53, 54}};
  const std::array<int, 4> original = buffer;

  // mrecv of MPI_MESSAGE_NO_PROC: a matched receive with no buffer role.
  mp1::message matched = mp1::mprobe(world, MPI_PROC_NULL);
  BOOST_TEST(matched.is_no_proc());
  const mp1::status blocking = mp1::mrecv(mp1::region{buffer.data(), overlap}, matched);
  BOOST_TEST(is_proc_null_status(blocking));
  BOOST_TEST(buffer == original);
  BOOST_TEST(not matched.is_valid());
  world_barrier();

  // imrecv of MPI_MESSAGE_NO_PROC: same no-op role, completion in the request.
  mp1::message matched_nonblocking = mp1::mprobe(world, MPI_PROC_NULL);
  BOOST_TEST(matched_nonblocking.is_no_proc());
  mp1::request posted = mp1::imrecv(mp1::region{buffer.data(), overlap}, matched_nonblocking);
  BOOST_TEST(not matched_nonblocking.is_valid());
  const mp1::status nonblocking = posted.wait();
  BOOST_TEST(nonblocking.source() == MPI_PROC_NULL);
  BOOST_TEST(nonblocking.error() == MPI_SUCCESS);
  BOOST_TEST(buffer == original);
  world_barrier();

  // An active matched message keeps the full check.  The local rejection must leave the
  // message owned and consumable, so the sender cannot strand the message.
  if (size >= 2) {
    if (rank == 0) {
      mp1::send(world, 71, 1, mp1::tag{9010});
    } else if (rank == 1) {
      mp1::message active = mp1::mprobe(world, 0, mp1::tag{9010});
      BOOST_TEST(active.is_valid());
      BOOST_TEST(throws_invalid_argument(
          [&] { (void)mp1::mrecv(mp1::region{buffer.data(), overlap}, active); }));
      BOOST_TEST(active.is_valid());
      BOOST_TEST(buffer == original);
      int value{0};
      const mp1::status received = mp1::mrecv(value, active);
      BOOST_TEST(value == 71);
      BOOST_TEST(received.tag().value == 9010);
      BOOST_TEST(not active.is_valid());
    }
    world_barrier();

    if (rank == 0) {
      mp1::send(world, 72, 1, mp1::tag{9011});
    } else if (rank == 1) {
      mp1::message active = mp1::mprobe(world, 0, mp1::tag{9011});
      BOOST_TEST(active.is_valid());
      BOOST_TEST(throws_invalid_argument(
          [&] { (void)mp1::imrecv(mp1::region{buffer.data(), overlap}, active); }));
      BOOST_TEST(active.is_valid());
      BOOST_TEST(buffer == original);
      int value{0};
      mp1::request request = mp1::imrecv(value, active);
      (void)request.wait();
      BOOST_TEST(value == 72);
      BOOST_TEST(not active.is_valid());
    }
    world_barrier();
  }
}

BOOST_AUTO_TEST_CASE(sendrecv_family_gates_receive_layout_on_source) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const mp1::layout<int> overlap = overlap_layout();
  const mp1::layout<int> contiguous = mp1::layout<int>::contiguous(4);

  const std::array<int, 4> send_data{{10, 11, 12, 13}};
  std::array<int, 4> receive_data{{-1, -1, -1, -1}};
  const std::array<int, 4> untouched{{-1, -1, -1, -1}};

  // A real dest with a MPI_PROC_NULL source is no-op on the receive side: the overlapping
  // recvl must be accepted even though the shared buffer is sent, and the recv side must not
  // be updated.  The message still has to be consumed by the peer so it cannot match a later
  // receive.
  if (size >= 2) {
    if (rank == 0) {
      reset_wrap_counts();
      const mp1::status status =
          mp1::sendrecv(world, mp1::region{send_data.data(), contiguous}, 1, mp1::tag{9020},
                        mp1::region{receive_data.data(), overlap}, MPI_PROC_NULL, mp1::tag{9020});
      expect_wrap_count("MPI_Sendrecv_c", 1);
      BOOST_TEST(is_proc_null_status(status));
      BOOST_TEST(receive_data == untouched);
    } else if (rank == 1) {
      std::array<int, 4> received{{0, 0, 0, 0}};
      (void)mp1::recv(world, std::span(received), 0, mp1::tag{9020});
      BOOST_TEST(received == send_data);
    }
    world_barrier();

    if (rank == 0) {
      reset_wrap_counts();
      mp1::request request =
          mp1::isendrecv(world, mp1::region{send_data.data(), contiguous}, 1, mp1::tag{9021},
                         mp1::region{receive_data.data(), overlap}, MPI_PROC_NULL, mp1::tag{9021});
      expect_wrap_count("MPI_Isendrecv_c", 1);
      (void)request.wait();
      BOOST_TEST(receive_data == untouched);
    } else if (rank == 1) {
      std::array<int, 4> received{{0, 0, 0, 0}};
      (void)mp1::recv(world, std::span(received), 0, mp1::tag{9021});
      BOOST_TEST(received == send_data);
    }
    world_barrier();
  }

  // A real source makes recvl a written type map: the proven overlap is rejected locally,
  // before MPI, on both the blocking and the nonblocking combined call.
  if (rank == 0) {
    reset_wrap_counts();
    BOOST_TEST(throws_invalid_argument([&] {
      (void)mp1::sendrecv(world, mp1::region{send_data.data(), contiguous}, 1, mp1::tag{9022},
                          mp1::region{receive_data.data(), overlap}, 1, mp1::tag{9022});
    }));
    expect_wrap_count("MPI_Sendrecv_c", 0);

    reset_wrap_counts();
    BOOST_TEST(throws_invalid_argument([&] {
      (void)mp1::isendrecv(world, mp1::region{send_data.data(), contiguous}, 1, mp1::tag{9023},
                           mp1::region{receive_data.data(), overlap}, 1, mp1::tag{9023});
    }));
    expect_wrap_count("MPI_Isendrecv_c", 0);
    BOOST_TEST(receive_data == untouched);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(sendrecv_replace_family_gates_on_receive_direction) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();
  const mp1::layout<int> overlap = overlap_layout();

  std::array<int, 4> data{{20, 21, 22, 23}};
  const std::array<int, 4> untouched{{20, 21, 22, 23}};

  // Both directions MPI_PROC_NULL: the shared buffer is neither sent nor received, so the
  // overlapping receive layout is not applied and the buffer stays untouched.
  reset_wrap_counts();
  const mp1::status empty = mp1::sendrecv_replace(world, mp1::region{data.data(), overlap},
                                                  MPI_PROC_NULL, mp1::tag{9030}, MPI_PROC_NULL, mp1::tag{9030});
  expect_wrap_count("MPI_Sendrecv_replace_c", 1);
  BOOST_TEST(is_proc_null_status(empty));
  BOOST_TEST(data == untouched);
  world_barrier();

  // A real dest with a MPI_PROC_NULL source: the buffer is sent, no receive occurs, so the
  // overlapping layout is accepted and the buffer is not updated by the call.
  if (size >= 2) {
    if (rank == 0) {
      reset_wrap_counts();
      const mp1::status status = mp1::sendrecv_replace(world, mp1::region{data.data(), overlap},
                                                       1, mp1::tag{9031}, MPI_PROC_NULL, mp1::tag{9031});
      expect_wrap_count("MPI_Sendrecv_replace_c", 1);
      BOOST_TEST(is_proc_null_status(status));
      BOOST_TEST(data == untouched);
    } else if (rank == 1) {
      std::array<int, 4> received{{0, 0, 0, 0}};
      (void)mp1::recv(world, std::span(received), 0, mp1::tag{9031});
      // The send type map is indexed({2@0, 2@1}) = {data[0], data[1], data[1], data[2]}.
      BOOST_TEST(received == (std::array<int, 4>{{20, 21, 21, 22}}));
    }
    world_barrier();

    if (rank == 0) {
      reset_wrap_counts();
      mp1::request request = mp1::isendrecv_replace(world, mp1::region{data.data(), overlap}, 1, mp1::tag{9032}, MPI_PROC_NULL, mp1::tag{9032});
      expect_wrap_count("MPI_Isendrecv_replace_c", 1);
      (void)request.wait();
      BOOST_TEST(data == untouched);
    } else if (rank == 1) {
      std::array<int, 4> received{{0, 0, 0, 0}};
      (void)mp1::recv(world, std::span(received), 0, mp1::tag{9032});
      BOOST_TEST(received == (std::array<int, 4>{{20, 21, 21, 22}}));
    }
    world_barrier();
  }

  // A real source makes the receive direction active: the shared buffer is written through the
  // overlapping layout, which is rejected locally before MPI.
  if (rank == 0) {
    reset_wrap_counts();
    BOOST_TEST(throws_invalid_argument([&] {
      (void)mp1::sendrecv_replace(world, mp1::region{data.data(), overlap}, 1, mp1::tag{9033}, 1, mp1::tag{9033});
    }));
    expect_wrap_count("MPI_Sendrecv_replace_c", 0);

    reset_wrap_counts();
    BOOST_TEST(throws_invalid_argument([&] {
      (void)mp1::isendrecv_replace(world, mp1::region{data.data(), overlap}, 1, mp1::tag{9034}, 1, mp1::tag{9034});
    }));
    expect_wrap_count("MPI_Isendrecv_replace_c", 0);
    BOOST_TEST(data == untouched);
  }
  world_barrier();
}

BOOST_AUTO_TEST_SUITE_END()
