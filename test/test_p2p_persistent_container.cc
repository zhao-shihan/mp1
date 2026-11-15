// Persistent p2p containers and process-buffer contract:
//   * send_init/bsend_init/ssend_init/rsend_init contiguous-container forms in restart cycles,
//     with the container contents changed between starts but the size fixed at init;
//   * recv_init into a fixed-capacity resizable container: one allocation at init, shrink to
//   the
//     received count after completion, restart without resizing, PROC_NULL leaves it untouched;
//   * matched sequence/associative receive through mrecv(message&&) / imrecv;
//   * process-buffer attach/detach with count_t sizes (MPI_Buffer_attach_c/detach_c).
//
// MPI 4.1 basis: §3.9 (persistent requests), §3.6 (buffered send and the process buffer),
// §3.10 (matched probes and MPI_MESSAGE_NO_PROC), §6.12-equivalent lifetime rules for the
// persistent buffer.  Runs at 2 and 4 ranks; every ring exchange is collective in the test's
// control flow, so no rank leaves a case while a peer waits.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <limits>
#include <list>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>


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

  struct persistent_fixture {
    persistent_fixture() {
      mp1::init();
    }

    ~persistent_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture cleanup must not throw
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(persistent_fixture);

BOOST_AUTO_TEST_SUITE(p2p_persistent_container)

BOOST_AUTO_TEST_CASE(contiguous_send_containers_restart_cycles) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int dest = next_rank(c, rank);
  const int source = previous_rank(c, rank);
  const std::size_t n = 3;

  std::vector<int> send(n, rank);
  std::vector<int> recv(n, -1);

  // Standard send.
  {
    mp1::request sender = mp1::send_init(c, send, dest, mp1::tag{70});
    mp1::request receiver = mp1::recv_init(c, recv, mp1::capacity{n}, source, mp1::tag{70});
    for (int cycle = 0; cycle < 4; ++cycle) {
      for (std::size_t i = 0; i < n; ++i)
        send[i] = rank * 100 + cycle;
      receiver.start();
      sender.start();
      receiver.wait();
      sender.wait();
      for (std::size_t i = 0; i < n; ++i)
        BOOST_TEST(recv[i] == source * 100 + cycle);
      BOOST_TEST(recv.size() == n);
    }
    sender.wait();  // inactive persistent requests are safe to wait on
    receiver.wait();
  }

  // Synchronous send.
  {
    mp1::request sender = mp1::ssend_init(c, send, dest, mp1::tag{71});
    mp1::request receiver = mp1::recv_init(c, recv, mp1::capacity{n}, source, mp1::tag{71});
    for (int cycle = 0; cycle < 3; ++cycle) {
      send[0] = rank + cycle;
      receiver.start();
      sender.start();
      sender.wait();
      receiver.wait();
      BOOST_TEST(recv[0] == source + cycle);
    }
  }

  // Buffered send: a process buffer must be attached while the request runs.
  {
    mp1::bsend_buffer buffer(4096);
    mp1::request sender = mp1::bsend_init(c, send, dest, mp1::tag{72});
    mp1::request receiver = mp1::recv_init(c, recv, mp1::capacity{n}, source, mp1::tag{72});
    for (int cycle = 0; cycle < 3; ++cycle) {
      send[1] = rank + 10 * cycle;
      receiver.start();
      sender.start();
      receiver.wait();
      sender.wait();
      BOOST_TEST(recv[1] == source + 10 * cycle);
    }
  }

  // Ready send: post the persistent receive before the ready send is started.
  {
    mp1::request sender = mp1::rsend_init(c, send, dest, mp1::tag{73});
    mp1::request receiver = mp1::recv_init(c, recv, mp1::capacity{n}, source, mp1::tag{73});
    for (int cycle = 0; cycle < 3; ++cycle) {
      send[2] = rank + 20 * cycle;
      receiver.start();
      sender.start();
      sender.wait();
      receiver.wait();
      BOOST_TEST(recv[2] == source + 20 * cycle);
    }
  }

  // The size is fixed at init: resizing the borrowed container is detectable and rejected at
  // start before any MPI call.
  {
    std::vector<int> shrinking(n, rank);
    mp1::request sender = mp1::send_init(c, shrinking, dest, mp1::tag{74});
    shrinking.resize(1);
    BOOST_TEST(throws_invalid_argument([&sender] { sender.start(); }));
    sender.release();
    (void)size;
  }
}

BOOST_AUTO_TEST_CASE(capacity_receive_and_proc_null) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int dest = next_rank(c, rank);
  const int source = previous_rank(c, rank);

  // The receive container is resized to the capacity at init and shrunk to the received count
  // on completion; the extra capacity is not part of the logical result.
  std::vector<int> send{rank, rank + 1, rank + 2};
  std::vector<int> recv;
  mp1::request receiver = mp1::recv_init(c, recv, mp1::capacity{3}, source, mp1::tag{80});
  BOOST_TEST(recv.size() == 3u);
  for (int cycle = 0; cycle < 3; ++cycle) {
    send[0] = rank + cycle;
    receiver.start();
    mp1::send(c, send, dest, mp1::tag{80});
    const mp1::status status = receiver.wait();
    BOOST_TEST(recv.size() == 3u);
    BOOST_TEST(status.get_count<int>() == 3);
    BOOST_TEST(recv[0] == source + cycle);
  }

  // A zero capacity clears the container and posts a count-0 receive.
  std::vector<int> empty_recv{1, 2, 3};
  mp1::request zero_receiver = mp1::recv_init(c, empty_recv, mp1::capacity{0}, source, mp1::tag{81});
  BOOST_TEST(empty_recv.empty());
  const std::vector<int> empty_send;
  zero_receiver.start();
  mp1::send(c, empty_send, dest, mp1::tag{81});
  const mp1::status zero_status = zero_receiver.wait();
  BOOST_TEST(zero_status.get_count<int>() == 0);
  BOOST_TEST(empty_recv.empty());

  // MPI_PROC_NULL leaves the container untouched, at init and at completion.  MPI 4.1 §3.10
  // defines the completion status (source MPI_PROC_NULL, tag MPI_ANY_TAG, count 0), and the
  // wrapper synthesizes it on every provider: MPICH 4.3.2 leaves SOURCE at MPI_ANY_SOURCE for
  // this persistent receive, so forwarding the provider's status would not be portable.
  std::vector<int> untouched{7, 8};
  mp1::request null_receiver =
      mp1::recv_init(c, untouched, mp1::capacity{2}, MPI_PROC_NULL, mp1::tag{82});
  BOOST_TEST(untouched.size() == 2u);
  for (int cycle = 0; cycle < 2; ++cycle) {
    null_receiver.start();
    const mp1::status null_status = null_receiver.wait();
    BOOST_TEST(null_status.source() == MPI_PROC_NULL);
    BOOST_TEST(null_status.tag().value == mp1::any_tag.value);
    BOOST_TEST(null_status.error() == MPI_SUCCESS);
    BOOST_TEST(null_status.get_count<int>() == 0);
    BOOST_TEST(null_status.get_elements<int>() == 0);
    BOOST_TEST(not null_status.is_cancelled());
    BOOST_TEST(untouched.size() == 2u);
    BOOST_TEST(untouched[0] == 7);
    BOOST_TEST(untouched[1] == 8);
  }

  // The fixed-capacity spelling is the strong capacity type; a raw count in that position is
  // not a valid container call (compile-only probes cover this).
}

BOOST_AUTO_TEST_CASE(matched_container_policies) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int dest = next_rank(c, rank);
  const int source = previous_rank(c, rank);

  // mrecv auto-size (policy 4): the probed count sizes the container before MPI_Mrecv.
  std::vector<int> payload{rank, rank + 1, rank + 2, rank + 3};
  mp1::send(c, payload, dest, mp1::tag{83});
  std::vector<int> received;
  mp1::message auto_message = mp1::mprobe(c, source, mp1::tag{83});
  const mp1::status auto_status = mp1::mrecv(received, auto_message);
  BOOST_TEST(auto_status.get_count<int>() == 4);
  const std::vector<int> expected{source, source + 1, source + 2, source + 3};
  BOOST_TEST(received == expected);

  // imrecv bounded (policy 5): capacity is the upper bound; the container shrinks to the
  // received count on completion.
  mp1::send(c, payload, dest, mp1::tag{84});
  std::vector<int> async_received;
  mp1::message bounded_message = mp1::mprobe(c, source, mp1::tag{84});
  mp1::request request = mp1::imrecv(async_received, mp1::capacity{8}, bounded_message);
  const mp1::status async_status = request.wait();
  BOOST_TEST(async_status.get_count<int>() == 4);
  BOOST_TEST(async_received == expected);

  // mrecv fixed (policy 2): the span extent bounds the message; no resize.
  mp1::send(c, payload, dest, mp1::tag{85});
  std::array<int, 4> fixed_received{};
  mp1::message fixed_message = mp1::mprobe(c, source, mp1::tag{85});
  (void)mp1::mrecv(std::span(fixed_received), fixed_message);
  BOOST_TEST(fixed_received[0] == source);
  BOOST_TEST(fixed_received[3] == source + 3);
}

BOOST_AUTO_TEST_CASE(process_buffer_count_t) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  // Explicit buffer: the detach returns the same address and the exact count_t size.
  constexpr std::size_t buffer_bytes = 1u << 20;
  std::vector<char> buffer(buffer_bytes, 0);
  mp1::buffer_attach(buffer.data(), static_cast<mp1::count_t>(buffer_bytes));
  const std::pair<void*, mp1::count_t> detached = mp1::buffer_detach();
  BOOST_TEST(detached.first == static_cast<void*>(buffer.data()));
  BOOST_TEST(detached.second == static_cast<mp1::count_t>(buffer_bytes));

  // Explicit buffer with a sizeof() argument selects the count_t overload unambiguously.
  std::vector<char> second_buffer(sizeof(int) * 64, 0);
  mp1::buffer_attach(second_buffer.data(), sizeof(int) * 64);
  const std::pair<void*, mp1::count_t> second_detached = mp1::buffer_detach();
  BOOST_TEST(second_detached.first == static_cast<void*>(second_buffer.data()));
  BOOST_TEST(second_detached.second == static_cast<mp1::count_t>(sizeof(int) * 64));

  // Automatic buffering: MPI ignores the size, but the wrapper must still accept a count_t
  // value beyond the old 32-bit int ceiling.
  mp1::buffer_attach(mp1::buffer_automatic, std::numeric_limits<mp1::count_t>::max());
  const std::pair<void*, mp1::count_t> automatic_detached = mp1::buffer_detach();
  BOOST_TEST(automatic_detached.first == mp1::buffer_automatic);

  // A negative size is rejected for an explicit buffer (automatic normalizes it).
  BOOST_TEST(throws_invalid_argument([&] { mp1::buffer_attach(buffer.data(), -1); }));

  // A buffered send through the explicit attachment still completes.
  std::vector<char> send_buffer(4096, 0);
  mp1::buffer_attach(send_buffer.data(), static_cast<mp1::count_t>(send_buffer.size()));
  std::vector<int> message{rank};
  std::vector<int> received(1, -1);
  const int dest = next_rank(c, rank);
  const int source = previous_rank(c, rank);
  mp1::request request = mp1::ibsend(c, message, dest, mp1::tag{86});
  mp1::request receive = mp1::irecv(c, received, mp1::capacity{1}, source, mp1::tag{86});
  request.wait();
  receive.wait();
  BOOST_TEST(received[0] == source);
  const std::pair<void*, mp1::count_t> final_detached = mp1::buffer_detach();
  BOOST_TEST(final_detached.first == static_cast<void*>(send_buffer.data()));
}

BOOST_AUTO_TEST_SUITE_END()
