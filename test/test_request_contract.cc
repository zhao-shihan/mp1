// Contract test for mp1::request and the free completion functions.
//
// The test talks to raw MPI (MPI_COMM_WORLD and raw MPI_Isend/Irecv/Send/Recv/*_init) and
// wraps the resulting MPI_Request with the named factories. It provides no main: the shared
// test harness links the Boost test main.

#include "mp1/impl/collective.hpp"
#include "mp1/impl/environment.hpp"
#include "mp1/impl/p2p.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/status.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(MP1_TEST_WRAP)
#include "support/mpi_wrap.hpp"
#define MP1_TEST_HAVE_MPI_WRAP 1
#else
#define MP1_TEST_HAVE_MPI_WRAP 0
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

  bool is_empty_status(const mp1::status& s) {
    return s.source() == MPI_ANY_SOURCE and s.tag() == mp1::any_tag;
  }

  template<class F>
  bool throws_logic_error(F&& f) {
    try {
      f();
    } catch (const std::logic_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
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

  // Provider error codes may be dynamically allocated (MPICH), so two occurrences of the same
  // failure can have different code values; the error class is stable.
  int error_class(int code) {
    int result{code};
    if (MPI_Error_class(code, &result) != MPI_SUCCESS)
      return code;
    return result;
  }

  mp1::request wrap_send(MPI_Request handle) {
    return mp1::request::from_send(handle);
  }

  mp1::request wrap_recv(MPI_Request handle) {
    return mp1::request::from_recv(handle);
  }

}  // namespace

struct mp1_request_contract_fixture {
  mp1_request_contract_fixture() {
    mp1::init();
  }

  ~mp1_request_contract_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Already finalized by another path; the request test must not throw from a dtor.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_request_contract_fixture);

BOOST_AUTO_TEST_SUITE(request_contract)

BOOST_AUTO_TEST_CASE(type_and_factory_state) {
  static_assert(not std::is_copy_constructible_v<mp1::request>);
  static_assert(not std::is_copy_assignable_v<mp1::request>);
  static_assert(std::is_nothrow_move_constructible_v<mp1::request>);
  static_assert(std::is_nothrow_move_assignable_v<mp1::request>);
  static_assert(std::is_nothrow_destructible_v<mp1::request>);

  mp1::request null_request;
  BOOST_TEST(not null_request.is_valid());
  BOOST_TEST(not null_request.is_active());
  BOOST_TEST(not null_request.is_persistent());
  BOOST_TEST(not static_cast<bool>(null_request));

  BOOST_TEST(not mp1::request::from_send(MPI_REQUEST_NULL).is_valid());
  BOOST_TEST(not mp1::request::from_recv(MPI_REQUEST_NULL).is_valid());
  BOOST_TEST(not mp1::request::from_collective(MPI_REQUEST_NULL).is_valid());
  BOOST_TEST(not mp1::request::from_file(MPI_REQUEST_NULL).is_valid());
  BOOST_TEST(not mp1::request::from_matched_recv(MPI_REQUEST_NULL).is_valid());
  BOOST_TEST(not mp1::request::from_persistent_send(MPI_REQUEST_NULL).is_valid());
  BOOST_TEST(not mp1::request::from_persistent_recv(MPI_REQUEST_NULL).is_valid());
  BOOST_TEST(mp1::request::from_persistent_send(MPI_REQUEST_NULL).is_persistent());
}

BOOST_AUTO_TEST_CASE(persistent_state_machine) {
  const int rank = world_rank();

  // Persistent send: inactive at construction, start/restart, handle stays valid.
  {
    int send_buffer = 7;
    int recv_buffer = 0;
    MPI_Request raw_send = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Send_init(&send_buffer, 1, MPI_INT, rank, 500, MPI_COMM_WORLD, &raw_send) ==
               MPI_SUCCESS);
    mp1::request r = mp1::request::from_persistent_send(raw_send);
    BOOST_TEST(r.is_valid());
    BOOST_TEST(not r.is_active());
    BOOST_TEST(r.is_persistent());
    BOOST_TEST(is_empty_status(r.wait()));  // inactive: no MPI, empty status
    BOOST_TEST(r.test().has_value());
    BOOST_TEST(r.get_status().has_value());
    BOOST_TEST(throws_logic_error([&r] { r.cancel(); }));  // inactive: valid non-cancelable
    BOOST_TEST(not r.is_active());

    r.start();
    BOOST_TEST(r.is_active());
    BOOST_TEST(throws_logic_error([&r] { r.start(); }));  // active: state misuse
    BOOST_TEST(MPI_Recv(&recv_buffer, 1, MPI_INT, rank, 500, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE) == MPI_SUCCESS);
    const mp1::status completed = r.wait();
    // A send completion status has no defined source/tag; only the state matters here.
    BOOST_TEST(not completed.is_cancelled());
    BOOST_TEST(r.is_valid());
    BOOST_TEST(not r.is_active());
    BOOST_TEST(is_empty_status(r.wait()));  // completed persistent: inactive again
  }

  // Persistent receive: cancel, restart and complete twice.
  {
    int send_buffer = 9;
    int recv_buffer = 0;
    MPI_Request raw_recv = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Recv_init(&recv_buffer, 1, MPI_INT, rank, 501, MPI_COMM_WORLD, &raw_recv) ==
               MPI_SUCCESS);
    mp1::request r = mp1::request::from_persistent_recv(raw_recv);
    BOOST_TEST(r.is_valid());
    BOOST_TEST(not r.is_active());

    r.start();
    r.cancel();
    const mp1::status cancelled = r.wait();
    BOOST_TEST(cancelled.is_cancelled());
    BOOST_TEST(r.is_valid());
    BOOST_TEST(not r.is_active());

    r.start();
    BOOST_TEST(MPI_Send(&send_buffer, 1, MPI_INT, rank, 501, MPI_COMM_WORLD) == MPI_SUCCESS);
    const mp1::status first = r.wait();
    BOOST_TEST(first.source() == rank);
    BOOST_TEST(first.tag().value == 501);
    BOOST_TEST(not first.is_cancelled());

    r.start();
    BOOST_TEST(MPI_Send(&send_buffer, 1, MPI_INT, rank, 501, MPI_COMM_WORLD) == MPI_SUCCESS);
    const mp1::status second = r.wait();
    BOOST_TEST(second.source() == rank);
    BOOST_TEST(second.tag().value == 501);
    BOOST_TEST(r.is_valid());
    BOOST_TEST(not r.is_active());
  }

  // Inactive persistent requests are freed by the destructor (no communication, no block).
  {
    int send_buffer = 0;
    MPI_Request raw_send = MPI_REQUEST_NULL;
    MPI_Send_init(&send_buffer, 1, MPI_INT, rank, 502, MPI_COMM_WORLD, &raw_send);
    mp1::request r = mp1::request::from_persistent_send(raw_send);
    BOOST_TEST(r.is_valid());
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(wait_test_and_get_status_shapes) {
  const int rank = world_rank();
  BOOST_TEST(world_size() >= 2);
  if (world_size() < 2) {
    BOOST_TEST_MESSAGE("wait/test/get_status shapes need at least 2 ranks");
    return;
  }

  // test() must return nullopt only while an active request is incomplete.
  {
    int recv_buffer = 0;
    mp1::request receiver;
    if (rank == 0) {
      MPI_Request raw = MPI_REQUEST_NULL;
      MPI_Irecv(&recv_buffer, 1, MPI_INT, 1, 510, MPI_COMM_WORLD, &raw);
      receiver = wrap_recv(raw);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      const std::optional<mp1::status> pending = receiver.test();
      BOOST_TEST(not pending.has_value());
      BOOST_TEST(receiver.is_active());
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
      int send_buffer = 11;
      MPI_Request raw = MPI_REQUEST_NULL;
      MPI_Isend(&send_buffer, 1, MPI_INT, 0, 510, MPI_COMM_WORLD, &raw);
      wrap_send(raw).wait();
    }
    if (rank == 0) {
      std::optional<mp1::status> completed;
      for (int i = 0; i < 1000000 and not completed; ++i)
        completed = receiver.test();
      BOOST_TEST(completed.has_value());
      if (completed) {
        BOOST_TEST(completed->source() == 1);
        BOOST_TEST(completed->tag().value == 510);
        BOOST_TEST(not receiver.is_active());
        BOOST_TEST(not receiver.is_valid());
        const std::optional<mp1::status> inactive = receiver.test();
        BOOST_TEST(inactive.has_value());
        BOOST_TEST(is_empty_status(*inactive));
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // get_status() never reaps and never changes state.
  {
    int recv_buffer = 0;
    mp1::request receiver;
    if (rank == 0) {
      MPI_Request raw = MPI_REQUEST_NULL;
      MPI_Irecv(&recv_buffer, 1, MPI_INT, 1, 511, MPI_COMM_WORLD, &raw);
      receiver = wrap_recv(raw);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
      int send_buffer = 12;
      MPI_Request raw = MPI_REQUEST_NULL;
      MPI_Isend(&send_buffer, 1, MPI_INT, 0, 511, MPI_COMM_WORLD, &raw);
      wrap_send(raw).wait();
    }
    if (rank == 0) {
      std::optional<mp1::status> polled;
      for (int i = 0; i < 1000000 and not polled; ++i)
        polled = receiver.get_status();
      BOOST_TEST(polled.has_value());
      BOOST_TEST(receiver.is_active());  // not reaped
      if (polled) {
        BOOST_TEST(polled->source() == 1);
        BOOST_TEST(polled->tag().value == 511);
      }
      const mp1::status reaped = receiver.wait();
      BOOST_TEST(reaped.source() == 1);
      BOOST_TEST(reaped.tag().value == 511);
      BOOST_TEST(not receiver.is_active());
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

BOOST_AUTO_TEST_CASE(cancel_semantics) {
  const int rank = world_rank();

  // Invalid and inactive cancel are documented no-ops.
  {
    mp1::request null_request;
    null_request.cancel();
    BOOST_TEST(not null_request.is_valid());
  }

  // Cancel a receive that can never match, then reap it.
  {
    int recv_buffer = 0;
    MPI_Request raw = MPI_REQUEST_NULL;
    MPI_Irecv(&recv_buffer, 1, MPI_INT, rank, 520, MPI_COMM_WORLD, &raw);
    mp1::request r = wrap_recv(raw);
    BOOST_TEST(r.is_active());
    r.cancel();
    const mp1::status cancelled = r.wait();
    BOOST_TEST(cancelled.is_cancelled());
    BOOST_TEST(not r.is_active());
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Cancel a send whose matching receive is already posted locally; both outcomes are safe.
  {
    int send_buffer[1024] = {};
    int recv_buffer[1024] = {};
    MPI_Request raw_recv = MPI_REQUEST_NULL;
    MPI_Request raw_send = MPI_REQUEST_NULL;
    MPI_Irecv(recv_buffer, 1024, MPI_INT, rank, 521, MPI_COMM_WORLD, &raw_recv);
    MPI_Isend(send_buffer, 1024, MPI_INT, rank, 521, MPI_COMM_WORLD, &raw_send);
    mp1::request r = wrap_send(raw_send);
    r.cancel();
    const mp1::status send_status = r.wait();
    int recv_complete{0};
    MPI_Test(&raw_recv, &recv_complete, MPI_STATUS_IGNORE);
    if (recv_complete != 0) {
      BOOST_TEST(not send_status.is_cancelled());
    } else {
      BOOST_TEST(send_status.is_cancelled());
      MPI_Cancel(&raw_recv);
      MPI_Wait(&raw_recv, MPI_STATUS_IGNORE);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Collective requests have wait teardown: cancel throws before MPI.
  {
    MPI_Request raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Ibarrier(MPI_COMM_WORLD, &raw) == MPI_SUCCESS);
    mp1::request r = mp1::request::from_collective(raw);
    BOOST_TEST(throws_logic_error([&r] { r.cancel(); }));
    r.wait();  // all ranks enter the barrier
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Persistent receive cancel: still valid/inactive and restartable.
  if (world_size() >= 2) {
    int recv_buffer = 0;
    int send_buffer = 13;
    MPI_Request raw_recv = MPI_REQUEST_NULL;
    MPI_Recv_init(&recv_buffer, 1, MPI_INT, 1, 522, MPI_COMM_WORLD, &raw_recv);
    mp1::request r = mp1::request::from_persistent_recv(raw_recv);
    r.start();
    r.cancel();
    const mp1::status cancelled = r.wait();
    BOOST_TEST(cancelled.is_cancelled());
    BOOST_TEST(r.is_valid());
    BOOST_TEST(not r.is_active());

    // The cancel above must be complete on every rank before rank 1 sends.
    // Otherwise the send can match rank 0's still-active persistent receive
    // (is_cancelled() then fails), and rank 0's restart below would wait
    // forever for a message that the first wait already consumed.
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
      MPI_Send(&send_buffer, 1, MPI_INT, 0, 522, MPI_COMM_WORLD);
    }
    if (rank == 0) {
      r.start();
      const mp1::status completed = r.wait();
      BOOST_TEST(completed.source() == 1);
      BOOST_TEST(completed.tag().value == 522);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(release_and_drop_policies) {
  const int rank = world_rank();

  // release(): no MPI call; the in-flight send is still delivered by MPI.
  if (world_size() >= 2) {
    int recv_buffer = 0;
    MPI_Request raw_recv = MPI_REQUEST_NULL;
    if (rank == 1)
      MPI_Irecv(&recv_buffer, 1, MPI_INT, 0, 530, MPI_COMM_WORLD, &raw_recv);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      int send_buffer = 21;
      MPI_Request raw_send = MPI_REQUEST_NULL;
      MPI_Isend(&send_buffer, 1, MPI_INT, 1, 530, MPI_COMM_WORLD, &raw_send);
      mp1::request r = wrap_send(raw_send);
      r.release();
      BOOST_TEST(not r.is_valid());
      BOOST_TEST(not r.is_active());
    }
    if (rank == 1) {
      MPI_Wait(&raw_recv, MPI_STATUS_IGNORE);
      BOOST_TEST(recv_buffer == 21);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Dropped send: MPI_Request_free; MPI still delivers the message.
  if (world_size() >= 2) {
    int recv_buffer = 0;
    MPI_Request raw_recv = MPI_REQUEST_NULL;
    if (rank == 1)
      MPI_Irecv(&recv_buffer, 1, MPI_INT, 0, 531, MPI_COMM_WORLD, &raw_recv);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      int send_buffer = 22;
      MPI_Request raw_send = MPI_REQUEST_NULL;
      MPI_Isend(&send_buffer, 1, MPI_INT, 1, 531, MPI_COMM_WORLD, &raw_send);
      {
        mp1::request r = wrap_send(raw_send);
        BOOST_TEST(r.is_active());
      }  // drop: MPI_Request_free, payload abandoned (none here)
    }
    if (rank == 1) {
      MPI_Wait(&raw_recv, MPI_STATUS_IGNORE);
      BOOST_TEST(recv_buffer == 22);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Dropped receive: MPI_Cancel + MPI_Wait; must not block.
  {
    int recv_buffer = 0;
    MPI_Request raw_recv = MPI_REQUEST_NULL;
    MPI_Irecv(&recv_buffer, 1, MPI_INT, rank, 532, MPI_COMM_WORLD, &raw_recv);
    {
      mp1::request r = wrap_recv(raw_recv);
      BOOST_TEST(r.is_active());
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Dropped collective: MPI_Wait; every rank drops the same barrier request.
  {
    MPI_Request raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Ibarrier(MPI_COMM_WORLD, &raw) == MPI_SUCCESS);
    {
      mp1::request r = mp1::request::from_collective(raw);
      BOOST_TEST(r.is_active());
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Move assignment releases the target with the target's own policy (cancel+wait here).
  {
    int recv_buffer = 0;
    MPI_Request first_raw = MPI_REQUEST_NULL;
    MPI_Request second_raw = MPI_REQUEST_NULL;
    MPI_Irecv(&recv_buffer, 1, MPI_INT, rank, 533, MPI_COMM_WORLD, &first_raw);
    MPI_Irecv(&recv_buffer, 1, MPI_INT, rank, 534, MPI_COMM_WORLD, &second_raw);
    mp1::request target = wrap_recv(first_raw);
    mp1::request source = wrap_recv(second_raw);
    target = std::move(source);
    BOOST_TEST(target.is_active());
    // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
    BOOST_TEST(not source.is_valid());
    target.cancel();
    const mp1::status cancelled = target.wait();
    BOOST_TEST(cancelled.is_cancelled());
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(wrap_teardown_call_counts) {
#if MP1_TEST_HAVE_MPI_WRAP
  if (not mpi_wrap::supported()) {
    BOOST_TEST_MESSAGE("MPI wrap counters are not active on this platform");
    return;
  }
  const int rank = world_rank();

  // Factory construction and inactive requests are MPI-free.
  {
    mp1::request request;
    mpi_wrap::reset_counts();
    (void)request.wait();
    (void)request.test();
    (void)request.get_status();
    request.cancel();
    request.release();
    (void)mp1::request::from_send(MPI_REQUEST_NULL);
    (void)mp1::request::from_recv(MPI_REQUEST_NULL);
    (void)mp1::request::from_persistent_send(MPI_REQUEST_NULL);
    BOOST_TEST(mpi_wrap::total() == 0);
  }

  // wait() on an inactive request makes no MPI call.
  {
    mp1::request request;
    mpi_wrap::reset_counts();
    (void)request.wait();
    BOOST_TEST(mpi_wrap::total() == 0);
  }

  // A dropped send uses MPI_Request_free only.
  {
    int buffer = 0;
    int guard_buffer = 0;
    MPI_Request guard = MPI_REQUEST_NULL;
    MPI_Irecv(&guard_buffer, 1, MPI_INT, rank, 700, MPI_COMM_WORLD, &guard);
    MPI_Request raw = MPI_REQUEST_NULL;
    MPI_Isend(&buffer, 1, MPI_INT, rank, 700, MPI_COMM_WORLD, &raw);
    mpi_wrap::reset_counts();
    {
      mp1::request request = wrap_send(raw);
    }
    BOOST_TEST(mpi_wrap::count("MPI_Request_free") == 1);
    BOOST_TEST(mpi_wrap::count("MPI_Cancel") == 0);
    BOOST_TEST(mpi_wrap::count("MPI_Wait") == 0);
    MPI_Wait(&guard, MPI_STATUS_IGNORE);
  }

  // A dropped receive uses MPI_Cancel + MPI_Wait.
  {
    int buffer = 0;
    MPI_Request raw = MPI_REQUEST_NULL;
    MPI_Irecv(&buffer, 1, MPI_INT, rank, 701, MPI_COMM_WORLD, &raw);
    mpi_wrap::reset_counts();
    {
      mp1::request request = wrap_recv(raw);
    }
    BOOST_TEST(mpi_wrap::count("MPI_Cancel") == 1);
    BOOST_TEST(mpi_wrap::count("MPI_Wait") == 1);
    BOOST_TEST(mpi_wrap::count("MPI_Request_free") == 0);
  }

  // A dropped inactive persistent request frees its handle only.
  {
    int buffer = 0;
    MPI_Request raw = MPI_REQUEST_NULL;
    MPI_Send_init(&buffer, 1, MPI_INT, rank, 702, MPI_COMM_WORLD, &raw);
    mpi_wrap::reset_counts();
    {
      mp1::request request = mp1::request::from_persistent_send(raw);
    }
    BOOST_TEST(mpi_wrap::count("MPI_Request_free") == 1);
    BOOST_TEST(mpi_wrap::count("MPI_Cancel") == 0);
    BOOST_TEST(mpi_wrap::count("MPI_Wait") == 0);
  }

  // release() makes no MPI call at all (the handle and payload are leaked intentionally).
  {
    int buffer = 0;
    int guard_buffer = 0;
    MPI_Request guard = MPI_REQUEST_NULL;
    MPI_Irecv(&guard_buffer, 1, MPI_INT, rank, 703, MPI_COMM_WORLD, &guard);
    MPI_Request raw = MPI_REQUEST_NULL;
    MPI_Isend(&buffer, 1, MPI_INT, rank, 703, MPI_COMM_WORLD, &raw);
    mp1::request request = wrap_send(raw);
    mpi_wrap::reset_counts();
    request.release();
    BOOST_TEST(mpi_wrap::total() == 0);
    MPI_Wait(&guard, MPI_STATUS_IGNORE);
  }

  // A dropped collective request waits (MPI_Wait only).
  {
    MPI_Request raw = MPI_REQUEST_NULL;
    MPI_Ibarrier(MPI_COMM_WORLD, &raw);
    mp1::request request = mp1::request::from_collective(raw);
    mpi_wrap::reset_counts();
    {
      mp1::request moved = std::move(request);
    }
    BOOST_TEST(mpi_wrap::count("MPI_Wait") == 1);
    BOOST_TEST(mpi_wrap::count("MPI_Request_free") == 0);
    BOOST_TEST(mpi_wrap::count("MPI_Cancel") == 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // get_status never reaps; the later wait is the only MPI_Wait.
  {
    int buffer = 0;
    int value = 3;
    MPI_Request raw = MPI_REQUEST_NULL;
    MPI_Irecv(&buffer, 1, MPI_INT, rank, 704, MPI_COMM_WORLD, &raw);
    mp1::request request = wrap_recv(raw);
    MPI_Send(&value, 1, MPI_INT, rank, 704, MPI_COMM_WORLD);
    mpi_wrap::reset_counts();
    (void)request.get_status();
    BOOST_TEST(mpi_wrap::count("MPI_Request_get_status") == 1);
    BOOST_TEST(mpi_wrap::count("MPI_Wait") == 0);
    (void)request.wait();
  }

  // Empty multi-request input is MPI-free; a non-empty vector uses one MPI array call.
  {
    std::vector<mp1::request> empty;
    mpi_wrap::reset_counts();
    (void)mp1::wait_all(empty);
    (void)mp1::test_all(empty);
    (void)mp1::wait_any(empty);
    (void)mp1::test_any(empty);
    (void)mp1::wait_some(empty);
    (void)mp1::test_some(empty);
    mp1::start_all(empty);
    BOOST_TEST(mpi_wrap::total() == 0);

    std::vector<mp1::request> inactive(1);
    mpi_wrap::reset_counts();
    (void)mp1::wait_all(inactive);
    BOOST_TEST(mpi_wrap::count("MPI_Waitall") == 1);
    mpi_wrap::reset_counts();
    (void)mp1::test_all(inactive);
    BOOST_TEST(mpi_wrap::count("MPI_Testall") == 1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
#else
  BOOST_TEST_MESSAGE("built without the MPI wrap shim; call-count checks skipped");
#endif
}

BOOST_AUTO_TEST_CASE(empty_and_inactive_completion_shapes) {
  std::vector<mp1::request> empty;
  BOOST_TEST(mp1::wait_all(empty).empty());
  const auto empty_all = mp1::test_all(empty);
  BOOST_TEST(empty_all.has_value());
  BOOST_TEST(empty_all->empty());
  const mp1::completion empty_any = mp1::wait_any(empty);
  BOOST_TEST(empty_any.first == MPI_UNDEFINED);
  const auto empty_test_any = mp1::test_any(empty);
  BOOST_TEST(empty_test_any.has_value());
  BOOST_TEST(empty_test_any->first == MPI_UNDEFINED);
  BOOST_TEST(mp1::wait_some(empty).empty());
  BOOST_TEST(not mp1::test_some(empty).has_value());
  mp1::start_all(empty);  // no-op

  std::vector<mp1::request> inactive(2);
  const std::vector<mp1::status> statuses = mp1::wait_all(inactive);
  BOOST_TEST(statuses.size() == 2);
  BOOST_TEST(is_empty_status(statuses[0]));
  BOOST_TEST(is_empty_status(statuses[1]));

  std::vector<mp1::request> inactive_test(1);
  const auto test_all_statuses = mp1::test_all(inactive_test);
  BOOST_TEST(test_all_statuses.has_value());
  BOOST_TEST(test_all_statuses->size() == 1);
  BOOST_TEST(is_empty_status((*test_all_statuses)[0]));

  std::vector<mp1::request> no_active(2);
  BOOST_TEST(mp1::wait_any(no_active).first == MPI_UNDEFINED);
  const auto test_any_none = mp1::test_any(no_active);
  BOOST_TEST(test_any_none.has_value());
  BOOST_TEST(test_any_none->first == MPI_UNDEFINED);
  BOOST_TEST(mp1::wait_some(no_active).empty());
  BOOST_TEST(not mp1::test_some(no_active).has_value());
}

BOOST_AUTO_TEST_CASE(wait_all_completion) {
  const int rank = world_rank();
  int first_buffer = 0;
  int second_buffer = 0;
  MPI_Request first_raw = MPI_REQUEST_NULL;
  MPI_Request second_raw = MPI_REQUEST_NULL;
  MPI_Irecv(&first_buffer, 1, MPI_INT, rank, 600, MPI_COMM_WORLD, &first_raw);
  MPI_Irecv(&second_buffer, 1, MPI_INT, rank, 601, MPI_COMM_WORLD, &second_raw);

  std::vector<mp1::request> requests;
  requests.push_back(wrap_recv(first_raw));
  requests.push_back(wrap_recv(second_raw));

  int send_first = 31;
  int send_second = 32;
  MPI_Request send_first_raw = MPI_REQUEST_NULL;
  MPI_Request send_second_raw = MPI_REQUEST_NULL;
  MPI_Isend(&send_first, 1, MPI_INT, rank, 600, MPI_COMM_WORLD, &send_first_raw);
  MPI_Isend(&send_second, 1, MPI_INT, rank, 601, MPI_COMM_WORLD, &send_second_raw);

  const std::vector<mp1::status> statuses = mp1::wait_all(requests);
  BOOST_TEST(statuses.size() == 2);
  BOOST_TEST(statuses[0].tag().value == 600);
  BOOST_TEST(statuses[1].tag().value == 601);
  BOOST_TEST(statuses[0].source() == rank);
  BOOST_TEST(not requests[0].is_active());
  BOOST_TEST(not requests[1].is_active());
  BOOST_TEST(not requests[0].is_valid());
  BOOST_TEST(not requests[1].is_valid());

  MPI_Wait(&send_first_raw, MPI_STATUS_IGNORE);
  MPI_Wait(&send_second_raw, MPI_STATUS_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(test_all_completion) {
  const int rank = world_rank();
  int buffer = 0;
  MPI_Request raw_recv = MPI_REQUEST_NULL;
  MPI_Irecv(&buffer, 1, MPI_INT, rank, 602, MPI_COMM_WORLD, &raw_recv);
  std::vector<mp1::request> requests;
  requests.push_back(wrap_recv(raw_recv));

  const auto pending = mp1::test_all(requests);
  BOOST_TEST(not pending.has_value());
  BOOST_TEST(requests[0].is_active());

  int send_value = 33;
  MPI_Request raw_send = MPI_REQUEST_NULL;
  MPI_Isend(&send_value, 1, MPI_INT, rank, 602, MPI_COMM_WORLD, &raw_send);

  std::optional<std::vector<mp1::status>> completed;
  for (int i = 0; i < 1000000 and not completed; ++i)
    completed = mp1::test_all(requests);
  BOOST_TEST(completed.has_value());
  if (completed) {
    BOOST_TEST(completed->size() == 1);
    BOOST_TEST((*completed)[0].tag().value == 602);
    BOOST_TEST((*completed)[0].source() == rank);
  }
  BOOST_TEST(not requests[0].is_active());
  MPI_Wait(&raw_send, MPI_STATUS_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(wait_any_and_test_any_completion) {
  const int rank = world_rank();

  // wait_any returns a slot index and completes exactly that slot.
  {
    int first_buffer = 0;
    int second_buffer = 0;
    MPI_Request first_raw = MPI_REQUEST_NULL;
    MPI_Request second_raw = MPI_REQUEST_NULL;
    MPI_Irecv(&first_buffer, 1, MPI_INT, rank, 603, MPI_COMM_WORLD, &first_raw);
    MPI_Irecv(&second_buffer, 1, MPI_INT, rank, 604, MPI_COMM_WORLD, &second_raw);
    std::vector<mp1::request> requests;
    requests.push_back(wrap_recv(first_raw));
    requests.push_back(wrap_recv(second_raw));

    int send_first = 34;
    int send_second = 35;
    MPI_Request send_first_raw = MPI_REQUEST_NULL;
    MPI_Request send_second_raw = MPI_REQUEST_NULL;
    MPI_Isend(&send_first, 1, MPI_INT, rank, 603, MPI_COMM_WORLD, &send_first_raw);
    MPI_Isend(&send_second, 1, MPI_INT, rank, 604, MPI_COMM_WORLD, &send_second_raw);

    const mp1::completion completed = mp1::wait_any(requests);
    BOOST_TEST((completed.first == 0 or completed.first == 1));
    BOOST_TEST(completed.second.source() == rank);
    BOOST_TEST(not requests[static_cast<std::size_t>(completed.first)].is_active());
    const std::vector<mp1::status> rest = mp1::wait_all(requests);
    BOOST_TEST(rest.size() == 2);
    MPI_Wait(&send_first_raw, MPI_STATUS_IGNORE);
    MPI_Wait(&send_second_raw, MPI_STATUS_IGNORE);
  }

  // test_any: nullopt while incomplete, then a completion or undefined-when-drained.
  {
    int buffer = 0;
    MPI_Request raw_recv = MPI_REQUEST_NULL;
    MPI_Irecv(&buffer, 1, MPI_INT, rank, 605, MPI_COMM_WORLD, &raw_recv);
    std::vector<mp1::request> requests;
    requests.push_back(wrap_recv(raw_recv));

    const auto pending = mp1::test_any(requests);
    BOOST_TEST(not pending.has_value());

    int send_value = 36;
    MPI_Request raw_send = MPI_REQUEST_NULL;
    MPI_Isend(&send_value, 1, MPI_INT, rank, 605, MPI_COMM_WORLD, &raw_send);
    std::optional<mp1::completion> completed;
    for (int i = 0; i < 1000000 and not completed; ++i)
      completed = mp1::test_any(requests);
    BOOST_TEST(completed.has_value());
    if (completed) {
      BOOST_TEST(completed->first == 0);
      BOOST_TEST(completed->second.source() == rank);
    }
    MPI_Wait(&raw_send, MPI_STATUS_IGNORE);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(wait_some_and_test_some_completion) {
  const int rank = world_rank();

  // wait_some returns every completed slot, possibly in several batches.
  {
    std::vector<int> buffers(3, 0);
    std::vector<mp1::request> requests;
    std::vector<MPI_Request> raw_recvs(3, MPI_REQUEST_NULL);
    for (int i = 0; i < 3; ++i) {
      MPI_Irecv(&buffers[i], 1, MPI_INT, rank, 610 + i, MPI_COMM_WORLD, &raw_recvs[i]);
      requests.push_back(wrap_recv(raw_recvs[i]));
    }
    std::vector<int> sends(3, 40);
    std::vector<MPI_Request> raw_sends(3, MPI_REQUEST_NULL);
    for (int i = 0; i < 3; ++i)
      MPI_Isend(&sends[i], 1, MPI_INT, rank, 610 + i, MPI_COMM_WORLD, &raw_sends[i]);

    std::vector<mp1::completion> completions;
    for (int round = 0; round < 10; ++round) {
      const std::vector<mp1::completion> batch = mp1::wait_some(requests);
      if (batch.empty())
        break;
      for (const auto& completion : batch)
        completions.push_back(completion);
    }
    BOOST_TEST(completions.size() == 3);
    for (const auto& completion : completions) {
      BOOST_TEST((completion.first >= 0 and completion.first < 3));
      BOOST_TEST(not requests[static_cast<std::size_t>(completion.first)].is_active());
    }
    for (auto& raw : raw_sends)
      MPI_Wait(&raw, MPI_STATUS_IGNORE);
  }

  // test_some: empty vector = active but nothing complete; nullopt = no active slots.
  {
    int buffer = 0;
    MPI_Request raw_recv = MPI_REQUEST_NULL;
    MPI_Irecv(&buffer, 1, MPI_INT, rank, 620, MPI_COMM_WORLD, &raw_recv);
    std::vector<mp1::request> requests;
    requests.push_back(wrap_recv(raw_recv));

    const auto nothing_complete = mp1::test_some(requests);
    BOOST_TEST(nothing_complete.has_value());
    BOOST_TEST(nothing_complete->empty());

    int send_value = 41;
    MPI_Request raw_send = MPI_REQUEST_NULL;
    MPI_Isend(&send_value, 1, MPI_INT, rank, 620, MPI_COMM_WORLD, &raw_send);

    std::vector<mp1::completion> completions;
    for (int i = 0; i < 1000000; ++i) {
      const auto batch = mp1::test_some(requests);
      if (not batch.has_value())
        break;  // no active slots
      for (const auto& completion : *batch)
        completions.push_back(completion);
    }
    BOOST_TEST(completions.size() == 1);
    if (not completions.empty()) {
      BOOST_TEST(completions[0].first == 0);
      BOOST_TEST(completions[0].second.source() == rank);
    }
    MPI_Wait(&raw_send, MPI_STATUS_IGNORE);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(start_all_completion) {
  const int rank = world_rank();

  std::vector<int> buffers(2, 0);
  std::vector<mp1::request> requests;
  std::vector<MPI_Request> raw_recvs(2, MPI_REQUEST_NULL);
  for (int i = 0; i < 2; ++i) {
    MPI_Recv_init(&buffers[i], 1, MPI_INT, rank, 630 + i, MPI_COMM_WORLD, &raw_recvs[i]);
    requests.push_back(mp1::request::from_persistent_recv(raw_recvs[i]));
  }
  BOOST_TEST(requests[0].is_persistent());
  BOOST_TEST(not requests[0].is_active());

  mp1::start_all(requests);
  BOOST_TEST(requests[0].is_active());
  BOOST_TEST(requests[1].is_active());
  BOOST_TEST(throws_logic_error([&requests] { mp1::start_all(requests); }));

  std::vector<int> sends(2, 42);
  std::vector<MPI_Request> raw_sends(2, MPI_REQUEST_NULL);
  for (int i = 0; i < 2; ++i)
    MPI_Isend(&sends[i], 1, MPI_INT, rank, 630 + i, MPI_COMM_WORLD, &raw_sends[i]);

  const std::vector<mp1::status> first = mp1::wait_all(requests);
  BOOST_TEST(first.size() == 2);
  BOOST_TEST(first[0].tag().value == 630);
  BOOST_TEST(first[1].tag().value == 631);
  BOOST_TEST(not requests[0].is_active());
  for (auto& raw : raw_sends)
    MPI_Wait(&raw, MPI_STATUS_IGNORE);

  // Persistent requests stay valid and restartable.
  mp1::start_all(requests);
  BOOST_TEST(requests[0].is_active());
  std::vector<MPI_Request> second_raw_sends(2, MPI_REQUEST_NULL);
  for (int i = 0; i < 2; ++i)
    MPI_Isend(&sends[i], 1, MPI_INT, rank, 630 + i, MPI_COMM_WORLD, &second_raw_sends[i]);
  const std::vector<mp1::status> second = mp1::wait_all(requests);
  BOOST_TEST(second.size() == 2);
  for (auto& raw : second_raw_sends)
    MPI_Wait(&raw, MPI_STATUS_IGNORE);

  std::vector<mp1::request> non_persistent(1);
  BOOST_TEST(throws_invalid_argument([&non_persistent] { mp1::start_all(non_persistent); }));
  MPI_Barrier(MPI_COMM_WORLD);
}


// -----------------------------------------------------------------------------------------
// Regression tests: state machine, drop delivery, per-slot batch errors, span slices,
// and status provenance. MPI 4.1 citations are in mp1/impl/request.hpp and status.hpp.
// -----------------------------------------------------------------------------------------

namespace {

  class throwing_payload final : public mp1::detail::request_payload {
  public:
    void complete(const mp1::status&) override {
      throw std::runtime_error{"request contract: throwing payload"};
    }
  };

  class counting_payload final : public mp1::detail::request_payload {
  public:
    explicit counting_payload(int* count) : count_{count} {
    }

    void complete(const mp1::status&) override {
      ++*count_;
    }

  private:
    int* count_;
  };

}  // namespace

BOOST_AUTO_TEST_CASE(kind_teardown_and_drop_table) {
  using mp1::request_kind;
  using mp1::request_state;
  using mp1::request_teardown;

  mp1::request empty;
  BOOST_TEST(static_cast<int>(empty.kind()) == static_cast<int>(request_kind::none));
  BOOST_TEST(static_cast<int>(empty.teardown_policy()) ==
             static_cast<int>(request_teardown::none));
  BOOST_TEST(static_cast<int>(empty.state()) == static_cast<int>(request_state::freed));
  BOOST_TEST(not empty.is_valid());
  BOOST_TEST(not empty.cancelable());
  BOOST_TEST(not empty.drop_would_block());
  BOOST_TEST(empty.native_handle() == MPI_REQUEST_NULL);

  const auto kind_is = [](const mp1::request& r, request_kind kind) {
    return static_cast<int>(r.kind()) == static_cast<int>(kind);
  };
  const auto teardown_is = [](const mp1::request& r, request_teardown policy) {
    return static_cast<int>(r.teardown_policy()) == static_cast<int>(policy);
  };

  const mp1::request send_null = mp1::request::from_send(MPI_REQUEST_NULL);
  BOOST_TEST(kind_is(send_null, request_kind::send));
  BOOST_TEST(teardown_is(send_null, request_teardown::request_free));
  BOOST_TEST(not send_null.is_valid());
  BOOST_TEST(not send_null.is_persistent());

  const mp1::request recv_null = mp1::request::from_recv(MPI_REQUEST_NULL);
  BOOST_TEST(kind_is(recv_null, request_kind::receive));
  BOOST_TEST(teardown_is(recv_null, request_teardown::cancel_wait));
  const mp1::request matched_null = mp1::request::from_matched_recv(MPI_REQUEST_NULL);
  BOOST_TEST(kind_is(matched_null, request_kind::receive));
  BOOST_TEST(teardown_is(matched_null, request_teardown::cancel_wait));

  const mp1::request collective_null = mp1::request::from_collective(MPI_REQUEST_NULL);
  BOOST_TEST(kind_is(collective_null, request_kind::collective));
  BOOST_TEST(teardown_is(collective_null, request_teardown::wait));
  const mp1::request persistent_collective_null =
      mp1::request::from_persistent_collective(MPI_REQUEST_NULL);
  BOOST_TEST(kind_is(persistent_collective_null, request_kind::collective));
  BOOST_TEST(teardown_is(persistent_collective_null, request_teardown::wait_and_free));
  BOOST_TEST(persistent_collective_null.is_persistent());
  BOOST_TEST(static_cast<int>(persistent_collective_null.state()) ==
             static_cast<int>(request_state::inactive));

  const mp1::request rma_null = mp1::request::from_rma(MPI_REQUEST_NULL);
  BOOST_TEST(kind_is(rma_null, request_kind::rma));
  BOOST_TEST(teardown_is(rma_null, request_teardown::wait));
  const mp1::request file_null = mp1::request::from_file(MPI_REQUEST_NULL);
  BOOST_TEST(kind_is(file_null, request_kind::file));
  BOOST_TEST(teardown_is(file_null, request_teardown::wait));
  const mp1::request flush_null = mp1::request::from_flush(MPI_REQUEST_NULL);
  BOOST_TEST(kind_is(flush_null, request_kind::flush));
  BOOST_TEST(teardown_is(flush_null, request_teardown::wait));

  // Active receive: cancelable, drop would block (cancel+wait). Active send: not blocking.
  {
    int recv_buffer = 0;
    MPI_Request raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&recv_buffer, 1, MPI_INT, world_rank(), 720, MPI_COMM_WORLD, &raw) ==
               MPI_SUCCESS);
    mp1::request r = mp1::request::from_recv(raw);
    BOOST_TEST(r.is_active());
    BOOST_TEST(r.cancelable());
    BOOST_TEST(r.drop_would_block());
    BOOST_TEST(r.native_handle() == raw);
    r.cancel();
    (void)r.wait();
  }
  {
    int buffer = 0;
    int guard_buffer = 0;
    MPI_Request guard = MPI_REQUEST_NULL;
    MPI_Request raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&guard_buffer, 1, MPI_INT, world_rank(), 721, MPI_COMM_WORLD,
                         &guard) == MPI_SUCCESS);
    BOOST_TEST(MPI_Isend(&buffer, 1, MPI_INT, world_rank(), 721, MPI_COMM_WORLD, &raw) ==
               MPI_SUCCESS);
    mp1::request r = mp1::request::from_send(raw);
    BOOST_TEST(r.is_active());
    BOOST_TEST(r.cancelable());
    BOOST_TEST(not r.drop_would_block());
    (void)r.wait();
    MPI_Wait(&guard, MPI_STATUS_IGNORE);
  }

  // Move-from resets the source; release() resets kind/teardown/persistence.
  {
    int buffer = 0;
    MPI_Request raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&buffer, 1, MPI_INT, world_rank(), 722, MPI_COMM_WORLD, &raw) ==
               MPI_SUCCESS);
    mp1::request source = mp1::request::from_recv(raw);
    mp1::request target = std::move(source);
    BOOST_TEST(kind_is(target, request_kind::receive));
    BOOST_TEST(not source.is_valid());
    BOOST_TEST(not source.is_persistent());
    BOOST_TEST(kind_is(source, request_kind::none));
    BOOST_TEST(static_cast<int>(source.state()) == static_cast<int>(request_state::moved_from));
    target.cancel();
    (void)target.wait();
  }
  {
    int buffer = 0;
    int guard_buffer = 0;
    MPI_Request guard = MPI_REQUEST_NULL;
    MPI_Request raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&guard_buffer, 1, MPI_INT, world_rank(), 723, MPI_COMM_WORLD,
                         &guard) == MPI_SUCCESS);
    BOOST_TEST(MPI_Isend(&buffer, 1, MPI_INT, world_rank(), 723, MPI_COMM_WORLD, &raw) ==
               MPI_SUCCESS);
    mp1::request r = mp1::request::from_send(raw);
    r.release();
    BOOST_TEST(not r.is_valid());
    BOOST_TEST(not r.is_persistent());
    BOOST_TEST(kind_is(r, request_kind::none));
    BOOST_TEST(teardown_is(r, request_teardown::none));
    BOOST_TEST(static_cast<int>(r.state()) == static_cast<int>(request_state::freed));
    MPI_Wait(&guard, MPI_STATUS_IGNORE);
  }
}

BOOST_AUTO_TEST_CASE(batch_span_from_containers) {
  std::array<mp1::request, 2> array_requests{};
  const std::vector<mp1::status> array_statuses = mp1::wait_all(array_requests);
  BOOST_TEST(array_statuses.size() == 2);
  BOOST_TEST(is_empty_status(array_statuses[0]));

  mp1::request c_array[2]{};
  const std::vector<mp1::status> c_statuses = mp1::wait_all(c_array);
  BOOST_TEST(c_statuses.size() == 2);

  std::vector<mp1::request> vector_requests(3);
  const std::span<mp1::request> requests{vector_requests};
  const std::span<mp1::request> slice = requests.subspan(1, 2);
  BOOST_TEST(slice.size() == 2);
  const std::vector<mp1::status> slice_statuses = mp1::wait_all(slice);
  BOOST_TEST(slice_statuses.size() == 2);

  BOOST_TEST(mp1::wait_all(std::span<mp1::request>{}).empty());
  const auto empty_any = mp1::wait_any(std::span<mp1::request>{});
  BOOST_TEST(empty_any.first == MPI_UNDEFINED);
}

BOOST_AUTO_TEST_CASE(throwing_payload_does_not_abandon_batch) {
  int first_buffer = 0;
  int second_buffer = 0;
  MPI_Request first_raw = MPI_REQUEST_NULL;
  MPI_Request second_raw = MPI_REQUEST_NULL;
  BOOST_TEST(MPI_Irecv(&first_buffer, 1, MPI_INT, 0, 730, MPI_COMM_SELF, &first_raw) ==
             MPI_SUCCESS);
  BOOST_TEST(MPI_Irecv(&second_buffer, 1, MPI_INT, 0, 731, MPI_COMM_SELF, &second_raw) ==
             MPI_SUCCESS);
  int first_value = 1;
  int second_value = 2;
  MPI_Request send_first = MPI_REQUEST_NULL;
  MPI_Request send_second = MPI_REQUEST_NULL;
  BOOST_TEST(MPI_Isend(&first_value, 1, MPI_INT, 0, 730, MPI_COMM_SELF, &send_first) ==
             MPI_SUCCESS);
  BOOST_TEST(MPI_Isend(&second_value, 1, MPI_INT, 0, 731, MPI_COMM_SELF, &send_second) ==
             MPI_SUCCESS);

  int delivered{0};
  std::vector<mp1::request> requests;
  requests.push_back(wrap_recv(first_raw));
  requests.push_back(wrap_recv(second_raw));
  mp1::detail::attach_payload(requests[0], std::make_unique<throwing_payload>());
  mp1::detail::attach_payload(requests[1], std::make_unique<counting_payload>(&delivered));

  bool threw{false};
  try {
    (void)mp1::wait_all(requests);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  BOOST_TEST(threw);
  BOOST_TEST(delivered == 1);  // slot 1 still finished after slot 0 threw
  BOOST_TEST(not requests[0].is_valid());
  BOOST_TEST(not requests[0].is_active());
  BOOST_TEST(not requests[1].is_valid());
  BOOST_TEST(not requests[1].is_active());
  BOOST_TEST(is_empty_status(requests[1].wait()));  // no second MPI Wait / stale handle

  MPI_Wait(&send_first, MPI_STATUS_IGNORE);
  MPI_Wait(&send_second, MPI_STATUS_IGNORE);
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(wait_any_error_commits_only_completed_slot) {
  if (world_size() < 2) {
    BOOST_TEST_MESSAGE("wait_any per-slot error test needs at least 2 ranks");
    return;
  }

  // Error handlers are per-communicator and MPI_COMM_DUP copies them (MPI 4.1 §7.4.2,
  // §9.3.1).  Install MPI_ERRORS_RETURN on MPI_COMM_WORLD before the duplicate so the
  // per-operation error from MPI_Waitany is returned to the wrapper instead of invoking the
  // default handler.
  BOOST_TEST(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) == MPI_SUCCESS);
  MPI_Comm dup = MPI_COMM_NULL;
  BOOST_TEST(MPI_Comm_dup(MPI_COMM_WORLD, &dup) == MPI_SUCCESS);
  const int rank = world_rank();

  if (rank == 0) {
    int first = 0;
    int second = 0;
    MPI_Request first_raw = MPI_REQUEST_NULL;
    MPI_Request second_raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&first, 1, MPI_INT, 1, 740, dup, &first_raw) == MPI_SUCCESS);
    BOOST_TEST(MPI_Irecv(&second, 1, MPI_INT, 1, 741, dup, &second_raw) == MPI_SUCCESS);
    std::vector<mp1::request> requests;
    requests.push_back(wrap_recv(first_raw));
    requests.push_back(wrap_recv(second_raw));

    MPI_Barrier(dup);  // rank 1 sends the truncated message after both receives are posted

    bool threw{false};
    int reported_slot{-1};
    try {
      (void)mp1::wait_any(requests);
    } catch (const mp1::mpi_error& e) {
      threw = true;
      reported_slot = e.slot();
      BOOST_TEST(std::string{e.what()}.find("slot") != std::string::npos);
    }
    BOOST_TEST(threw);

    // Only the completed (errored) slot was committed; the other receive is still active.
    const int active_count =
        (requests[0].is_active() ? 1 : 0) + (requests[1].is_active() ? 1 : 0);
    BOOST_TEST(active_count == 1);
    const std::size_t remaining = requests[0].is_active() ? 0 : 1;
    // mpi_error::slot() must name the completed MPI index (not -1); it is the other slot.
    BOOST_TEST(reported_slot == static_cast<int>(1 - remaining));
    BOOST_TEST(requests[remaining].is_valid());
    BOOST_TEST(not requests[remaining].is_broken());

    MPI_Barrier(dup);  // let rank 1 send the late normal message
    const mp1::status late = requests[remaining].wait();
    BOOST_TEST(static_cast<int>(late.kind()) == static_cast<int>(mp1::status_kind::native));
    BOOST_TEST((remaining == 0 ? first : second) == 77);
    MPI_Barrier(dup);
  } else if (rank == 1) {
    int four[4] = {1, 2, 3, 4};
    int good = 77;
    MPI_Barrier(dup);
    BOOST_TEST(MPI_Send(four, 4, MPI_INT, 0, 740, dup) == MPI_SUCCESS);  // truncated
    MPI_Barrier(dup);
    BOOST_TEST(MPI_Send(&good, 1, MPI_INT, 0, 741, dup) == MPI_SUCCESS);
    MPI_Barrier(dup);
  } else {
    MPI_Barrier(dup);
    MPI_Barrier(dup);
    MPI_Barrier(dup);
  }

  BOOST_TEST(MPI_Comm_free(&dup) == MPI_SUCCESS);
  BOOST_TEST(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL) == MPI_SUCCESS);
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(wait_any_test_any_use_function_error_code) {
  if (world_size() < 2) {
    BOOST_TEST_MESSAGE("wait_any/test_any error-code test needs at least 2 ranks");
    return;
  }

  // MPI 4.1 3.7.3: for functions that take a single MPI_Status the error code is returned by
  // the function and MPI_Status.MPI_ERROR is undefined. The wrapper must therefore throw the
  // function return code. A raw MPI control call on the same truncation scenario records that
  // code, then the wrapped call must throw exactly it.
  // Set the handler on MPI_COMM_WORLD before dup: the duplicate inherits it, and MPICH's
  // per-operation error path for a single-status completion consults the inherited handler.
  BOOST_TEST(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) == MPI_SUCCESS);
  MPI_Comm dup = MPI_COMM_NULL;
  BOOST_TEST(MPI_Comm_dup(MPI_COMM_WORLD, &dup) == MPI_SUCCESS);
  const int rank = world_rank();

  if (rank == 0) {
    int buffer{0};
    MPI_Request raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&buffer, 1, MPI_INT, 1, 760, dup, &raw) == MPI_SUCCESS);
    MPI_Barrier(dup);
    MPI_Status raw_status{};
    int raw_index{MPI_UNDEFINED};
    const int raw_code = MPI_Waitany(1, &raw, &raw_index, &raw_status);
    BOOST_TEST(raw_code != MPI_SUCCESS);
    BOOST_TEST(raw_index == 0);
    MPI_Barrier(dup);

    int wrapped_buffer{0};
    MPI_Request wrapped_raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&wrapped_buffer, 1, MPI_INT, 1, 761, dup, &wrapped_raw) ==
               MPI_SUCCESS);
    std::vector<mp1::request> requests;
    requests.push_back(wrap_recv(wrapped_raw));
    MPI_Barrier(dup);
    bool threw{false};
    int thrown_code{MPI_SUCCESS};
    int thrown_slot{-1};
    try {
      (void)mp1::wait_any(requests);
    } catch (const mp1::mpi_error& e) {
      threw = true;
      thrown_code = e.code();
      thrown_slot = e.slot();
    }
    BOOST_TEST(threw);
    BOOST_TEST(thrown_code != MPI_SUCCESS);
    BOOST_TEST(error_class(thrown_code) == error_class(raw_code));
    BOOST_TEST(thrown_slot == 0);
    // Only the errored slot completed; it is committed, not abandoned.
    BOOST_TEST(not requests[0].is_active());
    BOOST_TEST(not requests[0].is_valid());
    MPI_Barrier(dup);

    int testany_buffer{0};
    MPI_Request testany_raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&testany_buffer, 1, MPI_INT, 1, 762, dup, &testany_raw) ==
               MPI_SUCCESS);
    MPI_Barrier(dup);
    int raw_flag{0};
    int raw_testany_index{MPI_UNDEFINED};
    MPI_Status raw_testany_status{};
    int raw_testany_code{MPI_SUCCESS};
    for (int i = 0; i < 1000000; ++i) {
      raw_testany_code =
          MPI_Testany(1, &testany_raw, &raw_testany_index, &raw_flag, &raw_testany_status);
      if (raw_flag != 0 or raw_testany_code != MPI_SUCCESS)
        break;
    }
    BOOST_TEST(raw_testany_code != MPI_SUCCESS);
    BOOST_TEST(raw_testany_index == 0);
    MPI_Barrier(dup);

    int wrapped_testany_buffer{0};
    MPI_Request wrapped_testany_raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&wrapped_testany_buffer, 1, MPI_INT, 1, 763, dup,
                         &wrapped_testany_raw) == MPI_SUCCESS);
    std::vector<mp1::request> testany_requests;
    testany_requests.push_back(wrap_recv(wrapped_testany_raw));
    MPI_Barrier(dup);
    threw = false;
    thrown_code = MPI_SUCCESS;
    thrown_slot = -1;
    for (int i = 0; i < 1000000 and not threw; ++i) {
      try {
        (void)mp1::test_any(testany_requests);
      } catch (const mp1::mpi_error& e) {
        threw = true;
        thrown_code = e.code();
        thrown_slot = e.slot();
      }
    }
    BOOST_TEST(threw);
    BOOST_TEST(thrown_code != MPI_SUCCESS);
    BOOST_TEST(error_class(thrown_code) == error_class(raw_testany_code));
    BOOST_TEST(thrown_slot == 0);
    BOOST_TEST(not testany_requests[0].is_active());
    BOOST_TEST(not testany_requests[0].is_valid());
    MPI_Barrier(dup);
  } else if (rank == 1) {
    int four[4] = {1, 2, 3, 4};
    MPI_Barrier(dup);
    BOOST_TEST(MPI_Send(four, 4, MPI_INT, 0, 760, dup) == MPI_SUCCESS);
    MPI_Barrier(dup);
    MPI_Barrier(dup);
    BOOST_TEST(MPI_Send(four, 4, MPI_INT, 0, 761, dup) == MPI_SUCCESS);
    MPI_Barrier(dup);
    MPI_Barrier(dup);
    BOOST_TEST(MPI_Send(four, 4, MPI_INT, 0, 762, dup) == MPI_SUCCESS);
    MPI_Barrier(dup);
    MPI_Barrier(dup);
    BOOST_TEST(MPI_Send(four, 4, MPI_INT, 0, 763, dup) == MPI_SUCCESS);
    MPI_Barrier(dup);
  } else {
    for (int i = 0; i < 8; ++i)
      MPI_Barrier(dup);
  }

  BOOST_TEST(MPI_Comm_free(&dup) == MPI_SUCCESS);
  BOOST_TEST(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL) == MPI_SUCCESS);
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(active_destructor_delivers_capacity_receive) {
  if (world_size() < 2) {
    BOOST_TEST_MESSAGE("active destructor test needs at least 2 ranks");
    return;
  }
  const int rank = world_rank();
  const int tag = 750;
  if (rank == 0) {
    const std::vector<int> sent{1, 2, 3};
    mp1::send(mp1::comm_world(), sent, 1, mp1::tag{tag});
  } else if (rank == 1) {
    std::vector<int> received(8, -1);
    {
      mp1::request r = mp1::irecv(mp1::comm_world(), received, mp1::capacity{8}, 0, mp1::tag{tag});
      BOOST_TEST(r.drop_would_block());
      int guard{0};
      while (not r.get_status().has_value() and guard++ < 1000000)
        ;
    }  // drop: cancel + wait + complete (capacity resize)
    BOOST_TEST(received.size() == 3);
    BOOST_TEST(received[0] == 1);
    BOOST_TEST(received[1] == 2);
    BOOST_TEST(received[2] == 3);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(dropped_staged_receive_is_delivered) {
  if (world_size() < 2) {
    BOOST_TEST_MESSAGE("staged drop test needs at least 2 ranks");
    return;
  }
  const auto world = mp1::comm_world();
  const int rank = world.rank();
  const int size = world.size();
  const int root = 0;
  const mp1::layout<int> block = mp1::layout<int>::vector(2, 1, 2);  // holey -> staged
  std::vector<int> send(3, -1);
  send[0] = 100 + rank;
  send[2] = 200 + rank;

  if (rank == root) {
    std::vector<int> received(static_cast<std::size_t>(3 * size), -1);
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(size), block);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(3 * i);
    {
      mp1::request r = mp1::igatherv(world, root, mp1::region{send.data(), block}, received,
                                     recv_layouts, displs);
      BOOST_TEST(r.drop_would_block());
    }  // drop: MPI_Wait + payload complete -> staged receive unpacked
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(received[static_cast<std::size_t>(3 * i)] == 100 + i);
      BOOST_TEST(received[static_cast<std::size_t>(3 * i + 2)] == 200 + i);
    }
  } else {
    mp1::request r = mp1::igatherv(world, root, mp1::region{send.data(), block},
                                   std::span<int>{}, mp1::blocks{});
    (void)r;  // destructor waits for the collective
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(collective_completion_status_is_normalized) {
  // Nonblocking collective status has undefined SOURCE/TAG (MPI 4.1 6.12); the wrapper
  // normalizes it and never runs MPI_Test_cancelled for a collective completion (6.12/3.8.4).
  MPI_Request barrier_raw = MPI_REQUEST_NULL;
  BOOST_TEST(MPI_Ibarrier(MPI_COMM_WORLD, &barrier_raw) == MPI_SUCCESS);
  mp1::request barrier_request = mp1::request::from_collective(barrier_raw);
  const mp1::status barrier_status = barrier_request.wait();
  BOOST_TEST(static_cast<int>(barrier_status.kind()) ==
             static_cast<int>(mp1::status_kind::normalized));
  BOOST_TEST(barrier_status.source() == MPI_ANY_SOURCE);
  BOOST_TEST(barrier_status.tag().value == mp1::any_tag.value);
  BOOST_TEST(barrier_status.error() == MPI_SUCCESS);
  BOOST_TEST(barrier_status.get_count<int>() == 0);
  BOOST_TEST(barrier_status.get_elements<int>() == 0);
  BOOST_TEST(not barrier_status.is_cancelled());

  // A point-to-point receive completion keeps native provenance so get_count and
  // MPI_Test_cancelled remain meaningful (MPI 4.1 3.2.5, 3.8.4).
  int received = 0;
  int sent = 5;
  MPI_Request recv_raw = MPI_REQUEST_NULL;
  BOOST_TEST(MPI_Irecv(&received, 1, MPI_INT, world_rank(), 751, MPI_COMM_WORLD, &recv_raw) ==
             MPI_SUCCESS);
  BOOST_TEST(MPI_Send(&sent, 1, MPI_INT, world_rank(), 751, MPI_COMM_WORLD) == MPI_SUCCESS);
  mp1::request recv_request = mp1::request::from_recv(recv_raw);
  const mp1::status recv_status = recv_request.wait();
  BOOST_TEST(static_cast<int>(recv_status.kind()) ==
             static_cast<int>(mp1::status_kind::native));
  BOOST_TEST(recv_status.source() == world_rank());
  BOOST_TEST(recv_status.get_count<int>() == 1);
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_CASE(error_taxonomy_for_start_and_cancel) {
  // State/policy misuse is std::logic_error; a non-persistent start is invalid_argument.
  int buffer = 0;
  MPI_Request raw = MPI_REQUEST_NULL;
  BOOST_TEST(MPI_Recv_init(&buffer, 1, MPI_INT, world_rank(), 760, MPI_COMM_WORLD, &raw) ==
             MPI_SUCCESS);
  mp1::request persistent = mp1::request::from_persistent_recv(raw);
  persistent.start();
  BOOST_TEST(throws_logic_error([&] { persistent.start(); }));
  persistent.cancel();
  (void)persistent.wait();
  BOOST_TEST(throws_logic_error([&] { persistent.cancel(); }));  // inactive, non-cancelable

  int send_buffer = 0;
  MPI_Request send_raw = MPI_REQUEST_NULL;
  BOOST_TEST(MPI_Send_init(&send_buffer, 1, MPI_INT, world_rank(), 761, MPI_COMM_WORLD,
                           &send_raw) == MPI_SUCCESS);
  mp1::request persistent_send = mp1::request::from_persistent_send(send_raw);
  BOOST_TEST(throws_logic_error([&] { persistent_send.cancel(); }));
  MPI_Request leaked_send = persistent_send.native_handle();
  persistent_send.release();
  persistent_send.cancel();  // released: plain null request, no-op
  BOOST_TEST(throws_invalid_argument([&] { persistent_send.start(); }));
  BOOST_TEST(MPI_Request_free(&leaked_send) == MPI_SUCCESS);
  MPI_Barrier(MPI_COMM_WORLD);
}

BOOST_AUTO_TEST_SUITE_END()
