// Contract test for the MPI_ERR_PENDING slot of the multiple-completion functions.
//
// MPI 4.1 3.7.5: MPI_Waitall returns MPI_ERR_IN_STATUS and reports MPI_ERR_PENDING for a slot
// that "has neither failed nor completed". Such a slot is still active: the wrapper must not
// commit it, must keep its MPI_Request and payload, and must finish it on a later completion
// call.
//
// Two layers are tested:
//   1. forced_pending_batch_transaction drives the shared completion transaction directly with
//      forced per-slot statuses. It is provider-neutral and covers both the commit and the
//      PENDING branch of the transaction.
//   2. The end-to-end cases force the real Waitall state with a truncation error on one slot
//      and an unreceived message on another, using a point-to-point signal handshake instead
//      of a sleep. Only a provider that returns from Waitall as soon as one slot fails can
//      report PENDING for the others (Open MPI does; MPICH 4.3.2 waits for every slot and
//      cannot produce this state), so those cases run when OMPI_MAJOR_VERSION is defined.
//
// It provides no main: the shared test harness links the Boost test main.

#include "mp1/impl/detail/request_payload.hpp"
#include "mp1/impl/environment.hpp"
#include "mp1/impl/p2p.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/status.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace {

  [[maybe_unused]] int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  int world_size() {
    int size{1};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  // A PENDING status carries only error(): source/tag are normalized and count/elements are 0
  // without an MPI call; cancellation cannot be queried.
  void check_pending_status(const mp1::status& s) {
    BOOST_TEST(s.error() == MPI_ERR_PENDING);
    BOOST_TEST(s.source() == MPI_ANY_SOURCE);
    BOOST_TEST(s.tag().value == mp1::any_tag.value);
    BOOST_TEST(s.get_count<int>() == 0);
    BOOST_TEST(s.get_elements<int>() == 0);
    BOOST_TEST(not s.is_cancelled());
    BOOST_TEST(static_cast<int>(s.kind()) == static_cast<int>(mp1::status_kind::native));
  }

  [[maybe_unused]] bool throws_logic_error(const mp1::request& request) {
    try {
      // start() is non-const only because it changes state; a bad start must leave it
      // unchanged.
      const_cast<mp1::request&>(request).start();
    } catch (const std::logic_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  class counting_payload final : public mp1::detail::request_payload {
  public:
    explicit counting_payload(int* calls) : calls_{calls} {
    }

    void complete(const mp1::status&) override {
      ++*calls_;
    }

  private:
    int* calls_;
  };

}  // namespace

struct mp1_waitall_pending_fixture {
  mp1_waitall_pending_fixture() {
    mp1::init();
  }

  ~mp1_waitall_pending_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Already finalized by another path; never throw from a fixture dtor.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_waitall_pending_fixture);

BOOST_AUTO_TEST_SUITE(waitall_pending)

BOOST_AUTO_TEST_CASE(forced_pending_batch_transaction) {
  const auto self = mp1::comm_self();
  int first_value{-1};
  int second_value{-1};
  std::vector<mp1::request> requests;
  requests.push_back(mp1::recv_init(self, first_value, 0, mp1::tag{91}));
  requests.push_back(mp1::recv_init(self, second_value, 0, mp1::tag{92}));
  mp1::start_all(requests);
  BOOST_TEST(requests[0].is_active());
  BOOST_TEST(requests[1].is_active());

  // Complete slot 0 for real (persistent handle stays valid) and leave slot 1 pending.
  int value{42};
  mp1::request send_request = mp1::isend(self, value, 0, mp1::tag{91});
  (void)send_request.wait();
  MPI_Request first_handle = requests[0].native_handle();
  MPI_Status completed_native{};
  int flag{0};
  for (int i = 0; i < 1000000 and flag == 0; ++i)
    BOOST_TEST(MPI_Test(&first_handle, &flag, &completed_native) == MPI_SUCCESS);
  BOOST_TEST(flag != 0);

  int completed_count{0};
  mp1::detail::attach_payload(requests[1],
                              std::make_unique<counting_payload>(&completed_count));

  mp1::detail::request_batch batch{requests, "test forced pending", requests.size()};
  batch.native(0) = completed_native;
  batch.native(0).MPI_ERROR = MPI_SUCCESS;
  batch.native(1).MPI_ERROR = MPI_ERR_PENDING;
  batch.native(1).MPI_SOURCE = 5;
  batch.native(1).MPI_TAG = 6;

  // Classification is fixed by the function code and the per-slot error only.
  BOOST_TEST(static_cast<int>(batch.verdict(0, MPI_ERR_IN_STATUS)) ==
             static_cast<int>(mp1::detail::completion_verdict::success));
  BOOST_TEST(static_cast<int>(batch.verdict(1, MPI_ERR_IN_STATUS)) ==
             static_cast<int>(mp1::detail::completion_verdict::pending));
  batch.native(0).MPI_ERROR = MPI_ERR_OTHER;
  BOOST_TEST(static_cast<int>(batch.verdict(0, MPI_ERR_IN_STATUS)) ==
             static_cast<int>(mp1::detail::completion_verdict::error));
  // MPI_SUCCESS means every slot completed; a function-level code is never treated as success.
  BOOST_TEST(static_cast<int>(batch.verdict(0, MPI_SUCCESS)) ==
             static_cast<int>(mp1::detail::completion_verdict::success));
  BOOST_TEST(static_cast<int>(batch.verdict(0, MPI_ERR_OTHER)) ==
             static_cast<int>(mp1::detail::completion_verdict::error));
  batch.native(0) = completed_native;
  batch.native(0).MPI_ERROR = MPI_SUCCESS;

  std::vector<mp1::status> result(2);
  std::exception_ptr failure;
  batch.finish_all(MPI_ERR_IN_STATUS, result, failure);
  BOOST_TEST(not failure);

  // Completed slot: committed persistent request, native receive status with the real count.
  BOOST_TEST(result[0].get_count<int>() == 1);
  BOOST_TEST(result[0].source() == 0);
  BOOST_TEST(result[0].tag().value == 91);
  BOOST_TEST(not requests[0].is_active());
  BOOST_TEST(requests[0].is_valid());
  BOOST_TEST(static_cast<int>(requests[0].state()) ==
             static_cast<int>(mp1::request_state::inactive));

  // PENDING slot: never committed, never finished, payload untouched.
  check_pending_status(result[1]);
  BOOST_TEST(requests[1].is_active());
  BOOST_TEST(requests[1].is_valid());
  BOOST_TEST(requests[1].native_handle() != MPI_REQUEST_NULL);
  BOOST_TEST(completed_count == 0);
}

BOOST_AUTO_TEST_CASE(nonpersistent_pending_slot_stays_active) {
  if (world_size() < 2) {
    BOOST_TEST_MESSAGE("wait_all pending test needs at least 2 ranks");
    return;
  }
#if !defined(OMPI_MAJOR_VERSION)
  BOOST_TEST_MESSAGE(
      "wait_all PENDING end-to-end case skipped: MPICH 4.3.2 MPI_Waitall waits for every slot "
      "and cannot report MPI_ERR_PENDING before the other slot completes; "
      "forced_pending_batch_transaction covers the state machine");
  return;
#else
  // A truncation must be reported to the wrapper as a per-slot completion error instead of
  // invoking the default fatal handler. Set the handler on MPI_COMM_WORLD before dup: MPICH
  // ignores a handler set on a duped communicator for this error path, and the world handler
  // also governs the case's MPI_COMM_WORLD messages.
  BOOST_TEST(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) == MPI_SUCCESS);

  const int rank = world_rank();
  if (rank == 0) {
    int four[4] = {1, 2, 3, 4};
    BOOST_TEST(MPI_Send(four, 4, MPI_INT, 1, 1, MPI_COMM_WORLD) == MPI_SUCCESS);
    int signal{-1};
    BOOST_TEST(MPI_Recv(&signal, 1, MPI_INT, 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE) ==
               MPI_SUCCESS);
    // Sent only after rank 1 observed the PENDING slot: the first wait_all cannot see it.
    int two[2] = {7, 8};
    BOOST_TEST(MPI_Send(two, 2, MPI_INT, 1, 2, MPI_COMM_WORLD) == MPI_SUCCESS);
    int done{-1};
    BOOST_TEST(MPI_Recv(&done, 1, MPI_INT, 1, 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE) ==
               MPI_SUCCESS);
  } else if (rank == 1) {
    int one{-1};
    std::vector<int> capacity;  // resized to the posted capacity before MPI, then to the count
    std::vector<mp1::request> requests;
    requests.push_back(mp1::irecv(mp1::comm_world(), std::span<int>(&one, 1), 0, mp1::tag{1}));
    requests.push_back(mp1::irecv(mp1::comm_world(), capacity, mp1::capacity{4}, 0, mp1::tag{2}));
    BOOST_TEST(capacity.size() == 4);
    BOOST_TEST(requests[0].is_active());
    BOOST_TEST(requests[1].is_active());

    const std::vector<mp1::status> first = mp1::wait_all(requests);
    BOOST_TEST(first.size() == 2);
    BOOST_TEST(first[0].error() !=
               MPI_SUCCESS);  // truncated completion committed with an error
    check_pending_status(first[1]);
    BOOST_TEST(not requests[0].is_active());
    BOOST_TEST(not requests[0].is_valid());
    // The PENDING slot must still own its live handle and its payload.
    BOOST_TEST(requests[1].is_active());
    BOOST_TEST(requests[1].is_valid());
    BOOST_TEST(requests[1].native_handle() != MPI_REQUEST_NULL);
    BOOST_TEST(capacity.size() == 4);  // payload was not destroyed or finished

    int signal{1};
    BOOST_TEST(MPI_Send(&signal, 1, MPI_INT, 0, 99, MPI_COMM_WORLD) == MPI_SUCCESS);

    const std::vector<mp1::status> second = mp1::wait_all(requests);
    BOOST_TEST(second.size() == 2);
    BOOST_TEST(second[1].source() == 0);
    BOOST_TEST(second[1].tag().value == 2);
    BOOST_TEST(second[1].get_count<int>() == 2);
    // The capacity receive payload completes on the retry: the vector shrinks to the count.
    BOOST_TEST(capacity.size() == 2);
    BOOST_TEST(capacity[0] == 7);
    BOOST_TEST(capacity[1] == 8);
    BOOST_TEST(static_cast<int>(requests[1].state()) ==
               static_cast<int>(mp1::request_state::freed));
    BOOST_TEST(not requests[1].is_valid());

    int done{1};
    BOOST_TEST(MPI_Send(&done, 1, MPI_INT, 0, 98, MPI_COMM_WORLD) == MPI_SUCCESS);
  }

  BOOST_TEST(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  BOOST_TEST(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL) == MPI_SUCCESS);
  BOOST_TEST(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
#endif
}

BOOST_AUTO_TEST_CASE(persistent_pending_slot_stays_active) {
  if (world_size() < 2) {
    BOOST_TEST_MESSAGE("wait_all persistent pending test needs at least 2 ranks");
    return;
  }
#if !defined(OMPI_MAJOR_VERSION)
  BOOST_TEST_MESSAGE(
      "wait_all persistent PENDING end-to-end case skipped: MPICH 4.3.2 MPI_Waitall waits for "
      "every slot; forced_pending_batch_transaction covers the state machine");
  return;
#else
  BOOST_TEST(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN) == MPI_SUCCESS);

  const int rank = world_rank();
  if (rank == 0) {
    int four[4] = {1, 2, 3, 4};
    BOOST_TEST(MPI_Send(four, 4, MPI_INT, 1, 1, MPI_COMM_WORLD) == MPI_SUCCESS);
    int signal{-1};
    BOOST_TEST(MPI_Recv(&signal, 1, MPI_INT, 1, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE) ==
               MPI_SUCCESS);
    int two{7};
    BOOST_TEST(MPI_Send(&two, 1, MPI_INT, 1, 2, MPI_COMM_WORLD) == MPI_SUCCESS);
    BOOST_TEST(MPI_Recv(&signal, 1, MPI_INT, 1, 98, MPI_COMM_WORLD, MPI_STATUS_IGNORE) ==
               MPI_SUCCESS);
    // A clean third cycle, restarting both persistent receives.
    int one{11};
    int second_two{22};
    BOOST_TEST(MPI_Send(&one, 1, MPI_INT, 1, 1, MPI_COMM_WORLD) == MPI_SUCCESS);
    BOOST_TEST(MPI_Send(&second_two, 1, MPI_INT, 1, 2, MPI_COMM_WORLD) == MPI_SUCCESS);
    BOOST_TEST(MPI_Recv(&signal, 1, MPI_INT, 1, 97, MPI_COMM_WORLD, MPI_STATUS_IGNORE) ==
               MPI_SUCCESS);
  } else if (rank == 1) {
    int one{-1};
    int two{-1};
    std::vector<mp1::request> requests;
    requests.push_back(mp1::recv_init(mp1::comm_world(), one, 0, mp1::tag{1}));
    requests.push_back(mp1::recv_init(mp1::comm_world(), two, 0, mp1::tag{2}));
    BOOST_TEST(requests[0].is_persistent());
    BOOST_TEST(requests[1].is_persistent());
    BOOST_TEST(not requests[0].is_active());
    mp1::start_all(requests);
    BOOST_TEST(requests[0].is_active());
    BOOST_TEST(requests[1].is_active());

    const std::vector<mp1::status> first = mp1::wait_all(requests);
    BOOST_TEST(first.size() == 2);
    BOOST_TEST(first[0].error() != MPI_SUCCESS);
    check_pending_status(first[1]);
    // The truncated persistent receive completed with an error: inactive but still valid.
    BOOST_TEST(not requests[0].is_active());
    BOOST_TEST(requests[0].is_valid());
    BOOST_TEST(static_cast<int>(requests[0].state()) ==
               static_cast<int>(mp1::request_state::inactive));
    // The PENDING slot stays active, so a restart is rejected as "already active".
    BOOST_TEST(requests[1].is_active());
    BOOST_TEST(requests[1].is_valid());
    BOOST_TEST(requests[1].native_handle() != MPI_REQUEST_NULL);
    BOOST_TEST(throws_logic_error(requests[1]));

    int signal{1};
    BOOST_TEST(MPI_Send(&signal, 1, MPI_INT, 0, 99, MPI_COMM_WORLD) == MPI_SUCCESS);

    const std::vector<mp1::status> second = mp1::wait_all(requests);
    BOOST_TEST(second.size() == 2);
    BOOST_TEST(two == 7);
    BOOST_TEST(second[1].source() == 0);
    BOOST_TEST(second[1].tag().value == 2);
    BOOST_TEST(static_cast<int>(requests[1].state()) ==
               static_cast<int>(mp1::request_state::inactive));
    BOOST_TEST(requests[1].is_valid());

    // Third cycle: start() again and complete normally through the same persistent handles.
    BOOST_TEST(MPI_Send(&signal, 1, MPI_INT, 0, 98, MPI_COMM_WORLD) == MPI_SUCCESS);
    mp1::start_all(requests);
    const std::vector<mp1::status> third = mp1::wait_all(requests);
    BOOST_TEST(third.size() == 2);
    BOOST_TEST(one == 11);
    BOOST_TEST(two == 22);
    BOOST_TEST(third[0].source() == 0);
    BOOST_TEST(third[1].source() == 0);
    BOOST_TEST(static_cast<int>(requests[0].state()) ==
               static_cast<int>(mp1::request_state::inactive));
    BOOST_TEST(static_cast<int>(requests[1].state()) ==
               static_cast<int>(mp1::request_state::inactive));
    BOOST_TEST(MPI_Send(&signal, 1, MPI_INT, 0, 97, MPI_COMM_WORLD) == MPI_SUCCESS);
  }

  BOOST_TEST(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
  BOOST_TEST(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL) == MPI_SUCCESS);
  BOOST_TEST(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS);
#endif
}

BOOST_AUTO_TEST_CASE(test_all_control) {
  // No-complete: flag = false modifies no request (MPI 4.1 3.7.5) and returns nullopt.
  {
    int buffer{0};
    MPI_Request raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&buffer, 1, MPI_INT, 0, 55, MPI_COMM_SELF, &raw) == MPI_SUCCESS);
    std::vector<mp1::request> requests;
    requests.push_back(mp1::request::from_recv(raw));
    const std::optional<std::vector<mp1::status>> incomplete = mp1::test_all(requests);
    BOOST_TEST(not incomplete.has_value());
    BOOST_TEST(requests[0].is_active());
    BOOST_TEST(requests[0].is_valid());
    BOOST_TEST(requests[0].native_handle() != MPI_REQUEST_NULL);
  }  // destructor: cancel + wait

  // All-complete: flag = true commits every completed slot and returns its status.
  {
    int received{0};
    MPI_Request recv_raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&received, 1, MPI_INT, 0, 56, MPI_COMM_SELF, &recv_raw) ==
               MPI_SUCCESS);
    int value{42};
    MPI_Request send_raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Isend(&value, 1, MPI_INT, 0, 56, MPI_COMM_SELF, &send_raw) == MPI_SUCCESS);
    std::vector<mp1::request> requests;
    requests.push_back(mp1::request::from_send(send_raw));
    requests.push_back(mp1::request::from_recv(recv_raw));

    std::optional<std::vector<mp1::status>> completed;
    for (int i = 0; i < 1000000 and not completed; ++i)
      completed = mp1::test_all(requests);
    BOOST_TEST(completed.has_value());
    if (completed) {
      BOOST_TEST(completed->size() == 2);
      BOOST_TEST((*completed)[1].get_count<int>() == 1);
      BOOST_TEST((*completed)[1].source() == 0);
      BOOST_TEST((*completed)[1].tag().value == 56);
    }
    BOOST_TEST(received == 42);
    BOOST_TEST(not requests[0].is_active());
    BOOST_TEST(not requests[1].is_active());
    BOOST_TEST(not requests[0].is_valid());
    BOOST_TEST(not requests[1].is_valid());
  }
}

BOOST_AUTO_TEST_SUITE_END()
