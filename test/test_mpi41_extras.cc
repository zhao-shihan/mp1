// Tests for the MPI 4.1 additions that fit mp1's existing categories: non-destructive
// multi-request status queries, hardware resource info, process/communicator buffer APIs and
// the resource-guided split type.
//
// The shared test harness links the Boost.Test main; this file defines neither main nor
// BOOST_TEST_MODULE. MPI is initialized by the global fixture below.
//
// Every test is safe at 2 and 4 ranks: rank 0/1 exchange messages and the remaining ranks wait
// at MPI_COMM_WORLD barriers.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
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

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  bool is_empty_status(const mp1::status& s) {
    return s.source() == MPI_ANY_SOURCE and s.tag() == mp1::any_tag and s.error() == MPI_SUCCESS;
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

}  // namespace


struct mp1_mpi41_extras_fixture {
  mp1_mpi41_extras_fixture() {
    mp1::init();
  }

  ~mp1_mpi41_extras_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Another path already finalized; a test fixture destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_mpi41_extras_fixture);

BOOST_AUTO_TEST_SUITE(mpi41_extras)

BOOST_AUTO_TEST_CASE(request_get_status_empty_and_inactive) {
  // Empty vectors follow the existing test_* shapes and never call MPI.
  std::vector<mp1::request> empty;
  const std::optional<std::vector<mp1::status>> empty_all = mp1::get_status_all(empty);
  BOOST_TEST(empty_all.has_value());
  BOOST_TEST(empty_all->empty());
  const std::optional<mp1::completion> empty_any = mp1::get_status_any(empty);
  BOOST_TEST(empty_any.has_value());
  BOOST_TEST(empty_any->first == MPI_UNDEFINED);
  BOOST_TEST(is_empty_status(empty_any->second));
  BOOST_TEST(not mp1::get_status_some(empty).has_value());

  // All-null vectors: all -> one empty status per slot, any -> undefined index,
  // some -> MPI_UNDEFINED (nullopt). No slot is modified.
  std::vector<mp1::request> inactive(3);
  const std::optional<std::vector<mp1::status>> all = mp1::get_status_all(inactive);
  BOOST_TEST(all.has_value());
  BOOST_TEST(all->size() == 3);
  if (all) {
    for (const mp1::status& slot : *all)
      BOOST_TEST(is_empty_status(slot));
  }
  const std::optional<mp1::completion> any = mp1::get_status_any(inactive);
  BOOST_TEST(any.has_value());
  BOOST_TEST(any->first == MPI_UNDEFINED);
  BOOST_TEST(is_empty_status(any->second));
  BOOST_TEST(not mp1::get_status_some(inactive).has_value());
  for (const mp1::request& r : inactive) {
    BOOST_TEST(not r.is_valid());
    BOOST_TEST(not r.is_active());
  }
}

BOOST_AUTO_TEST_CASE(request_get_status_all_is_non_destructive) {
  const int rank = world_rank();
  const int size = world_size();
  BOOST_TEST(size >= 2);
  if (size < 2)
    return;

  int recv_value = 0;
  std::vector<mp1::request> requests(2);
  if (rank == 0) {
    MPI_Request raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&recv_value, 1, MPI_INT, 1, 9601, MPI_COMM_WORLD, &raw) ==
               MPI_SUCCESS);
    requests[1] = mp1::request::from_recv(raw);
  }
  world_barrier();
  if (rank == 0) {
    BOOST_TEST(not mp1::get_status_all(requests).has_value());
    BOOST_TEST(requests[1].is_active());
    BOOST_TEST(requests[1].is_valid());
  }
  world_barrier();
  if (rank == 1) {
    int value = 11;
    BOOST_TEST(MPI_Send(&value, 1, MPI_INT, 0, 9601, MPI_COMM_WORLD) == MPI_SUCCESS);
  }
  if (rank == 0) {
    std::optional<std::vector<mp1::status>> completed;
    for (int i = 0; i < 1000000 and not completed; ++i)
      completed = mp1::get_status_all(requests);
    BOOST_TEST(completed.has_value());
    if (completed) {
      BOOST_TEST(completed->size() == 2);
      BOOST_TEST(is_empty_status((*completed)[0]));
      BOOST_TEST((*completed)[1].source() == 1);
      BOOST_TEST((*completed)[1].tag().value == 9601);
    }
    // The query must not reap or inactivate the active request; the null slot stays null.
    BOOST_TEST(requests[1].is_active());
    BOOST_TEST(requests[1].is_valid());
    BOOST_TEST(not requests[0].is_valid());
    const mp1::status reaped = requests[1].wait();
    BOOST_TEST(reaped.source() == 1);
    BOOST_TEST(reaped.tag().value == 9601);
    BOOST_TEST(not requests[1].is_active());
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(request_get_status_any_reports_index) {
  const int rank = world_rank();
  const int size = world_size();
  BOOST_TEST(size >= 2);
  if (size < 2)
    return;

  int recv_value = 0;
  std::vector<mp1::request> requests(3);
  if (rank == 0) {
    MPI_Request raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&recv_value, 1, MPI_INT, 1, 9602, MPI_COMM_WORLD, &raw) ==
               MPI_SUCCESS);
    requests[1] = mp1::request::from_recv(raw);
    BOOST_TEST(not mp1::get_status_any(requests).has_value());
  }
  world_barrier();
  if (rank == 1) {
    int value = 12;
    BOOST_TEST(MPI_Send(&value, 1, MPI_INT, 0, 9602, MPI_COMM_WORLD) == MPI_SUCCESS);
  }
  if (rank == 0) {
    std::optional<mp1::completion> completed;
    for (int i = 0; i < 1000000 and not completed; ++i)
      completed = mp1::get_status_any(requests);
    BOOST_TEST(completed.has_value());
    if (completed) {
      BOOST_TEST(completed->first == 1);
      BOOST_TEST(completed->second.source() == 1);
      BOOST_TEST(completed->second.tag().value == 9602);
    }
    BOOST_TEST(requests[1].is_active());
    BOOST_TEST(not requests[0].is_valid());
    BOOST_TEST(not requests[2].is_valid());
    const std::vector<mp1::status> reaped = mp1::wait_all(requests);
    BOOST_TEST(reaped.size() == 3);
    BOOST_TEST(is_empty_status(reaped[0]));
    BOOST_TEST(reaped[1].tag().value == 9602);
    BOOST_TEST(is_empty_status(reaped[2]));
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(request_get_status_some_reports_completed_subset) {
  const int rank = world_rank();
  const int size = world_size();
  BOOST_TEST(size >= 2);
  if (size < 2)
    return;

  int first_value = 0;
  int second_value = 0;
  std::vector<mp1::request> requests(3);
  if (rank == 0) {
    MPI_Request first_raw = MPI_REQUEST_NULL;
    MPI_Request second_raw = MPI_REQUEST_NULL;
    BOOST_TEST(MPI_Irecv(&first_value, 1, MPI_INT, 1, 9603, MPI_COMM_WORLD, &first_raw) ==
               MPI_SUCCESS);
    BOOST_TEST(MPI_Irecv(&second_value, 1, MPI_INT, 1, 9604, MPI_COMM_WORLD, &second_raw) ==
               MPI_SUCCESS);
    requests[1] = mp1::request::from_recv(first_raw);
    requests[2] = mp1::request::from_recv(second_raw);
  }
  world_barrier();
  if (rank == 0) {
    const std::optional<std::vector<mp1::completion>> pending = mp1::get_status_some(requests);
    BOOST_TEST(pending.has_value());
    BOOST_TEST(pending->empty());
  }
  world_barrier();
  if (rank == 1) {
    int first = 13;
    int second = 14;
    BOOST_TEST(MPI_Send(&first, 1, MPI_INT, 0, 9603, MPI_COMM_WORLD) == MPI_SUCCESS);
    BOOST_TEST(MPI_Send(&second, 1, MPI_INT, 0, 9604, MPI_COMM_WORLD) == MPI_SUCCESS);
  }
  if (rank == 0) {
    std::optional<std::vector<mp1::completion>> completed;
    for (int i = 0; i < 1000000 and not completed; ++i) {
      completed = mp1::get_status_some(requests);
      if (completed and completed->empty())
        completed.reset();
    }
    BOOST_TEST(completed.has_value());
    if (completed) {
      BOOST_TEST((completed->size() == 1 or completed->size() == 2));
      for (const mp1::completion& entry : *completed) {
        BOOST_TEST((entry.first == 1 or entry.first == 2));
        const int expected_tag = entry.first == 1 ? 9603 : 9604;
        BOOST_TEST(entry.second.source() == 1);
        BOOST_TEST(entry.second.tag().value == expected_tag);
      }
    }
    // The query did not reap the requests; the later wait_all does.
    BOOST_TEST(requests[1].is_active());
    BOOST_TEST(requests[2].is_active());
    const std::vector<mp1::status> reaped = mp1::wait_all(requests);
    BOOST_TEST(reaped.size() == 3);
    BOOST_TEST(is_empty_status(reaped[0]));
    BOOST_TEST(reaped[1].tag().value == 9603);
    BOOST_TEST(reaped[2].tag().value == 9604);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(request_get_status_call_counts) {
#if MP1_TEST_HAVE_MPI_WRAP
  if (not mpi_wrap::supported()) {
    BOOST_TEST_MESSAGE("MPI wrap counters are not active on this platform");
    return;
  }

  std::vector<mp1::request> empty;
  mpi_wrap::reset_counts();
  (void)mp1::get_status_all(empty);
  (void)mp1::get_status_any(empty);
  (void)mp1::get_status_some(empty);
  BOOST_TEST(mpi_wrap::total() == 0);

  std::vector<mp1::request> inactive(2);
  mpi_wrap::reset_counts();
  const auto inactive_all = mp1::get_status_all(inactive);
  BOOST_TEST(inactive_all.has_value());
  BOOST_TEST(inactive_all->size() == 2);
  BOOST_TEST(is_empty_status((*inactive_all)[0]));
  // Per-slot query over only-inactive slots: every slot is skipped, so no MPI call is made
  // and the bulk MPI_Request_get_status_all error path is unreachable.
  BOOST_TEST(mpi_wrap::count("MPI_Request_get_status_all") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Request_get_status_any") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Request_get_status_some") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Request_get_status") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Wait") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Test") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Request_free") == 0);

  mpi_wrap::reset_counts();
  const auto inactive_any = mp1::get_status_any(inactive);
  BOOST_TEST(inactive_any.has_value());
  BOOST_TEST(inactive_any->first == MPI_UNDEFINED);
  BOOST_TEST(mpi_wrap::total() == 0);

  mpi_wrap::reset_counts();
  BOOST_TEST(not mp1::get_status_some(inactive).has_value());
  BOOST_TEST(mpi_wrap::total() == 0);
#else
  BOOST_TEST_MESSAGE("built without the MPI wrap shim; call-count checks skipped");
#endif
}

BOOST_AUTO_TEST_CASE(hw_resource_info) {
  const mp1::info hardware = mp1::get_hw_resource_info();
  BOOST_TEST(hardware.size() >= 0);
  if (not hardware.empty()) {
    const std::string key = hardware.key(0);
    BOOST_TEST(not key.empty());
    BOOST_TEST(hardware.value(key).has_value());
  }

  // The result is an ordinary owning info and remains usable/copyable.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): copy under test
  const mp1::info copy{hardware};
  BOOST_TEST(copy.size() == hardware.size());
}

BOOST_AUTO_TEST_CASE(process_buffer_flush) {
  BOOST_TEST(mp1::buffer_automatic == MPI_BUFFER_AUTOMATIC);
  mp1::buffer_attach(mp1::buffer_automatic, 0);

  if (world_size() >= 2) {
    if (world_rank() == 0) {
      mp1::bsend(mp1::comm_world(), 77, 1, mp1::tag{9701});
    } else if (world_rank() == 1) {
      int value = 0;
      (void)mp1::recv(mp1::comm_world(), value, 0, mp1::tag{9701});
      BOOST_TEST(value == 77);
    }
  }
  world_barrier();

  mp1::buffer_flush();
  mp1::request flush = mp1::buffer_iflush();
  BOOST_TEST(flush.is_valid() == flush.is_active());
  if (flush.is_valid())
    (void)flush.wait();

  const std::pair<void*, int> detached = mp1::buffer_detach();
  BOOST_TEST(detached.first == mp1::buffer_automatic);
  world_barrier();
}

BOOST_AUTO_TEST_CASE(buffer_flush_request_teardown) {
  mp1::buffer_attach(mp1::buffer_automatic, 0);
  mp1::request flush = mp1::buffer_iflush();
  if (flush.is_valid()) {
    BOOST_TEST(flush.is_active());
    BOOST_TEST(throws_logic_error([&flush] { flush.cancel(); }));
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      mpi_wrap::reset_counts();
      {
        mp1::request dropped = std::move(flush);
      }
      // Wait-only teardown: dropping an iflush request waits instead of freeing it.
      BOOST_TEST(mpi_wrap::count("MPI_Wait") == 1);
      BOOST_TEST(mpi_wrap::count("MPI_Request_free") == 0);
      BOOST_TEST(mpi_wrap::count("MPI_Cancel") == 0);
    } else
#endif
    {
      (void)flush.wait();
    }
  }
  (void)mp1::buffer_detach();
  world_barrier();
}

BOOST_AUTO_TEST_CASE(communicator_buffer_lifecycle) {
  mp1::communicator comm = mp1::communicator::dup(mp1::comm_world());
  const std::size_t capacity = 4096 + MPI_BSEND_OVERHEAD;
  std::vector<char> explicit_buffer(capacity);

  BOOST_TEST(throws_invalid_argument(
      [&comm, &explicit_buffer] { mp1::attach_buffer(comm, explicit_buffer.data(), -1); }));
  mp1::attach_buffer(comm, explicit_buffer.data(),
                     static_cast<mp1::count_t>(explicit_buffer.size()));
  if (world_size() >= 2) {
    if (world_rank() == 0) {
      mp1::bsend(comm, 88, 1, mp1::tag{9711});
    } else if (world_rank() == 1) {
      int value = 0;
      (void)mp1::recv(comm, value, 0, mp1::tag{9711});
      BOOST_TEST(value == 88);
    }
  }
  world_barrier();
  mp1::flush_buffer(comm);
  mp1::request explicit_flush = mp1::iflush_buffer(comm);
  if (explicit_flush.is_valid())
    (void)explicit_flush.wait();
  const std::pair<void*, mp1::count_t> detached = mp1::detach_buffer(comm);
  BOOST_TEST(detached.first == static_cast<void*>(explicit_buffer.data()));
  BOOST_TEST(detached.second == static_cast<mp1::count_t>(explicit_buffer.size()));

  // The same communicator can switch to automatic buffering after an explicit detach.  MPI
  // ignores the size for an automatic buffer, so a negative size is accepted and normalized
  // to 0, matching buffer_attach.
  mp1::attach_buffer(comm, mp1::buffer_automatic, -1);
  if (world_size() >= 2) {
    if (world_rank() == 0) {
      mp1::bsend(comm, 89, 1, mp1::tag{9712});
    } else if (world_rank() == 1) {
      int value = 0;
      (void)mp1::recv(comm, value, 0, mp1::tag{9712});
      BOOST_TEST(value == 89);
    }
  }
  world_barrier();
  mp1::flush_buffer(comm);
  mp1::request automatic_flush = mp1::iflush_buffer(comm);
  if (automatic_flush.is_valid())
    (void)automatic_flush.wait();
  // The undefined automatic-buffer size is reported, not thrown after the successful detach.
  const std::pair<void*, mp1::count_t> automatic_detached = mp1::detach_buffer(comm);
  BOOST_TEST(automatic_detached.first == mp1::buffer_automatic);

  comm.free();
  world_barrier();
}

// detach_buffer uses the same non-throwing size conversion as buffer_detach: an undefined or
// unrepresentable detached size becomes MPI_UNDEFINED instead of throwing after MPI has
// already detached the buffer.  On a host where MPI_Count and count_t have the same width
// every MPI_Count fits, so the out-of-range branch is pinned only where the widths differ.
BOOST_AUTO_TEST_CASE(buffer_detach_size_conversion_reports_undefined) {
  BOOST_TEST(mp1::detail::checked_count_or_undefined(MPI_UNDEFINED) == MPI_UNDEFINED);
  BOOST_TEST(mp1::detail::checked_count_or_undefined(0) == 0);
  if constexpr (sizeof(MPI_Count) > sizeof(mp1::count_t)) {
    // Every MPI_Count outside count_t's range is undefined; taking the type limits avoids an
    // overflowing expression (which a compiler may diagnose even in a discarded constexpr
    // branch on this same-width host).
    BOOST_TEST(mp1::detail::checked_count_or_undefined(std::numeric_limits<MPI_Count>::max()) ==
               MPI_UNDEFINED);
    BOOST_TEST(mp1::detail::checked_count_or_undefined(std::numeric_limits<MPI_Count>::min()) ==
               MPI_UNDEFINED);
  }
}

BOOST_AUTO_TEST_CASE(comm_type_resource_guided) {
  static_assert(mp1::comm_type_resource_guided == MPI_COMM_TYPE_RESOURCE_GUIDED,
                "resource-guided split type must mirror the MPI constant");

  mp1::info split_info;
  const mp1::info hardware = mp1::get_hw_resource_info();
  if (not hardware.empty())
    split_info.set("mpi_hw_resource_type", hardware.key(0));

  mp1::communicator split = mp1::communicator::split_type(
      mp1::comm_world(), mp1::comm_type_resource_guided, 0, split_info);
  if (split.is_valid()) {
    BOOST_TEST(split.size() >= 1);
    BOOST_TEST(not split.is_inter());
    split.free();
  }
  world_barrier();
}

BOOST_AUTO_TEST_SUITE_END()
