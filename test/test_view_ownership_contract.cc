// Owner/view contract test: one owner, borrowed views, explicit release policy.
//
// MPI 4.1 basis: §2.5.1 (Opaque Objects) — a handle is a copied value, not a reference, and a
// deallocate routine invalidates it; §7.4.3 (Communicator Destructors) — MPI_COMM_FREE is
// collective and sets only the caller's handle to MPI_COMM_NULL; §11.10.4 (Releasing
// Connections) — MPI_COMM_DISCONNECT is the collective release that waits for decoupled
// activities, while MPI_COMM_FREE does not disconnect; §12.2.5 (Window Destruction) —
// MPI_WIN_FREE is collective and returns a null handle.  The contract tested here is the mp1
// borrow model, not runtime use-after-free detection: is_valid() is a null-sentinel test and
// never promises liveness.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>


#if MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_OWNERSHIP_WRAP 1
#else
#define MP1_OWNERSHIP_WRAP 0
#endif

namespace {

  struct ownership_fixture {
    ownership_fixture() {
      mp1::init();
    }
    ~ownership_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
      }
    }
  };

  template<class Exception, class F>
  bool throws_as(F&& function) {
    try {
      function();
    } catch (const Exception&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

#if MP1_OWNERSHIP_WRAP
  std::size_t call_count(const char* symbol) {
    if (not mpi_wrap::supported())
      return 0;
    return mpi_wrap::count(symbol);
  }

  std::size_t total_calls() {
    return mpi_wrap::supported() ? mpi_wrap::total() : 0;
  }
#endif

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(ownership_fixture);

BOOST_AUTO_TEST_SUITE(view_ownership_contract)

BOOST_AUTO_TEST_CASE(owner_and_view_type_contract) {
  BOOST_TEST(true);
  // Views are zero-cost borrowed values.
  static_assert(std::is_trivially_copyable_v<mp1::mpi_communicator>);
  static_assert(std::is_trivially_copyable_v<mp1::mpi_window>);
  static_assert(std::is_trivially_destructible_v<mp1::mpi_communicator>);
  static_assert(std::is_trivially_destructible_v<mp1::mpi_window>);
  static_assert(std::is_nothrow_destructible_v<mp1::communicator>);
  static_assert(std::is_nothrow_destructible_v<mp1::window>);

  // Owners compose their view: no public base, no mutable view reference, no view-to-owner
  // conversion; an owner converts to a view by value and parent/owner remains move-only.
  static_assert(not std::is_base_of_v<mp1::mpi_communicator, mp1::communicator>);
  static_assert(not std::is_base_of_v<mp1::mpi_window, mp1::window>);
  static_assert(std::is_convertible_v<const mp1::communicator&, mp1::mpi_communicator>);
  static_assert(std::is_convertible_v<const mp1::window&, mp1::mpi_window>);
  static_assert(not std::is_convertible_v<mp1::communicator&, mp1::mpi_communicator&>);
  static_assert(not std::is_convertible_v<mp1::window&, mp1::mpi_window&>);
  static_assert(not std::is_constructible_v<mp1::communicator, const mp1::mpi_communicator&>);
  static_assert(not std::is_constructible_v<mp1::window, const mp1::mpi_window&>);
  static_assert(not std::is_copy_constructible_v<mp1::communicator>);
  static_assert(not std::is_copy_constructible_v<mp1::window>);

  const mp1::mpi_communicator world = mp1::comm_world();
  mp1::communicator owner = mp1::communicator::dup(world);
  BOOST_TEST(owner.is_valid());
  BOOST_TEST(static_cast<int>(owner.policy()) == static_cast<int>(mp1::release_policy::free));

  // An explicit view copy is the documented borrow and shares the MPI handle.  view() returns
  // the view by value, so the copy is not a reference into the movable owner.
  const mp1::mpi_communicator borrowed = owner;
  BOOST_TEST(borrowed.is_valid());
  BOOST_TEST(borrowed.native_handle() == owner.native_handle());
  BOOST_TEST(owner.view().native_handle() == owner.native_handle());

  // Moved-from owner is invalid and its own view is invalid; the destination owns the handle.
  // The view copy taken before the move stays valid and keeps the same snapshot.
  const MPI_Comm handle = owner.native_handle();
  const mp1::mpi_communicator copied_view = owner.view();
  mp1::communicator moved{std::move(owner)};
  BOOST_TEST(not owner.is_valid());
  BOOST_TEST(not owner.view().is_valid());
  BOOST_TEST(copied_view.is_valid());
  BOOST_TEST(copied_view.native_handle() == handle);
  BOOST_TEST(copied_view.size() == moved.size());
  BOOST_TEST(moved.is_valid());
  BOOST_TEST(moved.native_handle() == handle);
  BOOST_TEST(static_cast<int>(moved.policy()) == static_cast<int>(mp1::release_policy::free));

  // The invalid owner performs no MPI call when destroyed.
  owner.free();
  BOOST_TEST(not owner.is_valid());
}

BOOST_AUTO_TEST_CASE(owner_view_is_a_snapshot_not_liveness) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  mp1::communicator owner = mp1::communicator::dup(world);
  const int size_before = owner.size();
  const MPI_Comm handle_before = owner.native_handle();

  // Characterization of the documented contract: the copied view keeps the old handle and
  // the cached snapshot after the owner frees.  It must not be used for MPI calls; this test
  // deliberately never does (use-after-free is a precondition violation, not detected).
  const mp1::mpi_communicator stale = owner;
  owner.free();
  BOOST_TEST(not owner.is_valid());
  BOOST_TEST(not owner.view().is_valid());
  BOOST_TEST(stale.is_valid());
  BOOST_TEST(stale.native_handle() == handle_before);
  BOOST_TEST(stale.size() == size_before);
}

BOOST_AUTO_TEST_CASE(communicator_release_is_call_free) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  mp1::communicator owner = mp1::communicator::dup(world);
  const MPI_Comm handle = owner.native_handle();

#if MP1_OWNERSHIP_WRAP
  if (mpi_wrap::supported())
    mpi_wrap::reset_counts();
#endif
  const MPI_Comm released = owner.release();
  BOOST_TEST(released == handle);
  BOOST_TEST(not owner.is_valid());
  BOOST_TEST(not owner.view().is_valid());
#if MP1_OWNERSHIP_WRAP
  if (mpi_wrap::supported())
    BOOST_TEST(total_calls() == 0);
#endif
  // release() transfers responsibility: the caller frees the raw handle.
  MPI_Comm raw = released;
  BOOST_TEST(MPI_Comm_free(&raw) == MPI_SUCCESS);
  BOOST_TEST(raw == MPI_COMM_NULL);
}

BOOST_AUTO_TEST_CASE(release_policy_rejection_and_disconnect) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world.rank();
  const int size = world.size();
  if (size < 2) {
    BOOST_TEST_MESSAGE("skipping intercommunicator policy case: need at least 2 ranks");
    return;
  }

  // Build an intercommunicator collectively, then wrap its raw handle with the
  // disconnect policy.  This is the adopt_connected path used for raw connect/accept/join.
  const int half = (size + 1) / 2;
  const int color = rank < half ? 0 : 1;
  const int remote_leader = color == 0 ? half : 0;
  mp1::communicator local = mp1::communicator::split(world, color, rank);
  mp1::communicator raw_inter =
      mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{7611});
  BOOST_TEST(raw_inter.is_inter());

  MPI_Comm released_handle = raw_inter.release();
  mp1::communicator connected = mp1::communicator::adopt_connected(released_handle);
  BOOST_TEST(connected.is_valid());
  BOOST_TEST(connected.is_inter());
  BOOST_TEST(static_cast<int>(connected.policy()) ==
             static_cast<int>(mp1::release_policy::disconnect));

  // free() is rejected on a disconnect-policy owner before any MPI release call.
#if MP1_OWNERSHIP_WRAP
  if (mpi_wrap::supported())
    mpi_wrap::reset_counts();
#endif
  BOOST_TEST(throws_as<std::logic_error>([&] { connected.free(); }));
  BOOST_TEST(connected.is_valid());
#if MP1_OWNERSHIP_WRAP
  if (mpi_wrap::supported()) {
    BOOST_TEST(call_count("MPI_Comm_free") == 0);
    BOOST_TEST(call_count("MPI_Comm_disconnect") == 0);
  }
#endif

  // The explicit collective release is observed exactly once on this rank.
#if MP1_OWNERSHIP_WRAP
  if (mpi_wrap::supported())
    mpi_wrap::reset_counts();
#endif
  connected.disconnect();
  BOOST_TEST(not connected.is_valid());
#if MP1_OWNERSHIP_WRAP
  if (mpi_wrap::supported()) {
    BOOST_TEST(call_count("MPI_Comm_disconnect") == 1);
    BOOST_TEST(call_count("MPI_Comm_free") == 0);
  }
#endif

  // release() of a disconnect-policy owner makes no MPI call and leaves the peer protocol to
  // the caller; the raw handle is freed (free/free is the non-cooperating pair).
  mp1::communicator second = mp1::communicator::dup(world);
  MPI_Comm second_raw = second.release();
  mp1::communicator second_connected = mp1::communicator::adopt_connected(second_raw);
#if MP1_OWNERSHIP_WRAP
  if (mpi_wrap::supported())
    mpi_wrap::reset_counts();
#endif
  MPI_Comm detached = second_connected.release();
  BOOST_TEST(not second_connected.is_valid());
#if MP1_OWNERSHIP_WRAP
  if (mpi_wrap::supported())
    BOOST_TEST(total_calls() == 0);
#endif
  BOOST_TEST(MPI_Comm_free(&detached) == MPI_SUCCESS);
}

BOOST_AUTO_TEST_CASE(window_release_policy) {
  BOOST_TEST(true);
  const mp1::mpi_communicator self = mp1::comm_self();

  mp1::window owner =
      mp1::window::allocate(self, {.size = 4 * sizeof(int), .disp_unit = sizeof(int)});
  BOOST_TEST(owner.is_valid());

  const mp1::mpi_window view = owner;
  BOOST_TEST(view.is_valid());
  BOOST_TEST(view.native_handle() == owner.native_handle());
  BOOST_TEST(view.byte_size() == owner.byte_size());

  const MPI_Win handle = owner.native_handle();
#if MP1_OWNERSHIP_WRAP
  if (mpi_wrap::supported())
    mpi_wrap::reset_counts();
#endif
  const MPI_Win released = owner.release();
  BOOST_TEST(released == handle);
  BOOST_TEST(not owner.is_valid());
  BOOST_TEST(not owner.view().is_valid());
  // view() returns by value: the copy taken before release keeps the old handle and its
  // cached snapshot.  It must not be used for MPI calls.
  BOOST_TEST(view.is_valid());
  BOOST_TEST(view.native_handle() == handle);
#if MP1_OWNERSHIP_WRAP
  if (mpi_wrap::supported())
    BOOST_TEST(total_calls() == 0);
#endif
  // The caller owns the raw window now; MPI_Win_free is collective over MPI_COMM_SELF here.
  MPI_Win raw = released;
  BOOST_TEST(MPI_Win_free(&raw) == MPI_SUCCESS);

  // free() is idempotent and makes no further MPI call on the invalid owner.
  mp1::window freed = mp1::window::allocate(self, {.size = 0, .disp_unit = 1});
  freed.free();
  BOOST_TEST(not freed.is_valid());
  freed.free();
  BOOST_TEST(not freed.is_valid());

  // A copied view keeps the pre-free snapshot; it must not be used for MPI calls.
  mp1::window stale_owner =
      mp1::window::allocate(self, {.size = sizeof(int), .disp_unit = sizeof(int)});
  const mp1::mpi_window stale_view = stale_owner;
  const MPI_Win stale_handle = stale_owner.native_handle();
  stale_owner.free();
  BOOST_TEST(stale_view.is_valid());
  BOOST_TEST(stale_view.native_handle() == stale_handle);
}

BOOST_AUTO_TEST_CASE(invalid_owner_operations_are_local) {
  BOOST_TEST(true);
  mp1::communicator invalid;
  invalid.free();
  invalid.disconnect();
  BOOST_TEST(invalid.release() == MPI_COMM_NULL);
  mp1::window invalid_window;
  invalid_window.free();
  BOOST_TEST(invalid_window.release() == MPI_WIN_NULL);
}

BOOST_AUTO_TEST_SUITE_END()
