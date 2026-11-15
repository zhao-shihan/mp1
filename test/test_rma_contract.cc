// Contract test for mp1::mpi_window / mp1::window and the RMA data-movement/synchronization
// API.
//
// The shared test harness links the Boost.Test main; this file defines neither main nor
// BOOST_TEST_MODULE. MPI is initialized by the global fixture below.
//
// The test runs at 2 and 4 ranks.  Every collective (window creation/free, fence, info) is
// called by all ranks with identical control flow.  Shared-memory windows need a shared-memory
// domain and are covered by the separate scenario test test_rma_shared.cc, which returns 77
// when the domain is unavailable (for example a provider without a usable shared-memory
// transport); this file only uses the
// shared_query forms that MPI 4.1 allows on CREATE/ALLOCATE windows.
//
// Erroneous MPI programs are avoided on purpose: no RMA to a zero-size window, no out-of-range
// target_disp, no MPI_Win_free with an open epoch or an outstanding RMA request, and no
// incomplete fence.  Local-validation checks throw before any MPI call.  MPI_PROC_NULL is a
// standard no-op target for the ten communication calls (MPI 4.1 §12.3) and is exercised as
// such; lock/unlock/flush keep rejecting it (§12.5.3/§12.5.4).  The atomic datatype/op table
// (MPI 4.1 §6.9.2/§12.3.4) is enforced at compile time for built-in element types and before
// the MPI call for raw handles and custom traits.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_RMA_TEST_WRAP 1
#else
#define MP1_RMA_TEST_WRAP 0
#endif


// Custom predefined-datatype mappings: MPI_INT is a supported basic type, MPI_2INT is a
// predefined pair datatype that mp1 does not model in the typed RMA op table.  Both are
// classified from the committed handle at run time (one MPI query per T), before the RMA call.
struct rma_custom_int {
  int value;
};

struct rma_custom_pair {
  int first;
  int second;
};

MP1_DATATYPE_TRAITS(rma_custom_int, MPI_INT)
MP1_DATATYPE_TRAITS(rma_custom_pair, MPI_2INT)

namespace {

  struct rma_fixture {
    rma_fixture() {
      mp1::init();
    }

    ~rma_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): finalize cleanup must not throw
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

  [[nodiscard]] mp1::mpi_communicator world() {
    return mp1::comm_world();
  }

  template<class Function>
  bool throws_logic(Function&& function) {
    try {
      function();
    } catch (const std::logic_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class Function>
  bool throws_invalid(Function&& function) {
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
  bool throws_runtime(Function&& function) {
    try {
      function();
    } catch (const std::runtime_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  [[nodiscard]] bool is_unified_or_separate(int model) {
    return model == MPI_WIN_UNIFIED or model == MPI_WIN_SEPARATE;
  }

  // Window memory can be written by another process; a volatile load keeps the compiler from
  // caching the value across the MPI synchronization calls (MPI 4.1 12.7.4).
  [[nodiscard]] int window_load(const int* p) {
    return *static_cast<const volatile int*>(p);
  }

  // After a passive-target epoch, make the target's private/public copies consistent and
  // ensure every origin has completed its own flush/unlock before the local load.
  void sync_after_passive(const mp1::mpi_communicator& c, const mp1::mpi_window& w) {
    mp1::barrier(c);
    mp1::lock_all(w);
    mp1::sync(w);
    mp1::unlock_all(w);
  }

  /// Poll MPI_Win_test until it reports completion; test may legitimately return false first.
  [[nodiscard]] bool wait_until_tested(const mp1::mpi_window& w) {
    for (int i = 0; i < 2000000; ++i) {
      if (mp1::test(w))
        return true;
    }
    return false;
  }

  // A real user-defined MPI_Op for the "unknown handle is rejected for RMA" regression.
  // NOLINTNEXTLINE(readability-non-const-parameter): MPI_User_function signature
  void user_add_int(void* in, void* inout, int* len, MPI_Datatype*) {
    const int* input = static_cast<const int*>(in);
    int* output = static_cast<int*>(inout);
    for (int i = 0; i < *len; ++i)
      output[i] += input[i];
  }

  /// Make an intercommunicator whose two groups partition MPI_COMM_WORLD.  Used only to prove
  /// that the window factories reject intercommunicators locally (a provider can crash inside
  /// MPI_Win_create for an intercommunicator, so this must never reach MPI).
  [[nodiscard]] mp1::communicator make_intercomm() {
    const mp1::mpi_communicator w = world();
    const int color = w.rank() % 2;
    mp1::communicator local = mp1::communicator::split(w, color, w.rank());
    const int remote_leader = (w.rank() % 2 == 0) ? 1 : 0;
    mp1::communicator inter =
        mp1::communicator::intercomm_create(local, 0, w, remote_leader, mp1::tag{7311});
    local.free();
    return inter;
  }

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(rma_fixture);

BOOST_AUTO_TEST_SUITE(rma_contract)

BOOST_AUTO_TEST_CASE(window_types_views_adopt_move_free) {
  static_assert(not std::is_base_of_v<mp1::mpi_window, mp1::window>,
                "the owner composes a view; it must not publicly derive from it");
  static_assert(std::is_convertible_v<const mp1::window&, mp1::mpi_window>,
                "an owner converts to a view by value");
  static_assert(not std::is_convertible_v<mp1::window&, mp1::mpi_window&>,
                "the view base must not be mutable through the owner");
  static_assert(std::is_copy_constructible_v<mp1::mpi_window>, "mpi_window is a copyable view");
  static_assert(std::is_copy_assignable_v<mp1::mpi_window>, "mpi_window is a copyable view");
  static_assert(not std::is_copy_constructible_v<mp1::window>, "window is the move-only owner");
  static_assert(not std::is_copy_assignable_v<mp1::window>, "window is the move-only owner");
  static_assert(std::is_move_constructible_v<mp1::window>, "window must be movable");
  static_assert(std::is_move_assignable_v<mp1::window>, "window must be movable");

  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  mp1::mpi_window invalid_view;
  mp1::window invalid_owner;
  BOOST_TEST(not invalid_view.is_valid());
  BOOST_TEST(not invalid_owner.is_valid());
  BOOST_TEST(invalid_view.native_handle() == MPI_WIN_NULL);
  BOOST_TEST(throws_invalid([&] { (void)invalid_view.base(); }));
  BOOST_TEST(throws_invalid([&] { (void)invalid_view.get_group(); }));
  invalid_owner.free();  // invalid free is a no-op

  mp1::window owner =
      mp1::window::allocate(c, {.size = 4 * sizeof(int), .disp_unit = sizeof(int)});
  BOOST_TEST(owner.is_valid());
  BOOST_TEST(owner.base() != nullptr);
  BOOST_TEST(owner.byte_size() == static_cast<MPI_Aint>(4 * sizeof(int)));
  BOOST_TEST(owner.disp_unit() == static_cast<int>(sizeof(int)));
  BOOST_TEST(owner.rank() == rank);
  BOOST_TEST(owner.group_size() == c.size());

  // A sliced view shares the handle and never frees it.
  mp1::mpi_window view = owner;
  BOOST_TEST(view.native_handle() == owner.native_handle());
  BOOST_TEST(view.byte_size() == owner.byte_size());

  mp1::window moved = std::move(owner);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not owner.is_valid());
  BOOST_TEST(moved.is_valid());
  BOOST_TEST(moved.native_handle() == view.native_handle());

  mp1::window second =
      mp1::window::allocate(c, {.size = sizeof(int), .disp_unit = sizeof(int)});
  second = std::move(moved);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not moved.is_valid());
  BOOST_TEST(second.is_valid());
  second.free();
  second.free();  // idempotent
  BOOST_TEST(not second.is_valid());

  // borrow never frees: the caller keeps ownership of the raw handle.
  {
    std::vector<int> buffer(4, 0);
    MPI_Win raw{MPI_WIN_NULL};
    BOOST_REQUIRE_EQUAL(
        (MPI_Win_create(buffer.data(), static_cast<MPI_Aint>(buffer.size() * sizeof(int)),
                        static_cast<int>(sizeof(int)), MPI_INFO_NULL, c.native_handle(), &raw)),
        MPI_SUCCESS);
    {
      mp1::mpi_window borrowed = mp1::mpi_window::borrow(raw);
      BOOST_TEST(borrowed.is_valid());
      BOOST_TEST(borrowed.base() == buffer.data());
      BOOST_TEST(borrowed.create_flavor() == MPI_WIN_FLAVOR_CREATE);
    }
    BOOST_REQUIRE_EQUAL(MPI_Win_free(&raw), MPI_SUCCESS);
    BOOST_TEST(raw == MPI_WIN_NULL);
  }

  // adopt takes ownership and frees exactly once.
  {
    std::vector<int> buffer(2, 0);
    MPI_Win raw{MPI_WIN_NULL};
    BOOST_REQUIRE_EQUAL(
        (MPI_Win_create(buffer.data(), static_cast<MPI_Aint>(buffer.size() * sizeof(int)),
                        static_cast<int>(sizeof(int)), MPI_INFO_NULL, c.native_handle(), &raw)),
        MPI_SUCCESS);
    mp1::window adopted = mp1::window::adopt(raw);
    BOOST_TEST(adopted.is_valid());
    BOOST_TEST(adopted.base() == buffer.data());
    adopted.free();
    BOOST_TEST(not adopted.is_valid());
  }

  BOOST_TEST(throws_invalid([] { (void)mp1::window::adopt(MPI_WIN_NULL); }));

  int object = 0;
  const MPI_Aint expected = mp1::address_of(object);
  MPI_Aint reference{0};
  BOOST_REQUIRE_EQUAL(MPI_Get_address(&object, &reference), MPI_SUCCESS);
  BOOST_TEST(expected == reference);
}

BOOST_AUTO_TEST_CASE(window_create_flavors_and_attributes) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  // Per-rank different size and disp_unit: MPI allows both.
  {
    const int elements = rank + 1;
    std::vector<int> buffer(static_cast<std::size_t>(elements), rank);
    mp1::window w =
        mp1::window::create(c, {.base = buffer.data(),
                                .size = static_cast<MPI_Aint>(elements * sizeof(int)),
                                .disp_unit = rank + 1});
    BOOST_TEST(w.is_valid());
    BOOST_TEST(w.base() == buffer.data());
    BOOST_TEST(w.byte_size() == static_cast<MPI_Aint>(elements * sizeof(int)));
    BOOST_TEST(w.disp_unit() == rank + 1);
    BOOST_TEST(w.create_flavor() == MPI_WIN_FLAVOR_CREATE);
    BOOST_TEST(is_unified_or_separate(w.memory_model()));
    BOOST_TEST(w.rank() == rank);
    BOOST_TEST(w.group_size() == size);
    mp1::group window_group = w.get_group();
    BOOST_TEST(window_group.size() == size);
    BOOST_TEST(window_group.rank() == rank);
    BOOST_TEST(static_cast<int>(window_group.compare(mp1::get_group(c))) ==
               static_cast<int>(mp1::equality_type::identical));
    mp1::group forwarder = mp1::get_group(w);
    BOOST_TEST(forwarder.size() == size);
    w.free();
  }

  // allocate: MPI owns the memory and frees it through MPI_Win_free.
  {
    mp1::window w =
        mp1::window::allocate(c, {.size = 8 * sizeof(double), .disp_unit = sizeof(double)});
    BOOST_TEST(w.base() != nullptr);
    BOOST_TEST(w.byte_size() == static_cast<MPI_Aint>(8 * sizeof(double)));
    BOOST_TEST(w.disp_unit() == static_cast<int>(sizeof(double)));
    BOOST_TEST(w.create_flavor() == MPI_WIN_FLAVOR_ALLOCATE);
    BOOST_TEST(is_unified_or_separate(w.memory_model()));
    w.free();
  }

  // Data movement is not attempted on zero-size windows (erroneous in MPI and crash-prone).
  {
    mp1::window w = mp1::window::create(c, {.base = nullptr, .size = 0, .disp_unit = 1});
    BOOST_TEST(w.byte_size() == 0);
    BOOST_TEST(w.disp_unit() == 1);
    w.free();
  }
  {
    mp1::window w = mp1::window::allocate(c, {.size = 0, .disp_unit = 1});
    BOOST_TEST(w.byte_size() == 0);
    w.free();
  }

  // dynamic: base MPI_BOTTOM, size 0, disp_unit 1.
  {
    mp1::window w = mp1::window::create_dynamic(c);
    BOOST_TEST(w.base() == MPI_BOTTOM);
    BOOST_TEST(w.byte_size() == 0);
    BOOST_TEST(w.disp_unit() == 1);
    BOOST_TEST(w.create_flavor() == MPI_WIN_FLAVOR_DYNAMIC);
    w.free();
  }

  // Local factory validation happens before the collective MPI call.
  {
    std::vector<int> buffer(4, 0);
    BOOST_TEST(throws_invalid(
        [&] { (void)mp1::window::create(c, {.base = nullptr, .size = 4, .disp_unit = 1}); }));
    BOOST_TEST(throws_invalid([&] {
      (void)mp1::window::create(c, {.base = buffer.data(), .size = -1, .disp_unit = 1});
    }));
    BOOST_TEST(throws_invalid([&] {
      (void)mp1::window::create(c, {.base = buffer.data(), .size = 4, .disp_unit = 0});
    }));
    BOOST_TEST(throws_invalid([&] {
      (void)mp1::window::create(c, {.base = buffer.data(), .size = 4, .disp_unit = -4});
    }));
    BOOST_TEST(
        throws_invalid([&] { (void)mp1::window::allocate(c, {.size = -1, .disp_unit = 1}); }));
    BOOST_TEST(
        throws_invalid([&] { (void)mp1::window::allocate(c, {.size = 4, .disp_unit = 0}); }));
    BOOST_TEST(throws_invalid(
        [&] { (void)mp1::window::allocate_shared(c, {.size = -1, .disp_unit = 1}); }));
    BOOST_TEST(throws_invalid(
        [&] { (void)mp1::window::allocate_shared(c, {.size = 4, .disp_unit = 0}); }));
    BOOST_TEST(throws_invalid([&] {
      const mp1::mpi_communicator invalid;
      (void)mp1::window::create_dynamic(invalid);
    }));
  }

  // Intercommunicator is rejected locally (never forwarded to MPI_Win_create).
  {
    mp1::communicator inter = make_intercomm();
    BOOST_TEST(inter.is_inter());
    std::vector<int> buffer(4, 0);
    BOOST_TEST(throws_invalid([&] {
      (void)mp1::window::create(inter, {.base = buffer.data(),
                                        .size = static_cast<MPI_Aint>(buffer.size()),
                                        .disp_unit = 1});
    }));
    BOOST_TEST(throws_invalid(
        [&] { (void)mp1::window::allocate(inter, {.size = 4, .disp_unit = 1}); }));
    BOOST_TEST(throws_invalid(
        [&] { (void)mp1::window::allocate_shared(inter, {.size = 4, .disp_unit = 1}); }));
    BOOST_TEST(throws_invalid([&] { (void)mp1::window::create_dynamic(inter); }));
    inter.free();
  }
}

BOOST_AUTO_TEST_CASE(window_option_bag_factories) {
  const mp1::mpi_communicator c = world();

  // The option bags are designated-initializer aggregates with the documented defaults.
  const mp1::window_create_options create_defaults;
  BOOST_TEST(create_defaults.base == nullptr);
  BOOST_TEST(create_defaults.size == 0);
  BOOST_TEST(create_defaults.disp_unit == 1);
  BOOST_TEST(create_defaults.info.native_handle() == MPI_INFO_NULL);

  const mp1::window_allocate_options allocate_defaults;
  BOOST_TEST(allocate_defaults.size == 0);
  BOOST_TEST(allocate_defaults.disp_unit == 1);
  BOOST_TEST(allocate_defaults.info.native_handle() == MPI_INFO_NULL);

  // A default (zero-size) payload is legal: no data movement is attempted on it.
  {
    mp1::window w = mp1::window::allocate(c, {});
    BOOST_TEST(w.is_valid());
    BOOST_TEST(w.byte_size() == 0);
    BOOST_TEST(w.disp_unit() == 1);
    w.free();
  }
  {
    mp1::window w = mp1::window::create(c, {});
    BOOST_TEST(w.is_valid());
    BOOST_TEST(w.byte_size() == 0);
    BOOST_TEST(w.disp_unit() == 1);
    w.free();
  }

  // A stored payload is reusable and passes the info hint through to MPI.
  {
    mp1::info hints;
    hints.set("same_size", "true");
    const mp1::window_create_options options{
        .base = nullptr, .size = 0, .disp_unit = 1, .info = hints};
    mp1::window w = mp1::window::create(c, options);
    BOOST_TEST(w.create_flavor() == MPI_WIN_FLAVOR_CREATE);
    w.free();
  }
  {
    mp1::info hints;
    hints.set("same_size", "true");
    const mp1::window_allocate_options options{
        .size = 2 * sizeof(int), .disp_unit = sizeof(int), .info = hints};
    mp1::window w = mp1::window::allocate(c, options);
    BOOST_TEST(w.is_valid());
    BOOST_TEST(w.base() != nullptr);
    BOOST_TEST(w.byte_size() == static_cast<MPI_Aint>(2 * sizeof(int)));
    BOOST_TEST(w.disp_unit() == static_cast<int>(sizeof(int)));
    BOOST_TEST(w.create_flavor() == MPI_WIN_FLAVOR_ALLOCATE);
    w.free();
  }

  // The designated-initializer spelling binds a temporary directly.
  {
    std::vector<int> buffer(2, 0);
    mp1::window w =
        mp1::window::create(c, {.base = buffer.data(),
                                .size = static_cast<MPI_Aint>(buffer.size() * sizeof(int)),
                                .disp_unit = sizeof(int)});
    BOOST_TEST(w.base() == buffer.data());
    BOOST_TEST(w.byte_size() == static_cast<MPI_Aint>(buffer.size() * sizeof(int)));
    w.free();
  }
}

BOOST_AUTO_TEST_CASE(window_info_accessors) {
  const mp1::mpi_communicator c = world();
  mp1::window w = mp1::window::allocate(c, {.size = 4 * sizeof(int), .disp_unit = sizeof(int)});

  mp1::info initial = w.get_info();
  BOOST_TEST(initial.size() >= 0);

  mp1::info hints;
  hints.set("same_size", "true");
  hints.set("same_disp_unit", "true");
  w.set_info(hints);
  mp1::info updated = w.get_info();
  BOOST_TEST(updated.size() >= 0);
  // Hints may be ignored by the implementation; only the call contract is checked.

  BOOST_TEST(throws_invalid([&] { w.set_info(mp1::info{}); }));

  w.free();
}

BOOST_AUTO_TEST_CASE(window_shared_query_flavors) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  // MPI 4.1: shared_query is legal on ALLOCATE (and CREATE), not only on shared windows.
  {
    mp1::window w =
        mp1::window::allocate(c, {.size = 4 * sizeof(int), .disp_unit = sizeof(int)});
    mp1::shared_segment self = w.shared_query(rank);
    BOOST_TEST(self.size >= 0);
    if (self.size > 0) {
      BOOST_TEST(self.base != nullptr);
      BOOST_TEST(self.disp_unit == static_cast<int>(sizeof(int)));
    }
    mp1::shared_segment any = w.shared_query(MPI_PROC_NULL);
    BOOST_TEST(any.size >= 0);
    BOOST_TEST(throws_invalid([&] { (void)w.shared_query(-3); }));
    BOOST_TEST(throws_invalid([&] { (void)w.shared_query(c.size()); }));
    w.free();
  }

  // CREATE windows may also be queried under MPI 4.1; no shared memory is required.
  {
    std::vector<int> buffer(4, rank);
    mp1::window w =
        mp1::window::create(c, {.base = buffer.data(),
                                .size = static_cast<MPI_Aint>(buffer.size() * sizeof(int)),
                                .disp_unit = static_cast<int>(sizeof(int))});
    mp1::shared_segment self = w.shared_query(rank);
    BOOST_TEST(self.size >= 0);
    w.free();
  }

  // DYNAMIC is the one flavor rejected by MPI 4.1; reject it locally (OpenMPI is too lenient).
  {
    mp1::window w = mp1::window::create_dynamic(c);
    BOOST_TEST(throws_invalid([&] { (void)w.shared_query(0); }));
    BOOST_TEST(throws_invalid([&] { (void)w.shared_query(MPI_PROC_NULL); }));
    w.free();
  }
}

BOOST_AUTO_TEST_CASE(window_dynamic_attach_detach_and_rma) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  mp1::window w = mp1::window::create_dynamic(c);
  BOOST_TEST(w.create_flavor() == MPI_WIN_FLAVOR_DYNAMIC);

  // Non-dynamic windows reject attach/detach before MPI.
  {
    mp1::window flat =
        mp1::window::allocate(c, {.size = sizeof(int), .disp_unit = sizeof(int)});
    int value = 0;
    BOOST_TEST(throws_invalid([&] { flat.attach(&value, sizeof(int)); }));
    BOOST_TEST(throws_invalid([&] { flat.detach(&value); }));
    flat.free();
  }

  int value = 0;
  BOOST_TEST(throws_invalid([&] { w.attach(&value, -1); }));
  BOOST_TEST(throws_invalid([&] { w.attach(&value, 0); }));
  BOOST_TEST(throws_invalid([&] { w.attach(nullptr, 0); }));
  BOOST_TEST(throws_invalid([&] { w.attach(nullptr, sizeof(int)); }));
  BOOST_TEST(throws_invalid([&] { w.detach(nullptr); }));

  std::vector<int> buffer(4, -1);
  w.attach(buffer.data(), static_cast<MPI_Aint>(buffer.size() * sizeof(int)));

  // Pair even origin ranks with the following odd target rank.  The target sends the address
  // of its attached buffer; the origin uses it as the dynamic target_disp (bytes).
  if (rank % 2 == 0) {
    MPI_Aint address{0};
    MPI_Recv(&address, 1, MPI_AINT, rank + 1, 7401, c.native_handle(), MPI_STATUS_IGNORE);
    const int payload = 900 + rank;
    mp1::lock(w, mp1::lock_type::exclusive, rank + 1);
    mp1::put(w, mp1::span_from(&payload, 1), rank + 1, address, 1);
    mp1::unlock(w, rank + 1);
  } else {
    const MPI_Aint address = mp1::address_of(buffer[0]);
    MPI_Send(&address, 1, MPI_AINT, rank - 1, 7401, c.native_handle());
  }

  mp1::barrier(c);
  mp1::lock_all(w);
  mp1::sync(w);
  mp1::unlock_all(w);
  if (rank % 2 == 1) {
    BOOST_TEST(window_load(buffer.data()) == 900 + rank - 1);
  }

  w.detach(buffer.data());
  w.free();
}

BOOST_AUTO_TEST_CASE(fence_put_get) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  constexpr mp1::count_t n = 8;
  mp1::window w = mp1::window::allocate(
      c, {.size = static_cast<MPI_Aint>(n * sizeof(int)), .disp_unit = sizeof(int)});
  int* local = static_cast<int*>(w.base());
  for (mp1::count_t i = 0; i < n; ++i)
    local[i] = rank * 100 + static_cast<int>(i);

  mp1::fence(w);
  const int send[2] = {1000, 1001};
  if (rank == 0)
    mp1::put(w, mp1::span_from(send, 2), 1, 3, 2);  // 2 ints at rank 1's element 3
  mp1::fence(w);
  if (rank == 1) {
    BOOST_TEST(window_load(local + 3) == 1000);
    BOOST_TEST(window_load(local + 4) == 1001);
    BOOST_TEST(window_load(local + 5) == 105);
  }

  int received[2] = {-1, -1};
  if (rank == 0)
    mp1::get(w, mp1::span_from(received, 2), 1, 3, 2);
  mp1::fence(w);
  if (rank == 0) {
    BOOST_TEST(window_load(received) == 1000);
    BOOST_TEST(window_load(received + 1) == 1001);
  }

  // MPI_PROC_NULL is a valid target rank for every RMA communication call and is a no-op
  // (MPI 4.1 §12.3 Communication Calls); the epoch must still be closed by fence.
  received[0] = -7;
  received[1] = -8;
  mp1::fence(w);
  mp1::put(w, mp1::span_from(send, 2), MPI_PROC_NULL, 0, 2);
  mp1::get(w, mp1::span_from(received, 2), MPI_PROC_NULL, 0, 2);
  // The ignored payload is not validated: an empty buffer, a bogus count and even a negative
  // displacement on this non-dynamic window are accepted because MPI ignores them.
  mp1::put(w, std::span<const float>{}, MPI_PROC_NULL, -12345, 1234);
  mp1::get(w, std::span<float>{}, MPI_PROC_NULL, -12345, 1234);
  mp1::fence(w);
  BOOST_TEST(window_load(received) == -7);
  BOOST_TEST(window_load(received + 1) == -8);

  // Every other out-of-range rank (and lock/flush/...) still throws before MPI.
  BOOST_TEST(throws_invalid([&] { mp1::put(w, mp1::span_from(send, 1), -1000, 0, 1); }));
  BOOST_TEST(throws_invalid([&] { mp1::get(w, mp1::span_from(received, 1), -1000, 0, 1); }));

  w.free();

  // MPI_MODE_NOPRECEDE is legal on a fresh window with no preceding epoch.
  {
    mp1::window fresh = mp1::window::allocate(
        c, {.size = static_cast<MPI_Aint>(4 * sizeof(int)), .disp_unit = sizeof(int)});
    *static_cast<int*>(fresh.base()) = rank;
    mp1::fence(fresh, MPI_MODE_NOPRECEDE);
    const int payload = 55;
    if (rank == 0)
      mp1::put(fresh, mp1::span_from(&payload, 1), 1, 0, 1);
    mp1::fence(fresh);
    if (rank == 1)
      BOOST_TEST(window_load(static_cast<int*>(fresh.base())) == 55);
    fresh.free();
  }
}

BOOST_AUTO_TEST_CASE(lock_accumulate) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  mp1::window w = mp1::window::allocate(c, {.size = sizeof(int), .disp_unit = sizeof(int)});
  int* counter = static_cast<int*>(w.base());
  if (rank == 0)
    *counter = 0;
  mp1::fence(w);  // publish the initial store

  const int one = 1;
  for (int target = 0; target < size; ++target) {
    // Every rank touches target 0 only; this is a regular lock/unlock pair per origin.
    mp1::lock(w, mp1::lock_type::exclusive, 0);
    mp1::accumulate(w, mp1::span_from(&one, 1), 0, 0, 1, MPI_SUM);
    mp1::unlock(w, 0);
  }
  sync_after_passive(c, w);
  if (rank == 0)
    BOOST_TEST(window_load(counter) == size * size);

  // Shared lock plus get: target completion is guaranteed by unlock.
  mp1::lock(w, mp1::lock_type::shared, 0);
  int observed = -1;
  mp1::get(w, mp1::span_from(&observed, 1), 0, 0, 1);
  mp1::unlock(w, 0);
  mp1::barrier(c);
  BOOST_TEST(window_load(&observed) == size * size);

  // Locked target ranks must be valid window ranks.
  BOOST_TEST(throws_invalid([&] { mp1::lock(w, mp1::lock_type::exclusive, size); }));
  BOOST_TEST(throws_invalid([&] { mp1::lock(w, mp1::lock_type::exclusive, MPI_PROC_NULL); }));
  BOOST_TEST(throws_invalid([&] { mp1::lock(w, static_cast<mp1::lock_type>(1234), 0); }));
  BOOST_TEST(throws_invalid([&] { mp1::unlock(w, size); }));
  BOOST_TEST(
      throws_invalid([&] { mp1::accumulate(w, mp1::span_from(&one, 1), 0, -1, 1, MPI_SUM); }));

  w.free();
}

BOOST_AUTO_TEST_CASE(accumulate_builtin_and_standard_operators) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  constexpr mp1::count_t n = 4;
  mp1::window w = mp1::window::allocate(
      c, {.size = static_cast<MPI_Aint>(n * sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  if (rank == 0) {
    target[0] = 0;
    target[1] = 1;
    target[2] = 1;
    target[3] = 0;
  }
  mp1::fence(w);  // publish the initial values before the passive epoch

  mp1::lock_all(w);
  mp1::barrier(c);

  const int one = 1;
  const int two = 2;
  int old = -1;

  // Blocking accumulate with std::plus<int> and mp1::max<int>.
  mp1::accumulate(w, mp1::span_from(&one, 1), 0, 0, 1, std::plus<int>{});
  mp1::accumulate(w, mp1::span_from(&two, 1), 0, 0, 1, std::plus<int>{});
  mp1::accumulate(w, mp1::span_from(&rank, 1), 0, 2, 1, mp1::max<int>{});

  // get_accumulate and fetch_and_op with standard/built-in functors.
  mp1::get_accumulate(w, mp1::span_from(&one, 1), mp1::span_from(&old, 1), 0, 3, 1,
                      std::plus<int>{});
  int fetched = -1;
  mp1::fetch_and_op(w, &two, &fetched, 0, 3, std::plus<int>{});

  // Request-based accumulate and get_accumulate with standard functors.
  int rget_old = -1;
  mp1::request acc_request =
      mp1::raccumulate(w, mp1::span_from(&one, 1), 0, 0, 1, std::plus<int>{});
  mp1::request get_acc_request =
      mp1::rget_accumulate(w, mp1::span_from(&two, 1), mp1::span_from(&rget_old, 1), 0, 1, 1,
                           std::multiplies<int>{});
  acc_request.wait();
  get_acc_request.wait();

  mp1::flush_all(w);
  mp1::barrier(c);
  mp1::unlock_all(w);

  sync_after_passive(c, w);
  if (rank == 0) {
    BOOST_TEST(window_load(target + 0) == 4 * size);
    BOOST_TEST(window_load(target + 1) == (1 << size));
    BOOST_TEST(window_load(target + 2) == size - 1);
    BOOST_TEST(window_load(target + 3) == 3 * size);
  }

  w.free();
}

BOOST_AUTO_TEST_CASE(lock_all_flush_sync) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  mp1::window w = mp1::window::allocate(c, {.size = 2 * sizeof(int), .disp_unit = sizeof(int)});
  int* local = static_cast<int*>(w.base());
  local[0] = -1;
  local[1] = -1;
  mp1::lock_all(w);
  mp1::barrier(c);  // every rank is inside the passive epoch before any RMA is issued

  const int next = (rank + 1) % size;
  int first = 100 + rank;
  mp1::put(w, mp1::span_from(&first, 1), next, 0, 1);
  mp1::flush(w, next);
  int second = 200 + rank;
  mp1::put(w, mp1::span_from(&second, 1), next, 1, 1);
  mp1::flush_local(w, next);
  // The origin buffer may be reused after flush_local; the target may not see the new value.
  second = -5;

  mp1::flush_all(w);
  mp1::flush_local_all(w);
  mp1::sync(w);
  mp1::barrier(c);  // all origin-side flushes are complete before any target leaves its epoch
  mp1::unlock_all(w);

  // Re-acquire a passive epoch and sync after every origin has completed its flush; this is
  // the portable way to read local window memory after a passive epoch in the separate model.
  sync_after_passive(c, w);
  const int previous = (rank + size - 1) % size;
  BOOST_TEST(window_load(local) == 100 + previous);
  BOOST_TEST(window_load(local + 1) == 200 + previous);

  BOOST_TEST(throws_invalid([&] { mp1::flush(w, size); }));
  BOOST_TEST(throws_invalid([&] { mp1::flush_local(w, MPI_PROC_NULL); }));

  w.free();
}

BOOST_AUTO_TEST_CASE(pscw_post_start_complete_wait_test) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  mp1::window w = mp1::window::allocate(c, {.size = sizeof(int), .disp_unit = sizeof(int)});
  int* local = static_cast<int*>(w.base());
  *local = 0;
  mp1::group all = w.get_group();

  // Round 1: wait().
  mp1::post(w, all);
  mp1::start(w, all);
  const int five = 5;
  if (rank == 0)
    mp1::put(w, mp1::span_from(&five, 1), 1, 0, 1);
  mp1::complete(w);
  mp1::wait(w);
  mp1::barrier(c);
  if (rank == 1)
    BOOST_TEST(window_load(local) == 5);

  // Round 2: test() until the exposure epoch completes.
  mp1::post(w, all);
  mp1::start(w, all);
  const int six = 6;
  if (rank == 0)
    mp1::put(w, mp1::span_from(&six, 1), 1, 0, 1);
  mp1::complete(w);
  BOOST_TEST(wait_until_tested(w));
  mp1::barrier(c);
  if (rank == 1)
    BOOST_TEST(window_load(local) == 6);

  // Invalid groups are rejected before MPI.
  mp1::group moved = std::move(all);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(all.is_valid() == false);
  BOOST_TEST(throws_invalid([&] { mp1::post(w, all); }));
  BOOST_TEST(throws_invalid([&] { mp1::start(w, all); }));

  w.free();
}

BOOST_AUTO_TEST_CASE(four_origin_target_shapes) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  constexpr mp1::count_t n = 64;
  mp1::window w = mp1::window::allocate(
      c, {.size = static_cast<MPI_Aint>(n * sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (mp1::count_t i = 0; i < n; ++i)
    target[i] = -1;

  mp1::fence(w);
  // MPI_Put is nonblocking: the origin buffers and the datatype layouts stay borrowed until
  // the closing fence completes the access epoch (MPI 4.1 §12.5.3), so they are declared
  // outside the origin rank's block.
  const int flat[3] = {1, 2, 3};
  // flat origin -> layout target: elements 0 and 4 of the layout are elements 8 and 12 of the
  // target buffer (disp 8, disp_unit sizeof(int), stride 4 ints).
  const int flat_to_layout[2] = {4, 5};
  const mp1::layout<int> target_vector = mp1::layout<int>::vector(2, 1, 4);
  // layout origin -> flat target
  const int layout_origin[3] = {6, 99, 7};
  const mp1::layout<int> origin_vector = mp1::layout<int>::vector(2, 1, 2);
  // layout origin -> layout target
  const int layout_to_layout[3] = {8, 99, 9};
  if (rank == 0) {
    mp1::put(w, mp1::span_from(flat, 3), 1, 2, 3);
    mp1::put(w, std::span{flat_to_layout}, 1, 8, target_vector);
    mp1::put(w, mp1::region{layout_origin, origin_vector}, 1, 16, 2);
    mp1::put(w, mp1::region{layout_to_layout, origin_vector}, 1, 24, target_vector);
  }
  mp1::fence(w);
  if (rank == 1) {
    BOOST_TEST(window_load(target + 2) == 1);
    BOOST_TEST(window_load(target + 3) == 2);
    BOOST_TEST(window_load(target + 4) == 3);
    BOOST_TEST(window_load(target + 8) == 4);
    BOOST_TEST(window_load(target + 12) == 5);
    BOOST_TEST(window_load(target + 16) == 6);
    BOOST_TEST(window_load(target + 17) == 7);
    BOOST_TEST(window_load(target + 24) == 8);
    BOOST_TEST(window_load(target + 28) == 9);
  }
  mp1::barrier(c);

  // Prepare a known region for the get shapes (disjoint from the put region).
  if (rank == 1) {
    for (mp1::count_t i = 32; i < n; ++i)
      target[i] = 200 + static_cast<int>(i);
  }

  int got_flat[2] = {-1, -1};
  int got_layout[2] = {-1, -1};
  int got_origin_layout[4] = {-1, -1, -1, -1};
  int got_both[3] = {-1, -1, -1};
  mp1::fence(w);
  // MPI_Get is nonblocking too: the result buffers and the datatype layouts stay borrowed
  // until the closing fence (MPI 4.1 §12.5.3).
  const mp1::layout<int> get_target_vector = mp1::layout<int>::vector(2, 1, 4);
  const mp1::layout<int> get_origin_vector = mp1::layout<int>::vector(2, 1, 3);
  const mp1::layout<int> get_origin_pair = mp1::layout<int>::vector(2, 1, 2);
  if (rank == 0) {
    mp1::get(w, mp1::span_from(got_flat, 2), 1, 32, 2);
    mp1::get(w, std::span{got_layout}, 1, 40, get_target_vector);
    mp1::get(w, mp1::region{got_origin_layout, get_origin_vector}, 1, 32, 2);
    mp1::get(w, mp1::region{got_both, get_origin_pair}, 1, 48, get_target_vector);
  }
  mp1::fence(w);  // get data is valid only after this synchronization

  if (rank == 0) {
    BOOST_TEST(window_load(got_flat + 0) == 232);
    BOOST_TEST(window_load(got_flat + 1) == 233);
    BOOST_TEST(window_load(got_layout + 0) == 240);
    BOOST_TEST(window_load(got_layout + 1) == 244);
    BOOST_TEST(window_load(got_origin_layout + 0) == 232);
    BOOST_TEST(window_load(got_origin_layout + 3) == 233);
    BOOST_TEST(window_load(got_both + 0) == 248);
    BOOST_TEST(window_load(got_both + 2) == 252);
  }

  w.free();
}

BOOST_AUTO_TEST_CASE(get_accumulate_fetch_and_op_compare_and_swap) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  constexpr mp1::count_t n = 4;
  mp1::window w = mp1::window::allocate(
      c, {.size = static_cast<MPI_Aint>(n * sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (mp1::count_t i = 0; i < n; ++i)
    target[i] = 0;
  if (rank == 0)
    target[0] = 10;
  mp1::fence(w);  // publish the initial values before the passive epoch

  // get_accumulate: every rank adds 1.  The result is an origin-side output whose completion
  // point is the next synchronization call, not the function return (MPI 4.1 12.7 rule 1);
  // check it only after unlock.
  mp1::lock(w, mp1::lock_type::exclusive, 0);
  const int one = 1;
  int old = -1;
  mp1::get_accumulate(w, mp1::span_from(&one, 1), mp1::span_from(&old, 1), 0, 0, 1, MPI_SUM);

  // fetch_and_op: every rank adds 2.
  int fetched = -1;
  const int two = 2;
  mp1::fetch_and_op(w, &two, &fetched, 0, 0, MPI_SUM);
  mp1::unlock(w, 0);

  // Ranks interleave get_accumulate and fetch_and_op, so a snapshot can already include some
  // of both families.  Only the total is deterministic.
  BOOST_TEST(window_load(&old) >= 10);
  BOOST_TEST(window_load(&old) < 10 + 3 * size);
  BOOST_TEST(window_load(&fetched) >= 10);
  BOOST_TEST(window_load(&fetched) < 10 + 3 * size);

  sync_after_passive(c, w);
  if (rank == 0)
    BOOST_TEST(window_load(target) == 10 + 3 * size);

  // compare_and_swap: only rank 0 performs it, with the now-known value.
  int cas_old = -1;
  const int expected_value = 10 + 3 * size;
  if (rank == 0) {
    mp1::lock(w, mp1::lock_type::exclusive, 0);
    const int replacement = 100;
    mp1::compare_and_swap(w, &replacement, &expected_value, &cas_old, 0, 0);
    mp1::unlock(w, 0);
    BOOST_TEST(window_load(&cas_old) == expected_value);
  }
  sync_after_passive(c, w);
  if (rank == 0)
    BOOST_TEST(window_load(target) == 100);

  // MPI_NO_OP forms: origin is ignored; the old value is returned unchanged.
  int no_op_old = -1;
  const int ignored = 12345;
  mp1::lock(w, mp1::lock_type::shared, 0);
  mp1::fetch_and_op(w, &ignored, &no_op_old, 0, 0, MPI_NO_OP);
  // MPI_NO_OP ignores the origin, but span_from rejects a null pointer with a positive
  // count; a real (ignored) buffer satisfies the checked pointer+count contract.
  mp1::get_accumulate(w, mp1::span_from(&ignored, 1), mp1::span_from(&no_op_old, 1), 0, 0, 1,
                      MPI_NO_OP);
  mp1::unlock(w, 0);
  BOOST_TEST(window_load(&no_op_old) == 100);

  // MPI_REPLACE replaces the target value.
  int replaced_old = -1;
  if (rank == 0) {
    mp1::lock(w, mp1::lock_type::exclusive, 0);
    const int seven = 7;
    mp1::fetch_and_op(w, &seven, &replaced_old, 0, 1, MPI_REPLACE);
    mp1::unlock(w, 0);
    BOOST_TEST(window_load(&replaced_old) == 0);
  }
  sync_after_passive(c, w);
  if (rank == 0)
    BOOST_TEST(window_load(target + 1) == 7);

  w.free();
}

BOOST_AUTO_TEST_CASE(request_based_rma_and_cancel) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  constexpr mp1::count_t n = 4;
  mp1::window w = mp1::window::allocate(
      c, {.size = static_cast<MPI_Aint>(n * sizeof(int)), .disp_unit = sizeof(int)});
  int* local = static_cast<int*>(w.base());
  for (mp1::count_t i = 0; i < n; ++i)
    local[i] = rank * 10 + static_cast<int>(i);
  local[2] = 0;
  local[3] = 0;
  mp1::fence(w);

  const int next = (rank + 1) % size;

  // Request-based RMA is legal only inside a passive-target epoch.
  mp1::lock_all(w);
  mp1::barrier(c);

  // RMA requests must be reaped by wait/test; cancel() is rejected.
  mp1::request cancelled = mp1::rput(w, mp1::span_from(&local[0], 1), next, 0, 1);
  BOOST_TEST(not cancelled.is_persistent());
  BOOST_TEST(throws_logic([&] { cancelled.cancel(); }));
  cancelled.wait();

  // The request is no longer active; cancel on a completed request is a no-op.
  cancelled.cancel();

  // rget: the result is usable once wait() returns (origin-side completion).
  int got = -1;
  mp1::request get_request = mp1::rget(w, mp1::span_from(&got, 1), next, 1, 1);
  get_request.wait();
  BOOST_TEST(window_load(&got) == next * 10 + 1);

  // raccumulate and rget_accumulate to rank 0.
  const int one = 1;
  mp1::request acc_request = mp1::raccumulate(w, mp1::span_from(&one, 1), 0, 2, 1, MPI_SUM);
  acc_request.wait();
  int old = -1;
  mp1::request get_acc_request = mp1::rget_accumulate(
      w, mp1::span_from(&one, 1), mp1::span_from(&old, 1), 0, 3, 1, MPI_SUM);
  get_acc_request.wait();

  // Mixed vector completion: two RMA requests in one wait_all is legal.
  std::vector<mp1::request> requests;
  int payload = 700 + rank;
  requests.push_back(mp1::rput(w, mp1::span_from(&payload, 1), next, 0, 1));
  int read_back = -1;
  requests.push_back(mp1::rget(w, mp1::span_from(&read_back, 1), next, 1, 1));
  const std::vector<mp1::status> statuses = mp1::wait_all(requests);
  BOOST_TEST(statuses.size() == 2);
  BOOST_TEST(window_load(&read_back) == next * 10 + 1);

  mp1::flush_all(w);
  mp1::barrier(c);
  mp1::unlock_all(w);

  // Target completion after unlock_all; re-sync before the local load.
  sync_after_passive(c, w);
  if (rank == 0) {
    BOOST_TEST(window_load(local + 2) == size);
    BOOST_TEST(window_load(local + 3) == size);
  }

  w.free();
}

BOOST_AUTO_TEST_CASE(accumulate_predefined_complex_bool_byte) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  // Complex is a first-class SUM/PROD element (MPI 4.1 §6.9.2).
  {
    using complex_type = std::complex<double>;
    mp1::window w =
        mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(2 * sizeof(complex_type)),
                                  .disp_unit = sizeof(complex_type)});
    complex_type* target = static_cast<complex_type*>(w.base());
    if (rank == 0) {
      target[0] = complex_type{1.0, 2.0};
      target[1] = complex_type{3.0, 4.0};
    }
    mp1::fence(w);
    mp1::lock_all(w);
    mp1::barrier(c);
    const complex_type addend{1.0, 2.0};
    mp1::accumulate(w, mp1::span_from(&addend, 1), 0, 0, 1, std::plus<complex_type>{});
    if (rank == 0) {
      const complex_type factor{1.0, 2.0};
      mp1::accumulate(w, mp1::span_from(&factor, 1), 0, 1, 1, std::multiplies<complex_type>{});
    }
    mp1::flush_all(w);
    mp1::barrier(c);
    mp1::unlock_all(w);
    sync_after_passive(c, w);
    if (rank == 0) {
      const complex_type expected_sum{1.0 + static_cast<double>(size),
                                      2.0 + 2.0 * static_cast<double>(size)};
      BOOST_TEST(target[0] == expected_sum);
      const complex_type expected_product{-5.0, 10.0};  // (3+4i)(1+2i)
      BOOST_TEST(target[1] == expected_product);
    }
    w.free();
  }

  // bool is the Logical category: LAND/LOR map exactly to MPI_CXX_BOOL (MPI 4.1 §6.9.2).
  {
    mp1::window w = mp1::window::allocate(
        c, {.size = static_cast<MPI_Aint>(2 * sizeof(bool)), .disp_unit = sizeof(bool)});
    bool* target = static_cast<bool*>(w.base());
    if (rank == 0) {
      target[0] = false;
      target[1] = false;
    }
    mp1::fence(w);
    mp1::lock_all(w);
    mp1::barrier(c);
    const bool truth = true;
    mp1::accumulate(w, mp1::span_from(&truth, 1), 0, 0, 1, std::logical_and<bool>{});
    mp1::accumulate(w, mp1::span_from(&truth, 1), 0, 1, 1, std::logical_or<bool>{});
    mp1::flush_all(w);
    mp1::barrier(c);
    mp1::unlock_all(w);
    sync_after_passive(c, w);
    if (rank == 0) {
      BOOST_TEST(not target[0]);
      BOOST_TEST(target[1]);
    }
    w.free();
  }

  // std::byte is the Byte category: BAND/BOR/BXOR map to the byte operation (§6.9.2).
  {
    mp1::window w = mp1::window::allocate(
        c,
        {.size = static_cast<MPI_Aint>(2 * sizeof(std::byte)), .disp_unit = sizeof(std::byte)});
    std::byte* target = static_cast<std::byte*>(w.base());
    if (rank == 0) {
      target[0] = static_cast<std::byte>(0x0f);
      target[1] = static_cast<std::byte>(0xf0);
    }
    mp1::fence(w);
    mp1::lock_all(w);
    mp1::barrier(c);
    const std::byte high = static_cast<std::byte>(0xf0);
    const std::byte low = static_cast<std::byte>(0x0f);
    mp1::accumulate(w, mp1::span_from(&high, 1), 0, 0, 1, std::bit_or<std::byte>{});
    mp1::accumulate(w, mp1::span_from(&low, 1), 0, 1, 1, std::bit_and<std::byte>{});
    mp1::flush_all(w);
    mp1::barrier(c);
    mp1::unlock_all(w);
    sync_after_passive(c, w);
    if (rank == 0) {
      BOOST_TEST(static_cast<unsigned>(target[0]) == 0xffu);
      BOOST_TEST(static_cast<unsigned>(target[1]) == 0x00u);
    }
    w.free();
  }
}

BOOST_AUTO_TEST_CASE(byte_category_rejects_non_bitwise_operations) {
  // Regression: std::byte is `enum class byte : unsigned char`.  Classifying it through the
  // enum branch first mapped it to its underlying unsigned char (the §6.9.2 C integer
  // category) even though its MPI datatype is MPI_BYTE (the Byte category), so MPI_SUM,
  // MPI_PROD, MPI_MAX/MIN and the logical ops were accepted and sent undefined operations
  // to MPI.  §6.9.2 defines only BAND/BOR/BXOR for Byte.
  static_assert(
      mp1::detail::static_rma_category<std::byte>() == mp1::detail::rma_datatype_category::byte,
      "std::byte maps to MPI_BYTE: the Byte category, not its C integer underlying "
      "type");
  enum class rma_enum_uchar : unsigned char { first, second };
  static_assert(mp1::detail::static_rma_category<rma_enum_uchar>() ==
                    mp1::detail::rma_datatype_category::c_integer,
                "a plain enum still follows its underlying type's C integer datatype");

  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  mp1::window w = mp1::window::allocate(
      c,
      {.size = static_cast<MPI_Aint>(2 * sizeof(std::byte)), .disp_unit = sizeof(std::byte)});
  std::byte* target = static_cast<std::byte*>(w.base());
  if (rank == 0) {
    target[0] = std::byte{0x00};
    target[1] = std::byte{0x0f};
  }

  const std::byte origin = std::byte{0x0f};
  std::byte result = std::byte{0x00};

  // The raw-handle gate and every typed call reject the ops that
  // §6.9.2 does not define for Byte, before any MPI call.
  BOOST_TEST(throws_invalid([&] {
    mp1::predefined_op<std::byte> op{MPI_SUM};
    (void)op;
  }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_SUM); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_PROD); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_MAX); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_MIN); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_LAND); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_LOR); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_LXOR); }));
  BOOST_TEST(throws_invalid([&] { mp1::fetch_and_op(w, &origin, &result, 0, 0, MPI_SUM); }));
  BOOST_TEST(throws_invalid([&] { mp1::fetch_and_op(w, &origin, &result, 0, 0, MPI_LAND); }));
  BOOST_TEST(throws_invalid([&] {
    mp1::get_accumulate(w, mp1::span_from(&origin, 1), mp1::span_from(&result, 1), 0, 0, 1,
                        MPI_MAX);
  }));

  // MPI_PROC_NULL no-ops still build the predefined_op<T> first, so the gate cannot be
  // bypassed by using an ignored target rank.
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), MPI_PROC_NULL, 0, 1, MPI_SUM); }));

  // The Byte rows of §6.9.2 still reach MPI, and so do the family-wide REPLACE/NO_OP.
  mp1::fence(w);
  mp1::lock_all(w);
  mp1::barrier(c);
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_BOR);   // 0x00 | 0x0f = 0x0f
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 1, 1, MPI_BAND);  // 0x0f & 0x0f = 0x0f
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_REPLACE);
  mp1::fetch_and_op(w, &origin, &result, 0, 1, MPI_NO_OP);  // target unchanged, old value out
  mp1::flush_all(w);
  mp1::barrier(c);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  if (rank == 0) {
    BOOST_TEST(static_cast<unsigned>(target[0]) == 0x0fu);
    BOOST_TEST(static_cast<unsigned>(target[1]) == 0x0fu);
  }
  BOOST_TEST(static_cast<unsigned>(result) == 0x0fu);
  w.free();
}

BOOST_AUTO_TEST_CASE(atomic_op_and_datatype_rejections) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  mp1::window w = mp1::window::allocate(c, {.size = 2 * sizeof(int), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  if (rank == 0) {
    target[0] = 7;
    target[1] = 11;
  }
  mp1::fence(w);

  const int one = 1;
  int old = -1;
  int cas_old = -1;

  // A user-created MPI_Op is not a predefined handle: RMA rejects it before the MPI call
  // (MPI 4.1 §12.3.4: "User-defined operators cannot be used").
  MPI_Op user_op = MPI_OP_NULL;
  BOOST_REQUIRE_EQUAL(MPI_Op_create(&user_add_int, 1, &user_op), MPI_SUCCESS);
  BOOST_TEST(
      throws_invalid([&] { mp1::accumulate(w, mp1::span_from(&one, 1), 0, 0, 1, user_op); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::raccumulate(w, mp1::span_from(&one, 1), 0, 0, 1, user_op).wait(); }));
  BOOST_REQUIRE_EQUAL(MPI_Op_free(&user_op), MPI_SUCCESS);

  // Family gating and the §6.9.2 table.
  BOOST_TEST(
      throws_invalid([&] { mp1::accumulate(w, mp1::span_from(&one, 1), 0, 0, 1, MPI_NO_OP); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&one, 1), 0, 0, 1, MPI_MAXLOC); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&one, 1), 0, 0, 1, MPI_MINLOC); }));
  BOOST_TEST(throws_invalid([&] { mp1::fetch_and_op(w, &one, &old, 0, 0, MPI_MAXLOC); }));

  // MPI_MAX is not defined for complex; MPI_BAND is not defined for floating point.
  {
    using complex_type = std::complex<double>;
    mp1::window cw =
        mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(sizeof(complex_type)),
                                  .disp_unit = sizeof(complex_type)});
    complex_type z{1.0, 2.0};
    complex_type zr{};
    BOOST_TEST(
        throws_invalid([&] { mp1::accumulate(cw, mp1::span_from(&z, 1), 0, 0, 1, MPI_MAX); }));
    BOOST_TEST(throws_invalid([&] { mp1::fetch_and_op(cw, &z, &zr, 0, 0, MPI_MAX); }));
    cw.free();
  }
  {
    mp1::window dw =
        mp1::window::allocate(c, {.size = sizeof(double), .disp_unit = sizeof(double)});
    const double d = 1.0;
    double dr = 0.0;
    BOOST_TEST(
        throws_invalid([&] { mp1::accumulate(dw, mp1::span_from(&d, 1), 0, 0, 1, MPI_BAND); }));
    BOOST_TEST(throws_invalid([&] { mp1::fetch_and_op(dw, &d, &dr, 0, 0, MPI_BAND); }));
    dw.free();
  }
  {
    mp1::window bw =
        mp1::window::allocate(c, {.size = sizeof(bool), .disp_unit = sizeof(bool)});
    bool flag = true;
    BOOST_TEST(throws_invalid(
        [&] { mp1::accumulate(bw, mp1::span_from(&flag, 1), 0, 0, 1, MPI_SUM); }));
    bw.free();
  }
  {
    mp1::window chw =
        mp1::window::allocate(c, {.size = sizeof(char), .disp_unit = sizeof(char)});
    char ch = 'a';
    BOOST_TEST(throws_invalid(
        [&] { mp1::accumulate(chw, mp1::span_from(&ch, 1), 0, 0, 1, MPI_SUM); }));
    // The functor-built predefined_op<char>{std::plus<char>{}} spelling is not statically
    // rejected (the functor maps to MPI_SUM at construction); the runtime op table rejects
    // it.  That shape is a runtime rejection, not a compile-time one.
    mp1::fence(chw);
    mp1::accumulate(chw, mp1::span_from(&ch, 1), 0, 0, 1,
                    MPI_REPLACE);  // REPLACE is defined for every datatype
    mp1::fence(chw);
    chw.free();
  }

  // None of the rejected calls may have touched the target.
  sync_after_passive(c, w);
  if (rank == 0) {
    BOOST_TEST(window_load(target) == 7);
    BOOST_TEST(window_load(target + 1) == 11);
  }
  (void)cas_old;
  w.free();
}

BOOST_AUTO_TEST_CASE(overlapping_layouts_rejected_for_updated_positions) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  mp1::window w = mp1::window::allocate(
      c, {.size = static_cast<MPI_Aint>(6 * sizeof(int)), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  for (int i = 0; i < 6; ++i)
    target[i] = i;
  mp1::fence(w);

  mp1::indexed_parameter overlap_parameter;
  overlap_parameter.add(2, 0);
  overlap_parameter.add(2, 1);
  const mp1::layout<int> overlap_indexed = mp1::layout<int>::indexed(overlap_parameter);
  const mp1::layout<int> overlap_vector = mp1::layout<int>::vector(2, 2, 1);
  mp1::indexed_parameter disjoint_parameter;
  disjoint_parameter.add(1, 2);
  disjoint_parameter.add(1, 0);
  const mp1::layout<int> disjoint_unsorted = mp1::layout<int>::indexed(disjoint_parameter);
  BOOST_TEST(overlap_indexed.has_overlapping_entries());
  BOOST_TEST(overlap_vector.has_overlapping_entries());
  BOOST_TEST(not disjoint_unsorted.has_overlapping_entries());

  const int origin[4] = {10, 11, 12, 13};
  int result[4] = {-1, -1, -1, -1};

  // An updated target may not have overlapping type-map entries (§12.3.1/§12.3.4 Accumulate).
  BOOST_TEST(throws_invalid([&] { mp1::put(w, std::span{origin}, 0, 0, overlap_indexed); }));
  BOOST_TEST(throws_invalid([&] { mp1::put(w, std::span{origin}, 0, 0, overlap_vector); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, std::span{origin}, 0, 0, overlap_indexed, MPI_SUM); }));

  // An updated result/origin may not have overlapping entries (§12.3.2/§12.3.4 Get Accumulate).
  BOOST_TEST(
      throws_invalid([&] { mp1::get(w, mp1::region{result, overlap_indexed}, 0, 0, 4); }));
  const mp1::layout<int> origin_layout = mp1::layout<int>::contiguous(4);
  BOOST_TEST(throws_invalid([&] {
    mp1::get_accumulate(w, mp1::region{origin, origin_layout},
                        mp1::region{result, overlap_indexed}, 0, 0, 4, MPI_REPLACE);
  }));

  // A read-only origin or target may contain overlapping entries: the general datatype rule
  // only forbids overlap "in association with a buffer updated by the operation" (§5.1.11).
  // origin map {0,1,1,2} is read-only
  mp1::put(w, mp1::region{origin, overlap_indexed}, 0, 0, 4);
  // target map {0,1,1,2} is read-only
  mp1::get(w, std::span{result}, 0, 0, overlap_indexed);
  mp1::fence(w);
  if (rank == 0) {
    BOOST_TEST(window_load(target) == 10);
    BOOST_TEST(window_load(target + 1) == 11);
    BOOST_TEST(window_load(target + 2) == 11);  // the third origin entry repeats element 1
    BOOST_TEST(window_load(target + 3) == 12);
  }
  BOOST_TEST(window_load(result) == 10);
  BOOST_TEST(window_load(result + 1) == 11);
  BOOST_TEST(window_load(result + 2) == 11);
  BOOST_TEST(window_load(result + 3) == 11);
  w.free();
}

BOOST_AUTO_TEST_CASE(proc_null_all_data_movement_families) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  mp1::window w = mp1::window::allocate(c, {.size = sizeof(int), .disp_unit = sizeof(int)});
  int* target = static_cast<int*>(w.base());
  if (rank == 0)
    *target = 42;
  mp1::fence(w);

  mp1::lock_all(w);
  mp1::barrier(c);
  const int one = 1;
  int got = -1;
  int old = -1;
  int cas_old = -1;

  mp1::put(w, mp1::span_from(&one, 1), MPI_PROC_NULL, 0, 1);
  mp1::get(w, mp1::span_from(&got, 1), MPI_PROC_NULL, 0, 1);
  mp1::accumulate(w, mp1::span_from(&one, 1), MPI_PROC_NULL, 0, 1, MPI_SUM);
  mp1::get_accumulate(w, mp1::span_from(&one, 1), mp1::span_from(&old, 1), MPI_PROC_NULL, 0, 1,
                      MPI_SUM);
  mp1::fetch_and_op(w, &one, &old, MPI_PROC_NULL, 0, MPI_SUM);
  mp1::compare_and_swap(w, &one, &one, &cas_old, MPI_PROC_NULL, 0);

  int rget_value = -1;
  mp1::request rput_request = mp1::rput(w, mp1::span_from(&one, 1), MPI_PROC_NULL, 0, 1);
  mp1::request rget_request = mp1::rget(w, mp1::span_from(&rget_value, 1), MPI_PROC_NULL, 0, 1);
  mp1::request raccumulate_request =
      mp1::raccumulate(w, mp1::span_from(&one, 1), MPI_PROC_NULL, 0, 1, MPI_SUM);
  mp1::request rget_accumulate_request = mp1::rget_accumulate(
      w, mp1::span_from(&one, 1), mp1::span_from(&old, 1), MPI_PROC_NULL, 0, 1, MPI_SUM);

  // RMA requests are a distinct kind; wait is the only legal teardown, and the completion
  // status is normalized (MPI 4.1 §12.3.5: every field except MPI_ERROR is undefined).
  for (mp1::request* request :
       {&rput_request, &rget_request, &raccumulate_request, &rget_accumulate_request}) {
    BOOST_TEST(static_cast<int>(request->kind()) == static_cast<int>(mp1::request_kind::rma));
    BOOST_TEST(static_cast<int>(request->teardown_policy()) ==
               static_cast<int>(mp1::request_teardown::wait));
    BOOST_TEST(request->drop_would_block());
    BOOST_TEST(not request->cancelable());
    const mp1::status completion = request->wait();
    BOOST_TEST(static_cast<int>(completion.kind()) ==
               static_cast<int>(mp1::status_kind::normalized));
    BOOST_TEST(completion.get_count<int>() == 0);
  }

  BOOST_TEST(got == -1);
  BOOST_TEST(old == -1);
  BOOST_TEST(cas_old == -1);
  BOOST_TEST(rget_value == -1);
  mp1::flush_all(w);
  mp1::barrier(c);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  if (rank == 0)
    BOOST_TEST(window_load(target) == 42);
  w.free();
}

BOOST_AUTO_TEST_CASE(custom_predefined_trait_classified_before_mpi) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  // A custom trait mapped to MPI_INT is a predefined C integer datatype: accepted.
  {
    mp1::window w = mp1::window::allocate(
        c, {.size = static_cast<MPI_Aint>(sizeof(rma_custom_int)), .disp_unit = sizeof(int)});
    rma_custom_int* target = static_cast<rma_custom_int*>(w.base());
    if (rank == 0)
      target->value = 10;
    mp1::fence(w);
    mp1::lock_all(w);
    mp1::barrier(c);
    const rma_custom_int one{1};
    mp1::accumulate(w, mp1::span_from(&one, 1), 0, 0, 1, MPI_SUM);
    mp1::flush_all(w);
    mp1::barrier(c);
    mp1::unlock_all(w);
    sync_after_passive(c, w);
    if (rank == 0)
      BOOST_TEST(target->value == 10 + c.size());
    w.free();
  }

  // A custom trait mapped to MPI_2INT is predefined (named) but not a category mp1 models:
  // the typed boundary rejects it before any MPI call instead of letting MPI interpret a pair.
  {
    mp1::window w = mp1::window::allocate(
        c, {.size = static_cast<MPI_Aint>(sizeof(rma_custom_pair)), .disp_unit = sizeof(int)});
    const rma_custom_pair pair{1, 2};
    rma_custom_pair result{0, 0};
    BOOST_TEST(throws_invalid(
        [&] { mp1::accumulate(w, mp1::span_from(&pair, 1), 0, 0, 1, MPI_SUM); }));
    BOOST_TEST(throws_invalid([&] { mp1::fetch_and_op(w, &pair, &result, 0, 0, MPI_SUM); }));
    w.free();
  }
}

BOOST_AUTO_TEST_CASE(local_validation) {
  const mp1::mpi_communicator c = world();
  mp1::window w = mp1::window::allocate(c, {.size = 8 * sizeof(int), .disp_unit = sizeof(int)});
  int* local = static_cast<int*>(w.base());
  *local = 0;

  const int value = 1;
  int result = 0;
  const mp1::mpi_window invalid;
  const mp1::layout<int> good = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> invalid_layout;

  BOOST_TEST(throws_invalid([&] { mp1::put(invalid, mp1::span_from(&value, 1), 0, 0, 1); }));
  BOOST_TEST(throws_invalid([&] { mp1::get(invalid, mp1::span_from(&result, 1), 0, 0, 1); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(invalid, mp1::span_from(&value, 1), 0, 0, 1, MPI_SUM); }));
  BOOST_TEST(throws_invalid([&] { mp1::fence(invalid); }));
  BOOST_TEST(throws_invalid([&] { mp1::lock_all(invalid); }));
  BOOST_TEST(throws_invalid([&] { mp1::flush_all(invalid); }));
  BOOST_TEST(throws_invalid([&] { mp1::sync(invalid); }));
  BOOST_TEST(throws_invalid([&] { mp1::wait(invalid); }));
  BOOST_TEST(throws_invalid([&] { (void)mp1::test(invalid); }));

  BOOST_TEST(throws_invalid([&] { mp1::put(w, mp1::span_from(&value, 1), c.size(), 0, 1); }));
  BOOST_TEST(throws_invalid([&] { mp1::put(w, mp1::span_from(&value, 1), -1000, 0, 1); }));
  BOOST_TEST(throws_invalid([&] { mp1::put(w, mp1::span_from(&value, 1), 0, -1, 1); }));
  BOOST_TEST(throws_invalid([&] { mp1::put(w, mp1::span_from(&value, 1), 0, 0, -1); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::put(w, mp1::span_from(static_cast<const int*>(nullptr), 1), 0, 0, 1); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::get(w, mp1::span_from(static_cast<int*>(nullptr), 1), 0, 0, 1); }));

  BOOST_TEST(throws_invalid([&] { mp1::put(w, mp1::region{&value, good}, 0, 0, 3); }));
  BOOST_TEST(
      throws_invalid([&] { mp1::put(w, mp1::span_from(&value, 1), 0, 0, invalid_layout); }));
  BOOST_TEST(
      throws_invalid([&] { mp1::get(w, mp1::span_from(&result, 1), 0, 0, invalid_layout); }));

  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&value, 1), 0, 0, 1, MPI_OP_NULL); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::accumulate(w, mp1::span_from(&value, 1), 0, 0, 1, MPI_NO_OP); }));
  BOOST_TEST(throws_invalid(
      [&] { (void)mp1::raccumulate(w, mp1::span_from(&value, 1), 0, 0, 1, MPI_NO_OP); }));
  BOOST_TEST(throws_invalid([&] { mp1::fetch_and_op(w, &value, &result, 0, 0, MPI_OP_NULL); }));
  BOOST_TEST(throws_invalid([&] {
    mp1::get_accumulate(w, mp1::span_from(&value, 1), mp1::span_from(&result, 1), 0, 0, 1,
                        MPI_OP_NULL);
  }));

  BOOST_TEST(throws_invalid([&] {
    mp1::compare_and_swap(w, static_cast<const int*>(nullptr), &value, &result, 0, 0);
  }));
  BOOST_TEST(throws_invalid([&] {
    mp1::compare_and_swap(w, &value, static_cast<const int*>(nullptr), &result, 0, 0);
  }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::compare_and_swap(w, &value, &value, static_cast<int*>(nullptr), 0, 0); }));

  mp1::group invalid_group;  // default group state is MPI_GROUP_NULL
  BOOST_TEST(not invalid_group.is_valid());
  mp1::group empty_group = mp1::group::empty();
  mp1::group moved_group = std::move(empty_group);
  BOOST_TEST(moved_group.is_valid());
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not empty_group.is_valid());
  BOOST_TEST(throws_invalid([&] { mp1::post(w, invalid_group); }));
  BOOST_TEST(throws_invalid([&] { mp1::start(w, invalid_group); }));

  w.free();

  // The window is validated exactly once, at the RMA entry, before the MPI_PROC_NULL no-op
  // branch and before every family-specific check.  A released window therefore still throws
  // for the no-op spellings instead of reaching a native call with MPI_WIN_NULL.
  BOOST_TEST(
      throws_invalid([&] { mp1::put(w, mp1::span_from(&value, 1), MPI_PROC_NULL, 0, 1); }));
  BOOST_TEST(
      throws_invalid([&] { mp1::get(w, mp1::span_from(&result, 1), MPI_PROC_NULL, 0, 1); }));
  BOOST_TEST(throws_invalid(
      [&] { (void)mp1::rput(w, mp1::span_from(&value, 1), MPI_PROC_NULL, 0, 1); }));
  BOOST_TEST(throws_invalid(
      [&] { (void)mp1::rget(w, mp1::span_from(&result, 1), MPI_PROC_NULL, 0, 1); }));
  BOOST_TEST(throws_invalid([&] {
    (void)mp1::raccumulate(w, mp1::span_from(&value, 1), MPI_PROC_NULL, 0, 1, MPI_SUM);
  }));
  BOOST_TEST(throws_invalid([&] {
    (void)mp1::rget_accumulate(w, mp1::span_from(&value, 1), mp1::span_from(&result, 1),
                               MPI_PROC_NULL, 0, 1, MPI_SUM);
  }));
}

BOOST_AUTO_TEST_SUITE_END()
