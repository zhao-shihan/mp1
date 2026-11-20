// Contract test for mp1::group, mp1::mpi_communicator, mp1::communicator and the free
// point-to-point functions.
//
// The shared test harness links the Boost.Test main; this file defines neither main nor
// BOOST_TEST_MODULE. MPI is initialized by the global fixture below.
//
// Every factory collective is called by all ranks. Point-to-point tests use ranks 0/1 and
// make the other ranks wait at MPI_Barrier(MPI_COMM_WORLD), so the same source runs at 2 and
// 4 ranks.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(MP1_TEST_WRAP)
#include "support/mpi_wrap.hpp"
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

  bool is_proc_null_status(const mp1::status& s) {
    // MPI 4.1 §3.10: the wrapper synthesizes the whole status, so it is normalized rather
    // than the provider's native envelope.
    return s.kind() == mp1::status_kind::normalized and s.source() == MPI_PROC_NULL and
           s.tag() == mp1::any_tag and s.error() == MPI_SUCCESS;
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

  template<class F>
  bool throws_runtime_error(F&& f) {
    try {
      f();
    } catch (const std::runtime_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  // An allocator that always throws on allocation.  The auto-size container receive probes
  // the message, resizes the container, and only then receives it; a resize failure must not
  // strand the matched message (bad_alloc path).
  template<class T>
  struct throwing_allocator {
    using value_type = T;

    throwing_allocator() noexcept = default;
    template<class U>
    throwing_allocator(const throwing_allocator<U>&) noexcept {
    }

    [[nodiscard]] T* allocate(std::size_t) {
      throw std::bad_alloc{};
    }

    void deallocate(T*, std::size_t) noexcept {
    }
  };

  template<class T, class U>
  bool operator==(const throwing_allocator<T>&, const throwing_allocator<U>&) noexcept {
    return true;
  }

  template<class T, class U>
  bool operator!=(const throwing_allocator<T>&, const throwing_allocator<U>&) noexcept {
    return false;
  }

  // Temporarily install MPI_ERRORS_RETURN on MPI_COMM_WORLD so a truncation error becomes an
  // mp1::mpi_error instead of aborting the job; restores the previous handler on scope exit.
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

  static_assert(not std::is_copy_constructible_v<mp1::message>,
                "mp1::message owns one MPI_Message and must be move-only");
  static_assert(not std::is_copy_assignable_v<mp1::message>,
                "mp1::message owns one MPI_Message and must be move-only");
  static_assert(std::is_move_constructible_v<mp1::message>, "mp1::message must be movable");
  static_assert(std::is_move_assignable_v<mp1::message>,
                "mp1::message must be move-assignable");

  // ---------------------------------------------------------------------------------------
  // Compile-time send_mode hardening. Every legal enumerator must produce its documented
  // native name and instantiate all three native send dispatches. An illegal enumerator is
  // rejected by the static_assert in detail/p2p_engine.hpp; test/compile_fail/
  // invalid_send_mode.cc proves that rejection does not silently fall into the ready branch.
  // ---------------------------------------------------------------------------------------

  template<mp1::detail::send_mode M>
  constexpr bool send_mode_names_match(const char* send, const char* isend,
                                       const char* send_init) {
    return std::string_view{mp1::detail::send_mode_name<M>()} == send and
           std::string_view{mp1::detail::isend_mode_name<M>()} == isend and
           std::string_view{mp1::detail::send_init_mode_name<M>()} == send_init;
  }

  static_assert(send_mode_names_match<mp1::detail::send_mode::standard>("MPI_Send_c",
                                                                        "MPI_Isend_c",
                                                                        "MPI_Send_init_c"),
                "standard send_mode names");
  static_assert(send_mode_names_match<mp1::detail::send_mode::buffered>("MPI_Bsend_c",
                                                                        "MPI_Ibsend_c",
                                                                        "MPI_Bsend_init_c"),
                "buffered send_mode names");
  static_assert(send_mode_names_match<mp1::detail::send_mode::synchronous>("MPI_Ssend_c",
                                                                           "MPI_Issend_c",
                                                                           "MPI_Ssend_init_c"),
                "synchronous send_mode names");
  static_assert(send_mode_names_match<mp1::detail::send_mode::ready>("MPI_Rsend_c",
                                                                     "MPI_Irsend_c",
                                                                     "MPI_Rsend_init_c"),
                "ready send_mode names");

  // Taking the address of a dispatch specialization forces the compiler to instantiate its
  // body, so a dispatch that does not compile fails this translation unit.  The check is a
  // runtime expression, not a constexpr function-pointer comparison: GCC 16 with
  // -fsanitize=undefined rejects the latter in a static_assert, so the check stays a runtime
  // expression.
  BOOST_AUTO_TEST_CASE(send_mode_dispatch_instantiates) {
    const auto standard_send = &mp1::detail::native_send<mp1::detail::send_mode::standard>;
    const auto buffered_send = &mp1::detail::native_send<mp1::detail::send_mode::buffered>;
    const auto synchronous_send =
        &mp1::detail::native_send<mp1::detail::send_mode::synchronous>;
    const auto ready_send = &mp1::detail::native_send<mp1::detail::send_mode::ready>;
    const auto standard_isend = &mp1::detail::native_isend<mp1::detail::send_mode::standard>;
    const auto buffered_isend = &mp1::detail::native_isend<mp1::detail::send_mode::buffered>;
    const auto synchronous_isend =
        &mp1::detail::native_isend<mp1::detail::send_mode::synchronous>;
    const auto ready_isend = &mp1::detail::native_isend<mp1::detail::send_mode::ready>;
    const auto standard_init = &mp1::detail::native_send_init<mp1::detail::send_mode::standard>;
    const auto buffered_init = &mp1::detail::native_send_init<mp1::detail::send_mode::buffered>;
    const auto synchronous_init =
        &mp1::detail::native_send_init<mp1::detail::send_mode::synchronous>;
    const auto ready_init = &mp1::detail::native_send_init<mp1::detail::send_mode::ready>;
    BOOST_TEST(standard_send != nullptr);
    BOOST_TEST(buffered_send != nullptr);
    BOOST_TEST(synchronous_send != nullptr);
    BOOST_TEST(ready_send != nullptr);
    BOOST_TEST(standard_isend != nullptr);
    BOOST_TEST(buffered_isend != nullptr);
    BOOST_TEST(synchronous_isend != nullptr);
    BOOST_TEST(ready_isend != nullptr);
    BOOST_TEST(standard_init != nullptr);
    BOOST_TEST(buffered_init != nullptr);
    BOOST_TEST(synchronous_init != nullptr);
    BOOST_TEST(ready_init != nullptr);
  }

  // ---------------------------------------------------------------------------------------
  // After-finalize static-destruction probe. Storage is placement-new'd by a test and the
  // destructors are called by this guard, which is constructed BEFORE the Boost global
  // fixture and therefore destroyed AFTER the fixture's mp1::finalize().
  // ---------------------------------------------------------------------------------------

  alignas(mp1::communicator) unsigned char owner_storage[sizeof(mp1::communicator)];
  alignas(mp1::communicator) unsigned char invalid_owner_storage[sizeof(mp1::communicator)];
  alignas(mp1::group) unsigned char group_storage[sizeof(mp1::group)];
  alignas(mp1::group) unsigned char empty_group_storage[sizeof(mp1::group)];
  bool owner_constructed = false;

  [[noreturn]] void after_finalize_failure(const char* what) {
    std::fprintf(stderr, "mp1 p2p after-finalize failure: %s\n", what);
    std::abort();
  }

  struct after_finalize_check {
    ~after_finalize_check() noexcept {
      if (not owner_constructed)
        return;
      if (not mp1::finalized())
        return;  // The global fixture did not finalize MPI; leave the objects alone.
      auto* owner = std::launder(reinterpret_cast<mp1::communicator*>(owner_storage));
      auto* invalid_owner =
          std::launder(reinterpret_cast<mp1::communicator*>(invalid_owner_storage));
      auto* group = std::launder(reinterpret_cast<mp1::group*>(group_storage));
      auto* empty_group = std::launder(reinterpret_cast<mp1::group*>(empty_group_storage));

      if (not owner->is_valid() or owner->native_handle() == MPI_COMM_NULL)
        after_finalize_failure("operator view invalid after finalize");
      if (not group->is_valid() or group->native_handle() == MPI_GROUP_NULL)
        after_finalize_failure("group view invalid after finalize");

      bool comm_world_threw{false};
      try {
        (void)mp1::comm_world();
      } catch (const std::runtime_error&) {
        comm_world_threw = true;
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
      }
      if (not comm_world_threw)
        after_finalize_failure("comm_world did not throw after finalize");
      if (mp1::comm_null().is_valid())
        after_finalize_failure("comm_null is not invalid");

      bool free_threw{false};
      try {
        owner->free();
      } catch (const std::runtime_error&) {
        free_threw = true;
      }
      if (not free_threw)
        after_finalize_failure("communicator::free did not throw after finalize");

      bool disconnect_threw{false};
      try {
        owner->disconnect();
      } catch (const std::runtime_error&) {
        disconnect_threw = true;
      }
      if (not disconnect_threw)
        after_finalize_failure("communicator::disconnect did not throw after finalize");

      bool group_size_threw{false};
      try {
        (void)group->size();
      } catch (const std::runtime_error&) {
        group_size_threw = true;
      }
      if (not group_size_threw)
        after_finalize_failure("group::size did not throw after finalize");

      const std::size_t before = mp1::detail::abandoned_handle_count();
      owner->~communicator();
      group->~group();
      invalid_owner->~communicator();
      empty_group->~group();
      const std::size_t after = mp1::detail::abandoned_handle_count();
      if (after != before + 2) {
        after_finalize_failure(
            "static owner/group destructors did not abandon exactly 2 handles");
      }
#if defined(MP1_TEST_WRAP)
      if (mpi_wrap::supported() and mpi_wrap::after_finalize_calls() != 0)
        after_finalize_failure("an MPI call was made after finalize");
#endif
    }
  };

  after_finalize_check after_finalize_check_instance;

}  // namespace

struct mp1_p2p_contract_fixture {
  mp1_p2p_contract_fixture() {
    mp1::init();
  }

  ~mp1_p2p_contract_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Already finalized through another path; a destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_p2p_contract_fixture);

BOOST_AUTO_TEST_SUITE(p2p_contract)

BOOST_AUTO_TEST_CASE(type_and_view_basics) {
  BOOST_TEST(true);
  static_assert(std::is_copy_constructible_v<mp1::mpi_communicator>);
  static_assert(std::is_copy_assignable_v<mp1::mpi_communicator>);
  static_assert(std::is_move_constructible_v<mp1::mpi_communicator>);
  static_assert(std::is_move_assignable_v<mp1::mpi_communicator>);
  static_assert(std::is_nothrow_destructible_v<mp1::mpi_communicator>);
  static_assert(not std::is_base_of_v<mp1::mpi_communicator, mp1::communicator>,
                "the owner composes a view; it must not publicly derive from it");
  static_assert(std::is_convertible_v<const mp1::communicator&, mp1::mpi_communicator>,
                "an owner converts to a view by value");
  static_assert(not std::is_convertible_v<mp1::communicator&, mp1::mpi_communicator&>,
                "the view base must not be mutable through the owner");
  static_assert(not std::is_copy_constructible_v<mp1::communicator>);
  static_assert(not std::is_copy_assignable_v<mp1::communicator>);
  static_assert(std::is_move_constructible_v<mp1::communicator>);
  static_assert(std::is_move_assignable_v<mp1::communicator>);
  static_assert(not std::is_constructible_v<mp1::communicator, const mp1::mpi_communicator&>);
  static_assert(not std::is_constructible_v<mp1::communicator, MPI_Comm>);
  static_assert(not std::is_assignable_v<mp1::communicator&, const mp1::mpi_communicator&>);
  static_assert(not std::is_copy_constructible_v<mp1::group>);
  static_assert(std::is_move_constructible_v<mp1::group>);

  mp1::mpi_communicator invalid;
  BOOST_TEST(not invalid.is_valid());
  BOOST_TEST(invalid.native_handle() == MPI_COMM_NULL);
  BOOST_TEST(not mp1::comm_null().is_valid());
  BOOST_TEST(static_cast<bool>(mp1::comm_null() == mp1::comm_null()));
  BOOST_TEST(static_cast<bool>(mp1::comm_null() != mp1::comm_world()));

  const mp1::mpi_communicator world = mp1::comm_world();
  BOOST_TEST(world.is_valid());
  BOOST_TEST(world.size() == world_size());
  BOOST_TEST(world.rank() == world_rank());
  BOOST_TEST(not world.is_inter());
  BOOST_TEST(static_cast<bool>(world == mp1::comm_world()));
  BOOST_TEST(
      static_cast<bool>(world.compare(mp1::comm_world()) == mp1::equality_type::identical));
  BOOST_TEST(throws_invalid_argument([&world] { (void)world.remote_size(); }));
  BOOST_TEST(throws_invalid_argument([&world] { (void)world.remote_group(); }));

  const mp1::mpi_communicator self = mp1::comm_self();
  BOOST_TEST(self.is_valid());
  BOOST_TEST(self.size() == 1);
  BOOST_TEST(self.rank() == 0);
  BOOST_TEST(static_cast<bool>(self != world));

  mp1::communicator owner;
  BOOST_TEST(not owner.is_valid());
  owner = mp1::communicator::dup(world);
  BOOST_TEST(owner.is_valid());
  const MPI_Comm handle = owner.native_handle();
  mp1::mpi_communicator slice = owner;
  BOOST_TEST(slice.native_handle() == handle);
  mp1::mpi_communicator other = mp1::comm_null();
  slice = other;  // view move/copy assignment must never free the borrowed handle
  BOOST_TEST(not slice.is_valid());
  BOOST_TEST(owner.is_valid());
  BOOST_TEST(owner.native_handle() == handle);
  BOOST_TEST(static_cast<bool>(owner.compare(world) == mp1::equality_type::congruent));

  BOOST_TEST(throws_invalid_argument([] { (void)mp1::communicator::adopt(MPI_COMM_NULL); }));
  BOOST_TEST(throws_invalid_argument([] { (void)mp1::communicator::adopt(MPI_COMM_WORLD); }));
  BOOST_TEST(throws_invalid_argument([] { (void)mp1::communicator::adopt(MPI_COMM_SELF); }));
  BOOST_TEST(throws_invalid_argument([] { (void)mp1::communicator::dup(mp1::comm_null()); }));

  const mp1::communicator parent_owner = mp1::parent();
  BOOST_TEST(not parent_owner.is_valid());
  BOOST_TEST(not mp1::parent().is_valid());
}

BOOST_AUTO_TEST_CASE(group_contract) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();

  mp1::group invalid_group;
  BOOST_TEST(not invalid_group.is_valid());
  mp1::group empty = mp1::group::empty();
  BOOST_TEST(empty.is_valid());
  BOOST_TEST(empty.is_empty());
  BOOST_TEST(empty.size() == 0);
  BOOST_TEST(empty.rank() == MPI_UNDEFINED);
  BOOST_TEST(empty.native_handle() == MPI_GROUP_EMPTY);

  mp1::group all = mp1::get_group(world);
  BOOST_TEST(all.is_valid());
  BOOST_TEST(all.size() == size);
  BOOST_TEST(all.rank() == rank);
  BOOST_TEST(static_cast<bool>(all.compare(all) == mp1::equality_type::identical));
  BOOST_TEST(static_cast<bool>(all == all));
  BOOST_TEST(all.translate(rank, all) == rank);

  mp1::group first = mp1::group_incl(all, std::vector<int>{0});
  BOOST_TEST(first.size() == 1);
  BOOST_TEST((rank == 0 ? first.rank() == 0 : first.rank() == MPI_UNDEFINED));
  BOOST_TEST(first.translate(0, all) == 0);

  mp1::group rest = mp1::group_excl(all, std::vector<int>{0});
  BOOST_TEST(rest.size() == size - 1);
  BOOST_TEST(static_cast<bool>(first.compare(rest) == mp1::equality_type::unequal));
  BOOST_TEST(mp1::group_union(first, rest).size() == size);
  BOOST_TEST(mp1::group_intersection(first, rest).size() == 0);
  BOOST_TEST(mp1::group_difference(all, first).size() == size - 1);

  BOOST_TEST(throws_invalid_argument([&all] { (void)mp1::group_incl(all, std::array{-1}); }));
  BOOST_TEST(
      throws_invalid_argument([&all, size] { (void)mp1::group_incl(all, std::array{size}); }));
  BOOST_TEST(
      throws_invalid_argument([&all, size] { (void)mp1::group_excl(all, std::array{size}); }));

  mp1::group moved = std::move(all);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not all.is_valid());
  BOOST_TEST(throws_invalid_argument([&all] { (void)all.size(); }));
  BOOST_TEST(moved.is_valid());
  mp1::group assigned;
  assigned = std::move(moved);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not moved.is_valid());
  BOOST_TEST(assigned.size() == size);
}

BOOST_AUTO_TEST_CASE(communicator_factories_and_owner) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int size = world_size();

  {
    mp1::communicator dup = mp1::communicator::dup(world);
    BOOST_TEST(dup.is_valid());
    BOOST_TEST(static_cast<bool>(dup.compare(world) == mp1::equality_type::congruent));
    mp1::communicator with_info = mp1::communicator::dup_with_info(world, mp1::info{});
    BOOST_TEST(with_info.is_valid());
    with_info.free();
    BOOST_TEST(not with_info.is_valid());
    with_info.free();  // idempotent
    dup.free();
    dup.free();
    BOOST_TEST(not dup.is_valid());
  }

  {
    mp1::communicator even = mp1::communicator::split(world, rank % 2, rank);
    BOOST_TEST(even.is_valid());
    const int expected =
        (size % 2 == 0) ? size / 2 : (rank % 2 == 0 ? (size + 1) / 2 : size / 2);
    BOOST_TEST(even.size() == expected);
    mp1::communicator excluded = mp1::communicator::split(world, MPI_UNDEFINED, rank);
    BOOST_TEST(not excluded.is_valid());
  }

  {
    mp1::communicator shared = mp1::communicator::split_type(world, mp1::comm_type_shared);
    BOOST_TEST(shared.is_valid());
    BOOST_TEST(shared.size() >= 1);
  }

  {
    const int half = (size + 1) / 2;
    std::vector<int> ranks;
    ranks.reserve(static_cast<std::size_t>(half));
    for (int i = 0; i < half; ++i)
      ranks.push_back(i);
    const mp1::group half_group = mp1::group_incl(mp1::get_group(world), ranks);
    mp1::communicator subset = mp1::communicator::create(world, half_group);
    BOOST_TEST((rank < half ? subset.is_valid() : not subset.is_valid()));
    BOOST_TEST(throws_invalid_argument(
        [&half_group] { (void)mp1::communicator::create(mp1::comm_null(), half_group); }));
  }

  {
    mp1::communicator created =
        mp1::communicator::create_group(world, mp1::get_group(world), mp1::tag{991});
    BOOST_TEST(created.is_valid());
    BOOST_TEST(created.size() == size);
  }

  {
    mp1::communicator a = mp1::communicator::dup(world);
    const MPI_Comm old_handle = a.native_handle();
    mp1::communicator b = std::move(a);
    // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
    BOOST_TEST(not a.is_valid());
    BOOST_TEST(b.is_valid());
    BOOST_TEST(b.native_handle() == old_handle);
    mp1::communicator c = mp1::communicator::dup(world);
    c = std::move(b);
    // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
    BOOST_TEST(not b.is_valid());
    BOOST_TEST(c.is_valid());
    BOOST_TEST(c.native_handle() == old_handle);
  }

  {
    mp1::communicator dup = mp1::communicator::dup(world);
    BOOST_TEST(throws_invalid_argument([&dup] { dup.set_info(mp1::info{}); }));
    const mp1::info table = dup.get_info();
    BOOST_TEST(table.size() >= 0);
  }

  // Objects whose destructors must run after MPI_Finalize (probed by the cleanup guard).
  new (owner_storage) mp1::communicator(mp1::communicator::dup(world));
  new (invalid_owner_storage) mp1::communicator();
  new (group_storage) mp1::group(mp1::group_incl(mp1::get_group(world), std::vector<int>{0}));
  new (empty_group_storage) mp1::group(mp1::group::empty());
  owner_constructed = true;
}

BOOST_AUTO_TEST_CASE(intercommunicator_contract) {
  BOOST_TEST(true);
  const int size = world_size();
  if (size < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();
  const int half = (size + 1) / 2;
  const int color = rank < half ? 0 : 1;
  const int local_size = color == 0 ? half : size - half;
  const int remote_size = size - local_size;
  const int remote_leader = color == 0 ? half : 0;
  const int tag = 4100;

  mp1::communicator local = mp1::communicator::split(world, color, rank);
  mp1::communicator inter =
      mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{tag});
  BOOST_TEST(inter.is_valid());
  BOOST_TEST(inter.is_inter());
  BOOST_TEST(inter.size() == local_size);
  BOOST_TEST(inter.rank() == local.rank());
  BOOST_TEST(inter.remote_size() == remote_size);
  BOOST_TEST(mp1::get_group(inter).size() == local_size);
  BOOST_TEST(inter.remote_group().size() == remote_size);

  const int local_rank = local.rank();
  const int value = 7000 + rank;
  mp1::send(inter, value, 0, mp1::tag{4200});
  if (local_rank == 0) {
    for (int i = 0; i < remote_size; ++i) {
      int received{0};
      const mp1::status st = mp1::recv(inter, received, MPI_ANY_SOURCE, mp1::tag{4200});
      BOOST_TEST(st.source() >= 0);
      BOOST_TEST(st.source() < remote_size);
    }
    for (int i = 0; i < remote_size; ++i)
      mp1::send(inter, 8000 + i, i, mp1::tag{4201});
  }
  // Every rank, including the root of each side, receives exactly one reply from the remote
  // root.  Without this receive for the root the reply sends stay unmatched (MPI 4.1 §3.2.1
  // requires matching sends/receives); a provider can report the unmatched message when the
  // intercommunicator is later disconnected at MPI_THREAD_SINGLE.
  {
    int reply{0};
    const mp1::status st = mp1::recv(inter, reply, 0, mp1::tag{4201});
    BOOST_TEST(reply >= 8000);
    BOOST_TEST(st.tag().value == 4201);
  }

  BOOST_TEST(throws_invalid_argument([&inter, remote_size] {
    int x{0};
    mp1::send(inter, x, remote_size, mp1::tag{4202});
  }));
  BOOST_TEST(throws_invalid_argument([&inter, remote_size] {
    int x{0};
    (void)mp1::recv(inter, x, remote_size, mp1::tag{4202});
  }));

  {
    mp1::communicator duplicate = mp1::communicator::dup(inter);
    BOOST_TEST(duplicate.is_valid());
    BOOST_TEST(duplicate.is_inter());
    duplicate.disconnect();
    BOOST_TEST(not duplicate.is_valid());
    duplicate.disconnect();
  }

  {
    mp1::communicator merged = mp1::communicator::merge(inter, color == 1);
    BOOST_TEST(merged.is_valid());
    BOOST_TEST(not merged.is_inter());
    BOOST_TEST(merged.size() == size);
  }

  inter.disconnect();
  BOOST_TEST(not inter.is_valid());
  inter.disconnect();  // idempotent
}

BOOST_AUTO_TEST_CASE(blocking_send_modes) {
  BOOST_TEST(true);
  const int size = world_size();
  if (size < 2)
    return;
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();

  if (rank == 0) {
    mp1::send(world, 11, 1, mp1::tag{5000});
  } else if (rank == 1) {
    int got{0};
    const mp1::status st = mp1::recv(world, got, 0, mp1::tag{5000});
    BOOST_TEST(got == 11);
    BOOST_TEST(st.tag().value == 5000);
  }
  world_barrier();

  if (rank == 0) {
    mp1::ssend(world, 12, 1, mp1::tag{5001});
  } else if (rank == 1) {
    int got{0};
    (void)mp1::recv(world, got, 0, mp1::tag{5001});
    BOOST_TEST(got == 12);
  }
  world_barrier();

  int ready_value = 0;
  mp1::request ready_recv;
  if (rank == 1)
    ready_recv = mp1::irecv(world, ready_value, 0, mp1::tag{5002});
  world_barrier();
  if (rank == 0)
    mp1::rsend(world, 13, 1, mp1::tag{5002});
  if (rank == 1) {
    (void)ready_recv.wait();
    BOOST_TEST(ready_value == 13);
  }
  world_barrier();

  if (rank == 0)
    mp1::send(world, 14, MPI_PROC_NULL, mp1::tag{5003});
  world_barrier();
}

BOOST_AUTO_TEST_CASE(buffered_send_modes_with_overhead) {
  BOOST_TEST(true);
  const int size = world_size();
  if (size < 2)
    return;
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();
  const int payload = 16384;  // 64 KiB of int
  const int count = 100;
  const int buffer_size =
      count * (payload * static_cast<int>(sizeof(int)) + MPI_BSEND_OVERHEAD);
  mp1::bsend_buffer buffer{buffer_size};

  if (rank == 0) {
    const std::vector<int> message(static_cast<std::size_t>(payload), 7);
    for (int i = 0; i < count; ++i)
      mp1::bsend(world, message, 1, mp1::tag{5100 + i});
  } else if (rank == 1) {
    for (int i = 0; i < count; ++i) {
      std::vector<int> received;
      (void)mp1::recv(world, received, 0, mp1::tag{5100 + i});
      BOOST_TEST(received.size() == static_cast<std::size_t>(payload));
    }
  }
  world_barrier();

  if (rank == 0) {
    std::vector<int> message(3, 9);
    int bsend_value = 9;
    {
      mp1::request first = mp1::ibsend(world, message, 1, mp1::tag{5200});
      (void)first.wait();
    }
    {
      mp1::request persistent = mp1::bsend_init(world, bsend_value, 1, mp1::tag{5201});
      persistent.start();
      (void)persistent.wait();
    }
    {
      mp1::request persistent = mp1::bsend_init(world, bsend_value, 1, mp1::tag{5202});
      BOOST_TEST(persistent.is_persistent());
      BOOST_TEST(not persistent.is_active());
      persistent.start();
      (void)persistent.wait();
    }
  } else if (rank == 1) {
    std::vector<int> received;
    (void)mp1::recv(world, received, 0, mp1::tag{5200});
    BOOST_TEST(received.size() == 3);
    int value{0};
    (void)mp1::recv(world, value, 0, mp1::tag{5201});
    (void)mp1::recv(world, value, 0, mp1::tag{5202});
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(nonblocking_send_and_persistent_modes) {
  BOOST_TEST(true);
  const int size = world_size();
  if (size < 2)
    return;
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();

  {
    int send_value = 31;
    int recv_value = 0;
    mp1::request send;
    mp1::request recv;
    if (rank == 0)
      send = mp1::isend(world, send_value, 1, mp1::tag{5300});
    else if (rank == 1)
      recv = mp1::irecv(world, recv_value, 0, mp1::tag{5300});
    if (rank == 0) {
      BOOST_TEST(not send.is_persistent());
      (void)send.wait();
    } else if (rank == 1) {
      (void)recv.wait();
      BOOST_TEST(recv_value == 31);
    }
  }
  world_barrier();

  if (rank == 0) {
    int issend_value = 32;
    mp1::request first = mp1::issend(world, issend_value, 1, mp1::tag{5301});
    (void)first.wait();
  } else if (rank == 1) {
    int value{0};
    (void)mp1::recv(world, value, 0, mp1::tag{5301});
    BOOST_TEST(value == 32);
  }
  world_barrier();

  {
    int ready_value = 0;
    mp1::request posted;
    if (rank == 1)
      posted = mp1::irecv(world, ready_value, 0, mp1::tag{5302});
    world_barrier();
    if (rank == 0) {
      int value = 33;
      mp1::request ready = mp1::irsend(world, value, 1, mp1::tag{5302});
      (void)ready.wait();
    }
    if (rank == 1) {
      (void)posted.wait();
      BOOST_TEST(ready_value == 33);
    }
    world_barrier();
  }

  if (rank == 0) {
    int value = 41;
    mp1::request persistent = mp1::send_init(world, value, 1, mp1::tag{5400});
    BOOST_TEST(persistent.is_persistent());
    BOOST_TEST(not persistent.is_active());
    persistent.start();
    BOOST_TEST(persistent.is_active());
    (void)persistent.wait();
    BOOST_TEST(not persistent.is_active());
    persistent.start();
    (void)persistent.wait();
  } else if (rank == 1) {
    int value{0};
    (void)mp1::recv(world, value, 0, mp1::tag{5400});
    BOOST_TEST(value == 41);
    (void)mp1::recv(world, value, 0, mp1::tag{5400});
    BOOST_TEST(value == 41);
  }
  world_barrier();

  {
    int ready_value = 0;
    mp1::request posted;
    if (rank == 1)
      posted = mp1::irecv(world, ready_value, 0, mp1::tag{5401});
    world_barrier();
    if (rank == 0) {
      int send_value = 42;
      mp1::request ready = mp1::rsend_init(world, send_value, 1, mp1::tag{5401});
      ready.start();
      (void)ready.wait();
    }
    if (rank == 1) {
      (void)posted.wait();
      BOOST_TEST(ready_value == 42);
    }
    world_barrier();
  }

  {
    int rank0_value = 0;
    int rank1_value = 0;
    mp1::request rank0_recv;
    mp1::request rank1_recv;
    if (rank == 0)
      rank0_recv = mp1::recv_init(world, rank0_value, 1, mp1::tag{5500});
    else if (rank == 1)
      rank1_recv = mp1::recv_init(world, rank1_value, 0, mp1::tag{5500});
    if (rank == 0)
      rank0_recv.start();
    else if (rank == 1)
      rank1_recv.start();
    world_barrier();
    if (rank == 0)
      mp1::send(world, 52, 1, mp1::tag{5500});
    else if (rank == 1)
      mp1::send(world, 51, 0, mp1::tag{5500});
    if (rank == 0) {
      (void)rank0_recv.wait();
      BOOST_TEST(rank0_value == 51);
    } else if (rank == 1) {
      (void)rank1_recv.wait();
      BOOST_TEST(rank1_value == 52);
    }
    world_barrier();
  }
}

BOOST_AUTO_TEST_CASE(containers_and_iterators) {
  BOOST_TEST(true);
  const int size = world_size();
  if (size < 2)
    return;
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();

  // vector auto-resize
  if (rank == 0) {
    const std::vector<int> sent{1, 2, 3, 4};
    mp1::send(world, sent, 1, mp1::tag{6000});
  } else if (rank == 1) {
    std::vector<int> received;
    (void)mp1::recv(world, received, 0, mp1::tag{6000});
    BOOST_TEST(received == (std::vector<int>{1, 2, 3, 4}));
  }
  world_barrier();

  // string
  if (rank == 0) {
    const std::string sent = "hello";
    mp1::send(world, sent, 1, mp1::tag{6001});
  } else if (rank == 1) {
    std::string received;
    (void)mp1::recv(world, received, 0, mp1::tag{6001});
    BOOST_TEST(received == "hello");
  }
  world_barrier();

  // list send (staged); a sequence container is send-only: the receive target is a vector.
  if (rank == 0) {
    const std::list<int> sent{5, 6, 7};
    mp1::send(world, sent, 1, mp1::tag{6002});
  } else if (rank == 1) {
    std::vector<int> received;
    (void)mp1::recv(world, received, 0, mp1::tag{6002});
    BOOST_TEST(received.size() == 3);
    BOOST_TEST(received.front() == 5);
    BOOST_TEST(received.back() == 7);
  }
  world_barrier();

  // flat sequence in explicit order (an associative container is rejected; see
  // test/compile_fail/p2p_map_send.cc)
  if (rank == 0) {
    const std::vector<int> sent{3, 1, 2};
    mp1::send(world, sent, 1, mp1::tag{6003});
  } else if (rank == 1) {
    std::vector<int> received;
    (void)mp1::recv(world, received, 0, mp1::tag{6003});
    BOOST_TEST(received == (std::vector<int>{3, 1, 2}));
  }
  world_barrier();

  // fixed span range (pointer + count is the std::span spelling)
  if (rank == 0) {
    const std::array<int, 3> sent{{8, 9, 10}};
    mp1::send(world, std::span(sent), 1, mp1::tag{6004});
  } else if (rank == 1) {
    std::array<int, 3> received{};
    (void)mp1::recv(world, std::span(received), 0, mp1::tag{6004});
    BOOST_TEST(received[0] == 8);
    BOOST_TEST(received[2] == 10);
  }
  world_barrier();

  // region (typed layout) send/recv
  if (rank == 0) {
    int sent[4] = {11, 12, 13, 14};
    const mp1::layout<int> l = mp1::layout<int>::contiguous(3);
    mp1::send(world, mp1::region{sent, l}, 1, mp1::tag{6005});
  } else if (rank == 1) {
    int received[4] = {0, 0, 0, 0};
    const mp1::layout<int> l = mp1::layout<int>::contiguous(3);
    (void)mp1::recv(world, mp1::region{received, l}, 0, mp1::tag{6005});
    BOOST_TEST(received[0] == 11);
    BOOST_TEST(received[2] == 13);
  }
  world_barrier();

  // non-contiguous send (staged) into a fixed receive extent
  if (rank == 0) {
    const std::list<int> sent{15, 16, 17};
    mp1::send(world, sent, 1, mp1::tag{6006});
  } else if (rank == 1) {
    std::array<int, 3> received{};
    (void)mp1::recv(world, received, 0, mp1::tag{6006});
    BOOST_TEST(received[0] == 15);
    BOOST_TEST(received[2] == 17);
  }
  world_barrier();

  // capacity receive resizes down to the actual message length
  if (rank == 0) {
    const std::vector<int> sent{21, 22, 23};
    mp1::send(world, sent, 1, mp1::tag{6007});
  } else if (rank == 1) {
    std::vector<int> received(8, -1);
    (void)mp1::recv(world, received, mp1::capacity{8}, 0, mp1::tag{6007});
    BOOST_TEST(received.size() == 3);
    BOOST_TEST(received[1] == 22);
  }
  world_barrier();

  // a named contiguous container is borrowed by the request (an rvalue container is a
  // compile-time error; see test/compile_fail/isend_container_rvalue.cc)
  if (rank == 0) {
    const std::vector<int> sent{31, 32};
    mp1::request request = mp1::isend(world, sent, 1, mp1::tag{6008});
    (void)request.wait();
  } else if (rank == 1) {
    std::vector<int> received;
    (void)mp1::recv(world, received, 0, mp1::tag{6008});
    BOOST_TEST(received == (std::vector<int>{31, 32}));
  }
  world_barrier();

  // sequence lvalue isend borrows the container (staged by the request)
  if (rank == 0) {
    std::list<int> sent{41, 42};
    mp1::request request = mp1::isend(world, sent, 1, mp1::tag{6009});
    (void)request.wait();
  } else if (rank == 1) {
    std::vector<int> received;
    (void)mp1::recv(world, received, 0, mp1::tag{6009});
    BOOST_TEST(received.size() == 2);
  }
  world_barrier();

  // borrowed span isend/irecv
  if (rank == 0) {
    std::vector<int> sent{51, 52, 53};
    mp1::request request = mp1::isend(world, std::span(sent), 1, mp1::tag{6010});
    (void)request.wait();
  } else if (rank == 1) {
    std::vector<int> received(3, 0);
    mp1::request request = mp1::irecv(world, std::span(received), 0, mp1::tag{6010});
    (void)request.wait();
    BOOST_TEST(received[2] == 53);
  }
  world_barrier();

  // capacity irecv resizes after completion
  if (rank == 0) {
    const std::vector<int> sent{61, 62};
    mp1::send(world, sent, 1, mp1::tag{6011});
  } else if (rank == 1) {
    std::vector<int> received(6, -1);
    mp1::request request = mp1::irecv(world, received, mp1::capacity{6}, 0, mp1::tag{6011});
    (void)request.wait();
    BOOST_TEST(received.size() == 2);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(probe_and_matched_receive) {
  BOOST_TEST(true);
  const int size = world_size();
  if (size < 2)
    return;
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();

  world_barrier();
  if (rank == 1)
    BOOST_TEST(not mp1::iprobe(world).has_value());
  world_barrier();

  if (rank == 0) {
    const std::vector<int> sent{71, 72, 73};
    mp1::send(world, sent, 1, mp1::tag{6100});
  } else if (rank == 1) {
    const mp1::status probed = mp1::probe(world, 0, mp1::tag{6100});
    BOOST_TEST(probed.tag().value == 6100);
    BOOST_TEST(probed.source() == 0);
    BOOST_TEST(probed.get_count<int>() == 3);
    std::vector<int> received;
    (void)mp1::recv(world, received, 0, mp1::tag{6100});
    BOOST_TEST(received.size() == 3);
  }
  world_barrier();

  if (rank == 0) {
    mp1::send(world, 74, 1, mp1::tag{6101});
  } else if (rank == 1) {
    mp1::message matched = mp1::mprobe(world, 0, mp1::tag{6101});
    BOOST_TEST(matched.is_valid());
    BOOST_TEST(not matched.is_no_proc());
    BOOST_TEST(matched.probed_status().get_count<int>() == 1);
    int value{0};
    const mp1::status received = mp1::mrecv(value, matched);
    BOOST_TEST(value == 74);
    BOOST_TEST(received.tag().value == 6101);
    BOOST_TEST(not matched.is_valid());
  }
  world_barrier();

  if (rank == 0) {
    const std::list<int> sent{75, 76};
    mp1::send(world, sent, 1, mp1::tag{6102});
  } else if (rank == 1) {
    mp1::message matched = mp1::mprobe(world, 0, mp1::tag{6102});
    std::array<int, 2> received{};
    (void)mp1::mrecv(std::span(received), matched);
    BOOST_TEST(received.front() == 75);
    BOOST_TEST(received.back() == 76);
  }
  world_barrier();

  if (rank == 0) {
    mp1::send(world, 77, 1, mp1::tag{6103});
  } else if (rank == 1) {
    mp1::message matched = mp1::mprobe(world, 0, mp1::tag{6103});
    int value{0};
    mp1::request request = mp1::imrecv(value, matched);
    (void)request.wait();
    BOOST_TEST(value == 77);
  }
  world_barrier();

  if (rank == 1) {
    BOOST_TEST(not mp1::improbe(world).has_value());
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(capacity_type_and_fixed_extent_receive) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();

  // The capacity is a strong type: the five-argument form with a raw count must not compile
  // (test/compile_fail/capacity_raw_count.cc); this call is the documented spelling.
  if (rank == 0) {
    const std::vector<int> sent{101, 102, 103};
    mp1::send(world, sent, 1, mp1::tag{6300});
  } else if (rank == 1) {
    std::vector<int> received;
    const mp1::status status = mp1::recv(world, received, mp1::capacity{8}, 0, mp1::tag{6300});
    BOOST_TEST(received.size() == 3);
    BOOST_TEST(received[0] == 101);
    BOOST_TEST(received[2] == 103);
    BOOST_TEST(status.get_count<int>() == 3);
  }
  world_barrier();

  // std::array fixed-extent receive: no capacity argument, no resize.
  if (rank == 0) {
    const std::array<int, 3> sent{{111, 112, 113}};
    mp1::send(world, sent, 1, mp1::tag{6301});
  } else if (rank == 1) {
    std::array<int, 3> received{};
    const mp1::status status = mp1::recv(world, received, 0, mp1::tag{6301});
    BOOST_TEST(received[0] == 111);
    BOOST_TEST(received[2] == 113);
    BOOST_TEST(status.get_count<int>() == 3);
  }
  world_barrier();

  // Fixed-extent irecv and persistent recv_init forms.
  if (rank == 0) {
    const std::array<int, 2> sent{{121, 122}};
    mp1::send(world, sent, 1, mp1::tag{6302});
    mp1::send(world, sent, 1, mp1::tag{6303});
  } else if (rank == 1) {
    std::array<int, 2> received{};
    mp1::request request = mp1::irecv(world, received, 0, mp1::tag{6302});
    (void)request.wait();
    BOOST_TEST(received[1] == 122);
    std::array<int, 2> persistent_received{};
    mp1::request persistent = mp1::recv_init(world, persistent_received, 0, mp1::tag{6303});
    persistent.start();
    (void)persistent.wait();
    BOOST_TEST(persistent_received[0] == 121);
  }
  world_barrier();

  // The four-argument container receive is the auto-size form: `recv(c, v, source, tag)`.
  // At 3+ ranks rank 2 receives from rank 1 by spelling `recv(c, v, 1, tag)`; the capacity
  // requires the explicit `mp1::capacity` type, so this call cannot be read as a capacity.
  if (world_size() >= 3) {
    if (rank == 0) {
      const std::vector<int> sent{131};
      mp1::send(world, sent, 2, mp1::tag{6304});
    } else if (rank == 1) {
      const std::vector<int> sent{132, 133};
      mp1::send(world, sent, 2, mp1::tag{6304});
    } else if (rank == 2) {
      std::vector<int> received;
      (void)mp1::recv(world, received, 1, mp1::tag{6304});  // source 1, tag 6304; auto-sized
      BOOST_TEST(received.size() == 2);
      BOOST_TEST(received[0] == 132);
      std::vector<int> from_zero;  // drain rank 0's message so the test leaves none matched
      (void)mp1::recv(world, from_zero, 0, mp1::tag{6304});
      BOOST_TEST(from_zero.size() == 1);
      BOOST_TEST(from_zero[0] == 131);
    }
    world_barrier();
  }

  // A too-small capacity is a truncation error (MPI 4.1 §3.2.4); the message is consumed and
  // the operation reports an mpi_error, not a silent short read.
  if (rank == 0) {
    const std::vector<int> sent{141, 142, 143};
    mp1::ssend(world, sent, 1, mp1::tag{6305});  // completes only after the truncated receive consumes it
  } else if (rank == 1) {
    world_return_errors_scope return_errors;
    std::vector<int> received(4, -1);
    bool threw{false};
    try {
      (void)mp1::recv(world, received, mp1::capacity{1}, 0, mp1::tag{6305});
    } catch (const mp1::mpi_error&) {
      threw = true;
    }
    BOOST_TEST(threw);
    BOOST_TEST(not mp1::iprobe(world, 0, mp1::tag{6305}).has_value());
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(matched_message_is_owning_and_move_only) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();

  // A matched message can be received exactly once; the handle is invalid afterwards.
  if (rank == 0) {
    mp1::send(world, 201, 1, mp1::tag{6310});
  } else if (rank == 1) {
    mp1::message matched = mp1::mprobe(world, 0, mp1::tag{6310});
    BOOST_TEST(matched.is_valid());
    BOOST_TEST(not matched.is_no_proc());
    BOOST_TEST(matched.probed_status().get_count<int>() == 1);
    int value{0};
    const mp1::status status = mp1::mrecv(value, matched);
    BOOST_TEST(value == 201);
    BOOST_TEST(status.tag().value == 6310);
    BOOST_TEST(not matched.is_valid());
    bool threw{false};
    try {
      (void)mp1::mrecv(value, matched);  // second use: rejected before MPI
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    BOOST_TEST(threw);
  }
  world_barrier();

  // Move transfers ownership and resets the source.
  if (rank == 0) {
    mp1::send(world, 202, 1, mp1::tag{6311});
  } else if (rank == 1) {
    mp1::message source = mp1::mprobe(world, 0, mp1::tag{6311});
    mp1::message target = std::move(source);
    BOOST_TEST(not source.is_valid());
    BOOST_TEST(target.is_valid());
    mp1::message assigned;
    assigned = std::move(target);
    BOOST_TEST(not target.is_valid());
    int value{0};
    (void)mp1::mrecv(value, assigned);
    BOOST_TEST(value == 202);
  }
  world_barrier();

  // discard() consumes without interpreting the type and unblocks a synchronous sender.
  if (rank == 0) {
    mp1::ssend(world, 203, 1, mp1::tag{6312});
  } else if (rank == 1) {
    mp1::message matched = mp1::mprobe(world, 0, mp1::tag{6312});
    matched.discard();
    BOOST_TEST(not matched.is_valid());
    BOOST_TEST(not mp1::iprobe(world, 0, mp1::tag{6312}).has_value());
  }
  world_barrier();

  // The destructor discards a still-owned matched message (no raw handle leak).
  if (rank == 0) {
    mp1::ssend(world, 204, 1, mp1::tag{6313});
  } else if (rank == 1) {
    {
      mp1::message matched = mp1::mprobe(world, 0, mp1::tag{6313});
      BOOST_TEST(matched.is_valid());
    }
    BOOST_TEST(not mp1::iprobe(world, 0, mp1::tag{6313}).has_value());
  }
  world_barrier();

  // A default/invalid message is rejected before MPI.
  {
    mp1::message invalid;
    int value{0};
    bool threw{false};
    try {
      (void)mp1::mrecv(value, invalid);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    BOOST_TEST(threw);
  }
  world_barrier();

  // improbe returns an engaged owning message; nullopt means no match.
  if (rank == 1) {
    BOOST_TEST(not mp1::improbe(world, 0, mp1::tag{6314}).has_value());
  }
  world_barrier();
  if (rank == 0) {
    mp1::send(world, 205, 1, mp1::tag{6314});
  } else if (rank == 1) {
    std::optional<mp1::message> matched;
    while (not matched.has_value())
      matched = mp1::improbe(world, 0, mp1::tag{6314});
    int value{0};
    (void)mp1::mrecv(value, *matched);
    BOOST_TEST(value == 205);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(matched_message_no_proc) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();

  mp1::message matched = mp1::mprobe(world, MPI_PROC_NULL);
  BOOST_TEST(matched.is_valid());
  BOOST_TEST(matched.is_no_proc());
  BOOST_TEST(matched.probed_status().source() == MPI_PROC_NULL);
  BOOST_TEST(matched.probed_status().tag().value == mp1::any_tag.value);
  BOOST_TEST(matched.probed_status().get_count<int>() == 0);
  matched.discard();
  BOOST_TEST(not matched.is_valid());

  std::optional<mp1::message> improbed = mp1::improbe(world, MPI_PROC_NULL);
  BOOST_TEST(improbed.has_value());
  BOOST_TEST(improbed->is_no_proc());
  improbed->discard();

  // Receiving MPI_MESSAGE_NO_PROC is legal and yields the PROC_NULL status.
  mp1::message received_no_proc = mp1::mprobe(world, MPI_PROC_NULL);
  int value{0};
  const mp1::status status = mp1::mrecv(value, received_no_proc);
  BOOST_TEST(status.source() == MPI_PROC_NULL);
  BOOST_TEST(status.tag().value == mp1::any_tag.value);
  BOOST_TEST(status.error() == MPI_SUCCESS);
  BOOST_TEST(status.get_count<int>() == 0);
  BOOST_TEST(status.get_elements<int>() == 0);
  BOOST_TEST(not status.is_cancelled());
  BOOST_TEST(not received_no_proc.is_valid());
}

// MPI 4.1 §3.10 defines the completion status of a receive from MPI_PROC_NULL, including the
// count (0).  The wrapper synthesizes that status for every request form; pinned MPICH 4.3.2
// leaves SOURCE at MPI_ANY_SOURCE for a persistent receive initialized with MPI_PROC_NULL, so
// forwarding the provider's envelope would make the status provider-dependent.
BOOST_AUTO_TEST_CASE(proc_null_request_status_is_standard) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();
  int value = 42;

  const auto check_status = [](const mp1::status& s) {
    BOOST_TEST(s.source() == MPI_PROC_NULL);
    BOOST_TEST(s.tag().value == mp1::any_tag.value);
    BOOST_TEST(s.error() == MPI_SUCCESS);
    BOOST_TEST(s.get_count<int>() == 0);
    BOOST_TEST(s.get_elements<int>() == 0);
    BOOST_TEST(not s.is_cancelled());
  };

  // Nonblocking receive from MPI_PROC_NULL.
  mp1::request nonblocking = mp1::irecv(world, value, MPI_PROC_NULL, mp1::tag{6400});
  check_status(nonblocking.wait());
  BOOST_TEST(static_cast<int>(nonblocking.state()) ==
             static_cast<int>(mp1::request_state::freed));
  BOOST_TEST(value == 42);

  // Persistent receive from MPI_PROC_NULL: every start/wait cycle reports the standard
  // status and leaves the buffer untouched.
  mp1::request persistent = mp1::recv_init(world, value, MPI_PROC_NULL, mp1::tag{6401});
  BOOST_TEST(persistent.is_persistent());
  for (int cycle = 0; cycle < 3; ++cycle) {
    persistent.start();
    check_status(persistent.wait());
    BOOST_TEST(static_cast<int>(persistent.state()) ==
               static_cast<int>(mp1::request_state::inactive));
    BOOST_TEST(persistent.is_valid());
    BOOST_TEST(value == 42);
  }
  persistent.release();

  // Nonblocking matched receive of MPI_MESSAGE_NO_PROC.
  mp1::message no_proc = mp1::mprobe(world, MPI_PROC_NULL);
  BOOST_TEST(no_proc.is_no_proc());
  mp1::request matched = mp1::imrecv(value, no_proc);
  check_status(matched.wait());
  BOOST_TEST(static_cast<int>(matched.state()) == static_cast<int>(mp1::request_state::freed));
  BOOST_TEST(not no_proc.is_valid());
  BOOST_TEST(value == 42);

  // The metadata survives moving a completed/inactive request.
  mp1::request moved_from = mp1::recv_init(world, value, MPI_PROC_NULL, mp1::tag{6402});
  mp1::request moved_to = std::move(moved_from);
  moved_to.start();
  check_status(moved_to.wait());
  moved_to.release();
}

BOOST_AUTO_TEST_CASE(auto_size_recv_is_loss_free) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();

  // Type-signature mismatch (MPI_UNDEFINED count): the matched message must be consumed
  // before the exception, so the synchronous sender completes and no message is stranded.
  if (rank == 0) {
    mp1::ssend(world, 301, 1, mp1::tag{6320});
  } else if (rank == 1) {
    std::vector<long> received;
    bool threw{false};
    try {
      (void)mp1::recv(world, received, 0, mp1::tag{6320});
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    BOOST_TEST(threw);
    BOOST_TEST(not mp1::iprobe(world, 0, mp1::tag{6320}).has_value());
  }
  world_barrier();

  // Container resize failure (bad_alloc): same loss-free guarantee.
  if (rank == 0) {
    mp1::ssend(world, 302, 1, mp1::tag{6321});
  } else if (rank == 1) {
    std::vector<int, throwing_allocator<int>> received;
    bool threw{false};
    try {
      (void)mp1::recv(world, received, 0, mp1::tag{6321});
    } catch (const std::bad_alloc&) {
      threw = true;
    }
    BOOST_TEST(threw);
    BOOST_TEST(not mp1::iprobe(world, 0, mp1::tag{6321}).has_value());
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(sendrecv_and_sendrecv_replace) {
  BOOST_TEST(true);
  const int size = world_size();
  if (size < 2)
    return;
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();
  const int next = (rank + 1) % size;
  const int previous = (rank + size - 1) % size;

  {
    int received{0};
    (void)mp1::sendrecv(world, rank, next, mp1::tag{6200}, received, previous, mp1::tag{6200});
    BOOST_TEST(received == previous);
  }
  world_barrier();

  {
    int value = rank;
    (void)mp1::sendrecv_replace(world, value, next, mp1::tag{6201}, previous, mp1::tag{6201});
    BOOST_TEST(value == previous);
  }
  world_barrier();

  {
    const std::vector<int> sent{rank, rank + 100};
    std::vector<int> received(2, 0);
    (void)mp1::sendrecv(world, sent, next, mp1::tag{6202}, received, previous, mp1::tag{6202});
    BOOST_TEST(received[0] == previous);
    BOOST_TEST(received[1] == previous + 100);
  }
  world_barrier();

  {
    int data = rank;
    (void)mp1::sendrecv_replace(world, data, next, mp1::tag{6203}, previous, mp1::tag{6203});
    BOOST_TEST(data == previous);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(proc_null_and_local_validation) {
  BOOST_TEST(true);
  const int size = world_size();
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();

  int scalar = 42;
  const mp1::status empty = mp1::recv(world, scalar, MPI_PROC_NULL, mp1::tag{6300});
  BOOST_TEST(is_proc_null_status(empty));
  BOOST_TEST(scalar == 42);
  mp1::send(world, 7, MPI_PROC_NULL, mp1::tag{6301});

  std::vector<int> container{1, 2, 3};
  const mp1::status container_status = mp1::recv(world, container, MPI_PROC_NULL, mp1::tag{6302});
  BOOST_TEST(is_proc_null_status(container_status));
  BOOST_TEST(container.size() == 3);
  BOOST_TEST(container[2] == 3);

  // MPI_PROC_NULL ignores the buffer, so a null span with a positive extent is legal.
  (void)mp1::recv(world, std::span<int>(static_cast<int*>(nullptr), 5), MPI_PROC_NULL, mp1::tag{6303});
  mp1::send(world, std::span<const int>(static_cast<const int*>(nullptr), 5), MPI_PROC_NULL, mp1::tag{6304});

  BOOST_TEST(throws_invalid_argument([&world] {
    int x{0};
    mp1::send(world, x, world_size(), mp1::tag{6400});
  }));
  BOOST_TEST(throws_invalid_argument([&world] {
    int x{0};
    (void)mp1::recv(world, x, world_size(), mp1::tag{6400});
  }));
  BOOST_TEST(throws_invalid_argument([&world] {
    int x{0};
    mp1::send(world, x, -10, mp1::tag{6400});  // neither MPI_PROC_NULL nor a rank
  }));
  BOOST_TEST(throws_invalid_argument([&world] {
    int x{0};
    mp1::send(world, x, 0, mp1::tag{-1});
  }));
  BOOST_TEST(throws_invalid_argument([&world] {
    int x{0};
    (void)mp1::recv(world, x, 0, mp1::tag{-2});
  }));
  BOOST_TEST(throws_invalid_argument([&world] { (void)mp1::probe(world, world_size(), mp1::tag{0}); }));
  BOOST_TEST(throws_invalid_argument([] {
    int x{0};
    (void)mp1::recv(mp1::comm_null(), x, 0, mp1::tag{0});
  }));
  BOOST_TEST(throws_invalid_argument([&world] {
    int x{0};
    mp1::request request = mp1::isend(world, x, world_size(), mp1::tag{6500});
    (void)request;
  }));
  (void)size;
  (void)rank;
}

BOOST_AUTO_TEST_CASE(blocking_proc_null_status_is_standard) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();

  // MPI 4.1 §3.10 defines the whole status object of a receive or probe with source
  // MPI_PROC_NULL.  Every blocking form (scalar, fixed buffer, region, auto-size container,
  // bounded capacity, sendrecv/sendrecv_replace receive side) synthesizes the standard
  // normalized status instead of forwarding the provider's envelope, so kind() is normalized
  // and the count/elements/cancellation answers need no MPI query.
  const auto check_standard = [](const mp1::status& s) {
    BOOST_TEST(static_cast<int>(s.kind()) == static_cast<int>(mp1::status_kind::normalized));
    BOOST_TEST(s.source() == MPI_PROC_NULL);
    BOOST_TEST(s.tag().value == mp1::any_tag.value);
    BOOST_TEST(s.error() == MPI_SUCCESS);
    BOOST_TEST(s.get_count<int>() == 0);
    BOOST_TEST(s.get_elements<int>() == 0);
    BOOST_TEST(not s.is_cancelled());
#if defined(MP1_TEST_WRAP)
    if (mpi_wrap::supported()) {
      // The same answers must not reach MPI: make_proc_null() records count 0 and no
      // cancellation query, exactly like the nonblocking/persistent/matched statuses.
      mpi_wrap::reset_counts();
      (void)s.get_count<int>();
      (void)s.get_elements<int>();
      (void)s.is_cancelled();
      BOOST_TEST(mpi_wrap::total() == 0);
    }
#endif
  };

  int scalar = 42;
  check_standard(mp1::recv(world, scalar, MPI_PROC_NULL, mp1::tag{6801}));
  BOOST_TEST(scalar == 42);

  std::array<int, 3> fixed{-1, -2, -3};
  check_standard(mp1::recv(world, fixed, MPI_PROC_NULL, mp1::tag{6802}));
  BOOST_TEST(fixed[0] == -1);
  BOOST_TEST(fixed[2] == -3);

  const mp1::layout<int> slots = mp1::layout<int>::contiguous(3);
  std::array<int, 3> region_data{-4, -5, -6};
  check_standard(
      mp1::recv(world, mp1::region{region_data.data(), slots}, MPI_PROC_NULL, mp1::tag{6803}));
  BOOST_TEST(region_data[0] == -4);

  std::vector<int> container{1, 2, 3};
  check_standard(mp1::recv(world, container, MPI_PROC_NULL, mp1::tag{6804}));
  BOOST_TEST(container.size() == 3);
  BOOST_TEST(container[2] == 3);

  std::vector<int> bounded{7, 8};
  check_standard(mp1::recv(world, bounded, mp1::capacity{16}, MPI_PROC_NULL, mp1::tag{6805}));
  BOOST_TEST(bounded.size() == 2);

  check_standard(mp1::probe(world, MPI_PROC_NULL, mp1::tag{6806}));

  int send_value = 7;
  int recv_value = -1;
  check_standard(mp1::sendrecv(world, send_value, MPI_PROC_NULL, mp1::tag{6807}, recv_value,
                               MPI_PROC_NULL, mp1::tag{6807}));
  BOOST_TEST(recv_value == -1);

  int replace_value = 11;
  check_standard(mp1::sendrecv_replace(world, replace_value, MPI_PROC_NULL, mp1::tag{6808},
                                       MPI_PROC_NULL, mp1::tag{6808}));
  BOOST_TEST(replace_value == 11);

  // A real receive keeps the provider's native, count-defining status: the PROC_NULL
  // normalization is selected by the requested source, never by looking at status fields.
  const int size = world_size();
  if (size >= 2) {
    const int rank = world_rank();
    if (rank == 0) {
      int out = 77;
      int unused = -1;
      const mp1::status sent =
          mp1::sendrecv(world, out, 1, mp1::tag{6809}, unused, MPI_PROC_NULL, mp1::tag{6809});
      check_standard(sent);  // receive side is MPI_PROC_NULL
      BOOST_TEST(unused == -1);
    } else if (rank == 1) {
      int unused = -1;
      int received = 0;
      const mp1::status status = mp1::sendrecv(world, unused, MPI_PROC_NULL, mp1::tag{6809},
                                               received, 0, mp1::tag{6809});
      BOOST_TEST(static_cast<int>(status.kind()) == static_cast<int>(mp1::status_kind::native));
      BOOST_TEST(status.source() == 0);
      BOOST_TEST(status.tag().value == mp1::tag{6809}.value);
      BOOST_TEST(status.get_count<int>() == 1);
      BOOST_TEST(received == 77);
    }
    world_barrier();

    if (rank == 0) {
      mp1::send(world, 88, 1, mp1::tag{6810});
    } else if (rank == 1) {
      const mp1::status probed = mp1::probe(world, 0, mp1::tag{6810});
      BOOST_TEST(static_cast<int>(probed.kind()) == static_cast<int>(mp1::status_kind::native));
      BOOST_TEST(probed.source() == 0);
      BOOST_TEST(probed.tag().value == mp1::tag{6810}.value);
      BOOST_TEST(probed.get_count<int>() == 1);
      int received = 0;
      const mp1::status status = mp1::recv(world, received, 0, mp1::tag{6810});
      BOOST_TEST(static_cast<int>(status.kind()) == static_cast<int>(mp1::status_kind::native));
      BOOST_TEST(status.source() == 0);
      BOOST_TEST(status.get_count<int>() == 1);
      BOOST_TEST(received == 88);
    }
    world_barrier();
  }
}

BOOST_AUTO_TEST_CASE(iprobe_proc_null_status_is_standard) {
  BOOST_TEST(true);
  const mp1::mpi_communicator world = mp1::comm_world();

  // MPI 4.1 §3.10 applies to a successful MPI_Iprobe with source MPI_PROC_NULL exactly like
  // the blocking probe: source MPI_PROC_NULL, tag any_tag, count 0 and success. The wrapper
  // synthesizes that status, so neither the count nor the cancellation query reaches MPI.
  const auto check_standard = [](const mp1::status& s) {
    BOOST_TEST(static_cast<int>(s.kind()) == static_cast<int>(mp1::status_kind::normalized));
    BOOST_TEST(s.source() == MPI_PROC_NULL);
    BOOST_TEST(s.tag().value == mp1::any_tag.value);
    BOOST_TEST(s.error() == MPI_SUCCESS);
    BOOST_TEST(s.get_count<int>() == 0);
    BOOST_TEST(s.get_elements<int>() == 0);
    BOOST_TEST(not s.is_cancelled());
#if defined(MP1_TEST_WRAP)
    if (mpi_wrap::supported()) {
      mpi_wrap::reset_counts();
      (void)s.get_count<int>();
      (void)s.get_elements<int>();
      (void)s.is_cancelled();
      BOOST_TEST(mpi_wrap::total() == 0);
    }
#endif
  };

  // A successful MPI_PROC_NULL iprobe matches immediately, even though the requested tag is
  // concrete: the synthesized status always reports the any_tag wildcard.
  const std::optional<mp1::status> proc_null =
      mp1::iprobe(world, MPI_PROC_NULL, mp1::tag{6811});
  BOOST_TEST(proc_null.has_value());
  if (proc_null)
    check_standard(*proc_null);

  // A real source keeps the provider's native, count-defining status.
  const int size = world_size();
  if (size >= 2) {
    const int rank = world_rank();
    if (rank == 0) {
      mp1::send(world, 88, 1, mp1::tag{6812});
    } else if (rank == 1) {
      std::optional<mp1::status> probed;
      while (not probed)
        probed = mp1::iprobe(world, 0, mp1::tag{6812});
      BOOST_TEST(static_cast<int>(probed->kind()) ==
                 static_cast<int>(mp1::status_kind::native));
      BOOST_TEST(probed->source() == 0);
      BOOST_TEST(probed->tag().value == mp1::tag{6812}.value);
      BOOST_TEST(probed->get_count<int>() == 1);
      BOOST_TEST(not probed->is_cancelled());
      int received = 0;
      const mp1::status status = mp1::recv(world, received, 0, mp1::tag{6812});
      BOOST_TEST(static_cast<int>(status.kind()) == static_cast<int>(mp1::status_kind::native));
      BOOST_TEST(status.source() == 0);
      BOOST_TEST(status.get_count<int>() == 1);
      BOOST_TEST(received == 88);
    }
    world_barrier();
  }
}

BOOST_AUTO_TEST_CASE(empty_variants) {
  BOOST_TEST(true);
  const int size = world_size();
  if (size < 2)
    return;
  const int rank = world_rank();
  const mp1::mpi_communicator world = mp1::comm_world();
  mp1::bsend_buffer buffer{1024};

  // The empty spelling is std::span<T>{} (zero elements with T's datatype); it matches any
  // zero-count message because a zero-count type signature contributes no elements.
  if (rank == 0) {
    mp1::send(world, std::span<int>{}, 1, mp1::tag{6600});
    mp1::bsend(world, std::span<int>{}, 1, mp1::tag{6601});
    mp1::ssend(world, std::span<int>{}, 1, mp1::tag{6602});
    mp1::rsend(world, std::span<int>{}, 1, mp1::tag{6603});
    mp1::request send = mp1::isend(world, std::span<int>{}, 1, mp1::tag{6604});
    (void)send.wait();
  } else if (rank == 1) {
    (void)mp1::recv(world, std::span<int>{}, 0, mp1::tag{6600});
    (void)mp1::recv(world, std::span<int>{}, 0, mp1::tag{6601});
    (void)mp1::recv(world, std::span<int>{}, 0, mp1::tag{6602});
    (void)mp1::recv(world, std::span<int>{}, 0, mp1::tag{6603});
    mp1::request receive = mp1::irecv(world, std::span<int>{}, 0, mp1::tag{6604});
    (void)receive.wait();
  }
  world_barrier();

  if (rank == 0) {
    mp1::request persistent = mp1::send_init(world, std::span<int>{}, 1, mp1::tag{6605});
    persistent.start();
    (void)persistent.wait();
  } else if (rank == 1) {
    mp1::request persistent = mp1::recv_init(world, std::span<int>{}, 0, mp1::tag{6605});
    persistent.start();
    (void)persistent.wait();
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(wrap_counters) {
  BOOST_TEST(true);
#if defined(MP1_TEST_WRAP)
  if (not mpi_wrap::supported())
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  {
    mpi_wrap::reset_counts();
    {
      mp1::communicator duplicate = mp1::communicator::dup(world);
      (void)duplicate;
    }
    BOOST_TEST(mpi_wrap::count("MPI_Comm_free") == 1);
  }

  {
    const mp1::group all = mp1::get_group(world);
    mpi_wrap::reset_counts();
    {
      mp1::group subset = mp1::group_incl(all, std::vector<int>{0});
      (void)subset;
    }
    BOOST_TEST(mpi_wrap::count("MPI_Group_free") == 1);
  }

  {
    int send_value = 1;
    int recv_value = 0;
    mp1::request posted = mp1::irecv(world, recv_value, rank, mp1::tag{6700});
    mpi_wrap::reset_counts();
    {
      mp1::request dropped = mp1::isend(world, send_value, rank, mp1::tag{6700});
      (void)dropped;
    }
    BOOST_TEST(mpi_wrap::count("MPI_Request_free") == 1);
    (void)posted.wait();
  }
#else
  BOOST_TEST(true);
#endif
}

BOOST_AUTO_TEST_SUITE_END()
