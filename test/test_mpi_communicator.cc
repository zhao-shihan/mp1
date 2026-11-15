// mpi_communicator view / communicator owner tests.
//
// The test covers a non-owning copyable view backed by an owning move-only communicator,
// the owner-only factories and free(), and always-on validation.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>


namespace {

  template<class F>
  bool throws_invalid_argument(F&& f) {
    try {
      f();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  }

}  // namespace


BOOST_AUTO_TEST_CASE(view_basics) {
  if (not mp1::initialized())
    mp1::init();

  static_assert(std::is_copy_constructible_v<mp1::mpi_communicator>);
  static_assert(std::is_copy_assignable_v<mp1::mpi_communicator>);
  static_assert(std::is_nothrow_destructible_v<mp1::mpi_communicator>);

  const auto comm_world = mp1::comm_world();
  BOOST_TEST(comm_world.is_valid());
  BOOST_TEST((comm_world.native_handle() == MPI_COMM_WORLD));
  BOOST_TEST((comm_world.size() >= 1));
  BOOST_TEST((comm_world.rank() >= 0));
  BOOST_TEST((comm_world.rank() < comm_world.size()));
  BOOST_TEST(not comm_world.is_inter());

  // Views are cheap copies with value semantics for identity.
  auto comm_world_copy = comm_world;
  BOOST_TEST((comm_world_copy == comm_world));
  BOOST_TEST((comm_world_copy.compare(comm_world) == mp1::equality_type::identical));
  BOOST_TEST((comm_world_copy.native_handle() == comm_world.native_handle()));

  // borrow() is the named form of the borrowing constructor.
  const auto borrowed = mp1::mpi_communicator::borrow(MPI_COMM_WORLD);
  BOOST_TEST(borrowed.is_valid());
  BOOST_TEST((borrowed == comm_world));

  // A default view and comm_null() are invalid and MPI-free.
  mp1::mpi_communicator invalid;
  BOOST_TEST(not invalid.is_valid());
  BOOST_TEST(not mp1::comm_null().is_valid());
  BOOST_TEST((invalid == mp1::comm_null()));
  BOOST_TEST((invalid != comm_world));

  // Invalid-view accessors throw std::invalid_argument (not a validity query).
  BOOST_TEST(throws_invalid_argument([&] { (void)invalid.size(); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)invalid.rank(); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)invalid.is_inter(); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)invalid.remote_size(); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)invalid.remote_group(); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)invalid.compare(comm_world); }));

  // remote_size/remote_group are intercommunicator-only.
  BOOST_TEST(throws_invalid_argument([&] { (void)comm_world.remote_size(); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)comm_world.remote_group(); }));

  // A duplicate view uses a different context: congruent, not identical.
  auto owner = mp1::communicator::dup(comm_world);
  mp1::mpi_communicator duplicate_view = static_cast<const mp1::mpi_communicator&>(owner);
  BOOST_TEST(duplicate_view.is_valid());
  BOOST_TEST((duplicate_view != comm_world));
  BOOST_TEST((duplicate_view.compare(comm_world) == mp1::equality_type::congruent));
}


BOOST_AUTO_TEST_CASE(owner_factories) {
  if (not mp1::initialized())
    mp1::init();

  static_assert(not std::is_copy_constructible_v<mp1::communicator>);
  static_assert(not std::is_copy_assignable_v<mp1::communicator>);
  static_assert(std::is_move_constructible_v<mp1::communicator>);

  const auto comm_world = mp1::comm_world();
  const int rank = comm_world.rank();

  // dup / dup_with_info
  auto dup = mp1::communicator::dup(comm_world);
  BOOST_TEST(dup.is_valid());
  BOOST_TEST((dup.size() == comm_world.size()));
  BOOST_TEST((dup.rank() == comm_world.rank()));
  BOOST_TEST((dup.compare(comm_world) == mp1::equality_type::congruent));

  mp1::info i;
  i.set("mp1_test_key", "mp1_test_value");
  auto dup_info = mp1::communicator::dup_with_info(comm_world, i);
  BOOST_TEST(dup_info.is_valid());
  BOOST_TEST((dup_info.compare(comm_world) == mp1::equality_type::congruent));
  // Whether a user key survives MPI_Comm_get_info is implementation-defined; only require the
  // owning info round trip to be usable.
  auto read_back = dup_info.get_info();
  BOOST_TEST((read_back.size() >= 0));

  // split: excluded ranks (MPI_UNDEFINED) get an invalid owning communicator.
  auto split = mp1::communicator::split(comm_world, rank == 0 ? 0 : MPI_UNDEFINED);
  BOOST_TEST((split.is_valid() == (rank == 0)));
  if (rank == 0) {
    BOOST_TEST((split.size() == 1));
    BOOST_TEST((split.rank() == 0));
  }

  // create(): collective over comm_world; only the group members receive a handle.
  auto group_one = mp1::group_incl(mp1::get_group(comm_world), {0});
  auto created = mp1::communicator::create(comm_world, group_one);
  BOOST_TEST((created.is_valid() == (rank == 0)));
  if (rank == 0)
    BOOST_TEST((created.size() == 1));

  // Adoption rejects the handles mp1 does not own.
  BOOST_TEST(throws_invalid_argument([] { (void)mp1::communicator::adopt(MPI_COMM_NULL); }));
  BOOST_TEST(throws_invalid_argument([] { (void)mp1::communicator::adopt(MPI_COMM_WORLD); }));
  BOOST_TEST(throws_invalid_argument([] { (void)mp1::communicator::adopt(MPI_COMM_SELF); }));

  // Move semantics: the moved-from owner is invalid; the target takes over the handle.
  auto moved_to = std::move(dup);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(not dup.is_valid());
  BOOST_TEST(moved_to.is_valid());
  BOOST_TEST((moved_to.size() == comm_world.size()));

  // free() is idempotent; a freed owner is invalid and every MPI operation rejects it.
  moved_to.free();
  BOOST_TEST(not moved_to.is_valid());
  moved_to.free();
  BOOST_TEST(throws_invalid_argument([&] { (void)moved_to.size(); }));

  // The destructor of a live owner frees its handle; dropping one is safe.
  {
    auto scoped = mp1::communicator::dup(comm_world);
    BOOST_TEST(scoped.is_valid());
  }

  // set_info/get_info are owner/view operations; a sliced view cannot free.
  mp1::mpi_communicator info_view = static_cast<const mp1::mpi_communicator&>(dup_info);
  BOOST_TEST(info_view.is_valid());
  auto view_info = info_view.get_info();
  BOOST_TEST((view_info.size() >= 0));
}


BOOST_AUTO_TEST_CASE(validation_errors) {
  if (not mp1::initialized())
    mp1::init();

  const auto comm_world = mp1::comm_world();

  // p2p envelope validation uses the cached local/remote sizes.
  int value = 1;
  BOOST_TEST(throws_invalid_argument([&] { mp1::send(comm_world, value, comm_world.size()); }));
  BOOST_TEST(
      throws_invalid_argument([&] { mp1::send(comm_world, value, comm_world.size() + 1); }));
  BOOST_TEST(throws_invalid_argument([&] { mp1::send(comm_world, value, 0, mp1::tag{-1}); }));
  BOOST_TEST(throws_invalid_argument([&] { mp1::recv(comm_world, value, comm_world.size()); }));

  // MPI_PROC_NULL is always a valid destination even when it is not a rank.
  mp1::send(comm_world, value, MPI_PROC_NULL);
  BOOST_TEST((mp1::recv(comm_world, value, MPI_PROC_NULL).source() == MPI_PROC_NULL));
}
