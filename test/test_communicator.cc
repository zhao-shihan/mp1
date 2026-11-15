// Communicator tests: predefined views, owning factories (dup/split/split_type/create_group),
// compare/equality, and always-on factory validation.  Intercommunicators are covered by
// test_inter_communicator.cc.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <stdexcept>


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


// test properties of the predefined communicator comm_world
bool communicator_comm_world_test() {
  const auto comm_world{mp1::comm_world()};
  if (not comm_world.is_valid())
    return false;
  const int size{comm_world.size()};
  const int rank{comm_world.rank()};
  if (size < 1)
    return false;
  if (rank < 0 or rank >= size)
    return false;
  if (comm_world.is_inter())
    return false;
  const auto comm_world_2{mp1::comm_world()};
  if (comm_world_2 != comm_world)
    return false;
  const auto comm_self{mp1::comm_self()};
  if (comm_self == comm_world)
    return false;
  if (size == 1 and comm_self.compare(comm_world) != mp1::equality_type::congruent)
    return false;
  if (size > 1 and comm_self.compare(comm_world) != mp1::equality_type::unequal)
    return false;
  return true;
}


// test properties of a newly created (owned) duplicate
bool communicator_comm_world_dup_test() {
  const auto comm_world{mp1::comm_world()};
  auto comm_new = mp1::communicator::dup(comm_world);
  if (not comm_new.is_valid())
    return false;
  if (comm_world.size() != comm_new.size())
    return false;
  if (comm_world.rank() != comm_new.rank())
    return false;
  if (comm_new.compare(comm_world) != mp1::equality_type::congruent)
    return false;
  // A duplicate is a different context: MPI_IDENT is false, so operator== is false.
  if (comm_new == comm_world)
    return false;
  return true;
}


// test properties of a newly created split communicator
bool communicator_comm_world_split_test() {
  const auto comm_world{mp1::comm_world()};
  const int size{comm_world.size()};
  const int rank{comm_world.rank()};
  auto comm_new = mp1::communicator::split(comm_world, rank % 2 == 0);
  if (not comm_new.is_valid())
    return false;
  const int size_new{comm_new.size()};
  if (size % 2 == 0) {
    if (size_new != size / 2)
      return false;
  } else {
    if (rank % 2 == 0 && size_new != (size + 1) / 2)
      return false;
    if (rank % 2 == 1 && size_new != (size - 1) / 2)
      return false;
  }
  if (comm_new.is_inter())
    return false;
  return true;
}


// test properties of a newly created shared-memory split communicator
bool communicator_comm_world_split_shared_memory_test() {
  const auto comm_world{mp1::comm_world()};
  const int rank{comm_world.rank()};
  auto comm_new =
      mp1::communicator::split_type(comm_world, mp1::comm_type_shared, rank % 2 == 0);
  if (not comm_new.is_valid())
    return false;
  if (comm_new.is_inter())
    return false;
  if (comm_new.size() < 1)
    return false;
  return true;
}


// test properties of a newly created communicator restricted to a group
bool communicator_comm_world_create_group_test() {
  const auto comm_world{mp1::comm_world()};
  const int rank{comm_world.rank()};
  auto group_world = mp1::get_group(comm_world);
  auto group_one = mp1::group_incl(group_world, {0});
  // create_group is collective over the group only: non-members must not enter it.
  mp1::communicator comm_new;
  if (rank == 0)
    comm_new = mp1::communicator::create_group(comm_world, group_one);
  if (rank == 0) {
    if (not comm_new.is_valid())
      return false;
    if (comm_new.size() != 1)
      return false;
    if (comm_new.rank() != 0)
      return false;
  } else {
    if (comm_new.is_valid())
      return false;
  }
  if (comm_new.is_valid() and comm_new.is_inter())
    return false;
  return true;
}


bool communicator_comm_self_test() {
  const auto comm_self{mp1::comm_self()};
  if (not comm_self.is_valid())
    return false;
  const int size{comm_self.size()};
  const int rank{comm_self.rank()};
  if (size != 1)
    return false;
  if (rank != 0)
    return false;
  const auto comm_self_2{mp1::comm_self()};
  if (comm_self_2 != comm_self)
    return false;
  const auto comm_world{mp1::comm_world()};
  if (comm_self == comm_world)
    return false;
  if (comm_world.size() == 1 and comm_self.compare(comm_world) != mp1::equality_type::congruent)
    return false;
  if (comm_world.size() > 1 and comm_self.compare(comm_world) != mp1::equality_type::unequal)
    return false;
  return true;
}


BOOST_AUTO_TEST_CASE(communicator) {
  if (not mp1::initialized())
    mp1::init();

  BOOST_TEST(communicator_comm_world_test());
  BOOST_TEST(communicator_comm_world_dup_test());
  BOOST_TEST(communicator_comm_world_split_test());
  BOOST_TEST(communicator_comm_world_split_shared_memory_test());
  BOOST_TEST(communicator_comm_world_create_group_test());
  BOOST_TEST(communicator_comm_self_test());

  const auto comm_world{mp1::comm_world()};

  // Invalid split colors are rejected before MPI.  (Positive split_type values are passed to
  // MPI as implementation constants; only the negative non-MPI_UNDEFINED values are local
  // validation errors.)
  BOOST_TEST(throws_invalid_argument([&] { (void)mp1::communicator::split(comm_world, -2); }));
  BOOST_TEST(
      throws_invalid_argument([&] { (void)mp1::communicator::split_type(comm_world, -2); }));
  // create_group is intra-only and validates its tag.
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::create_group(comm_world, mp1::get_group(comm_world), mp1::tag{-1});
  }));
  // duplicate factories reject an invalid communicator.
  BOOST_TEST(throws_invalid_argument([&] { (void)mp1::communicator::dup(mp1::comm_null()); }));

  // free() is idempotent and leaves an invalid owning communicator.
  auto owner = mp1::communicator::dup(comm_world);
  BOOST_TEST(owner.is_valid());
  owner.free();
  BOOST_TEST(not owner.is_valid());
  owner.free();
  BOOST_TEST(not owner.is_valid());

  // comm_null is always available and invalid.
  BOOST_TEST(not mp1::comm_null().is_valid());
  BOOST_TEST((mp1::comm_null() == mp1::comm_null()));
  BOOST_TEST((mp1::comm_null() != comm_world));
}
