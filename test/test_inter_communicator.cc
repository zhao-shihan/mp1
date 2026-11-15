// Intercommunicator tests: one communicator type with runtime is_inter()/remote_size()/
// remote_group(), creation via communicator::intercomm_create, intercomm p2p, merge, split,
// disconnect, and the inter-specific validation errors.

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

  // Split comm_world by parity and create the intercommunicator between the two halves.
  // All ranks call this (MPI_Intercomm_create is collective over the local group and its peer).
  mp1::communicator make_intercommunicator(const mp1::mpi_communicator& comm_world, int tag) {
    const int world_rank = comm_world.rank();
    const int my_group = world_rank % 2;
    auto local = mp1::communicator::split(comm_world, my_group);
    const int local_leader = 0;
    const int remote_leader = my_group == 0 ? 1 : 0;
    return mp1::communicator::intercomm_create(local, local_leader, comm_world, remote_leader, mp1::tag{tag});
  }

}  // namespace


// test inter-communicator creation and runtime inter properties
BOOST_AUTO_TEST_CASE(inter_communicator_create) {
  if (not mp1::initialized())
    mp1::init();

  const auto comm_world{mp1::comm_world()};
  const int world_rank{comm_world.rank()};
  const int world_size{comm_world.size()};
  const int my_group{world_rank % 2};

  auto inter_com = make_intercommunicator(comm_world, 0);

  BOOST_TEST(inter_com.is_valid());
  BOOST_TEST(inter_com.is_inter());
  BOOST_TEST((inter_com.size() + inter_com.remote_size() == world_size));
  if (my_group == 0) {
    BOOST_TEST((inter_com.size() == (world_size + 1) / 2));
    BOOST_TEST((inter_com.remote_size() == world_size / 2));
  } else {
    BOOST_TEST((inter_com.remote_size() == (world_size + 1) / 2));
    BOOST_TEST((inter_com.size() == world_size / 2));
  }

  // remote_group() is the peer group: the calling process is not a member of it.
  auto remote_group = inter_com.remote_group();
  BOOST_TEST(remote_group.is_valid());
  BOOST_TEST((remote_group.size() == inter_com.remote_size()));
  BOOST_TEST((remote_group.rank() == MPI_UNDEFINED));

  // get_group() returns the local group.
  auto local_group = mp1::get_group(inter_com);
  BOOST_TEST((local_group.size() == inter_com.size()));
  BOOST_TEST((local_group.rank() == inter_com.rank()));

  // p2p on an intercommunicator addresses ranks in the remote group.  Rank 0 of each half
  // exchanges one value with rank 0 of the peer half.
  if (inter_com.rank() == 0) {
    int send_value = 1000 + my_group;
    int recv_value = -1;
    mp1::sendrecv(inter_com, send_value, 0, mp1::tag{7}, recv_value, 0, mp1::tag{7});
    BOOST_TEST((recv_value == 1000 + (1 - my_group)));
  }

  // The remote peer count bounds dest/source validation.
  int value = 0;
  BOOST_TEST(
      throws_invalid_argument([&] { mp1::send(inter_com, value, inter_com.remote_size()); }));
  BOOST_TEST(
      throws_invalid_argument([&] { mp1::recv(inter_com, value, inter_com.remote_size()); }));

  // Inter-specific factory validation.
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::communicator::create_group(inter_com, local_group); }));

  // merge is inter-only and produces an intracommunicator covering both groups.
  BOOST_TEST(
      throws_invalid_argument([&] { (void)mp1::communicator::merge(comm_world, true); }));

  inter_com.disconnect();
  BOOST_TEST(not inter_com.is_valid());
  inter_com.disconnect();
  BOOST_TEST(not inter_com.is_valid());
}


// Regression test (MPI 4.1 §7.6.2): peer_comm and remote_leader are "significant only at
// the local_leader".  A legal call whose non-leaders pass MPI_COMM_NULL and -1 must still
// create the intercommunicator.  The old all-rank validation threw std::invalid_argument on
// the non-leaders before the MPI call and left the leaders blocked in MPI_Intercomm_create.
// Runs on the 4-rank CTest entry; the
// 2-rank entry has one rank per group, so every rank is the local leader.
BOOST_AUTO_TEST_CASE(inter_communicator_leader_only_arguments) {
  if (not mp1::initialized())
    mp1::init();

  const auto comm_world{mp1::comm_world()};
  const int world_size{comm_world.size()};
  if (world_size < 4)
    return;

  const int world_rank{comm_world.rank()};
  const int my_group{world_rank % 2};
  auto local = mp1::communicator::split(comm_world, my_group);
  const int local_leader{0};
  const int remote_leader{my_group == 0 ? 1 : 0};
  const bool is_leader{local.rank() == local_leader};

  // Non-leaders deliberately pass an MPI_COMM_NULL peer and remote_leader -1.
  const mp1::mpi_communicator peer = is_leader ? mp1::comm_world() : mp1::mpi_communicator{};
  mp1::communicator inter = mp1::communicator::intercomm_create(
      local, local_leader, peer, is_leader ? remote_leader : -1, mp1::tag{4242});

  BOOST_TEST(inter.is_valid());
  BOOST_TEST(inter.is_inter());
  BOOST_TEST((inter.size() == local.size()));
  BOOST_TEST((inter.size() + inter.remote_size() == world_size));

  // The intercommunicator is functional: the two group leaders exchange one value.
  if (is_leader) {
    int send_value = 1000 + my_group;
    int recv_value = -1;
    mp1::sendrecv(inter, send_value, 0, mp1::tag{0}, recv_value, 0, mp1::tag{0});
    BOOST_TEST((recv_value == 1000 + (1 - my_group)));
  }

  inter.disconnect();
  BOOST_TEST(not inter.is_valid());
  local.free();
}


// test inter-communicator creation error and merge
BOOST_AUTO_TEST_CASE(inter_communicator_merge) {
  if (not mp1::initialized())
    mp1::init();

  const auto comm_world{mp1::comm_world()};
  const int world_rank{comm_world.rank()};
  const int my_group{world_rank % 2};

  // Local, every-rank arguments are rejected on every local rank (no collective is entered):
  // a negative or out-of-range local_leader and a negative tag.  The leader-only arguments
  // are checked where MPI uses them: MPI 4.1 §7.6.2 makes peer_comm and remote_leader
  // "significant only at the local_leader", so a non-leader must not be forced to pass a
  // usable value and the old all-rank rejection is itself an erroneous program.
  {
    auto local = mp1::communicator::split(comm_world, my_group);
    const int remote_leader = my_group == 0 ? 1 : 0;
    if (comm_world.size() >= 2) {
      BOOST_TEST(throws_invalid_argument([&] {
        (void)mp1::communicator::intercomm_create(local, -1, comm_world, remote_leader);
      }));
      BOOST_TEST(throws_invalid_argument([&] {
        (void)mp1::communicator::intercomm_create(local, local.size(), comm_world,
                                                  remote_leader);
      }));
      BOOST_TEST(throws_invalid_argument([&] {
        (void)mp1::communicator::intercomm_create(local, 0, comm_world, remote_leader, mp1::tag{-1});
      }));
    }
    if (local.rank() == 0) {
      const int local_leader = 0;
      BOOST_TEST(throws_invalid_argument([&] {
        (void)mp1::communicator::intercomm_create(local, local_leader, mp1::comm_null(),
                                                  remote_leader);
      }));
      BOOST_TEST(throws_invalid_argument([&] {
        (void)mp1::communicator::intercomm_create(local, local_leader, comm_world,
                                                  world_rank + 5);
      }));
      BOOST_TEST(throws_invalid_argument([&] {
        (void)mp1::communicator::intercomm_create(local, local_leader, comm_world, -1);
      }));
    }
    local.free();
  }

  auto inter_comm = make_intercommunicator(comm_world, 1);

  // local/peer must be intracommunicators.
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::communicator::intercomm_create(inter_comm, 0, comm_world, 0); }));

  // split/split_type are valid on an intercommunicator and return an intercommunicator.
  auto inter_split = mp1::communicator::split(inter_comm, 0);
  BOOST_TEST(inter_split.is_valid());
  BOOST_TEST(inter_split.is_inter());
  // MPI 4.1 §7.4.2: MPI_COMM_SPLIT_TYPE creates both intra- and intercommunicators.  With
  // MPI_COMM_TYPE_SHARED and no MPI_UNDEFINED, MPI_COMM_NULL is not a legal result for any
  // rank, so the factory must return a valid intercommunicator spanning both groups.
  auto inter_split_type = mp1::communicator::split_type(inter_comm, mp1::comm_type_shared);
  BOOST_REQUIRE(inter_split_type.is_valid());
  BOOST_TEST(inter_split_type.is_inter());
  BOOST_TEST((inter_split_type.size() + inter_split_type.remote_size() == comm_world.size()));

  // Merge back into an intracommunicator; all local ranks are present.
  auto merged = mp1::communicator::merge(inter_comm, my_group == 0);
  BOOST_TEST(merged.is_valid());
  BOOST_TEST(not merged.is_inter());
  BOOST_TEST((merged.size() == comm_world.size()));
  const auto communicator_equality{merged.compare(comm_world)};
  BOOST_TEST((communicator_equality == mp1::equality_type::congruent or
              communicator_equality == mp1::equality_type::similar));

  // free() is the owner-only release; disconnect() is the intercomm-specific one.
  inter_comm.free();
  BOOST_TEST(not inter_comm.is_valid());
  inter_split.free();
  inter_split_type.free();
  merged.free();
}
