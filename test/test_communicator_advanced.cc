// MPI_Comm_create_from_group / MPI_Intercomm_create_from_groups tests.
//
// Both factories are collective over a group (or the union of two groups).  `info` defaults to
// MPI_INFO_NULL and the handler to MPI_ERRORS_ARE_FATAL; the tests cover explicit handlers and
// defaults, full and subset groups at 2 and 4 ranks,
// empty groups, the local validation paths, error-handler attachment, and the leader-only
// remote arguments.  Non-membership is deliberately not diagnosed locally (MPI is the
// authority), so no test expects a local throw for it.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>


namespace {

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

  void require_world() {
    if (not mp1::initialized())
      mp1::init();
  }

  std::string long_stringtag() {
    return std::string(static_cast<std::size_t>(MPI_MAX_STRINGTAG_LEN), 'x');
  }

  void check_created(const mp1::communicator& comm, const mp1::mpi_communicator& source) {
    BOOST_TEST(comm.is_valid());
    BOOST_TEST((comm.size() == source.size()));
    BOOST_TEST((comm.rank() == source.rank()));
    BOOST_TEST((comm.compare(source) == mp1::equality_type::congruent));
  }

  void check_handler(const mp1::communicator& comm, mp1::error_handler expected) {
    const std::optional<mp1::error_handler> handler = mp1::get_error_handler(comm);
    BOOST_REQUIRE(handler.has_value());
    BOOST_TEST(static_cast<int>(*handler) == static_cast<int>(expected));
  }

}  // namespace


BOOST_AUTO_TEST_CASE(create_from_group_world) {
  require_world();
  const auto world = mp1::comm_world();
  auto world_group = mp1::get_group(world);
  mp1::info hints;
  hints.set("mp1_create_key", "mp1_create_value");

  const std::size_t abandoned_before = mp1::detail::abandoned_handle_count();
  {
    auto created = mp1::communicator::create_from_group(world_group, "mp1-create-world", hints,
                                                        mp1::error_handler::returns);
    check_created(created, world);
    check_handler(created, mp1::error_handler::returns);
    created.free();
    BOOST_TEST(not created.is_valid());
  }
  BOOST_TEST(mp1::detail::abandoned_handle_count() == abandoned_before);
}

BOOST_AUTO_TEST_CASE(create_from_group_subset) {
  require_world();
  const auto world = mp1::comm_world();
  auto world_group = mp1::get_group(world);
  auto rank_zero_group = mp1::group_incl(world_group, std::array{0});

  // Collective only over the group's members: non-members do not call.
  if (world.rank() == 0) {
    auto created = mp1::communicator::create_from_group(
        rank_zero_group, "mp1-create-rank0", mp1::info{}, mp1::error_handler::returns);
    BOOST_TEST(created.is_valid());
    BOOST_TEST((created.size() == 1));
    BOOST_TEST((created.rank() == 0));
    BOOST_TEST(not created.is_inter());
    created.free();
  }

  // A two-member group at four ranks; local rank and size follow the group order.
  if (world.size() >= 4) {
    auto group_02 = mp1::group_incl(world_group, std::array{0, 2});
    const int local_rank = world.rank() == 0 ? 0 : (world.rank() == 2 ? 1 : -1);
    if (local_rank >= 0) {
      auto created = mp1::communicator::create_from_group(
          group_02, "mp1-create-02", mp1::info{}, mp1::error_handler::returns);
      BOOST_TEST(created.is_valid());
      BOOST_TEST((created.size() == 2));
      BOOST_TEST((created.rank() == local_rank));
      created.free();
    }
  }
}

BOOST_AUTO_TEST_CASE(create_from_group_are_fatal_success) {
  require_world();
  const auto world = mp1::comm_world();
  auto world_group = mp1::get_group(world);
  auto rank_zero_group = mp1::group_incl(world_group, std::array{0});
  if (world.rank() == 0) {
    // A successful creation, not an error path: MPI_ERRORS_ARE_FATAL is a valid handler.
    auto created = mp1::communicator::create_from_group(
        rank_zero_group, "mp1-create-fatal", mp1::info{}, mp1::error_handler::are_fatal);
    BOOST_TEST(created.is_valid());
    check_handler(created, mp1::error_handler::are_fatal);
    created.free();
  }
}

BOOST_AUTO_TEST_CASE(create_from_group_default_handler) {
  require_world();
  const auto world = mp1::comm_world();
  auto world_group = mp1::get_group(world);

  // No info/handler arguments: the defaults are MPI_INFO_NULL and MPI_ERRORS_ARE_FATAL.
  auto created = mp1::communicator::create_from_group(world_group, "mp1-create-default");
  check_created(created, world);
  check_handler(created, mp1::error_handler::are_fatal);
  created.free();
}

BOOST_AUTO_TEST_CASE(create_from_group_empty_and_validation) {
  require_world();
  const auto world = mp1::comm_world();
  auto world_group = mp1::get_group(world);

  // Empty group: local operation, invalid owner, no collective.
  const mp1::communicator empty = mp1::communicator::create_from_group(
      mp1::group::empty(), "", mp1::info{}, mp1::error_handler::returns);
  BOOST_TEST(not empty.is_valid());

  // The largest legal stringtag (MPI_MAX_STRINGTAG_LEN - 1) is accepted.
  const std::string max_length_tag(static_cast<std::size_t>(MPI_MAX_STRINGTAG_LEN) - 1, 't');
  const mp1::communicator max_tag = mp1::communicator::create_from_group(
      mp1::group::empty(), max_length_tag, mp1::info{}, mp1::error_handler::returns);
  BOOST_TEST(not max_tag.is_valid());

  // Local validation happens before the empty-group short circuit, so every rank throws here
  // without entering MPI.
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::create_from_group(mp1::group::empty(), long_stringtag(),
                                               mp1::info{}, mp1::error_handler::returns);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::create_from_group(
        mp1::group::empty(), std::string_view{"tag\0with-nul", sizeof("tag\0with-nul") - 1},
        mp1::info{}, mp1::error_handler::returns);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::create_from_group(mp1::group::empty(), "mp1-create", mp1::info{},
                                               static_cast<mp1::error_handler>(200));
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::create_from_group(mp1::group{}, "mp1-create", mp1::info{},
                                               mp1::error_handler::returns);
  }));

  // A valid group with a too-long tag still rejects locally (no collective entered).
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::create_from_group(world_group, long_stringtag(), mp1::info{},
                                               mp1::error_handler::returns);
  }));
}

BOOST_AUTO_TEST_CASE(intercomm_from_groups_two_ranks) {
  require_world();
  const auto world = mp1::comm_world();
  if (world.size() != 2) {
    BOOST_TEST_MESSAGE("the 1+1 intercomm-from-groups case runs at 2 ranks");
    return;
  }

  auto world_group = mp1::get_group(world);
  auto group_0 = mp1::group_incl(world_group, std::array{0});
  auto group_1 = mp1::group_incl(world_group, std::array{1});
  const bool is_zero = world.rank() == 0;
  const mp1::group& local = is_zero ? group_0 : group_1;
  const mp1::group& remote = is_zero ? group_1 : group_0;

  auto inter = mp1::communicator::intercomm_create_from_groups(
      local, 0, remote, 0, "mp1-inter-2", mp1::info{}, mp1::error_handler::returns);
  BOOST_REQUIRE(inter.is_valid());
  BOOST_TEST(inter.is_inter());
  BOOST_TEST((inter.size() == 1));
  BOOST_TEST((inter.remote_size() == 1));
  BOOST_TEST((inter.rank() == 0));
  check_handler(inter, mp1::error_handler::returns);

  int send_value = 100 + world.rank();
  int recv_value = -1;
  mp1::sendrecv(inter, send_value, 0, mp1::tag{901}, recv_value, 0, mp1::tag{901});
  BOOST_TEST((recv_value == 100 + (1 - world.rank())));
  inter.free();

  // Default info/handler: MPI_INFO_NULL and MPI_ERRORS_ARE_FATAL.
  auto inter_default = mp1::communicator::intercomm_create_from_groups(local, 0, remote, 0,
                                                                       "mp1-inter-2-default");
  BOOST_REQUIRE(inter_default.is_valid());
  BOOST_TEST((inter_default.remote_size() == 1));
  check_handler(inter_default, mp1::error_handler::are_fatal);
  inter_default.free();
}

BOOST_AUTO_TEST_CASE(intercomm_from_groups_four_ranks) {
  require_world();
  const auto world = mp1::comm_world();
  if (world.size() < 4) {
    BOOST_TEST_MESSAGE("the 2+2 intercomm-from-groups case needs at least 4 ranks");
    return;
  }

  auto world_group = mp1::get_group(world);
  std::vector<int> even_ranks;
  std::vector<int> odd_ranks;
  for (int rank = 0; rank < world.size(); ++rank)
    (rank % 2 == 0 ? even_ranks : odd_ranks).push_back(rank);
  auto even_group = mp1::group_incl(world_group, even_ranks);
  auto odd_group = mp1::group_incl(world_group, odd_ranks);
  const bool is_even = world.rank() % 2 == 0;
  const mp1::group& local = is_even ? even_group : odd_group;
  const mp1::group& remote = is_even ? odd_group : even_group;
  const int local_rank = local.rank();
  const int local_leader = 0;

  // Every local rank passes a valid remote group and a valid remote leader.  MPI 4.1 says
  // both are significant only at local_leader; the portable form keeps them valid everywhere.
  const int remote_leader = 0;
  auto inter = mp1::communicator::intercomm_create_from_groups(
      local, local_leader, remote, remote_leader, "mp1-inter-4", mp1::info{},
      mp1::error_handler::returns);
  BOOST_REQUIRE(inter.is_valid());
  BOOST_TEST(inter.is_inter());
  BOOST_TEST(
      (inter.size() == static_cast<int>(is_even ? even_ranks.size() : odd_ranks.size())));
  BOOST_TEST((inter.remote_size() ==
              static_cast<int>(is_even ? odd_ranks.size() : even_ranks.size())));
  BOOST_TEST((inter.rank() == local_rank));

  if (local_rank == local_leader) {
    int send_value = 200 + world.rank();
    int recv_value = -1;
    mp1::sendrecv(inter, send_value, 0, mp1::tag{902}, recv_value, 0, mp1::tag{902});
    BOOST_TEST((recv_value == 200 + (is_even ? world.rank() + 1 : world.rank() - 1)));
  }
  inter.free();
}

BOOST_AUTO_TEST_CASE(intercomm_from_groups_empty_groups) {
  require_world();
  const auto world = mp1::comm_world();

  // Empty local group: local operation returning an invalid owner; the remote arguments are
  // not validated (they are significant only at the leader).
  const mp1::communicator empty_local = mp1::communicator::intercomm_create_from_groups(
      mp1::group::empty(), 0, mp1::group{}, 0, "", mp1::info{}, mp1::error_handler::returns);
  BOOST_TEST(not empty_local.is_valid());

  // MPI 4.1 §7.6.2: an empty remote group at the leader is a local operation returning
  // MPI_COMM_NULL.  Open MPI 6.0.0rc3 rejects it with MPI_ERR_ARG (see COMPATIBILITY.md), so
  // run this standard form on conforming providers and document the skip on Open MPI 6.
#if defined(OMPI_MAJOR_VERSION) && OMPI_MAJOR_VERSION >= 6
  if (world.size() == 2) {
    BOOST_TEST_MESSAGE(
        "empty remote group skipped on Open MPI 6: MPI_ERR_ARG, see COMPATIBILITY.md");
  }
#else
  auto world_group = mp1::get_group(world);
  if (world.size() == 2) {
    auto self_group = mp1::group_incl(world_group, std::array{world.rank()});
    const mp1::communicator empty_remote = mp1::communicator::intercomm_create_from_groups(
        self_group, 0, mp1::group::empty(), 0, "mp1-inter-empty-remote", mp1::info{},
        mp1::error_handler::returns);
    BOOST_TEST(not empty_remote.is_valid());
  }
#endif
}

BOOST_AUTO_TEST_CASE(intercomm_from_groups_validation) {
  require_world();
  const auto world = mp1::comm_world();
  auto world_group = mp1::get_group(world);

  // Local validation fires before MPI on every rank, so no rank enters the collective.
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::intercomm_create_from_groups(mp1::group{}, 0, mp1::group{}, 0,
                                                          "mp1-inter", mp1::info{},
                                                          mp1::error_handler::returns);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::intercomm_create_from_groups(mp1::group::empty(), 0, mp1::group{},
                                                          0, long_stringtag(), mp1::info{},
                                                          mp1::error_handler::returns);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::intercomm_create_from_groups(
        mp1::group::empty(), 0, mp1::group{}, 0, std::string_view{"a\0b", 3}, mp1::info{},
        mp1::error_handler::returns);
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::intercomm_create_from_groups(mp1::group::empty(), 0, mp1::group{},
                                                          0, "mp1-inter", mp1::info{},
                                                          static_cast<mp1::error_handler>(200));
  }));

  // Every rank belongs to its singleton local group, so an invalid local_leader throws
  // locally on every rank before MPI.
  auto self_group = mp1::group_incl(world_group, std::array{world.rank()});
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::communicator::intercomm_create_from_groups(self_group, 1, mp1::group{}, 0,
                                                          "mp1-inter-leader", mp1::info{},
                                                          mp1::error_handler::returns);
  }));

  // At two ranks every process is the local leader, so an invalid remote_leader also throws
  // locally on every rank before MPI.
  if (world.size() == 2) {
    auto remote_group = mp1::group_incl(world_group, std::array{1 - world.rank()});
    BOOST_TEST(throws_invalid_argument([&] {
      (void)mp1::communicator::intercomm_create_from_groups(
          self_group, 0, remote_group, 1, "mp1-inter-remote-leader", mp1::info{},
          mp1::error_handler::returns);
    }));
  }
}
