// MPI_Comm_idup / MPI_Comm_idup_with_info tests.
//
// The request stays the single request type: a successful idup completion enters
// request_state::result_pending and the request owns the duplicate until take_communicator()
// transfers it.  The test covers wait/test, every bulk completion function, the take error
// paths, destructor/release/move ownership, and the intercommunicator form.  MPI 4.1 §7.4
// (communicator constructors) and §6.12 (nonblocking collectives).

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>


namespace {

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

  bool is_empty_status(const mp1::status& s) {
    return s.source() == MPI_ANY_SOURCE and s.tag() == mp1::any_tag;
  }

  void require_world() {
    if (not mp1::initialized())
      mp1::init();
  }

  // Split comm_world by parity and create the intercommunicator between the two halves.
  mp1::communicator make_intercommunicator(const mp1::mpi_communicator& world, int tag) {
    const int world_rank = world.rank();
    const int my_group = world_rank % 2;
    auto local = mp1::communicator::split(world, my_group);
    return mp1::communicator::intercomm_create(local, 0, world, my_group == 0 ? 1 : 0,
                                               mp1::tag{tag});
  }

  void check_duplicate(const mp1::communicator& duplicate,
                       const mp1::mpi_communicator& source) {
    BOOST_TEST(duplicate.is_valid());
    BOOST_TEST((duplicate.size() == source.size()));
    BOOST_TEST((duplicate.rank() == source.rank()));
    BOOST_TEST((duplicate.compare(source) == mp1::equality_type::congruent));
  }

}  // namespace


BOOST_AUTO_TEST_CASE(idup_wait_test_and_take) {
  require_world();
  const auto world = mp1::comm_world();
  const std::size_t abandoned_before = mp1::detail::abandoned_handle_count();
  {
    mp1::request r = mp1::communicator::idup(world);
    BOOST_TEST(r.is_valid());
    BOOST_TEST(r.is_active());
    BOOST_TEST(static_cast<int>(r.kind()) == static_cast<int>(mp1::request_kind::collective));
    BOOST_TEST(not r.has_result());
    BOOST_TEST(r.drop_would_block());  // active collective: destructor waits
    BOOST_TEST(throws_logic_error([&] { (void)mp1::take_communicator(r); }));
    BOOST_TEST(r.is_active());

    const mp1::status completion = r.wait();
    BOOST_TEST(static_cast<int>(completion.kind()) ==
               static_cast<int>(mp1::status_kind::normalized));
    BOOST_TEST(r.has_result());
    BOOST_TEST(not r.is_valid());
    BOOST_TEST(not static_cast<bool>(r));
    BOOST_TEST(static_cast<int>(r.state()) ==
               static_cast<int>(mp1::request_state::result_pending));
    BOOST_TEST(r.drop_would_block());

    // After result_pending, wait/test/get_status are MPI-free no-ops that keep the result.
    BOOST_TEST(is_empty_status(r.wait()));
    const std::optional<mp1::status> tested = r.test();
    BOOST_REQUIRE(tested.has_value());
    BOOST_TEST(is_empty_status(*tested));
    const std::optional<mp1::status> status_only = r.get_status();
    BOOST_REQUIRE(status_only.has_value());
    BOOST_TEST(r.has_result());

    mp1::communicator duplicate = mp1::take_communicator(r);
    check_duplicate(duplicate, world);
    BOOST_TEST(not r.has_result());
    BOOST_TEST(static_cast<int>(r.state()) == static_cast<int>(mp1::request_state::freed));
    BOOST_TEST(not r.is_valid());
    BOOST_TEST(throws_invalid_argument([&] { (void)mp1::take_communicator(r); }));
    duplicate.free();
  }
  BOOST_TEST(mp1::detail::abandoned_handle_count() == abandoned_before);
}

BOOST_AUTO_TEST_CASE(idup_with_info_take) {
  require_world();
  const auto world = mp1::comm_world();
  mp1::info hints;
  hints.set("mp1_idup_key", "mp1_idup_value");
  mp1::request r = mp1::communicator::idup_with_info(world, hints);
  (void)r.wait();
  BOOST_TEST(r.has_result());
  mp1::communicator duplicate = mp1::take_communicator(r);
  check_duplicate(duplicate, world);
  // Whether a user hint is retained is implementation-defined; only require the owning info
  // round trip to be usable.
  BOOST_TEST((duplicate.get_info().size() >= 0));
  duplicate.free();
}

BOOST_AUTO_TEST_CASE(idup_inherits_source_error_handler) {
  require_world();
  const auto world = mp1::comm_world();
  auto source = mp1::communicator::dup(world);
  mp1::set_error_handler(source, mp1::error_handler::returns);
  mp1::request r = mp1::communicator::idup(source);
  (void)r.wait();
  mp1::communicator duplicate = mp1::take_communicator(r);
  const std::optional<mp1::error_handler> handler = mp1::get_error_handler(duplicate);
  BOOST_REQUIRE(handler.has_value());
  BOOST_TEST(static_cast<int>(*handler) == static_cast<int>(mp1::error_handler::returns));
  duplicate.free();
  source.free();
}

BOOST_AUTO_TEST_CASE(idup_wait_all_take_each) {
  require_world();
  const auto world = mp1::comm_world();
  std::vector<mp1::request> requests;
  requests.reserve(3);
  for (int i = 0; i < 3; ++i)
    requests.push_back(mp1::communicator::idup(world));

  const std::vector<mp1::status> statuses = mp1::wait_all(requests);
  BOOST_REQUIRE(statuses.size() == requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    BOOST_TEST(static_cast<int>(statuses[i].kind()) ==
               static_cast<int>(mp1::status_kind::normalized));
    BOOST_TEST(requests[i].has_result());
    mp1::communicator duplicate = mp1::take_communicator(requests[i]);
    check_duplicate(duplicate, world);
    duplicate.free();
  }
}

BOOST_AUTO_TEST_CASE(idup_test_all_take_each) {
  require_world();
  const auto world = mp1::comm_world();
  std::vector<mp1::request> requests;
  requests.reserve(2);
  for (int i = 0; i < 2; ++i)
    requests.push_back(mp1::communicator::idup(world));

  std::optional<std::vector<mp1::status>> statuses;
  int guard = 0;
  while (not statuses.has_value() and guard++ < 1000000)
    statuses = mp1::test_all(requests);
  BOOST_REQUIRE(statuses.has_value());
  BOOST_TEST(statuses->size() == requests.size());
  for (mp1::request& request : requests) {
    BOOST_TEST(request.has_result());
    mp1::communicator duplicate = mp1::take_communicator(request);
    check_duplicate(duplicate, world);
    duplicate.free();
  }
}

BOOST_AUTO_TEST_CASE(idup_wait_any_take_completed) {
  require_world();
  const auto world = mp1::comm_world();
  std::vector<mp1::request> requests;
  requests.reserve(2);
  for (int i = 0; i < 2; ++i)
    requests.push_back(mp1::communicator::idup(world));

  const mp1::completion first = mp1::wait_any(requests);
  BOOST_REQUIRE(first.first != MPI_UNDEFINED);
  BOOST_REQUIRE(first.first >= 0 and static_cast<std::size_t>(first.first) < requests.size());
  BOOST_TEST(requests[static_cast<std::size_t>(first.first)].has_result());
  mp1::communicator first_duplicate =
      mp1::take_communicator(requests[static_cast<std::size_t>(first.first)]);
  check_duplicate(first_duplicate, world);
  first_duplicate.free();

  // The remaining slot may still be active; wait_all completes it.  The already-taken slot is
  // inactive and contributes the empty status.
  (void)mp1::wait_all(requests);
  for (mp1::request& request : requests) {
    if (not request.has_result())
      continue;
    mp1::communicator duplicate = mp1::take_communicator(request);
    check_duplicate(duplicate, world);
    duplicate.free();
  }
}

BOOST_AUTO_TEST_CASE(idup_test_any_take_completed) {
  require_world();
  const auto world = mp1::comm_world();
  std::vector<mp1::request> requests;
  requests.reserve(2);
  for (int i = 0; i < 2; ++i)
    requests.push_back(mp1::communicator::idup(world));

  std::optional<mp1::completion> completed;
  int guard = 0;
  while (not completed.has_value() and guard++ < 1000000)
    completed = mp1::test_any(requests);
  BOOST_REQUIRE(completed.has_value());
  BOOST_REQUIRE(completed->first != MPI_UNDEFINED);
  const std::size_t slot = static_cast<std::size_t>(completed->first);
  BOOST_REQUIRE(slot < requests.size());
  BOOST_TEST(requests[slot].has_result());
  mp1::communicator duplicate = mp1::take_communicator(requests[slot]);
  check_duplicate(duplicate, world);
  duplicate.free();

  (void)mp1::wait_all(requests);
  for (mp1::request& request : requests) {
    if (not request.has_result())
      continue;
    mp1::communicator rest = mp1::take_communicator(request);
    check_duplicate(rest, world);
    rest.free();
  }
}

BOOST_AUTO_TEST_CASE(idup_wait_some_take_completed) {
  require_world();
  const auto world = mp1::comm_world();
  std::vector<mp1::request> requests;
  requests.reserve(3);
  for (int i = 0; i < 3; ++i)
    requests.push_back(mp1::communicator::idup(world));

  const std::vector<mp1::completion> completed = mp1::wait_some(requests);
  BOOST_REQUIRE(not completed.empty());
  for (const mp1::completion& entry : completed) {
    const std::size_t slot = static_cast<std::size_t>(entry.first);
    BOOST_REQUIRE(slot < requests.size());
    BOOST_TEST(requests[slot].has_result());
    mp1::communicator duplicate = mp1::take_communicator(requests[slot]);
    check_duplicate(duplicate, world);
    duplicate.free();
  }

  (void)mp1::wait_all(requests);
  for (mp1::request& request : requests) {
    if (not request.has_result())
      continue;
    mp1::communicator duplicate = mp1::take_communicator(request);
    check_duplicate(duplicate, world);
    duplicate.free();
  }
}

BOOST_AUTO_TEST_CASE(idup_test_some_take_completed) {
  require_world();
  const auto world = mp1::comm_world();
  std::vector<mp1::request> requests;
  requests.reserve(3);
  for (int i = 0; i < 3; ++i)
    requests.push_back(mp1::communicator::idup(world));

  std::array<bool, 3> taken{false, false, false};
  int guard = 0;
  while ((not taken[0] or not taken[1] or not taken[2]) and guard++ < 1000000) {
    const std::optional<std::vector<mp1::completion>> completed = mp1::test_some(requests);
    BOOST_REQUIRE(completed.has_value());
    for (const mp1::completion& entry : *completed) {
      const std::size_t slot = static_cast<std::size_t>(entry.first);
      BOOST_REQUIRE(slot < requests.size());
      BOOST_TEST(not taken[slot]);
      mp1::communicator duplicate = mp1::take_communicator(requests[slot]);
      check_duplicate(duplicate, world);
      duplicate.free();
      taken[slot] = true;
    }
  }
  BOOST_TEST(taken[0]);
  BOOST_TEST(taken[1]);
  BOOST_TEST(taken[2]);
}

BOOST_AUTO_TEST_CASE(take_error_axes) {
  require_world();
  const auto world = mp1::comm_world();

  // No result at all: every non-idup request and a default request.
  mp1::request invalid;
  BOOST_TEST(throws_invalid_argument([&] { (void)mp1::take_communicator(invalid); }));

  mp1::request barrier = mp1::ibarrier(world);
  BOOST_TEST(throws_invalid_argument([&] { (void)mp1::take_communicator(barrier); }));
  (void)barrier.wait();

  // Active idup: retained result but not completed -> std::logic_error, request unchanged.
  mp1::request active = mp1::communicator::idup(world);
  BOOST_TEST(throws_logic_error([&] { (void)mp1::take_communicator(active); }));
  BOOST_TEST(active.is_active());
  (void)active.wait();
  BOOST_TEST(active.has_result());
  mp1::communicator duplicate = mp1::take_communicator(active);
  check_duplicate(duplicate, world);
  duplicate.free();
}

BOOST_AUTO_TEST_CASE(result_pending_destruction_frees) {
  require_world();
  const auto world = mp1::comm_world();
  const std::size_t abandoned_before = mp1::detail::abandoned_handle_count();
  const std::size_t dropped_before = mp1::detail::dropped_request_count();
  {
    mp1::request r = mp1::communicator::idup(world);
    (void)r.wait();
    BOOST_TEST(r.has_result());
    BOOST_TEST(r.drop_would_block());
  }  // result_pending destructor: MPI_Comm_free
  BOOST_TEST(mp1::detail::abandoned_handle_count() == abandoned_before);
  BOOST_TEST(mp1::detail::dropped_request_count() == dropped_before);
}

BOOST_AUTO_TEST_CASE(active_idup_destruction_waits_and_frees) {
  require_world();
  const auto world = mp1::comm_world();
  const std::size_t abandoned_before = mp1::detail::abandoned_handle_count();
  {
    mp1::request r = mp1::communicator::idup(world);
    BOOST_TEST(r.is_valid());
  }  // active destructor: MPI_Wait + payload complete + MPI_Comm_free
  BOOST_TEST(mp1::detail::abandoned_handle_count() == abandoned_before);
}

BOOST_AUTO_TEST_CASE(release_leaks_without_mpi) {
  require_world();
  const auto world = mp1::comm_world();

  // Active: leaks the request handle and the payload (MPI may still write the result slot).
  {
    const std::size_t dropped_before = mp1::detail::dropped_request_count();
    const std::size_t abandoned_before = mp1::detail::abandoned_handle_count();
    mp1::request r = mp1::communicator::idup(world);
    r.release();
    BOOST_TEST(not r.is_valid());
    BOOST_TEST(not r.has_result());
    BOOST_TEST(static_cast<int>(r.state()) == static_cast<int>(mp1::request_state::freed));
    BOOST_TEST(mp1::detail::dropped_request_count() == dropped_before + 1);
    BOOST_TEST(mp1::detail::abandoned_handle_count() == abandoned_before + 1);
  }

  // Completed: leaks the retained communicator handle; no MPI call.
  {
    const std::size_t dropped_before = mp1::detail::dropped_request_count();
    const std::size_t abandoned_before = mp1::detail::abandoned_handle_count();
    mp1::request r = mp1::communicator::idup(world);
    (void)r.wait();
    BOOST_TEST(r.has_result());
    r.release();
    BOOST_TEST(not r.has_result());
    BOOST_TEST(not r.is_valid());
    BOOST_TEST(static_cast<int>(r.state()) == static_cast<int>(mp1::request_state::freed));
    BOOST_TEST(mp1::detail::dropped_request_count() == dropped_before);
    BOOST_TEST(mp1::detail::abandoned_handle_count() == abandoned_before + 1);
  }
}

BOOST_AUTO_TEST_CASE(move_transfers_result) {
  require_world();
  const auto world = mp1::comm_world();

  {
    mp1::request source = mp1::communicator::idup(world);
    (void)source.wait();
    mp1::request target = std::move(source);
    BOOST_TEST(target.has_result());
    BOOST_TEST(not source.has_result());
    BOOST_TEST(static_cast<int>(source.state()) ==
               static_cast<int>(mp1::request_state::moved_from));
    mp1::communicator duplicate = mp1::take_communicator(target);
    check_duplicate(duplicate, world);
    duplicate.free();
  }

  {
    mp1::request source = mp1::communicator::idup(world);
    mp1::request target = mp1::communicator::idup(world);
    (void)source.wait();
    (void)target.wait();
    BOOST_TEST(source.has_result());
    BOOST_TEST(target.has_result());
    target = std::move(source);  // drops (frees) the target's duplicate, then steals source's
    BOOST_TEST(target.has_result());
    BOOST_TEST(not source.has_result());
    mp1::communicator duplicate = mp1::take_communicator(target);
    check_duplicate(duplicate, world);
    duplicate.free();
  }
}

BOOST_AUTO_TEST_CASE(intercommunicator_idup) {
  require_world();
  const auto world = mp1::comm_world();
  if (world.size() < 4) {
    BOOST_TEST_MESSAGE("intercommunicator idup needs at least 4 ranks");
    return;
  }

  auto inter = make_intercommunicator(world, 900);
  BOOST_REQUIRE(inter.is_valid());
  BOOST_REQUIRE(inter.is_inter());
  const int local_size = inter.size();
  const int remote_size = inter.remote_size();

  mp1::request r = mp1::communicator::idup(inter);
  (void)r.wait();
  BOOST_TEST(r.has_result());
  mp1::communicator duplicate = mp1::take_communicator(r);
  BOOST_TEST(duplicate.is_valid());
  BOOST_TEST(duplicate.is_inter());
  BOOST_TEST((duplicate.size() == local_size));
  BOOST_TEST((duplicate.remote_size() == remote_size));
  duplicate.free();
  inter.free();
}
