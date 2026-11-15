// Contract tests for persistent collective requests (MPI 4 section 6.13 / 8.8).
//
// Covers the request lifecycle that is not visible in the data-oriented unit test: the
// inactive/active state machine, payload retention across completions, the inactive-slot rule
// in wait_all/test_all, active-destructor wait-then-free, refusal of cancel(), the release()
// escape (no MPI call), per-start re-packing of staged payloads, and the local validation
// rules (no MPI collective is entered on an invalid call).
//
// The unit is registered at 2 and 4 ranks. MPI is initialized by the global fixture below.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
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

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  template<class F>
  bool throws_logic_error(F&& function) {
    try {
      function();
    } catch (const std::logic_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class F>
  bool throws_invalid_argument(F&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  mp1::layout<int> holey_two() {
    return mp1::layout<int>::vector(2, 1, 2);
  }

  // One holey_two() block per rank, as an aggregate layout over the whole buffer.
  mp1::layout<int> aggregate_holey_int(int blocks) {
    mp1::indexed_parameter parameter;
    for (int i = 0; i < blocks; ++i) {
      parameter.add(1, static_cast<mp1::count_t>(3 * i));
      parameter.add(1, static_cast<mp1::count_t>(3 * i + 2));
    }
    return mp1::layout<int>::indexed(parameter);
  }

  mp1::communicator make_intercommunicator(const mp1::mpi_communicator& world, int tag) {
    const int my_group = world.rank() % 2;
    auto local = mp1::communicator::split(world, my_group);
    const int remote_leader = my_group == 0 ? 1 : 0;
    return mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{tag});
  }

  bool wrap_available() {
#if MP1_TEST_HAVE_MPI_WRAP
    return mpi_wrap::supported();
#else
    return false;
#endif
  }

}  // namespace


struct mp1_persistent_collective_contract_fixture {
  mp1_persistent_collective_contract_fixture() {
    mp1::init();
  }

  ~mp1_persistent_collective_contract_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor must not throw
    } catch (...) {
      // Another owner may finalize; a fixture destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_persistent_collective_contract_fixture);

BOOST_AUTO_TEST_SUITE(persistent_collective_contract)

BOOST_AUTO_TEST_CASE(factory_state) {
  // A null handle still identifies the persistent-collective teardown policy.
  mp1::request null_request = mp1::request::from_persistent_collective(MPI_REQUEST_NULL);
  BOOST_TEST(not null_request.is_valid());
  BOOST_TEST(not null_request.is_active());
  BOOST_TEST(null_request.is_persistent());
  BOOST_TEST(null_request.wait().error() == MPI_SUCCESS);
  BOOST_TEST(null_request.test().has_value());

  const auto c = mp1::comm_world();
  mp1::request r = mp1::barrier_init(c);
  BOOST_TEST(r.is_valid());
  BOOST_TEST(r.is_persistent());
  BOOST_TEST(not r.is_active());
  BOOST_TEST(r.get_status().has_value());  // inactive: empty status, no MPI
  world_barrier();
}

BOOST_AUTO_TEST_CASE(inactive_and_active_state_machine) {
  const auto c = mp1::comm_world();

  {
    mp1::request r = mp1::barrier_init(c);
    if (wrap_available()) {
      mpi_wrap::reset_counts();
      BOOST_TEST(r.wait().error() == MPI_SUCCESS);
      BOOST_TEST(r.test().has_value());
      BOOST_TEST(r.get_status().has_value());
      BOOST_TEST(mpi_wrap::total() == 0);
    }
    // A second start while active is a local error and makes no MPI call.
    r.start();
    BOOST_TEST(r.is_active());
    if (wrap_available()) {
      mpi_wrap::reset_counts();
      BOOST_TEST(throws_logic_error([&] { r.start(); }));
      BOOST_TEST(mpi_wrap::total() == 0);
    } else {
      BOOST_TEST(throws_logic_error([&] { r.start(); }));
    }
    BOOST_TEST(r.is_active());
    r.wait();
    BOOST_TEST(not r.is_active());
    // The handle stays valid and reusable after completion.
    BOOST_TEST(r.is_valid());
    r.start();
    r.wait();
  }

  // An inactive persistent collective request is freed with MPI_Request_free only.
  {
    mp1::request r = mp1::barrier_init(c);
    if (wrap_available()) {
      mpi_wrap::reset_counts();
      {
        mp1::request moved = std::move(r);
        (void)moved;
      }
      BOOST_TEST(mpi_wrap::count("MPI_Request_free") == 1);
      BOOST_TEST(mpi_wrap::count("MPI_Wait") == 0);
      BOOST_TEST(mpi_wrap::count("MPI_Cancel") == 0);
    } else {
      mp1::request moved = std::move(r);
      (void)moved;
    }
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(wait_all_skips_inactive_slots) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const mp1::layout<int> l = holey_two();
  std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), l);
  std::vector<mp1::count_t> displs(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    displs[static_cast<std::size_t>(i)] = i * 3;

  std::array<int, 3> send{{rank + 1, -1, rank + 2}};
  std::vector<int> first(static_cast<std::size_t>(size * 3), -1);
  std::vector<int> second(static_cast<std::size_t>(size * 3), -1);
  const std::span<int> first_span{first};
  const std::span<int> second_span{second};
  mp1::request a =
      mp1::allgatherv_init(c, mp1::region{send.data(), l}, first_span, layouts, displs);
  mp1::request b =
      mp1::allgatherv_init(c, mp1::region{send.data(), l}, second_span, layouts, displs);

  a.start();
  std::vector<mp1::request> requests;
  requests.push_back(std::move(a));
  requests.push_back(std::move(b));
  if (wrap_available())
    mpi_wrap::reset_counts();
  const std::vector<mp1::status> statuses = mp1::wait_all(requests);
  BOOST_TEST(statuses.size() == 2u);
  if (wrap_available())
    BOOST_TEST(mpi_wrap::count("MPI_Waitall") == 1);

  for (int i = 0; i < size; ++i) {
    BOOST_TEST(first[static_cast<std::size_t>(i * 3)] == i + 1);
    BOOST_TEST(first[static_cast<std::size_t>(i * 3 + 2)] == i + 2);
    // The inactive slot's payload did not unpack a second time: the buffer is untouched.
    BOOST_TEST(second[static_cast<std::size_t>(i * 3)] == -1);
    BOOST_TEST(second[static_cast<std::size_t>(i * 3 + 2)] == -1);
  }

  // The inactive slot is still startable on its own.
  requests[1].start();
  requests[1].wait();
  for (int i = 0; i < size; ++i) {
    BOOST_TEST(second[static_cast<std::size_t>(i * 3)] == i + 1);
    BOOST_TEST(second[static_cast<std::size_t>(i * 3 + 2)] == i + 2);
  }

  // A full start_all/wait_all cycle on both requests keeps working.
  mp1::start_all(requests);
  (void)mp1::wait_all(requests);
  for (int i = 0; i < size; ++i) {
    BOOST_TEST(first[static_cast<std::size_t>(i * 3)] == i + 1);
    BOOST_TEST(second[static_cast<std::size_t>(i * 3 + 2)] == i + 2);
  }
  BOOST_TEST(not requests[0].is_active());
  BOOST_TEST(not requests[1].is_active());
  world_barrier();
}

BOOST_AUTO_TEST_CASE(active_destructor_waits_then_frees) {
  const auto c = mp1::comm_world();

  // Every rank starts the barrier before the request leaves scope, so the destructor's
  // MPI_Wait can complete and the subsequent MPI_Request_free is legal.
  if (wrap_available())
    mpi_wrap::reset_counts();
  {
    mp1::request r = mp1::barrier_init(c);
    r.start();
  }
  if (wrap_available()) {
    BOOST_TEST(mpi_wrap::count("MPI_Wait") == 1);
    BOOST_TEST(mpi_wrap::count("MPI_Request_free") == 1);
  }
  world_barrier();

  // The same for a data-carrying request whose payload must be destroyed after the wait.
  int value = 0;
  if (world_rank() == 0)
    value = 5;
  {
    mp1::request r = mp1::bcast_init(c, 0, value);
    r.start();
  }
  BOOST_TEST(value == 5);
  world_barrier();

  // An active persistent request whose receive side is staged (holey layout) must deliver its
  // payload from the destructor, not just perform MPI_Wait.  The same path applies to
  // reduce_init; the variable family exercises it too.
  {
    const auto c = mp1::comm_world();
    const int rank = c.rank();
    const int size = c.size();
    const mp1::layout<int> block = holey_two();
    std::vector<mp1::layout<int>> receive_layouts(static_cast<std::size_t>(size), block);
    std::vector<mp1::count_t> displacements(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      displacements[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(3 * i);
    std::vector<int> received(static_cast<std::size_t>(3 * size), -1);
    // The receive block has two int components; the local contribution must match its type
    // signature (MPI 4.1 6.7: the local send block equals the local receive block intra).
    const std::array<int, 3> send{{rank + 1, -1, rank + 2}};
    {
      const std::span<int> received_span{received};
      mp1::request r = mp1::allgatherv_init(c, mp1::region{send.data(), block}, received_span,
                                            receive_layouts, displacements);
      BOOST_TEST(not r.drop_would_block());  // inactive persistent: request-free, no block
      r.start();
      BOOST_TEST(r.drop_would_block());  // active collective: the destructor waits
    }  // destructor: MPI_Wait -> payload complete -> MPI_Request_free
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(received[static_cast<std::size_t>(3 * i)] == i + 1);
      BOOST_TEST(received[static_cast<std::size_t>(3 * i + 2)] == i + 2);
      BOOST_TEST(received[static_cast<std::size_t>(3 * i + 1)] == -1);
    }
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(cancel_is_refused) {
  const auto c = mp1::comm_world();

  mp1::request r = mp1::barrier_init(c);
  BOOST_TEST(throws_logic_error([&] { r.cancel(); }));  // inactive: valid non-cancelable
  r.start();
  BOOST_TEST(throws_logic_error([&] { r.cancel(); }));  // active: valid non-cancelable
  if (wrap_available()) {
    mpi_wrap::reset_counts();
    BOOST_TEST(throws_logic_error([&] { r.cancel(); }));
    BOOST_TEST(mpi_wrap::count("MPI_Cancel") == 0);
  }
  r.wait();

  // release() resets the request to a plain null request: cancel is a no-op.
  r.release();
  BOOST_TEST(not r.is_valid());
  BOOST_TEST(not r.is_persistent());
  BOOST_TEST(static_cast<int>(r.kind()) == static_cast<int>(mp1::request_kind::none));
  r.cancel();  // no-op: a released request is a plain null request
  world_barrier();
}

BOOST_AUTO_TEST_CASE(release_makes_no_mpi_call) {
  const auto c = mp1::comm_world();

  mp1::request r = mp1::barrier_init(c);
  r.start();
  if (wrap_available()) {
    mpi_wrap::reset_counts();
    r.release();
    BOOST_TEST(mpi_wrap::total() == 0);
  } else {
    r.release();
  }
  BOOST_TEST(not r.is_valid());
  BOOST_TEST(not r.is_active());
  BOOST_TEST(throws_invalid_argument([&] { r.start(); }));

  // The peer ranks must still complete their starts, so this barrier both drains the
  // collective and proves the leaked handle does not disturb later communication.
  world_barrier();

  // An inactive request can similarly be released without any MPI call.
  mp1::request inactive = mp1::barrier_init(c);
  if (wrap_available()) {
    mpi_wrap::reset_counts();
    inactive.release();
    BOOST_TEST(mpi_wrap::total() == 0);
  } else {
    inactive.release();
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(staging_is_rebuilt_on_every_start) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const mp1::layout<int> l = holey_two();

  // Holey broadcast: the root's staged send buffer must be re-packed from the caller buffer
  // before every start.
  {
    std::array<int, 3> data{{-1, -1, -1}};
    mp1::request r = mp1::bcast_init(c, 0, mp1::region<int>{data.data(), l});
    for (int cycle = 0; cycle < 3; ++cycle) {
      if (rank == 0) {
        data[0] = 10 + cycle;
        data[2] = 20 + cycle;
      }
      r.start();
      r.wait();
      BOOST_TEST(data[0] == 10 + cycle);
      BOOST_TEST(data[2] == 20 + cycle);
    }
  }

  // Holey in-place allgatherv: the local contribution is primed from the receive buffer before
  // every start and the staged receive is unpacked once per completion.
  {
    const int size = c.size();
    std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), l);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      displs[static_cast<std::size_t>(i)] = i * 3;
    std::vector<int> recv(static_cast<std::size_t>(size * 3), -1);
    const std::span<int> recv_span{recv};
    mp1::request r = mp1::allgatherv_init(c, mp1::in_place, recv_span, layouts, displs);
    for (int cycle = 0; cycle < 3; ++cycle) {
      recv[static_cast<std::size_t>(rank * 3)] = rank * 10 + cycle;
      recv[static_cast<std::size_t>(rank * 3 + 2)] = rank * 10 + 50 + cycle;
      r.start();
      r.wait();
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(i * 3)] == i * 10 + cycle);
        BOOST_TEST(recv[static_cast<std::size_t>(i * 3 + 2)] == i * 10 + 50 + cycle);
      }
      // wait() on the now-inactive request is MPI-free and must not unpack again.
      if (wrap_available()) {
        mpi_wrap::reset_counts();
        (void)r.wait();
        BOOST_TEST(mpi_wrap::total() == 0);
      }
    }
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(local_validation) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  int value = 0;

  // Invalid arguments are rejected before the MPI init call; only the offending rank calls
  // them, so no rank enters a collective.
  if (rank == 0) {
    BOOST_TEST(throws_invalid_argument([&] { (void)mp1::bcast_init(c, size, value); }));
    BOOST_TEST(throws_invalid_argument([&] { (void)mp1::bcast_init(c, -1, value); }));
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)mp1::bcast_init(c, 0, mp1::span_from(&value, -1)); }));
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)mp1::bcast_init(c, 0, std::span<int>{static_cast<int*>(nullptr), 1}); }));
    BOOST_TEST(throws_invalid_argument([&] { (void)mp1::barrier_init(mp1::comm_null()); }));
    BOOST_TEST(throws_invalid_argument([&] {
      (void)mp1::reduce_init(
          c, [](int a, int b) noexcept { return a + b; }, 0, mp1::span_from(&value, -1),
          std::span<int>{});
    }));
  }
  world_barrier();

  // Intercommunicator-specific local rejections: in-place, scan and exscan.
  {
    mp1::communicator inter = make_intercommunicator(c, 700);
    const mp1::layout<int> one = mp1::layout<int>::contiguous(1);
    const std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(inter.remote_size()),
                                                one);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(inter.remote_size()), 0);
    int buffer = 0;
    if (rank == 0) {
      BOOST_TEST(throws_invalid_argument([&] {
        (void)mp1::allgatherv_init(inter, mp1::in_place, std::span<int>{&buffer, 1}, layouts,
                                   displs);
      }));
      BOOST_TEST(throws_invalid_argument([&] {
        (void)mp1::scan_init(
            inter, [](int a, int b) noexcept { return a + b; }, buffer, buffer);
      }));
      BOOST_TEST(throws_invalid_argument([&] {
        (void)mp1::exscan_init(
            inter, [](int a, int b) noexcept { return a + b; }, buffer, buffer);
      }));
      BOOST_TEST(throws_invalid_argument([&] {
        (void)mp1::alltoallv_init(inter, mp1::in_place, std::span<const mp1::layout<int>>{},
                                  std::span<const mp1::count_t>{}, std::span<int>{&buffer, 1},
                                  layouts, displs);
      }));
    }
    world_barrier();
  }
}

BOOST_AUTO_TEST_CASE(info_argument) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();

  // Default (null) info.
  {
    int value = 0;
    mp1::request r = mp1::bcast_init(c, 0, value);
    if (rank == 0)
      value = 1;
    r.start();
    r.wait();
    BOOST_TEST(value == 1);
  }
  // A populated info handle is passed through to MPI_Bcast_init.
  {
    mp1::info hints;
    hints.set("mp1_test_hint", "1");
    int value = 0;
    mp1::request r = mp1::bcast_init(c, 0, value, hints);
    if (rank == 0)
      value = 2;
    r.start();
    r.wait();
    BOOST_TEST(value == 2);
  }
  world_barrier();
}

// ---------------------------------------------------------------------------------------
// Persistent staged reductions own their descriptors.  MPI 4.1 6.13 requires the
// datatype arrays handed to the init call to stay unmodified until the request is freed; a
// staged plan additionally re-packs at each start and unpacks at completion.  The caller's
// layout may therefore be destroyed (or reused) right after *_init.  Under ASan a stored
// borrow fails instead of silently re-packing with the wrong type map.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(staged_reduction_layout_lifetime) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const std::plus<int> plus{};
  const int sum = size * (size - 1) / 2;

  // reduce_init: the layout dies before the first start; the root's staged receive still
  // unpacks at offsets 0/2 of the holey block, and the non-root's send re-packs from the
  // borrowed data buffer.
  {
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> l;
      l.emplace(holey_two());
      r = mp1::reduce_init(c, plus, 0, mp1::region<int>{send.data(), *l},
                           mp1::region<int>{recv.data(), *l});
    }
    r.start();
    r.wait();
    if (rank == 0) {
      BOOST_TEST(recv[0] == sum);
      BOOST_TEST(recv[2] == sum + 10 * size);
      BOOST_TEST(recv[1] == -1);
    }
    world_barrier();
  }

  // allreduce_init: reuse the same layout object with a dense type after init.  A borrow
  // would re-pack the dense map; the owned copy keeps the holey map.
  {
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    std::optional<mp1::layout<int>> l;
    l.emplace(holey_two());
    mp1::request r = mp1::allreduce_init(c, plus, mp1::region<int>{send.data(), *l},
                                         mp1::region<int>{recv.data(), *l});
    l.emplace(mp1::layout<int>::contiguous(2));
    r.start();
    r.wait();
    BOOST_TEST(recv[0] == sum);
    BOOST_TEST(recv[2] == sum + 10 * size);
    BOOST_TEST(recv[1] == -1);
    world_barrier();
  }

  // scan_init: prefix sums.
  {
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> l;
      l.emplace(holey_two());
      r = mp1::scan_init(c, plus, mp1::region<int>{send.data(), *l},
                         mp1::region<int>{recv.data(), *l});
    }
    r.start();
    r.wait();
    BOOST_TEST(recv[0] == rank * (rank + 1) / 2);
    BOOST_TEST(recv[2] == rank * (rank + 1) / 2 + 10 * (rank + 1));
    world_barrier();
  }

  // exscan_init: rank 0's receive buffer stays untouched.
  {
    std::vector<int> send(3, -1);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, 777);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> l;
      l.emplace(holey_two());
      r = mp1::exscan_init(c, plus, mp1::region<int>{send.data(), *l},
                           mp1::region<int>{recv.data(), *l});
    }
    r.start();
    r.wait();
    if (rank == 0) {
      BOOST_TEST(recv[0] == 777);
    } else {
      BOOST_TEST(recv[0] == rank * (rank - 1) / 2);
      BOOST_TEST(recv[2] == rank * (rank - 1) / 2 + 10 * rank);
    }
    world_barrier();
  }

  // reduce_scatter_init: one holey block per rank.
  {
    std::vector<int> send(static_cast<std::size_t>(3 * size), -1);
    for (int i = 0; i < size; ++i) {
      send[static_cast<std::size_t>(3 * i)] = i;
      send[static_cast<std::size_t>(3 * i + 2)] = 10 + i;
    }
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> recv(3, -1);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> l;
      l.emplace(holey_two());
      const mp1::layout<int> aggregate = aggregate_holey_int(size);
      const std::vector<mp1::count_t> per_rank(static_cast<std::size_t>(size), 2);
      r = mp1::reduce_scatter_init(c, plus, mp1::region<int>{send.data(), aggregate},
                                   mp1::region<int>{recv.data(), *l}, per_rank);
    }
    r.start();
    r.wait();
    // Each rank contributes the full flat vector; rank r's receive block is the summed
    // flat segment [2r, 2r+1], i.e. (size*r, size*(10+r)) for this identical input.
    BOOST_TEST(recv[0] == size * rank);
    BOOST_TEST(recv[2] == size * (10 + rank));
    BOOST_TEST(recv[1] == -1);
    world_barrier();
  }

  // reduce_scatter_block_init: the same block with a uniform repetition count.
  {
    std::vector<int> send(static_cast<std::size_t>(3 * size), -1);
    for (int i = 0; i < size; ++i) {
      send[static_cast<std::size_t>(3 * i)] = i;
      send[static_cast<std::size_t>(3 * i + 2)] = 10 + i;
    }
    std::vector<int> recv(3, -1);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> l;
      l.emplace(holey_two());
      const mp1::layout<int> aggregate = aggregate_holey_int(size);
      r = mp1::reduce_scatter_block_init(c, plus, mp1::region<int>{send.data(), aggregate},
                                         mp1::region<int>{recv.data(), *l}, 2);
    }
    r.start();
    r.wait();
    BOOST_TEST(recv[0] == size * rank);
    BOOST_TEST(recv[2] == size * (10 + rank));
    BOOST_TEST(recv[1] == -1);
    world_barrier();
  }
}

BOOST_AUTO_TEST_SUITE_END()
