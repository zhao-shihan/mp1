// Fixed-size collective contract regressions.
//
// The suite runs at 2 and 4 ranks.  It covers the zero-byte/null contract (an empty layout or
// span is presented as a dummy pointer/count 0/base datatype and never reaches MPI_Pack_c), the
// intercommunicator send/receive asymmetry (MPI 4.1 §6.7/§6.8 advice to users), bool
// reductions (MPI 4.1 §6.9.2: Logical is a legal MPI_LAND/MPI_LOR/MPI_LXOR type), the exact
// non-root scatter receive span (the former `block_count` is the span size), and the
// in-place/out-of-place exscan allocation contract.
//
// Every call is a valid MPI program; role-local failures are only exercised where all ranks
// stay out of MPI.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_FIXED_CONTRACT_WRAP 1
#else
#define MP1_FIXED_CONTRACT_WRAP 0
#endif


namespace {

  struct fixed_contract_fixture {
    fixed_contract_fixture() {
      mp1::init();
    }

    ~fixed_contract_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

  [[nodiscard]] mp1::mpi_communicator world() {
    return mp1::comm_world();
  }

  struct bool_or {
    bool operator()(bool left, bool right) const noexcept {
      return left or right;
    }
  };

  struct bool_and {
    bool operator()(bool left, bool right) const noexcept {
      return left and right;
    }
  };

  [[nodiscard]] bool expected_and(int size) {
    bool result = true;
    for (int i = 0; i < size; ++i)
      result = result and (i % 2 == 0);
    return result;
  }

  [[nodiscard]] bool expected_prefix_and(int rank) {
    bool result = true;
    for (int i = 0; i <= rank; ++i)
      result = result and (i % 2 == 0);
    return result;
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(fixed_contract_fixture);

BOOST_AUTO_TEST_SUITE(collective_fixed_contract)

// Empty layouts and count-0 spans are legal with null buffers and are
// presented to MPI as a dummy pointer, count 0 and the base datatype.
BOOST_AUTO_TEST_CASE(zero_byte_fixed_collectives) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const mp1::layout<int> empty = mp1::layout<int>::contiguous(0);
  std::vector<mp1::count_t> zeros(static_cast<std::size_t>(size), 0);

  // Barrier and broadcast.
  mp1::barrier(c);
  mp1::bcast(c, 0, mp1::region<int>{nullptr, empty});
  mp1::bcast(c, 0, std::span<int>{});

  // All-to-all and gather-to-all layout forms with null buffers on both sides.
  mp1::allgather(c, mp1::region<const int>{nullptr, empty}, mp1::region<int>{nullptr, empty});
  mp1::alltoall(c, mp1::region<const int>{nullptr, empty}, mp1::region<int>{nullptr, empty});
  mp1::allgather(c, std::span<const int>{}, std::span<int>{});
  mp1::alltoall(c, std::span<const int>{}, std::span<int>{});
  mp1::allgather(c, mp1::in_place, mp1::region<int>{nullptr, empty});
  mp1::alltoall(c, mp1::in_place, mp1::region<int>{nullptr, empty});

  // Root-taking families: the root form on the root and the non-root dummy elsewhere.
  if (rank == 0) {
    mp1::gather(c, 0, mp1::region<const int>{nullptr, empty}, mp1::region<int>{nullptr, empty});
    mp1::scatter(c, 0, mp1::region<const int>{nullptr, empty}, mp1::in_place);
    mp1::reduce(c, MPI_SUM, 0, mp1::in_place, mp1::region<int>{nullptr, empty});
  } else {
    mp1::gather(c, 0, mp1::region<const int>{nullptr, empty}, std::span<int>{});
    mp1::scatter(c, 0, std::span<const int>{}, mp1::region<int>{nullptr, empty});
    mp1::reduce(c, MPI_SUM, 0, mp1::region<const int>{nullptr, empty}, std::span<int>{});
  }

  // Every-rank families; a zero-byte typed region must skip MPI_Pack_c entirely.
  mp1::allreduce(c, MPI_SUM, mp1::in_place, mp1::region<int>{nullptr, empty});
  mp1::scan(c, MPI_SUM, mp1::in_place, mp1::region<int>{nullptr, empty});
  mp1::exscan(c, MPI_SUM, mp1::in_place, mp1::region<int>{nullptr, empty});
  mp1::reduce_scatter_block(c, MPI_SUM, mp1::in_place, mp1::region<int>{nullptr, empty}, 0);
  mp1::reduce_scatter(c, MPI_SUM, mp1::in_place, mp1::region<int>{nullptr, empty}, zeros);
  mp1::reduce_scatter_block(c, MPI_SUM, mp1::region<const int>{nullptr, empty},
                            mp1::region<int>{nullptr, empty}, 0);
  mp1::reduce_scatter(c, MPI_SUM, mp1::region<const int>{nullptr, empty},
                      mp1::region<int>{nullptr, empty}, zeros);

  // Empty containers.
  const std::vector<int> empty_send;
  std::vector<int> empty_recv;
  mp1::allgather(c, empty_send, empty_recv);
  mp1::alltoall(c, empty_send, empty_recv);

  // A non-empty argument with a null pointer is diagnosed locally, not handed to MPI.
  BOOST_CHECK_THROW(mp1::allgather(c, std::span<const int>{static_cast<const int*>(nullptr), 1},
                                   std::span<int>{}),
                    std::invalid_argument);
  BOOST_CHECK_THROW(mp1::alltoall(c, std::span<const int>{static_cast<const int*>(nullptr), 1},
                                  std::span<int>{}),
                    std::invalid_argument);
}

// The wrapper's own zero-byte staging never calls MPI_Pack_c/MPI_Unpack_c and the
// fixed families still issue exactly one native collective.
BOOST_AUTO_TEST_CASE(zero_byte_staging_call_gate) {
#if MP1_FIXED_CONTRACT_WRAP
  if (not mpi_wrap::supported())
    return;
  const mp1::mpi_communicator c = world();
  const int size = c.size();
  const mp1::layout<int> empty = mp1::layout<int>::contiguous(0);
  std::vector<mp1::count_t> zeros(static_cast<std::size_t>(size), 0);

  mpi_wrap::reset_counts();
  mpi_wrap::reset_allocation_count();
  mp1::allgather(c, mp1::region<const int>{nullptr, empty}, mp1::region<int>{nullptr, empty});
  BOOST_TEST(mpi_wrap::count("MPI_Pack_c") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Unpack_c") == 0);

  mpi_wrap::reset_counts();
  mp1::reduce_scatter_block(c, MPI_SUM, mp1::in_place, mp1::region<int>{nullptr, empty}, 0);
  BOOST_TEST(mpi_wrap::count("MPI_Reduce_scatter_block_c") == 1);
  BOOST_TEST(mpi_wrap::count("MPI_Pack_c") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Unpack_c") == 0);

  mpi_wrap::reset_counts();
  mp1::reduce_scatter(c, MPI_SUM, mp1::in_place, mp1::region<int>{nullptr, empty}, zeros);
  BOOST_TEST(mpi_wrap::count("MPI_Reduce_scatter_c") == 1);
  BOOST_TEST(mpi_wrap::count("MPI_Pack_c") == 0);
  BOOST_TEST(mpi_wrap::count("MPI_Unpack_c") == 0);
#else
  BOOST_TEST(true);
#endif
}

// Flat in-place exscan has no per-call scratch; out-of-place owns exactly one
// rank-0 buffer (the unchanged-rank-0 guarantee), and nobody memsets it.
BOOST_AUTO_TEST_CASE(exscan_flat_allocation_contract) {
#if MP1_FIXED_CONTRACT_WRAP
  if (not mpi_wrap::supported())
    return;
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();

  int in_place_value = rank + 1;
  // Warm the operator/request paths first so only the operation's own allocations count.
  mp1::exscan(c, MPI_SUM, mp1::in_place, std::span<int>{&in_place_value, 1});
  mpi_wrap::reset_allocation_count();
  mp1::exscan(c, MPI_SUM, mp1::in_place, std::span<int>{&in_place_value, 1});
  BOOST_TEST(mpi_wrap::allocation_count() == 0);

  const int send_value = rank + 1;
  int recv_value = -1;
  mp1::exscan(c, MPI_SUM, std::span<const int>{&send_value, 1}, std::span<int>{&recv_value, 1});
  mpi_wrap::reset_allocation_count();
  mp1::exscan(c, MPI_SUM, std::span<const int>{&send_value, 1}, std::span<int>{&recv_value, 1});
  BOOST_TEST(mpi_wrap::allocation_count() == (rank == 0 ? 1U : 0U));
  if (rank == 0)
    BOOST_TEST(recv_value == -1);  // rank 0's receive buffer stays untouched
#else
  BOOST_TEST(true);
#endif
}

// The persistent in-place form validates a null non-empty receive buffer before any
// pointer use, exactly like the blocking and nonblocking forms covered by
// test_reduction_layout_shift.cc.
BOOST_AUTO_TEST_CASE(exscan_init_null_receive_is_diagnosed) {
  const mp1::mpi_communicator c = world();
  BOOST_CHECK_THROW((void)mp1::exscan_init(c, MPI_SUM, mp1::in_place,
                                           std::span<int>{static_cast<int*>(nullptr), 1}),
                    std::invalid_argument);
  mp1::indexed_parameter indexed_parameter;
  indexed_parameter.add(1, 2);
  const mp1::layout<int> indexed = mp1::layout<int>::indexed(indexed_parameter);
  BOOST_CHECK_THROW(
      (void)mp1::exscan_init(c, MPI_SUM, mp1::in_place, mp1::region<int>{nullptr, indexed}),
      std::invalid_argument);
}

// A bool is a scalar data type and a Logical datatype; every reduction family and
// every mode must instantiate and produce the logical result.  Staging uses the raw
// stage_buffer (not std::vector<bool>), and a holey bool layout exercises the pack/unpack
// path.
BOOST_AUTO_TEST_CASE(bool_reductions_all_modes) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const bool flag = rank % 2 == 0;
  const bool all_and = expected_and(size);

  // Blocking scalar forms.
  {
    bool all = false;
    mp1::allreduce(c, std::logical_and<bool>{}, flag, all);
    BOOST_TEST(all == all_and);
    bool native = false;
    mp1::allreduce(c, MPI_LAND, flag, native);
    BOOST_TEST(native == all_and);
    bool custom = false;
    mp1::allreduce(c, bool_and{}, flag, custom);
    BOOST_TEST(custom == all_and);

    bool reduced = false;
    mp1::reduce(c, MPI_LAND, 0, flag, reduced);
    if (rank == 0)
      BOOST_TEST(reduced == all_and);

    bool prefix = false;
    mp1::scan(c, std::logical_and<bool>{}, flag, prefix);
    BOOST_TEST(prefix == expected_prefix_and(rank));

    bool exclusive = false;
    mp1::exscan(c, MPI_LOR, flag, exclusive);
    if (rank == 0)
      BOOST_TEST(exclusive == false);
    else
      BOOST_TEST(exclusive == true);
  }

  // Nonblocking forms.
  {
    bool all = false;
    mp1::request request = mp1::iallreduce(c, MPI_LAND, flag, all);
    request.wait();
    BOOST_TEST(all == all_and);
    bool reduced = false;
    mp1::request reduce_request = mp1::ireduce(c, std::logical_and<bool>{}, 0, flag, reduced);
    reduce_request.wait();
    if (rank == 0)
      BOOST_TEST(reduced == all_and);
    bool scanned = false;
    mp1::request scan_request = mp1::iscan(c, MPI_LAND, flag, scanned);
    scan_request.wait();
    BOOST_TEST(scanned == expected_prefix_and(rank));
    bool exc = false;
    mp1::request exscan_request = mp1::iexscan(c, MPI_LOR, flag, exc);
    exscan_request.wait();
    if (rank > 0)
      BOOST_TEST(exc == true);
  }

  // Persistent forms, two cycles: the plan must reuse the bool staging.  Every reduction
  // family is instantiated so a future std::vector<bool>-style staging regression fails to
  // compile here, not at a user call site.
  {
    bool all = false;
    mp1::request request = mp1::allreduce_init(c, std::logical_and<bool>{}, flag, all);
    for (int cycle = 0; cycle < 2; ++cycle) {
      request.start();
      request.wait();
      BOOST_TEST(all == all_and);
    }

    bool reduced = false;
    mp1::request reduce_request =
        mp1::reduce_init(c, std::logical_and<bool>{}, 0, flag, reduced);
    reduce_request.start();
    reduce_request.wait();
    if (rank == 0)
      BOOST_TEST(reduced == all_and);

    bool scanned = false;
    mp1::request scan_request = mp1::scan_init(c, MPI_LAND, flag, scanned);
    scan_request.start();
    scan_request.wait();
    BOOST_TEST(scanned == expected_prefix_and(rank));

    bool exclusive = false;
    mp1::request exscan_request = mp1::exscan_init(c, MPI_LOR, flag, exclusive);
    exscan_request.start();
    exscan_request.wait();
    if (rank > 0)
      BOOST_TEST(exclusive == true);

    std::array<bool, 8> rs_send{};
    std::array<bool, 8> rs_recv{};
    for (int i = 0; i < size; ++i)
      rs_send[static_cast<std::size_t>(i)] = i % 2 == 0;
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    mp1::request rs_request = mp1::reduce_scatter_init(
        c, MPI_LAND, std::span<bool>{rs_send.data(), static_cast<std::size_t>(size)},
        std::span<bool>{rs_recv.data(), 1}, counts);
    rs_request.start();
    rs_request.wait();
    BOOST_TEST(rs_recv[0] == (rank % 2 == 0));

    mp1::request rsb_request = mp1::reduce_scatter_block_init(
        c, MPI_LAND, std::span<bool>{rs_send.data(), static_cast<std::size_t>(size)},
        std::span<bool>{rs_recv.data(), 1}, 1);
    rsb_request.start();
    rsb_request.wait();
    BOOST_TEST(rs_recv[0] == (rank % 2 == 0));
  }

  // Holey bool layout: forces pack/unpack staging with T = bool.
  {
    const mp1::layout<bool> holey = mp1::layout<bool>::vector(2, 1, 2);
    std::array<bool, 3> send{true, false, flag};
    std::array<bool, 3> recv{false, false, false};
    mp1::allreduce(c, std::logical_and<bool>{}, mp1::region<bool>{send.data(), holey},
                   mp1::region<bool>{recv.data(), holey});
    BOOST_TEST(recv[0] == true);
    BOOST_TEST(recv[1] == false);
    BOOST_TEST(recv[2] == all_and);
  }

  // reduce_scatter[_block] with an exact per-rank element count of one bool.
  {
    std::array<bool, 8> send{};
    std::array<bool, 8> recv{};
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = i % 2 == 0;
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    // Rank r receives the reduction of block r over all ranks; every rank's send buffer has
    // the same alternating pattern, so block r reduces to (r % 2 == 0).
    const bool block_result = rank % 2 == 0;
    mp1::reduce_scatter(c, MPI_LAND,
                        std::span<bool>{send.data(), static_cast<std::size_t>(size)},
                        std::span<bool>{recv.data(), 1}, counts);
    BOOST_TEST(recv[0] == block_result);
    mp1::reduce_scatter_block(c, MPI_LAND,
                              std::span<bool>{send.data(), static_cast<std::size_t>(size)},
                              std::span<bool>{recv.data(), 1}, 1);
    BOOST_TEST(recv[0] == block_result);
    mp1::request iscatter_request = mp1::ireduce_scatter(
        c, MPI_LAND, std::span<bool>{send.data(), static_cast<std::size_t>(size)},
        std::span<bool>{recv.data(), 1}, counts);
    iscatter_request.wait();
    BOOST_TEST(recv[0] == block_result);
    mp1::request request = mp1::ireduce_scatter_block(
        c, MPI_LAND, std::span<bool>{send.data(), static_cast<std::size_t>(size)},
        std::span<bool>{recv.data(), 1}, 1);
    request.wait();
    BOOST_TEST(recv[0] == block_result);
  }
}

// Intercommunicator allgather/alltoall allow different block sizes in the two
// directions; intracommunicator local equality stays a diagnosed invariant.
BOOST_AUTO_TEST_CASE(intercommunicator_asymmetric_blocks) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int group = rank % 2;
  mp1::communicator local = mp1::communicator::split(c, group);
  const int local_leader = 0;
  const int remote_leader = group == 0 ? 1 : 0;
  mp1::communicator inter =
      mp1::communicator::intercomm_create(local, local_leader, c, remote_leader, mp1::tag{91});

  const int send_block = group == 0 ? 2 : 1;
  const int recv_block = group == 0 ? 1 : 2;
  const mp1::layout<int> send_layout = mp1::layout<int>::contiguous(send_block);
  const mp1::layout<int> recv_layout =
      mp1::layout<int>::contiguous(recv_block * inter.remote_size());
  const int peers = inter.remote_size();

  std::vector<int> send(static_cast<std::size_t>(send_block), 7 + group);
  std::vector<int> recv(static_cast<std::size_t>(recv_block * peers), -1);
  mp1::allgather(inter, mp1::region<int>{send.data(), send_layout},
                 mp1::region<int>{recv.data(), recv_layout});
  for (int i = 0; i < recv_block * peers; ++i)
    BOOST_TEST(recv[static_cast<std::size_t>(i)] == 7 + (1 - group));

  // Container allgather: one local block per rank, the receive side is the concatenation of
  // the remote group's blocks.
  std::vector<int> container_send(static_cast<std::size_t>(send_block), 17 + group);
  std::vector<int> container_recv(static_cast<std::size_t>(recv_block * peers), -1);
  mp1::allgather(inter, container_send, container_recv);
  for (int i = 0; i < recv_block * peers; ++i)
    BOOST_TEST(container_recv[static_cast<std::size_t>(i)] == 17 + (1 - group));

  std::vector<int> send_all(static_cast<std::size_t>(send_block * peers), 3 + group);
  std::vector<int> recv_all(static_cast<std::size_t>(recv_block * peers), -1);
  const mp1::layout<int> send_aggregate = mp1::layout<int>::contiguous(send_block * peers);
  const mp1::layout<int> recv_aggregate = mp1::layout<int>::contiguous(recv_block * peers);
  mp1::alltoall(inter, mp1::region<int>{send_all.data(), send_aggregate},
                mp1::region<int>{recv_all.data(), recv_aggregate});
  for (int i = 0; i < recv_block * peers; ++i)
    BOOST_TEST(recv_all[static_cast<std::size_t>(i)] == 3 + (1 - group));

  // Container alltoall: the two directions have separate totals and separate per-peer
  // blocks (the split that a single shared container block would get wrong).
  std::vector<int> container_all_send(static_cast<std::size_t>(send_block * peers), 23 + group);
  std::vector<int> container_all_recv(static_cast<std::size_t>(recv_block * peers), -1);
  mp1::alltoall(inter, container_all_send, container_all_recv);
  for (int i = 0; i < recv_block * peers; ++i)
    BOOST_TEST(container_all_recv[static_cast<std::size_t>(i)] == 23 + (1 - group));

  // The nonblocking forms share the helper; keep one live check on the same shape.
  std::vector<int> irecv(static_cast<std::size_t>(recv_block * peers), -1);
  mp1::request request = mp1::iallgather(inter, mp1::region<int>{send.data(), send_layout},
                                         mp1::region<int>{irecv.data(), recv_layout});
  request.wait();
  for (int i = 0; i < recv_block * peers; ++i)
    BOOST_TEST(irecv[static_cast<std::size_t>(i)] == 7 + (1 - group));

  // Intracommunicator local equality is still rejected on every rank before MPI.
  BOOST_CHECK_THROW(mp1::allgather(c, std::span<const int>{static_cast<const int*>(nullptr), 2},
                                   std::span<int>{static_cast<int*>(nullptr), 1}),
                    std::invalid_argument);
  BOOST_CHECK_THROW(mp1::alltoall(c, std::span<const int>{static_cast<const int*>(nullptr), 2},
                                  std::span<int>{static_cast<int*>(nullptr), 1}),
                    std::invalid_argument);

  inter.disconnect();
}

// The per-peer scatter block is the receive span itself; a root aggregate and a non-root
// receive span are the two spellings (the former `block_count` value type is deleted).
BOOST_AUTO_TEST_CASE(scatter_block_count_forms) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  // Scalar receive: the root's send is the aggregate of the per-rank scalars.
  std::vector<int> scalar_send(static_cast<std::size_t>(size), -1);
  for (int i = 0; i < size; ++i)
    scalar_send[static_cast<std::size_t>(i)] = 500 + i;
  int scalar = -1;
  if (rank == root)
    mp1::scatter(c, root, std::span<const int>{scalar_send}, scalar);
  else
    mp1::scatter(c, root, std::span<const int>{}, scalar);
  BOOST_TEST(scalar == 500 + rank);

  // Two-element receive block described by the receive span itself.
  std::vector<int> send(static_cast<std::size_t>(2 * size), -1);
  for (int i = 0; i < size; ++i) {
    send[static_cast<std::size_t>(2 * i)] = 600 + i;
    send[static_cast<std::size_t>(2 * i + 1)] = 700 + i;
  }
  std::array<int, 2> recv{-1, -1};
  if (rank == root)
    mp1::scatter(c, root, std::span<const int>{send}, std::span<int>{recv});
  else
    mp1::scatter(c, root, std::span<const int>{}, std::span<int>{recv});
  BOOST_TEST(recv[0] == 600 + rank);
  BOOST_TEST(recv[1] == 700 + rank);

  std::array<int, 2> irecv{-1, -1};
  mp1::request request{};
  if (rank == root)
    request = mp1::iscatter(c, root, std::span<const int>{send}, std::span<int>{irecv});
  else
    request = mp1::iscatter(c, root, std::span<const int>{}, std::span<int>{irecv});
  request.wait();
  BOOST_TEST(irecv[0] == 600 + rank);
  BOOST_TEST(irecv[1] == 700 + rank);

  std::array<int, 2> precv{-1, -1};
  mp1::request persistent{};
  if (rank == root)
    persistent = mp1::scatter_init(c, root, std::span<const int>{send}, std::span<int>{precv});
  else
    persistent = mp1::scatter_init(c, root, std::span<const int>{}, std::span<int>{precv});
  persistent.start();
  persistent.wait();
  BOOST_TEST(precv[0] == 600 + rank);
  BOOST_TEST(precv[1] == 700 + rank);
}

BOOST_AUTO_TEST_SUITE_END()
