// Role-spelling contract.
//
// Fixed families have one declaration per lifetime, so the non-root ignored position is a
// dummy buffer (empty span), not a separate overload.  reduce_scatter/reduce_scatter_block have
// no root and no ignored position: every rank passes the real receive buffer and identical
// counts.  Intracommunicator roots are valid local ranks; only an intercommunicator root group
// uses MPI_ROOT/MPI_PROC_NULL.  Runs at 2 and 4 ranks.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>


namespace {

  struct role_fixture {
    role_fixture() {
      mp1::init();
    }

    ~role_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
      }
    }
  };

  [[nodiscard]] mp1::mpi_communicator world() {
    return mp1::comm_world();
  }

  [[nodiscard]] int sum_ranks(int size) {
    return size * (size - 1) / 2;
  }

  template<class Function>
  [[nodiscard]] bool throws_invalid(Function&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(role_fixture);

BOOST_AUTO_TEST_SUITE(collective_role_spellings)

// Dummy spellings for the families whose non-root position MPI ignores.
BOOST_AUTO_TEST_CASE(dummy_non_root_spellings) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  // gather: real aggregate at the root, empty span elsewhere.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    if (rank == root) {
      mp1::gather(c, root, 200 + rank, recv);
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 200 + i);
    } else {
      mp1::gather(c, root, 200 + rank, std::span<int>{});
    }
  }

  // scatter: real aggregate at the root, empty span elsewhere.
  {
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = 300 + i;
    if (rank == root) {
      int recv = -1;
      mp1::scatter(c, root, send, recv);
      BOOST_TEST(recv == 300 + root);
    } else {
      int recv = -1;
      mp1::scatter(c, root, std::span<const int>{}, recv);
      BOOST_TEST(recv == 300 + rank);
    }
  }

  // gatherv: real aggregate at the root, empty span + empty blocks elsewhere.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    std::vector<mp1::count_t> displacements(static_cast<std::size_t>(size), 0);
    for (int i = 0; i < size; ++i)
      displacements[static_cast<std::size_t>(i)] = i;
    if (rank == root) {
      mp1::gatherv(c, root, 600 + rank, recv, mp1::blocks{counts, displacements});
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 600 + i);
    } else {
      mp1::gatherv(c, root, 600 + rank, std::span<int>{}, mp1::blocks{});
    }
  }

  // scatterv: real aggregate at the root, empty span + empty blocks elsewhere.
  {
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = 700 + i;
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    std::vector<mp1::count_t> displacements(static_cast<std::size_t>(size), 0);
    for (int i = 0; i < size; ++i)
      displacements[static_cast<std::size_t>(i)] = i;
    if (rank == root) {
      int recv = -1;
      mp1::scatterv(c, root, send, mp1::blocks{counts, displacements}, recv);
      BOOST_TEST(recv == 700 + root);
    } else {
      int recv = -1;
      mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{}, recv);
      BOOST_TEST(recv == 700 + rank);
    }
  }

  // reduce: the non-root receive buffer is ignored and is not compared with the real send.
  // A non-empty dummy is accepted at the ignored position and left untouched.
  {
    std::vector<int> dummy(static_cast<std::size_t>(size), -1);
    if (rank == root) {
      int recv = -1;
      mp1::reduce(c, std::plus<int>{}, root, rank, recv);
      BOOST_TEST(recv == sum_ranks(size));
    } else {
      mp1::reduce(c, std::plus<int>{}, root, rank, dummy);
      BOOST_TEST(std::ranges::all_of(dummy, [](int value) { return value == -1; }));
    }
  }
}

// No-root families: every rank passes the real receive buffer and identical counts; a
// result-sized receive in the in-place form and an empty/short counts array are rejected.
BOOST_AUTO_TEST_CASE(no_root_families) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
  counts[0] = 2;
  mp1::count_t total = 0;
  for (const mp1::count_t count : counts)
    total += count;
  const mp1::count_t local = counts[static_cast<std::size_t>(rank)];

  // Out-of-place: send is the full vector, recv is exactly the local block.
  {
    std::vector<int> send(static_cast<std::size_t>(total), rank);
    std::vector<int> recv(static_cast<std::size_t>(local), -1);
    mp1::reduce_scatter(c, std::plus<int>{}, send, recv, counts);
    for (const int value : recv)
      BOOST_TEST(value == sum_ranks(size));
  }

  // A result-sized in-place buffer is rejected.
  {
    std::vector<int> result_sized(static_cast<std::size_t>(local), -1);
    BOOST_TEST(throws_invalid([&] {
      mp1::reduce_scatter(c, std::plus<int>{}, mp1::in_place, result_sized, counts);
    }));
  }

  // An empty/short counts array is rejected by the local size check.
  {
    std::vector<int> send(static_cast<std::size_t>(total), rank);
    std::vector<int> recv(static_cast<std::size_t>(local), -1);
    const std::vector<mp1::count_t> short_counts(static_cast<std::size_t>(size - 1), 1);
    BOOST_TEST(throws_invalid(
        [&] { mp1::reduce_scatter(c, std::plus<int>{}, send, recv, short_counts); }));
    const std::span<const mp1::count_t> empty_counts{};
    BOOST_TEST(throws_invalid(
        [&] { mp1::reduce_scatter(c, std::plus<int>{}, send, recv, empty_counts); }));
  }

  // A dummy/empty receive for a non-zero local count is a local contract violation: every
  // rank passes a real receive buffer (there is no ignored root position).
  {
    std::vector<int> send(static_cast<std::size_t>(total), rank);
    const std::span<int> empty_recv{};
    BOOST_TEST(throws_invalid(
        [&] { mp1::reduce_scatter(c, std::plus<int>{}, send, empty_recv, counts); }));
  }

  // reduce_scatter_block: same rules with the scalar per-rank count.
  {
    std::vector<int> send(static_cast<std::size_t>(2 * size), rank);
    std::vector<int> recv(2, -1);
    mp1::reduce_scatter_block(c, std::plus<int>{}, send, recv, 2);
    for (const int value : recv)
      BOOST_TEST(value == sum_ranks(size));
    std::vector<int> result_sized(2, -1);
    BOOST_TEST(throws_invalid([&] {
      mp1::reduce_scatter_block(c, std::plus<int>{}, mp1::in_place, result_sized, 2);
    }));
  }
}

// Intracommunicator roots are valid local ranks; sentinels are not roots.
BOOST_AUTO_TEST_CASE(intra_root_encoding) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();

  int value = rank;
  BOOST_TEST(throws_invalid([&] { mp1::bcast(c, MPI_PROC_NULL, value); }));
  BOOST_TEST(throws_invalid([&] { mp1::bcast(c, MPI_ROOT, value); }));
  BOOST_TEST(throws_invalid([&] { mp1::bcast(c, size, value); }));
  BOOST_TEST(throws_invalid([&] { mp1::bcast(c, -1, value); }));
  std::vector<int> recv(static_cast<std::size_t>(size), -1);
  BOOST_TEST(throws_invalid([&] { mp1::gather(c, MPI_PROC_NULL, rank, std::span<int>{}); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::reduce(c, std::plus<int>{}, MPI_ROOT, rank, std::span<int>{}); }));
}

// Intercommunicator root encoding: MPI_ROOT at the root, MPI_PROC_NULL at the other members
// of the root's group, and the root's rank in the root's group for the remote group.
BOOST_AUTO_TEST_CASE(inter_root_encoding) {
  const mp1::mpi_communicator c = world();
  const int group = c.rank() % 2;
  mp1::communicator local = mp1::communicator::split(c, group);
  mp1::communicator inter =
      mp1::communicator::intercomm_create(local, 0, c, group == 0 ? 1 : 0, mp1::tag{9201});
  const int local_rank = inter.rank();
  const int remote_size = inter.remote_size();
  const int root_arg = group == 0 ? (local_rank == 0 ? MPI_ROOT : MPI_PROC_NULL) : 0;

  // bcast: the root group root sends, the remote group receives, the other root-group
  // members pass MPI_PROC_NULL and ignore the buffer.
  {
    int value = (group == 0 and local_rank == 0) ? 4321 : -1;
    mp1::bcast(inter, root_arg, value);
    if (group == 1)
      BOOST_TEST(value == 4321);
  }

  // gather: only the root group root receives; the remote group sends.
  {
    std::vector<int> recv;
    if (group == 0 and local_rank == 0)
      recv.assign(static_cast<std::size_t>(remote_size), -1);
    if (group == 0 and local_rank == 0) {
      mp1::gather(inter, root_arg, 0, recv);
      for (int i = 0; i < remote_size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 1000 + i);
    } else if (group == 1) {
      mp1::gather(inter, root_arg, 1000 + local_rank, std::span<int>{});
    } else {
      mp1::gather(inter, root_arg, 0, std::span<int>{});
    }
  }

  // scatter: only the root group root sends; the remote group receives.
  {
    std::vector<int> send;
    if (group == 0 and local_rank == 0) {
      send.assign(static_cast<std::size_t>(remote_size), -1);
      for (int i = 0; i < remote_size; ++i)
        send[static_cast<std::size_t>(i)] = 2000 + i;
    }
    if (group == 0 and local_rank == 0) {
      int ignored = 0;
      mp1::scatter(inter, root_arg, send, ignored);
    } else {
      int recv = -1;
      mp1::scatter(inter, root_arg, std::span<const int>{}, recv);
      if (group == 1)
        BOOST_TEST(recv == 2000 + local_rank);
    }
  }

  // reduce: the root group root receives the remote group's reduction.
  {
    const int root = group == 0 ? (local_rank == 0 ? MPI_ROOT : MPI_PROC_NULL) : 0;
    if (group == 0 and local_rank == 0) {
      int recv = -1;
      mp1::reduce(inter, std::plus<int>{}, root, 0, recv);
      int expected = 0;
      for (int i = 0; i < remote_size; ++i)
        expected += 3000 + i;
      BOOST_TEST(recv == expected);
    } else {
      mp1::reduce(inter, std::plus<int>{}, root, 3000 + local_rank, std::span<int>{});
    }
  }

  inter.disconnect();
}

BOOST_AUTO_TEST_SUITE_END()
