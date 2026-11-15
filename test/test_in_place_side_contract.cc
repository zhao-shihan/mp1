// In-place side contract.
//
// The side is compile-checked by the position concepts (the negative matrix lives in the
// compile-fail probes); this suite pins the runtime rules: intracommunicator-only, root-only
// for gather/scatter/reduce, the data-buffer sizes, and the sentinel/data-buffer
// positions.  Runs at 2 and 4 ranks.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>


namespace {

  struct in_place_fixture {
    in_place_fixture() {
      mp1::init();
    }

    ~in_place_fixture() {
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

BOOST_TEST_GLOBAL_FIXTURE(in_place_fixture);

BOOST_AUTO_TEST_SUITE(collective_in_place_side)

// The positive matrix and the sentinel/data-buffer positions.
BOOST_AUTO_TEST_CASE(in_place_positive_matrix) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  // gather: sentinel in the send slot, data in the root's aggregate receive buffer.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    if (rank == root) {
      recv[static_cast<std::size_t>(root)] = root * 10;
      mp1::gather(c, root, mp1::in_place, recv);
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == i * 10);
    } else {
      mp1::gather(c, root, rank * 10, std::span<int>{});
    }
  }

  // scatter: sentinel in the receive slot, data in the root's send aggregate.
  {
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = 100 + i;
    if (rank == root) {
      mp1::scatter(c, root, send, mp1::in_place);
      BOOST_TEST(send[static_cast<std::size_t>(root)] == 100 + root);
    } else {
      int recv = -1;
      mp1::scatter(c, root, std::span<const int>{}, recv);
      BOOST_TEST(recv == 100 + rank);
    }
  }

  // allgather / alltoall: in-place at every rank.
  {
    std::vector<int> values(static_cast<std::size_t>(size), -1);
    values[static_cast<std::size_t>(rank)] = rank + 5;
    mp1::allgather(c, mp1::in_place, values);
    for (int i = 0; i < size; ++i)
      BOOST_TEST(values[static_cast<std::size_t>(i)] == i + 5);
  }
  {
    std::vector<int> buffer(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      buffer[static_cast<std::size_t>(i)] = rank * 100 + i;
    mp1::alltoall(c, mp1::in_place, buffer);
    for (int i = 0; i < size; ++i)
      BOOST_TEST(buffer[static_cast<std::size_t>(i)] == i * 100 + rank);
  }

  // allreduce / scan / exscan: the receive buffer is the input and the output.
  {
    std::vector<int> values(static_cast<std::size_t>(size), rank);
    mp1::allreduce(c, std::plus<int>{}, mp1::in_place, values);
    for (const int value : values)
      BOOST_TEST(value == sum_ranks(size));
  }
  {
    std::vector<int> prefix(static_cast<std::size_t>(size), rank);
    mp1::scan(c, std::plus<int>{}, mp1::in_place, prefix);
    for (const int value : prefix)
      BOOST_TEST(value == rank * (rank + 1) / 2);
  }
  {
    std::vector<int> suffix(static_cast<std::size_t>(size), rank);
    mp1::exscan(c, std::plus<int>{}, mp1::in_place, suffix);
    if (rank == 0) {
      for (const int value : suffix)
        BOOST_TEST(value == 0);
    } else {
      for (const int value : suffix)
        BOOST_TEST(value == rank * (rank - 1) / 2);
    }
  }

  // reduce: the root's receive buffer is the input and the output; there is no aggregate.
  {
    int recv = rank;
    if (rank == root) {
      mp1::reduce(c, std::plus<int>{}, root, mp1::in_place, recv);
      BOOST_TEST(recv == sum_ranks(size));
    } else {
      mp1::reduce(c, std::plus<int>{}, root, rank, std::span<int>{});
    }
  }

  // Typed single-block in-place forms.
  {
    const mp1::layout<int> layout = mp1::layout<int>::contiguous(2);
    mp1::region<int> region{static_cast<int*>(nullptr), layout};
    (void)region;
    std::array<int, 2> values{rank, rank + 1};
    mp1::allreduce(c, std::plus<int>{}, mp1::in_place, mp1::region<int>{values.data(), layout});
    BOOST_TEST(values[0] == sum_ranks(size));
    BOOST_TEST(values[1] == sum_ranks(size) + size);
  }
}

// The in-place receive buffer is the full input vector, not the result block.
BOOST_AUTO_TEST_CASE(in_place_size_rules) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  // reduce_scatter: recv.size() == sum(recvcounts) passes; a result-sized recv is rejected.
  {
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    mp1::count_t total = 0;
    for (const mp1::count_t count : counts)
      total += count;
    std::vector<int> full(static_cast<std::size_t>(total));
    for (int i = 0; i < total; ++i)
      full[static_cast<std::size_t>(i)] = rank + i;
    mp1::reduce_scatter(c, std::plus<int>{}, mp1::in_place, full, counts);

    std::vector<int> result_sized(
        static_cast<std::size_t>(counts[static_cast<std::size_t>(rank)]), -1);
    BOOST_TEST(throws_invalid([&] {
      mp1::reduce_scatter(c, std::plus<int>{}, mp1::in_place, result_sized, counts);
    }));
  }

  // reduce_scatter_block: recv.size() == peers * recvcount passes; recvcount is rejected.
  {
    const mp1::count_t recvcount = 2;
    std::vector<int> full(static_cast<std::size_t>(recvcount * size));
    for (std::size_t i = 0; i < full.size(); ++i)
      full[i] = rank + static_cast<int>(i);
    mp1::reduce_scatter_block(c, std::plus<int>{}, mp1::in_place, full, recvcount);

    std::vector<int> result_sized(static_cast<std::size_t>(recvcount), -1);
    BOOST_TEST(throws_invalid([&] {
      mp1::reduce_scatter_block(c, std::plus<int>{}, mp1::in_place, result_sized, recvcount);
    }));
  }

  // Root in-place reduce: the count is the root's receive size; there is no aggregate term and
  // no `peers` multiplication, so a `peers*count`-sized buffer is simply a reduction over that
  // many elements.  It is not rejected: a local check cannot distinguish it from a caller's
  // larger count (the cross-rank count agreement is the caller's obligation, as out of place).
  {
    const int count = 3;
    const std::size_t length = static_cast<std::size_t>(count * size);
    if (rank == root) {
      std::vector<int> recv(length, -1);
      for (std::size_t j = 0; j < length; ++j)
        recv[j] = rank + static_cast<int>(j);
      mp1::reduce(c, std::plus<int>{}, root, mp1::in_place, recv);
      for (std::size_t j = 0; j < length; ++j)
        BOOST_TEST(recv[j] == sum_ranks(size) + size * static_cast<int>(j));
    } else {
      std::vector<int> send(length, -1);
      for (std::size_t j = 0; j < length; ++j)
        send[j] = rank + static_cast<int>(j);
      mp1::reduce(c, std::plus<int>{}, root, send, std::span<int>{});
    }
  }
}

// Non-flat in-place reduce_scatter / reduce_scatter_block.  The receive buffer is
// the full input vector described by one whole-buffer region; the local result block is
// unpacked into the first `recvcounts[rank]` / `recvcount` map occurrences (MPI 4.1 6.10) and
// the holes keep their sentinel value.  The blocking, nonblocking and persistent forms share
// the staged whole-image plan; the persistent cycles change the input between starts so a
// missing re-pack is observable.
BOOST_AUTO_TEST_CASE(in_place_nonflat_reduce_scatter) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int expected_all = sum_ranks(size);

  // reduce_scatter: counts vary per rank, two mapped elements of every three-element group.
  {
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size));
    mp1::count_t total = 0;
    for (int i = 0; i < size; ++i) {
      counts[static_cast<std::size_t>(i)] = 2 * ((i % 2) + 1);
      total += counts[static_cast<std::size_t>(i)];
    }
    const std::size_t groups = static_cast<std::size_t>(total / 2);
    mp1::indexed_parameter parameter;
    for (std::size_t g = 0; g < groups; ++g) {
      parameter.add(1, static_cast<mp1::count_t>(3 * g));
      parameter.add(1, static_cast<mp1::count_t>(3 * g + 2));
    }
    const mp1::layout<int> whole = mp1::layout<int>::indexed(parameter);

    mp1::count_t local_start = 0;
    for (int i = 0; i < rank; ++i)
      local_start += counts[static_cast<std::size_t>(i)];
    const mp1::count_t local = counts[static_cast<std::size_t>(rank)];
    const std::size_t local_groups = static_cast<std::size_t>(local / 2);

    const auto fill = [&](std::vector<int>& buffer, int phase) {
      mp1::count_t flat = 0;
      for (std::size_t g = 0; g < groups; ++g) {
        buffer[3 * g] = rank * 1000 + phase + static_cast<int>(flat++);
        buffer[3 * g + 2] = rank * 1000 + phase + static_cast<int>(flat++);
      }
    };
    const auto verify = [&](const std::vector<int>& buffer, int phase) {
      for (std::size_t g = 0; g < local_groups; ++g) {
        const int expected_first =
            1000 * expected_all + size * (phase + static_cast<int>(local_start + 2 * g));
        const int expected_second =
            1000 * expected_all + size * (phase + static_cast<int>(local_start + 2 * g + 1));
        BOOST_TEST(buffer[3 * g] == expected_first);
        BOOST_TEST(buffer[3 * g + 1] == -9);
        BOOST_TEST(buffer[3 * g + 2] == expected_second);
      }
    };

    // Blocking.
    {
      std::vector<int> buffer(static_cast<std::size_t>(3 * groups), -9);
      fill(buffer, 0);
      mp1::reduce_scatter(c, std::plus<int>{}, mp1::in_place,
                          mp1::region<int>{buffer.data(), whole}, counts);
      verify(buffer, 0);
    }

    // Nonblocking: the request owns the staged whole input image and the result unpacks at
    // completion, after the local plan is gone.
    {
      std::vector<int> buffer(static_cast<std::size_t>(3 * groups), -9);
      fill(buffer, 0);
      mp1::request request = mp1::ireduce_scatter(
          c, std::plus<int>{}, mp1::in_place, mp1::region<int>{buffer.data(), whole}, counts);
      (void)request.wait();
      verify(buffer, 0);
    }

    // Persistent: a new input image is re-packed before every start.
    {
      std::vector<int> buffer(static_cast<std::size_t>(3 * groups), -9);
      mp1::region<int> whole_region{buffer.data(), whole};
      mp1::request request =
          mp1::reduce_scatter_init(c, std::plus<int>{}, mp1::in_place, whole_region, counts);
      fill(buffer, 0);
      request.start();
      (void)request.wait();
      verify(buffer, 0);
      fill(buffer, 1);
      request.start();
      (void)request.wait();
      verify(buffer, 1);
    }
  }

  // reduce_scatter_block: recvcount = 2, the whole map has one two-element group per rank.
  {
    const mp1::count_t recvcount = 2;
    mp1::indexed_parameter parameter;
    for (int i = 0; i < size; ++i) {
      parameter.add(1, static_cast<mp1::count_t>(3 * i));
      parameter.add(1, static_cast<mp1::count_t>(3 * i + 2));
    }
    const mp1::layout<int> whole = mp1::layout<int>::indexed(parameter);

    const auto fill = [&](std::vector<int>& buffer) {
      for (int i = 0; i < size; ++i) {
        buffer[static_cast<std::size_t>(3 * i)] = rank * 1000 + 2 * i;
        buffer[static_cast<std::size_t>(3 * i + 2)] = rank * 1000 + 2 * i + 1;
      }
    };
    const auto verify = [&](const std::vector<int>& buffer) {
      const int expected_first = 1000 * expected_all + size * (2 * rank);
      const int expected_second = 1000 * expected_all + size * (2 * rank + 1);
      BOOST_TEST(buffer[0] == expected_first);
      BOOST_TEST(buffer[1] == -9);
      BOOST_TEST(buffer[2] == expected_second);
    };

    // Blocking.
    {
      std::vector<int> buffer(static_cast<std::size_t>(3 * size), -9);
      fill(buffer);
      mp1::reduce_scatter_block(c, std::plus<int>{}, mp1::in_place,
                                mp1::region<int>{buffer.data(), whole}, recvcount);
      verify(buffer);
    }

    // Nonblocking.
    {
      std::vector<int> buffer(static_cast<std::size_t>(3 * size), -9);
      fill(buffer);
      mp1::request request =
          mp1::ireduce_scatter_block(c, std::plus<int>{}, mp1::in_place,
                                     mp1::region<int>{buffer.data(), whole}, recvcount);
      (void)request.wait();
      verify(buffer);
    }
  }
}

// Runtime role rejections: in-place is intracommunicator-only and root-only for the root
// families.  Every rank performs its local check; no rank enters MPI.
BOOST_AUTO_TEST_CASE(in_place_role_rejections) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;

  if (rank != root) {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    BOOST_TEST(throws_invalid([&] { mp1::gather(c, root, mp1::in_place, recv); }));
    int recv_scalar = -1;
    BOOST_TEST(throws_invalid(
        [&] { mp1::reduce(c, std::plus<int>{}, root, mp1::in_place, recv_scalar); }));
    std::vector<int> send(static_cast<std::size_t>(size), 1);
    BOOST_TEST(throws_invalid([&] { mp1::scatter(c, root, send, mp1::in_place); }));
  }

  // A runtime check only; no MPI call follows a legitimate rejection.
  BOOST_TEST(
      throws_invalid([&] { mp1::gather(c, MPI_PROC_NULL, mp1::in_place, std::span<int>{}); }));
  BOOST_TEST(throws_invalid(
      [&] { mp1::scatter(c, MPI_ROOT, std::span<const int>{}, mp1::in_place); }));
}

// Intercommunicators have no in-place form at all.
BOOST_AUTO_TEST_CASE(inter_in_place_rejected) {
  const mp1::mpi_communicator c = world();
  const int group = c.rank() % 2;
  mp1::communicator local = mp1::communicator::split(c, group);
  mp1::communicator inter =
      mp1::communicator::intercomm_create(local, 0, c, group == 0 ? 1 : 0, mp1::tag{9101});
  const int peers = inter.remote_size();

  std::vector<int> buffer(static_cast<std::size_t>(peers), -1);
  BOOST_TEST(throws_invalid([&] { mp1::allgather(inter, mp1::in_place, buffer); }));
  BOOST_TEST(
      throws_invalid([&] { mp1::allreduce(inter, std::plus<int>{}, mp1::in_place, buffer); }));

  inter.disconnect();
}

BOOST_AUTO_TEST_SUITE_END()
