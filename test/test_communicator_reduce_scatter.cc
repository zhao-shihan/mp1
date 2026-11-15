// Reduce-scatter tests: free-function API, per-rank receive counts (including zero), flat and
// block-layout forms, derived/holey blocks, in-place opt-in, and nonblocking requests.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>


struct padded_value {
  int first{0};
  double second{0.0};
};

MP1_REFLECTION(padded_value, first, second)


namespace {

  struct mpi_fixture {
    mpi_fixture() {
      mp1::init();
    }

    ~mpi_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // Never throw from a fixture destructor.
      }
    }
  };

  struct padded_add {
    padded_value operator()(const padded_value& left,
                            const padded_value& right) const noexcept {
      return padded_value{left.first + right.first, left.second + right.second};
    }
  };

  [[nodiscard]] int rank_sum(int size) {
    return size * (size - 1) / 2;
  }

  template<class Call>
  [[nodiscard]] bool throws_invalid_argument(Call&& call) {
    try {
      call();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);


BOOST_AUTO_TEST_CASE(reduce_scatter_flat_counts) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const std::plus<int> plus{};
  const int expected_all = rank_sum(size);

  // Ragged counts, one distinct input block per destination rank.
  std::vector<mp1::count_t> counts(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    counts[static_cast<std::size_t>(i)] = i + 1;
  mp1::count_t total = 0;
  for (const mp1::count_t count : counts)
    total += count;

  {
    std::vector<int> send(static_cast<std::size_t>(total), 0);
    mp1::count_t offset = 0;
    for (int destination = 0; destination < size; ++destination) {
      for (mp1::count_t j = 0; j < counts[static_cast<std::size_t>(destination)]; ++j) {
        send[static_cast<std::size_t>(offset + j)] =
            rank * 1000 + destination * 100 + static_cast<int>(j);
      }
      offset += counts[static_cast<std::size_t>(destination)];
    }
    const mp1::count_t local = counts[static_cast<std::size_t>(rank)];
    std::vector<int> recv(static_cast<std::size_t>(local), -1);
    mp1::reduce_scatter(comm, plus, std::span<int>{send}, std::span<int>{recv}, counts);
    for (mp1::count_t j = 0; j < local; ++j) {
      const int expected = 1000 * expected_all + size * (rank * 100 + static_cast<int>(j));
      BOOST_TEST(recv[static_cast<std::size_t>(j)] == expected);
    }
  }

  // In-place form: recv is the full input vector and only the local block stays valid.
  {
    std::vector<int> buffer(static_cast<std::size_t>(total));
    for (mp1::count_t k = 0; k < total; ++k)
      buffer[static_cast<std::size_t>(k)] = static_cast<int>(k);
    mp1::count_t local_start = 0;
    for (int i = 0; i < rank; ++i)
      local_start += counts[static_cast<std::size_t>(i)];
    const mp1::count_t local = counts[static_cast<std::size_t>(rank)];
    mp1::reduce_scatter(comm, plus, mp1::in_place, std::span<int>{buffer}, counts);
    for (mp1::count_t j = 0; j < local; ++j) {
      const int expected = size * static_cast<int>(local_start + j);
      BOOST_TEST(buffer[static_cast<std::size_t>(j)] == expected);
    }
  }

  // Receive counts may be zero on some ranks; that rank's receive block is empty.
  {
    std::vector<mp1::count_t> sparse_counts(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      sparse_counts[static_cast<std::size_t>(i)] = (i % 2 == 0) ? 0 : 2;
    mp1::count_t sparse_total = 0;
    for (const mp1::count_t count : sparse_counts)
      sparse_total += count;
    std::vector<int> send(static_cast<std::size_t>(sparse_total), 0);
    mp1::count_t offset = 0;
    for (int destination = 0; destination < size; ++destination) {
      for (mp1::count_t j = 0; j < sparse_counts[static_cast<std::size_t>(destination)]; ++j) {
        send[static_cast<std::size_t>(offset + j)] =
            rank * 1000 + destination * 100 + static_cast<int>(j);
      }
      offset += sparse_counts[static_cast<std::size_t>(destination)];
    }
    const mp1::count_t local = sparse_counts[static_cast<std::size_t>(rank)];
    std::vector<int> recv(static_cast<std::size_t>(local), -1);
    mp1::reduce_scatter(comm, plus, std::span<int>{send}, std::span<int>{recv}, sparse_counts);
    for (mp1::count_t j = 0; j < local; ++j) {
      const int expected = 1000 * expected_all + size * (rank * 100 + static_cast<int>(j));
      BOOST_TEST(recv[static_cast<std::size_t>(j)] == expected);
    }
  }

  // Array length and negativity are rejected before MPI (all ranks throw locally).
  {
    std::vector<mp1::count_t> short_counts(counts.begin(), counts.end() - 1);
    std::vector<int> dummy(1, 0);
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::reduce_scatter(comm, plus, std::span<int>{dummy}, std::span<int>{dummy},
                          short_counts);
    }));
    std::vector<mp1::count_t> negative_counts = counts;
    negative_counts[0] = -1;
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::reduce_scatter(comm, plus, std::span<int>{dummy}, std::span<int>{dummy},
                          negative_counts);
    }));
  }
}


BOOST_AUTO_TEST_CASE(reduce_scatter_native_and_standard_operators) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
  const int sum_expected = size * (size + 1) / 2;
  int product_expected = 1;
  for (int i = 1; i <= size; ++i)
    product_expected *= i;

  // Native MPI_Op, std::plus<int> (mapped to MPI_SUM) and std::multiplies<int> (mapped to
  // MPI_PROD) through the blocking path.
  {
    std::vector<int> send(static_cast<std::size_t>(size));
    for (int j = 0; j < size; ++j)
      send[static_cast<std::size_t>(j)] = rank + 1;
    int recv = -1;
    mp1::reduce_scatter(comm, MPI_SUM, std::span<int>{send}, std::span<int>{&recv, 1}, counts);
    BOOST_TEST(recv == sum_expected);
  }
  {
    std::vector<int> send(static_cast<std::size_t>(size));
    for (int j = 0; j < size; ++j)
      send[static_cast<std::size_t>(j)] = rank + 1;
    int recv = -1;
    mp1::reduce_scatter(comm, std::plus<int>{}, std::span<int>{send}, std::span<int>{&recv, 1},
                        counts);
    BOOST_TEST(recv == sum_expected);
  }
  {
    std::vector<int> send(static_cast<std::size_t>(size));
    for (int j = 0; j < size; ++j)
      send[static_cast<std::size_t>(j)] = rank + 1;
    int recv = -1;
    mp1::reduce_scatter(comm, std::multiplies<int>{}, std::span<int>{send},
                        std::span<int>{&recv, 1}, counts);
    BOOST_TEST(recv == product_expected);
  }

  // Nonblocking path with a native op and a standard functor.
  {
    std::vector<int> send(static_cast<std::size_t>(size));
    for (int j = 0; j < size; ++j)
      send[static_cast<std::size_t>(j)] = rank + 1;
    int recv = -1;
    mp1::request request = mp1::ireduce_scatter(comm, MPI_SUM, std::span<int>{send},
                                                std::span<int>{&recv, 1}, counts);
    request.wait();
    BOOST_TEST(recv == sum_expected);
  }
  {
    std::vector<int> send(static_cast<std::size_t>(size));
    for (int j = 0; j < size; ++j)
      send[static_cast<std::size_t>(j)] = rank + 1;
    int recv = -1;
    mp1::request request = mp1::ireduce_scatter(
        comm, std::multiplies<int>{}, std::span<int>{send}, std::span<int>{&recv, 1}, counts);
    request.wait();
    BOOST_TEST(recv == product_expected);
  }

  // A native MPI_OP_NULL is rejected locally, before any collective MPI call.
  {
    std::vector<int> send(static_cast<std::size_t>(size), 0);
    int recv = -1;
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::reduce_scatter(comm, MPI_OP_NULL, std::span<int>{send}, std::span<int>{&recv, 1},
                          counts);
    }));
  }
}

BOOST_AUTO_TEST_CASE(reduce_scatter_layout_blocks) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const std::plus<int> plus{};
  const int expected_all = rank_sum(size);

  // Dense whole-buffer regions: the layout describes the whole send/recv buffer, and the
  // counts are base-element counts on every form.
  std::vector<mp1::count_t> counts(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    counts[static_cast<std::size_t>(i)] = 3 * ((i % 2) + 1);
  mp1::count_t total = 0;
  for (const mp1::count_t count : counts)
    total += count;
  const mp1::count_t local = counts[static_cast<std::size_t>(rank)];

  {
    std::vector<int> send(static_cast<std::size_t>(total), 0);
    mp1::count_t offset = 0;
    for (int destination = 0; destination < size; ++destination) {
      for (mp1::count_t j = 0; j < counts[static_cast<std::size_t>(destination)]; ++j) {
        send[static_cast<std::size_t>(offset + j)] =
            rank * 1000 + destination * 100 + static_cast<int>(j);
      }
      offset += counts[static_cast<std::size_t>(destination)];
    }
    std::vector<int> recv(static_cast<std::size_t>(local), -1);
    const mp1::layout<int> send_layout =
        mp1::layout<int>::contiguous(static_cast<mp1::count_t>(send.size()));
    const mp1::layout<int> recv_layout =
        mp1::layout<int>::contiguous(static_cast<mp1::count_t>(recv.size()));
    mp1::reduce_scatter(comm, plus, mp1::region<int>{send.data(), send_layout},
                        mp1::region<int>{recv.data(), recv_layout}, counts);
    for (mp1::count_t j = 0; j < local; ++j) {
      const int expected = 1000 * expected_all + size * (rank * 100 + static_cast<int>(j));
      BOOST_TEST(recv[static_cast<std::size_t>(j)] == expected);
    }
  }

  // In-place flat whole-buffer region: the receive buffer is the full input vector.
  {
    std::vector<int> buffer(static_cast<std::size_t>(total));
    for (mp1::count_t k = 0; k < total; ++k)
      buffer[static_cast<std::size_t>(k)] = static_cast<int>(k);
    mp1::count_t local_start = 0;
    for (int i = 0; i < rank; ++i)
      local_start += counts[static_cast<std::size_t>(i)];
    const mp1::layout<int> layout =
        mp1::layout<int>::contiguous(static_cast<mp1::count_t>(buffer.size()));
    mp1::reduce_scatter(comm, plus, mp1::in_place, mp1::region<int>{buffer.data(), layout},
                        counts);
    for (mp1::count_t j = 0; j < local; ++j) {
      const int expected = size * static_cast<int>(local_start + j);
      BOOST_TEST(buffer[static_cast<std::size_t>(j)] == expected);
    }
  }

  // Padded element type through whole-buffer regions: two padded elements per rank.
  {
    const std::vector<mp1::count_t> one_each(static_cast<std::size_t>(size), 2);
    std::vector<padded_value> send(static_cast<std::size_t>(size * 2), padded_value{-1, -1.0});
    for (int destination = 0; destination < size; ++destination) {
      send[static_cast<std::size_t>(destination * 2)] = padded_value{rank, 0.5 * rank};
      send[static_cast<std::size_t>(destination * 2 + 1)] =
          padded_value{rank + 1, 0.5 * rank + 1.0};
    }
    std::vector<padded_value> recv(2, padded_value{-1, -1.0});
    const mp1::layout<padded_value> send_layout =
        mp1::layout<padded_value>::contiguous(static_cast<mp1::count_t>(send.size()));
    const mp1::layout<padded_value> recv_layout = mp1::layout<padded_value>::contiguous(2);
    mp1::reduce_scatter(comm, padded_add{}, mp1::region<padded_value>{send.data(), send_layout},
                        mp1::region<padded_value>{recv.data(), recv_layout}, one_each);
    BOOST_TEST(recv[0].first == expected_all);
    BOOST_TEST(recv[0].second == 0.5 * static_cast<double>(expected_all));
    BOOST_TEST(recv[1].first == expected_all + size);
    BOOST_TEST(recv[1].second == 0.5 * static_cast<double>(expected_all) + size);
  }

  // Derived whole-buffer layout with a hole: two elements at offsets 0 and 2 of each
  // three-element group.  The plan stages both directions and only the mapped elements are
  // written back; the hole keeps its sentinel value.
  std::vector<mp1::count_t> holey_counts(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    holey_counts[static_cast<std::size_t>(i)] = 2 * ((i % 2) + 1);
  mp1::count_t holey_total = 0;
  for (const mp1::count_t count : holey_counts)
    holey_total += count;
  const mp1::count_t holey_local = holey_counts[static_cast<std::size_t>(rank)];
  const std::size_t holey_groups = static_cast<std::size_t>(holey_total / 2);
  const std::size_t local_groups = static_cast<std::size_t>(holey_local / 2);
  mp1::indexed_parameter holey_parameter;
  for (std::size_t group = 0; group < holey_groups; ++group) {
    holey_parameter.add(1, static_cast<mp1::count_t>(3 * group));
    holey_parameter.add(1, static_cast<mp1::count_t>(3 * group + 2));
  }
  const mp1::layout<int> holey = mp1::layout<int>::indexed(holey_parameter);
  mp1::indexed_parameter holey_local_parameter;
  for (std::size_t group = 0; group < local_groups; ++group) {
    holey_local_parameter.add(1, static_cast<mp1::count_t>(3 * group));
    holey_local_parameter.add(1, static_cast<mp1::count_t>(3 * group + 2));
  }
  const mp1::layout<int> holey_local_layout = mp1::layout<int>::indexed(holey_local_parameter);
  {
    std::vector<int> send(static_cast<std::size_t>(3 * holey_groups), -9);
    std::size_t group = 0;
    for (int destination = 0; destination < size; ++destination) {
      const mp1::count_t destination_count =
          holey_counts[static_cast<std::size_t>(destination)];
      for (mp1::count_t j = 0; j < destination_count; j += 2, ++group) {
        send[3 * group] = rank * 1000 + destination * 100 + static_cast<int>(j);
        send[3 * group + 2] = rank * 1000 + destination * 100 + static_cast<int>(j + 2);
      }
    }
    std::vector<int> recv(static_cast<std::size_t>(3 * local_groups), -1);
    mp1::reduce_scatter(comm, plus, mp1::region<int>{send.data(), holey},
                        mp1::region<int>{recv.data(), holey_local_layout}, holey_counts);
    for (std::size_t g = 0; g < local_groups; ++g) {
      const int expected_first =
          1000 * expected_all + size * (rank * 100 + static_cast<int>(2 * g));
      const int expected_second =
          1000 * expected_all + size * (rank * 100 + static_cast<int>(2 * g + 2));
      BOOST_TEST(recv[3 * g] == expected_first);
      BOOST_TEST(recv[3 * g + 1] == -1);
      BOOST_TEST(recv[3 * g + 2] == expected_second);
    }
  }

  // In-place whole-buffer non-flat region: the full input vector is the receive buffer and
  // `holey` maps it.  The plan stages the flat input image and calls the native reduction out
  // of place; the rank's local result block is unpacked into the first `holey_local` map
  // occurrences (the MPI in-place result position), while the holes keep their sentinel.
  {
    std::vector<int> buffer(static_cast<std::size_t>(3 * holey_groups), -9);
    mp1::count_t flat = 0;
    for (std::size_t g = 0; g < holey_groups; ++g) {
      buffer[3 * g] = rank * 1000 + static_cast<int>(flat++);
      buffer[3 * g + 2] = rank * 1000 + static_cast<int>(flat++);
    }
    mp1::count_t local_start = 0;
    for (int i = 0; i < rank; ++i)
      local_start += holey_counts[static_cast<std::size_t>(i)];
    mp1::reduce_scatter(comm, plus, mp1::in_place, mp1::region<int>{buffer.data(), holey},
                        holey_counts);
    for (std::size_t g = 0; g < local_groups; ++g) {
      const int expected_first =
          1000 * expected_all + size * static_cast<int>(local_start + 2 * g);
      const int expected_second =
          1000 * expected_all + size * static_cast<int>(local_start + 2 * g + 1);
      BOOST_TEST(buffer[3 * g] == expected_first);
      BOOST_TEST(buffer[3 * g + 1] == -9);
      BOOST_TEST(buffer[3 * g + 2] == expected_second);
    }
  }
}


BOOST_AUTO_TEST_CASE(reduce_scatter_nonblocking) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const std::plus<int> plus{};
  const int expected_all = rank_sum(size);

  std::vector<mp1::count_t> counts(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    counts[static_cast<std::size_t>(i)] = i + 1;
  mp1::count_t total = 0;
  for (const mp1::count_t count : counts)
    total += count;

  // Flat request.
  {
    std::vector<int> send(static_cast<std::size_t>(total), 0);
    mp1::count_t offset = 0;
    for (int destination = 0; destination < size; ++destination) {
      for (mp1::count_t j = 0; j < counts[static_cast<std::size_t>(destination)]; ++j) {
        send[static_cast<std::size_t>(offset + j)] =
            rank * 1000 + destination * 100 + static_cast<int>(j);
      }
      offset += counts[static_cast<std::size_t>(destination)];
    }
    const mp1::count_t local = counts[static_cast<std::size_t>(rank)];
    std::vector<int> recv(static_cast<std::size_t>(local), -1);
    mp1::request request =
        mp1::ireduce_scatter(comm, plus, std::span<int>{send}, std::span<int>{recv}, counts);
    request.wait();
    for (mp1::count_t j = 0; j < local; ++j) {
      const int expected = 1000 * expected_all + size * (rank * 100 + static_cast<int>(j));
      BOOST_TEST(recv[static_cast<std::size_t>(j)] == expected);
    }
  }

  // In-place flat request.
  {
    std::vector<int> buffer(static_cast<std::size_t>(total));
    for (mp1::count_t k = 0; k < total; ++k)
      buffer[static_cast<std::size_t>(k)] = static_cast<int>(k);
    mp1::count_t local_start = 0;
    for (int i = 0; i < rank; ++i)
      local_start += counts[static_cast<std::size_t>(i)];
    const mp1::count_t local = counts[static_cast<std::size_t>(rank)];
    mp1::request request =
        mp1::ireduce_scatter(comm, plus, mp1::in_place, std::span<int>{buffer}, counts);
    request.wait();
    for (mp1::count_t j = 0; j < local; ++j) {
      const int expected = size * static_cast<int>(local_start + j);
      BOOST_TEST(buffer[static_cast<std::size_t>(j)] == expected);
    }
  }

  // Whole-buffer holey region request: the staged send is owned by the request, so the
  // caller's send buffer is only borrowed by pointer; the type map is duplicated at
  // initiation, so the layout object may die before wait().
  {
    std::vector<mp1::count_t> layout_counts(static_cast<std::size_t>(size));
    mp1::count_t local = 0;
    mp1::count_t groups = 0;
    for (int i = 0; i < size; ++i) {
      const mp1::count_t elements = 2 * ((i % 2) + 1);
      layout_counts[static_cast<std::size_t>(i)] = elements;
      groups += elements / 2;
      if (i == rank)
        local = elements;
    }
    const std::size_t local_groups = static_cast<std::size_t>(local / 2);
    std::vector<int> send(static_cast<std::size_t>(3 * groups), -9);
    std::size_t group = 0;
    for (int destination = 0; destination < size; ++destination) {
      const mp1::count_t elements = layout_counts[static_cast<std::size_t>(destination)];
      for (mp1::count_t j = 0; j < elements; j += 2, ++group) {
        send[3 * group] = rank * 1000 + destination * 100 + static_cast<int>(j);
        send[3 * group + 2] = rank * 1000 + destination * 100 + static_cast<int>(j + 2);
      }
    }
    mp1::indexed_parameter send_parameter;
    for (std::size_t g = 0; g < static_cast<std::size_t>(groups); ++g) {
      send_parameter.add(1, static_cast<mp1::count_t>(3 * g));
      send_parameter.add(1, static_cast<mp1::count_t>(3 * g + 2));
    }
    mp1::indexed_parameter recv_parameter;
    for (std::size_t g = 0; g < local_groups; ++g) {
      recv_parameter.add(1, static_cast<mp1::count_t>(3 * g));
      recv_parameter.add(1, static_cast<mp1::count_t>(3 * g + 2));
    }
    std::vector<int> recv(static_cast<std::size_t>(3 * local_groups), -1);
    mp1::layout<int> send_layout = mp1::layout<int>::indexed(send_parameter);
    mp1::layout<int> recv_layout = mp1::layout<int>::indexed(recv_parameter);
    mp1::request request =
        mp1::ireduce_scatter(comm, plus, mp1::region<int>{send.data(), send_layout},
                             mp1::region<int>{recv.data(), recv_layout}, layout_counts);
    request.wait();
    for (std::size_t g = 0; g < local_groups; ++g) {
      const int expected_first =
          1000 * expected_all + size * (rank * 100 + static_cast<int>(2 * g));
      const int expected_second =
          1000 * expected_all + size * (rank * 100 + static_cast<int>(2 * g + 2));
      BOOST_TEST(recv[3 * g] == expected_first);
      BOOST_TEST(recv[3 * g + 1] == -1);
      BOOST_TEST(recv[3 * g + 2] == expected_second);
    }
  }
}
