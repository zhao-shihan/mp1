// Reduce-scatter-block tests: free-function API, flat and layout forms, padded/derived block
// layouts, in-place opt-in (full input vector), zero/negative counts and nonblocking requests.

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


BOOST_AUTO_TEST_CASE(reduce_scatter_block_flat) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const std::plus<int> plus{};
  const int expected_all = rank_sum(size);

  // Two elements per rank: element j of the result is sum over ranks of the local block.
  const mp1::count_t block_count = 2;
  {
    std::vector<int> send(static_cast<std::size_t>(block_count * size), 0);
    for (int destination = 0; destination < size; ++destination) {
      for (mp1::count_t j = 0; j < block_count; ++j) {
        send[static_cast<std::size_t>(destination * block_count + j)] =
            rank * 1000 + destination * 100 + static_cast<int>(j);
      }
    }
    std::vector<int> recv(static_cast<std::size_t>(block_count), -1);
    mp1::reduce_scatter_block(comm, plus, std::span<int>{send}, std::span<int>{recv},
                              block_count);
    for (mp1::count_t j = 0; j < block_count; ++j) {
      const int expected = 1000 * expected_all + size * (rank * 100 + static_cast<int>(j));
      BOOST_TEST(recv[static_cast<std::size_t>(j)] == expected);
    }
  }

  // Single-element block: the smallest non-empty flat form.
  {
    std::vector<int> send(static_cast<std::size_t>(size), 0);
    for (int destination = 0; destination < size; ++destination)
      send[static_cast<std::size_t>(destination)] = rank * 100 + destination;
    std::vector<int> recv(1, -1);
    mp1::reduce_scatter_block(comm, plus, std::span<int>{send}, std::span<int>{recv},
                              mp1::count_t{1});
    const int expected = 100 * expected_all + size * rank;
    BOOST_TEST(recv[0] == expected);
  }

  // In-place flat form: recv holds the full input vector; only the local block stays valid.
  {
    std::vector<int> buffer(static_cast<std::size_t>(block_count * size), 0);
    for (std::size_t k = 0; k < buffer.size(); ++k)
      buffer[k] = static_cast<int>(k);
    mp1::reduce_scatter_block(comm, plus, mp1::in_place, std::span<int>{buffer}, block_count);
    for (mp1::count_t j = 0; j < block_count; ++j) {
      const int expected = size * static_cast<int>(rank * block_count + j);
      BOOST_TEST(buffer[static_cast<std::size_t>(j)] == expected);
    }
  }

  // Zero count is a valid no-op; a negative count is rejected before MPI.
  mp1::reduce_scatter_block(comm, plus, std::span<const int>{}, std::span<int>{},
                            mp1::count_t{0});
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::reduce_scatter_block(comm, plus, std::span<const int>{}, std::span<int>{},
                              mp1::count_t{-1});
  }));
}


BOOST_AUTO_TEST_CASE(reduce_scatter_block_native_and_standard_operators) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int sum_expected = size * (size + 1) / 2;
  int product_expected = 1;
  for (int i = 1; i <= size; ++i)
    product_expected *= i;

  // Native MPI_Op and standard functors through the blocking path.
  {
    std::vector<int> send(static_cast<std::size_t>(size), rank + 1);
    int recv = -1;
    mp1::reduce_scatter_block(comm, MPI_SUM, std::span<int>{send}, std::span<int>{&recv, 1},
                              mp1::count_t{1});
    BOOST_TEST(recv == sum_expected);
  }
  {
    std::vector<int> send(static_cast<std::size_t>(size), rank + 1);
    int recv = -1;
    mp1::reduce_scatter_block(comm, std::plus<int>{}, std::span<int>{send},
                              std::span<int>{&recv, 1}, mp1::count_t{1});
    BOOST_TEST(recv == sum_expected);
  }
  {
    std::vector<int> send(static_cast<std::size_t>(size), rank + 1);
    int recv = -1;
    mp1::reduce_scatter_block(comm, std::multiplies<int>{}, std::span<int>{send},
                              std::span<int>{&recv, 1}, mp1::count_t{1});
    BOOST_TEST(recv == product_expected);
  }

  // Nonblocking path.
  {
    std::vector<int> send(static_cast<std::size_t>(size), rank + 1);
    int recv = -1;
    mp1::request request = mp1::ireduce_scatter_block(
        comm, MPI_SUM, std::span<int>{send}, std::span<int>{&recv, 1}, mp1::count_t{1});
    request.wait();
    BOOST_TEST(recv == sum_expected);
  }
  {
    std::vector<int> send(static_cast<std::size_t>(size), rank + 1);
    int recv = -1;
    mp1::request request =
        mp1::ireduce_scatter_block(comm, std::plus<int>{}, std::span<int>{send},
                                   std::span<int>{&recv, 1}, mp1::count_t{1});
    request.wait();
    BOOST_TEST(recv == sum_expected);
  }

  // Local validation before MPI.
  {
    std::vector<int> send(static_cast<std::size_t>(size), 0);
    int recv = -1;
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::reduce_scatter_block(comm, MPI_OP_NULL, std::span<int>{send},
                                std::span<int>{&recv, 1}, mp1::count_t{1});
    }));
  }
}

BOOST_AUTO_TEST_CASE(reduce_scatter_block_layouts) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const std::plus<int> plus{};
  const int expected_all = rank_sum(size);

  // Dense whole-buffer regions: recvcount is the per-rank base-element count and the layout
  // describes the whole buffer it is attached to.
  const mp1::count_t recvcount = 6;
  {
    std::vector<int> send(static_cast<std::size_t>(recvcount * size), 0);
    for (int destination = 0; destination < size; ++destination) {
      for (mp1::count_t r = 0; r < 2; ++r) {
        for (mp1::count_t j = 0; j < 3; ++j) {
          send[static_cast<std::size_t>((destination * 2 + r) * 3 + j)] =
              rank * 1000 + destination * 100 + static_cast<int>(r * 10 + j);
        }
      }
    }
    std::vector<int> recv(static_cast<std::size_t>(recvcount), -1);
    const mp1::layout<int> send_layout =
        mp1::layout<int>::contiguous(static_cast<mp1::count_t>(send.size()));
    const mp1::layout<int> recv_layout =
        mp1::layout<int>::contiguous(static_cast<mp1::count_t>(recv.size()));
    mp1::reduce_scatter_block(comm, plus, mp1::region<int>{send.data(), send_layout},
                              mp1::region<int>{recv.data(), recv_layout}, recvcount);
    for (mp1::count_t r = 0; r < 2; ++r) {
      for (mp1::count_t j = 0; j < 3; ++j) {
        const int expected =
            1000 * expected_all + size * (rank * 100 + static_cast<int>(r * 10 + j));
        BOOST_TEST(recv[static_cast<std::size_t>(r * 3 + j)] == expected);
      }
    }
  }

  // In-place flat whole-buffer region: the receive buffer is the full input vector.
  {
    std::vector<int> buffer(static_cast<std::size_t>(recvcount * size), 0);
    for (std::size_t k = 0; k < buffer.size(); ++k)
      buffer[k] = static_cast<int>(k);
    const mp1::layout<int> layout =
        mp1::layout<int>::contiguous(static_cast<mp1::count_t>(buffer.size()));
    mp1::reduce_scatter_block(comm, plus, mp1::in_place,
                              mp1::region<int>{buffer.data(), layout}, recvcount);
    for (mp1::count_t j = 0; j < recvcount; ++j) {
      const int expected = size * static_cast<int>(rank * recvcount + j);
      BOOST_TEST(buffer[static_cast<std::size_t>(j)] == expected);
    }
  }

  // Padded element type through whole-buffer regions.
  {
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
    mp1::reduce_scatter_block(
        comm, padded_add{}, mp1::region<padded_value>{send.data(), send_layout},
        mp1::region<padded_value>{recv.data(), recv_layout}, mp1::count_t{2});
    BOOST_TEST(recv[0].first == expected_all);
    BOOST_TEST(recv[0].second == 0.5 * static_cast<double>(expected_all));
    BOOST_TEST(recv[1].first == expected_all + size);
    BOOST_TEST(recv[1].second == 0.5 * static_cast<double>(expected_all) + size);
  }

  // Derived whole-buffer layout with a hole: two elements at offsets 0 and 2 of each
  // three-element group.  The plan stages both directions and only the mapped elements are
  // written back; the hole keeps its sentinel value.
  const std::size_t send_groups = static_cast<std::size_t>(size);
  const std::size_t local_groups = 1;
  mp1::indexed_parameter send_parameter;
  for (std::size_t group = 0; group < send_groups; ++group) {
    send_parameter.add(1, static_cast<mp1::count_t>(3 * group));
    send_parameter.add(1, static_cast<mp1::count_t>(3 * group + 2));
  }
  mp1::indexed_parameter local_parameter;
  for (std::size_t group = 0; group < local_groups; ++group) {
    local_parameter.add(1, static_cast<mp1::count_t>(3 * group));
    local_parameter.add(1, static_cast<mp1::count_t>(3 * group + 2));
  }
  const mp1::layout<int> holey = mp1::layout<int>::indexed(send_parameter);
  const mp1::layout<int> local_holey = mp1::layout<int>::indexed(local_parameter);
  {
    std::vector<int> send(static_cast<std::size_t>(3 * send_groups), -9);
    for (int destination = 0; destination < size; ++destination) {
      send[static_cast<std::size_t>(3 * destination)] = rank * 1000 + destination * 100;
      send[static_cast<std::size_t>(3 * destination + 2)] = rank * 1000 + destination * 100 + 2;
    }
    std::vector<int> recv(static_cast<std::size_t>(3 * local_groups), -1);
    mp1::reduce_scatter_block(comm, plus, mp1::region<int>{send.data(), holey},
                              mp1::region<int>{recv.data(), local_holey}, 2);
    const int expected_first = 1000 * expected_all + size * (rank * 100);
    const int expected_second = 1000 * expected_all + size * (rank * 100 + 2);
    BOOST_TEST(recv[0] == expected_first);
    BOOST_TEST(recv[1] == -1);
    BOOST_TEST(recv[2] == expected_second);
  }

  // In-place whole-buffer non-flat region: the full n*recvcount input vector is the receive
  // buffer and `holey` maps it.  The plan stages the flat input image and calls the native
  // reduction out of place; `recvcount` result elements are unpacked into the first `recvcount`
  // map occurrences (the MPI in-place result position), while the holes keep their sentinel.
  {
    std::vector<int> buffer(static_cast<std::size_t>(3 * size), -9);
    for (std::size_t g = 0; g < send_groups; ++g) {
      buffer[3 * g] = rank * 1000 + static_cast<int>(2 * g);
      buffer[3 * g + 2] = rank * 1000 + static_cast<int>(2 * g + 1);
    }
    mp1::reduce_scatter_block(comm, plus, mp1::in_place, mp1::region<int>{buffer.data(), holey},
                              2);
    const int expected_first = 1000 * expected_all + size * (2 * rank);
    const int expected_second = 1000 * expected_all + size * (2 * rank + 1);
    BOOST_TEST(buffer[0] == expected_first);
    BOOST_TEST(buffer[1] == -9);
    BOOST_TEST(buffer[2] == expected_second);
  }
}


BOOST_AUTO_TEST_CASE(reduce_scatter_block_nonblocking) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const std::plus<int> plus{};
  const int expected_all = rank_sum(size);

  // Flat request.
  {
    const mp1::count_t block_count = 2;
    std::vector<int> send(static_cast<std::size_t>(block_count * size), 0);
    for (int destination = 0; destination < size; ++destination) {
      for (mp1::count_t j = 0; j < block_count; ++j) {
        send[static_cast<std::size_t>(destination * block_count + j)] =
            rank * 1000 + destination * 100 + static_cast<int>(j);
      }
    }
    std::vector<int> recv(static_cast<std::size_t>(block_count), -1);
    mp1::request request = mp1::ireduce_scatter_block(comm, plus, std::span<int>{send},
                                                      std::span<int>{recv}, block_count);
    const mp1::status status = request.wait();
    BOOST_TEST(status.error() == MPI_SUCCESS);
    for (mp1::count_t j = 0; j < block_count; ++j) {
      const int expected = 1000 * expected_all + size * (rank * 100 + static_cast<int>(j));
      BOOST_TEST(recv[static_cast<std::size_t>(j)] == expected);
    }
  }

  // Whole-buffer holey region request: the staged send is request-owned and the type map is
  // duplicated at initiation, so the layout object may die before wait().
  {
    const std::size_t send_groups = static_cast<std::size_t>(size);
    const std::size_t local_groups = 1;
    mp1::indexed_parameter send_parameter;
    for (std::size_t group = 0; group < send_groups; ++group) {
      send_parameter.add(1, static_cast<mp1::count_t>(3 * group));
      send_parameter.add(1, static_cast<mp1::count_t>(3 * group + 2));
    }
    mp1::indexed_parameter local_parameter;
    for (std::size_t group = 0; group < local_groups; ++group) {
      local_parameter.add(1, static_cast<mp1::count_t>(3 * group));
      local_parameter.add(1, static_cast<mp1::count_t>(3 * group + 2));
    }
    mp1::layout<int> send_layout = mp1::layout<int>::indexed(send_parameter);
    mp1::layout<int> local_layout = mp1::layout<int>::indexed(local_parameter);
    std::vector<int> send(static_cast<std::size_t>(3 * send_groups), -9);
    for (int destination = 0; destination < size; ++destination) {
      send[static_cast<std::size_t>(3 * destination)] = rank * 1000 + destination * 100;
      send[static_cast<std::size_t>(3 * destination + 2)] = rank * 1000 + destination * 100 + 2;
    }
    std::vector<int> recv(static_cast<std::size_t>(3 * local_groups), -1);
    mp1::request request = mp1::ireduce_scatter_block(
        comm, plus, mp1::region<int>{send.data(), send_layout},
        mp1::region<int>{recv.data(), local_layout}, mp1::count_t{2});
    request.wait();
    const int expected_first = 1000 * expected_all + size * (rank * 100);
    const int expected_second = 1000 * expected_all + size * (rank * 100 + 2);
    BOOST_TEST(recv[0] == expected_first);
    BOOST_TEST(recv[1] == -1);
    BOOST_TEST(recv[2] == expected_second);
  }

  // In-place flat request.
  {
    const mp1::count_t block_count = 2;
    std::vector<int> buffer(static_cast<std::size_t>(block_count * size), 0);
    for (std::size_t k = 0; k < buffer.size(); ++k)
      buffer[k] = static_cast<int>(k);
    mp1::request request = mp1::ireduce_scatter_block(comm, plus, mp1::in_place,
                                                      std::span<int>{buffer}, block_count);
    request.wait();
    for (mp1::count_t j = 0; j < block_count; ++j) {
      const int expected = size * static_cast<int>(rank * block_count + j);
      BOOST_TEST(buffer[static_cast<std::size_t>(j)] == expected);
    }
  }
}
