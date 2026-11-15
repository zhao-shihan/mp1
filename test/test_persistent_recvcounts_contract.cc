// Persistent and nonblocking reduce_scatter descriptor lifetime.
//
// `recvcounts` (and the scalar reduce_scatter_block count) are copied at initiation, so the
// caller's array may be destroyed or rewritten immediately after the call and before
// start()/wait().  Runs at 2 and 4 ranks.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>


namespace {

  struct recvcounts_fixture {
    recvcounts_fixture() {
      mp1::init();
    }

    ~recvcounts_fixture() {
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

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(recvcounts_fixture);

BOOST_AUTO_TEST_SUITE(collective_persistent_recvcounts)

BOOST_AUTO_TEST_CASE(persistent_reduce_scatter_counts_are_owned) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const std::plus<int> plus{};
  const int expected = size * (size + 1) / 2;

  // The counts vector is scoped away right after init; the plan's copy must drive every
  // start, including a rewrite of the original storage.
  std::vector<int> send(static_cast<std::size_t>(size), 0);
  std::vector<int> recv(1, -1);
  mp1::request request;
  {
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    request =
        mp1::reduce_scatter_init(c, plus, std::span<int>{send}, std::span<int>{recv}, counts);
    for (auto& count : counts)
      count = 99;  // the plan's copy is unaffected
  }
  for (int cycle = 0; cycle < 2; ++cycle) {
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = rank + 1 + cycle;
    request.start();
    request.wait();
    BOOST_TEST(recv[0] == expected + cycle * size);
  }
}

BOOST_AUTO_TEST_CASE(persistent_reduce_scatter_block_count_is_owned) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const std::plus<int> plus{};
  const int expected = size * (size + 1) / 2;

  std::vector<int> send(static_cast<std::size_t>(size), 0);
  std::vector<int> recv(1, -1);
  mp1::request request =
      mp1::reduce_scatter_block_init(c, plus, std::span<int>{send}, std::span<int>{recv}, 1);
  for (int cycle = 0; cycle < 2; ++cycle) {
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = rank + 1 + cycle;
    request.start();
    request.wait();
    BOOST_TEST(recv[0] == expected + cycle * size);
  }
}

BOOST_AUTO_TEST_CASE(nonblocking_reduce_scatter_counts_are_owned) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const std::plus<int> plus{};
  const int expected = size * (size + 1) / 2;

  std::vector<int> send(static_cast<std::size_t>(size), rank + 1);
  std::array<int, 1> recv{-1};
  // The temporary counts vector dies at the end of the full-expression; the request payload
  // owns its copy until completion.
  mp1::request request =
      mp1::ireduce_scatter(c, plus, send, std::span<int>{recv},
                           std::vector<mp1::count_t>(static_cast<std::size_t>(size), 1));
  request.wait();
  BOOST_TEST(recv[0] == expected);
}

BOOST_AUTO_TEST_SUITE_END()
