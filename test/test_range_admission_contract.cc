// Range admission and exclusions, p2p subset.
//
// Admitted: contiguous/sized/element ranges (vector, array, string, span, C array of
// non-character T, string_view on send), sized forward non-contiguous sequences on send only
// (staged list/deque), region, and std::ranges::subrange over contiguous iterators.
// Excluded: vector<bool>, built-in character arrays, forward_list, sequence containers on
// receive, rvalue containers on receive, const buffers on receive, and every associative
// container (keyed, not flat).  The negative cases are compile-fail probes (test/compile_fail/
// p2p_*); this suite pins the concept lattice and the positive calls.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <forward_list>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>


namespace {

  [[nodiscard]] int world_rank() {
    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }

  [[nodiscard]] int world_size() {
    int size{1};
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

}  // namespace


struct mp1_range_admission_fixture {
  mp1_range_admission_fixture() {
    mp1::init();
  }

  ~mp1_range_admission_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Already finalized through another path; a destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_range_admission_fixture);

BOOST_AUTO_TEST_SUITE(range_admission_contract)

BOOST_AUTO_TEST_CASE(concept_lattice) {
  BOOST_TEST(true);
  // Admitted contiguous buffers.
  static_assert(mp1::element_buffer<std::vector<int>>);
  static_assert(mp1::element_buffer<std::array<int, 4>>);
  static_assert(mp1::element_buffer<std::string>);
  static_assert(mp1::element_buffer<std::span<int>>);
  static_assert(mp1::element_buffer<std::span<const int>>);
  static_assert(mp1::element_buffer<int[4]>);
  static_assert(mp1::element_buffer<std::array<char, 4>>);
  static_assert(mp1::element_buffer<std::string_view>);
  // Admitted sequences are send-only and sized.
  static_assert(mp1::element_sequence<std::list<int>>);
  static_assert(!mp1::element_buffer<std::list<int>>);
  static_assert(mp1::send_source<std::list<int>&>);
  static_assert(!mp1::recv_sink<std::list<int>&>);
  // Excluded as associative.
  static_assert(!mp1::element_buffer<std::vector<bool>>);
  static_assert(!mp1::send_source<std::vector<bool>&>);
  static_assert(!mp1::recv_sink<std::vector<bool>&>);
  static_assert(!mp1::element_buffer<std::forward_list<int>>);
  static_assert(!mp1::element_sequence<std::forward_list<int>>);
  using char_array = char[4];
  static_assert(!mp1::element_buffer<char_array>);
  static_assert(!mp1::send_source<char_array&>);
  static_assert(!mp1::recv_sink<char_array&>);
  static_assert(!mp1::element_buffer<std::map<int, int>>);
  static_assert(!mp1::element_buffer<std::set<int>>);
  static_assert(!mp1::element_buffer<std::unordered_map<int, int>>);
  static_assert(!mp1::element_buffer<std::unordered_set<int>>);
  static_assert(!mp1::send_source<std::map<int, int>&>);
  static_assert(!mp1::recv_sink<std::map<int, int>&>);
  static_assert(!mp1::send_source<std::set<int>&>);
  static_assert(!mp1::recv_sink<std::set<int>&>);
  static_assert(!mp1::send_source<std::unordered_map<int, int>&>);
  static_assert(!mp1::recv_sink<std::unordered_map<int, int>&>);
  static_assert(!mp1::send_source<std::unordered_set<int>&>);
  static_assert(!mp1::recv_sink<std::unordered_set<int>&>);
  static_assert(!mp1::recv_sink<std::list<int>&>);
  // Receive targets are writable and borrowed.
  static_assert(!mp1::recv_sink<std::span<const int>>);
  static_assert(!mp1::recv_sink<std::string_view>);
  static_assert(!mp1::recv_sink<std::vector<int>>,
                "an rvalue container is not a receive target");
  static_assert(mp1::recv_sink<std::vector<int>&>);
  static_assert(mp1::recv_sink<std::array<int, 4>&>);
  static_assert(mp1::send_source<const std::vector<int>&>);
}

BOOST_AUTO_TEST_CASE(vector_of_pairs_round_trip) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  // A vector of std::pair is the documented materialization of a map: an explicit, flat order.
  const std::vector<std::pair<int, int>> sent{{1, 10}, {2, 20}, {3, 30}};
  if (rank == 0) {
    mp1::send(world, sent, 1, mp1::tag{7300});
  } else if (rank == 1) {
    std::vector<std::pair<int, int>> received;
    (void)mp1::recv(world, received, 0, mp1::tag{7300});
    BOOST_TEST(received == sent);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(character_buffers) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  // std::array<char,N> and std::span<char> are the receive spellings; char[N] is excluded.
  const std::array<char, 5> sent{{'h', 'e', 'l', 'l', 'o'}};
  if (rank == 0) {
    mp1::send(world, sent, 1, mp1::tag{7310});
  } else if (rank == 1) {
    std::array<char, 5> received{};
    (void)mp1::recv(world, received, 0, mp1::tag{7310});
    BOOST_TEST(received == sent);
  }
  world_barrier();

  std::string text{"world"};
  if (rank == 0) {
    mp1::send(world, text, 1, mp1::tag{7311});
  } else if (rank == 1) {
    std::string received;
    (void)mp1::recv(world, received, 0, mp1::tag{7311});
    BOOST_TEST(received == text);
  }
  world_barrier();

  // std::array<bool,N> over real bool objects stays accepted.
  const std::array<bool, 3> flags{{true, false, true}};
  if (rank == 0) {
    mp1::send(world, flags, 1, mp1::tag{7312});
  } else if (rank == 1) {
    std::array<bool, 3> received{};
    (void)mp1::recv(world, received, 0, mp1::tag{7312});
    BOOST_TEST(received == flags);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(non_contiguous_send_is_staged) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  // A sized forward sequence with lvalue references is send-only; the wire form is one
  // contiguous staging buffer, and the receive target is a vector.
  const std::list<int> sent{5, 6, 7};
  if (rank == 0) {
    mp1::send(world, sent, 1, mp1::tag{7320});
  } else if (rank == 1) {
    std::vector<int> received;
    (void)mp1::recv(world, received, 0, mp1::tag{7320});
    BOOST_TEST(received == (std::vector<int>{5, 6, 7}));
  }
  world_barrier();

  // The nonblocking form stages into the request payload.
  const std::list<int> async_sent{8, 9};
  if (rank == 0) {
    mp1::request request = mp1::isend(world, async_sent, 1, mp1::tag{7321});
    (void)request.wait();
  } else if (rank == 1) {
    std::vector<int> received;
    (void)mp1::recv(world, received, 0, mp1::tag{7321});
    BOOST_TEST(received == (std::vector<int>{8, 9}));
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(c_array_range_round_trip) {
  BOOST_TEST(true);
  if (world_size() < 2)
    return;
  const mp1::mpi_communicator world = mp1::comm_world();
  const int rank = world_rank();

  int sent[3] = {11, 12, 13};
  if (rank == 0) {
    mp1::send(world, sent, 1, mp1::tag{7330});
  } else if (rank == 1) {
    int received[3] = {0, 0, 0};
    (void)mp1::recv(world, received, 0, mp1::tag{7330});
    BOOST_TEST(received[2] == 13);
  }
  world_barrier();
}

BOOST_AUTO_TEST_SUITE_END()
