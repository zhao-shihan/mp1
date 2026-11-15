// Descriptor-unit tests: a typed variable-collective displacement span holds element
// displacements (never bytes), the dense `blocks` descriptor counts native elements, and a
// variable collective interprets both in units of T elements, not bytes or layout blocks.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <type_traits>
#include <vector>


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

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);


BOOST_AUTO_TEST_CASE(blocks_descriptor_units) {
  // The dense descriptor is two borrowed const count spans in (counts, displacements) order.
  static_assert(std::is_same_v<decltype(mp1::blocks::counts), std::span<const mp1::count_t>>);
  static_assert(
      std::is_same_v<decltype(mp1::blocks::displacements), std::span<const mp1::count_t>>);
  static_assert(std::is_same_v<mp1::count_t, std::ptrdiff_t>);

  const std::vector<mp1::count_t> counts{1, 2, 3};
  const std::vector<mp1::count_t> displacements{0, 1, 3};
  const mp1::blocks dense{counts, displacements};
  BOOST_TEST(dense.counts.size() == 3);
  BOOST_TEST(dense.displacements.size() == 3);
  BOOST_TEST(dense.counts[1] == 2);
  BOOST_TEST(dense.displacements[2] == 3);

  // Element counts are signed: negative values are representable, and a negative count or
  // displacement is rejected by the collective before any MPI call.
  const std::vector<mp1::count_t> signed_values{-2, 0, 2};
  BOOST_TEST(signed_values[0] == -2);
  BOOST_TEST(signed_values[2] == 2);
}


BOOST_AUTO_TEST_CASE(displacements_are_element_counts) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int root = size - 1;

  // One element per rank at a displacement of 2 elements: a byte-unit implementation would
  // write at 2 * sizeof(int) bytes (i.e. element 2 * i * sizeof(int)), not element 2 * i.
  {
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> recv_displacements;
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(mp1::layout<int>::contiguous(1));
      recv_displacements.push_back(static_cast<mp1::count_t>(2 * i));
    }
    // Enlarged so that a byte-unit bug would stay in bounds and be detected by the checks.
    std::vector<int> recv(static_cast<std::size_t>(8 * size + 1), -1);
    const int value = 100 + rank;
    mp1::gatherv(comm, root, value, recv, recv_layouts, recv_displacements);
    if (rank == root) {
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(2 * i)] == 100 + i);
        BOOST_TEST(recv[static_cast<std::size_t>(2 * i + 1)] == -1);
      }
    }
  }

  // Two-element blocks at a displacement of 3 elements: the displacement counts T elements,
  // not blocks and not bytes.
  {
    const mp1::layout<int> block = mp1::layout<int>::contiguous(2);
    std::vector<mp1::layout<int>> recv_layouts;
    std::vector<mp1::count_t> recv_displacements;
    for (int i = 0; i < size; ++i) {
      recv_layouts.push_back(block);
      recv_displacements.push_back(static_cast<mp1::count_t>(3 * i));
    }
    std::vector<int> recv(static_cast<std::size_t>(3 * (size - 1) + 2), -1);
    const std::vector<int> send{100 + rank, 200 + rank};
    mp1::gatherv(comm, root, send, recv, recv_layouts, recv_displacements);
    if (rank == root) {
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == 100 + i);
        BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 1)] == 200 + i);
        if (i + 1 < size)
          BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == -1);
      }
    }
  }
}
