// Nonblocking fixed collectives with a non-contiguous `sequence` send must own the staged flat
// vector for the whole request lifetime (MPI 4.1 §3.7.2: the sender must not modify a
// nonblocking send buffer until the operation completes; §6.12 applies the same rule to
// nonblocking collectives).  The wrapper stages a sized non-contiguous forward range into a
// flat vector at initiation; that vector is destroyed with the local plan unless the request
// owns it.  This suite destroys the caller's source and reuses the freed staging storage
// between initiation and `wait()`, for all four staged nonblocking families.  The blocking
// forms keep their plan on the stack for the native call and are unaffected; contiguous
// `std::vector` sends are the zero-copy controls.  Runs at 2 and 4 ranks.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <deque>
#include <list>
#include <span>
#include <vector>


namespace {

  // Large enough that the freed staging allocation is a real heap block the poison vector can
  // reuse (the C-1 reproduction used 200,000 elements).
  constexpr std::size_t sequence_extent = 200000;

  struct sequence_lifetime_fixture {
    sequence_lifetime_fixture() {
      mp1::init();
    }

    ~sequence_lifetime_fixture() {
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

  [[nodiscard]] int sequence_value(int rank, std::size_t index) {
    return rank * 1000000 + static_cast<int>(index);
  }

  [[nodiscard]] int alltoall_value(int source, int destination, std::size_t index) {
    return destination * 1000000 + source * 100000 + static_cast<int>(index);
  }

  // The freed plan staging block has exactly this size; reading the poison after `wait()` keeps
  // the allocation and its stores observable, so a use-after-free reads poison values (or
  // faults) instead of silently passing on an implementation that copied during initiation.
  [[nodiscard]] std::vector<int> make_poison(int rank) {
    std::vector<int> poison(sequence_extent, rank + 7000);
    std::ranges::fill(poison, rank + 7000);
    return poison;
  }

  template<class Range>
  void fill_sequence(Range& range, int rank, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i)
      range.push_back(sequence_value(rank, i));
  }

  template<class Range>
  void fill_alltoall_sequence(Range& range, int rank, int size, std::size_t extent) {
    for (int destination = 0; destination < size; ++destination)
      for (std::size_t i = 0; i < extent; ++i)
        range.push_back(alltoall_value(rank, destination, i));
  }

  [[nodiscard]] bool gathered_values_match(const std::vector<int>& recv, int size,
                                           std::size_t extent) {
    for (int source = 0; source < size; ++source)
      for (std::size_t i = 0; i < extent; ++i)
        if (recv[static_cast<std::size_t>(source) * extent + i] != sequence_value(source, i))
          return false;
    return true;
  }

  [[nodiscard]] bool scattered_values_match(const std::vector<int>& recv, int root, int rank,
                                            std::size_t extent) {
    for (std::size_t i = 0; i < extent; ++i)
      if (recv[i] != sequence_value(root, rank * extent + i))
        return false;
    return true;
  }

  [[nodiscard]] bool alltoall_values_match(const std::vector<int>& recv, int rank, int size,
                                           std::size_t extent) {
    for (int source = 0; source < size; ++source)
      for (std::size_t i = 0; i < extent; ++i)
        if (recv[static_cast<std::size_t>(source) * extent + i] !=
            alltoall_value(source, rank, i))
          return false;
    return true;
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(sequence_lifetime_fixture);

BOOST_AUTO_TEST_SUITE(collective_sequence_lifetime)

// igather / iscatter / iallgather / ialltoall with a std::list or std::deque send: the source
// is destroyed and the freed staging storage is reused before wait().
BOOST_AUTO_TEST_CASE(nonblocking_sequence_send_survives_clobber) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  const std::size_t extent = sequence_extent;

  // igather: every rank's list is staged by its own request; the root's aggregate receive is
  // the real position at the root.
  {
    std::list<int> source;
    fill_sequence(source, rank, extent);
    std::vector<int> recv(static_cast<std::size_t>(size) * extent, -1);
    mp1::request request = mp1::igather(c, root, source, std::span<int>{recv});
    source.clear();
    std::vector<int> poison = make_poison(rank);
    const mp1::status status = request.wait();
    BOOST_TEST(status.error() == MPI_SUCCESS);
    if (rank == root)
      BOOST_TEST(gathered_values_match(recv, size, extent));
    BOOST_TEST(poison[extent / 2] == rank + 7000);
  }

  // iscatter: only the root's send aggregate is significant and staged; every rank's list is
  // destroyed after initiation.
  {
    std::list<int> source;
    if (rank == root)
      fill_sequence(source, rank, static_cast<std::size_t>(size) * extent);
    std::vector<int> recv(extent, -1);
    mp1::request request = mp1::iscatter(c, root, source, std::span<int>{recv});
    source.clear();
    std::vector<int> poison = make_poison(rank);
    const mp1::status status = request.wait();
    BOOST_TEST(status.error() == MPI_SUCCESS);
    BOOST_TEST(scattered_values_match(recv, root, rank, extent));
    BOOST_TEST(poison[extent / 2] == rank + 7000);
  }

  // iallgather: every rank stages its deque send and receives the full aggregate.
  {
    std::deque<int> source;
    fill_sequence(source, rank, extent);
    std::vector<int> recv(static_cast<std::size_t>(size) * extent, -1);
    mp1::request request = mp1::iallgather(c, source, std::span<int>{recv});
    source.clear();
    std::vector<int> poison = make_poison(rank);
    const mp1::status status = request.wait();
    BOOST_TEST(status.error() == MPI_SUCCESS);
    BOOST_TEST(gathered_values_match(recv, size, extent));
    BOOST_TEST(poison[extent / 2] == rank + 7000);
  }

  // ialltoall: every rank stages its deque send aggregate; the receive block from source `s`
  // at destination `d` carries alltoall_value(s, d, i).
  {
    std::deque<int> source;
    fill_alltoall_sequence(source, rank, size, extent);
    std::vector<int> recv(static_cast<std::size_t>(size) * extent, -1);
    mp1::request request = mp1::ialltoall(c, source, std::span<int>{recv});
    source.clear();
    std::vector<int> poison = make_poison(rank);
    const mp1::status status = request.wait();
    BOOST_TEST(status.error() == MPI_SUCCESS);
    BOOST_TEST(alltoall_values_match(recv, rank, size, extent));
    BOOST_TEST(poison[extent / 2] == rank + 7000);
  }
}

// Blocking list/deque sends and a nonblocking contiguous-vector send are the controls: the
// blocking plan lives through the native call, and a contiguous send is passed zero-copy.
BOOST_AUTO_TEST_CASE(blocking_and_contiguous_controls) {
  const mp1::mpi_communicator c = world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = size - 1;
  const std::size_t extent = 4096;

  // Blocking gather with a list send.
  {
    std::list<int> source;
    fill_sequence(source, rank, extent);
    std::vector<int> recv(static_cast<std::size_t>(size) * extent, -1);
    if (rank == root) {
      mp1::gather(c, root, source, recv);
      BOOST_TEST(gathered_values_match(recv, size, extent));
    } else {
      mp1::gather(c, root, source, std::span<int>{});
    }
  }

  // Blocking scatter with a deque send aggregate.
  {
    std::deque<int> source;
    if (rank == root)
      fill_sequence(source, rank, static_cast<std::size_t>(size) * extent);
    std::vector<int> recv(extent, -1);
    mp1::scatter(c, root, source, recv);
    BOOST_TEST(scattered_values_match(recv, root, rank, extent));
  }

  // Blocking allgather with a list send.
  {
    std::list<int> source;
    fill_sequence(source, rank, extent);
    std::vector<int> recv(static_cast<std::size_t>(size) * extent, -1);
    mp1::allgather(c, source, recv);
    BOOST_TEST(gathered_values_match(recv, size, extent));
  }

  // Nonblocking contiguous control: a vector send is borrowed (no staging), so there is no
  // staging lifetime to own and the result is the same.
  {
    std::vector<int> source;
    fill_sequence(source, rank, extent);
    std::vector<int> recv(static_cast<std::size_t>(size) * extent, -1);
    mp1::request request =
        mp1::iallgather(c, std::span<const int>{source}, std::span<int>{recv});
    const mp1::status status = request.wait();
    BOOST_TEST(status.error() == MPI_SUCCESS);
    BOOST_TEST(gathered_values_match(recv, size, extent));
  }
}

BOOST_AUTO_TEST_SUITE_END()
