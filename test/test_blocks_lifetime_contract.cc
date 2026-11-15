// `blocks` descriptor lifetime contract.
//
// The dense descriptor is a non-owning aggregate of two `std::span<const count_t>` values.  A
// blocking call borrows it for the call; a nonblocking/persistent initiation copies the wire
// counts/displacements into the plan before it returns, so the caller's arrays may be
// destroyed (or rewritten) immediately after initiation.  The persistent plan re-reads only
// the arrays it owns at MPI_Start, never the caller's.
//
// Cases: blocking borrow, nonblocking and persistent copy (arrays destroyed and rewritten
// after initiation), a stored `blocks` built from temporary vectors, the exact length and
// non-negative/bounds rejection matrix, zero blocks, MPI_PROC_NULL on an intercommunicator,
// and the dense direct path (one native call, no staging, no datatype duplication).
//
// MPI 4.1 basis: §6.5-§6.8 (variable collectives), §6.12/§6.13 (request lifetime), §5.1.8
// (extent), §6.2.3 (intercommunicator root encoding).

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_BLOCKS_HAVE_WRAP 1
#else
#define MP1_BLOCKS_HAVE_WRAP 0
#endif


namespace {

  struct blocks_fixture {
    blocks_fixture() {
      mp1::init();
    }

    ~blocks_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
      } catch (...) {
        // Never throw from a fixture destructor.
      }
    }
  };

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

  [[nodiscard]] std::vector<mp1::count_t> uniform(int peers, mp1::count_t value) {
    return std::vector<mp1::count_t>(static_cast<std::size_t>(peers), value);
  }

  [[nodiscard]] std::vector<mp1::count_t> prefix(const std::vector<mp1::count_t>& counts) {
    std::vector<mp1::count_t> displs(counts.size(), 0);
    mp1::count_t offset = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      displs[i] = offset;
      offset += counts[i];
    }
    return displs;
  }

  void barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(blocks_fixture);

BOOST_AUTO_TEST_SUITE(blocks_lifetime_contract)

// A blocking call borrows the descriptor: the arrays only need to live through the call.
BOOST_AUTO_TEST_CASE(blocking_borrow) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  std::vector<int> send(static_cast<std::size_t>(rank + 1), rank * 10);
  std::vector<int> recv(static_cast<std::size_t>(size * (size + 1) / 2), -1);
  {
    const std::vector<mp1::count_t> counts = uniform(size, 1);
    std::vector<mp1::count_t> counts_varying(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      counts_varying[static_cast<std::size_t>(i)] = i + 1;
    const std::vector<mp1::count_t> displs = prefix(counts_varying);
    (void)counts;
    mp1::gatherv(c, root, send, recv, mp1::blocks{counts_varying, displs});
  }
  if (rank == root) {
    mp1::count_t offset = 0;
    for (int peer = 0; peer < size; ++peer) {
      for (int j = 0; j <= peer; ++j)
        BOOST_TEST(recv[static_cast<std::size_t>(offset + j)] == peer * 10);
      offset += peer + 1;
    }
  }
  barrier();

  // A prvalue `blocks` built from temporary vectors is safe for a blocking call: the spans
  // live to the end of the full-expression and the wrapper must not store them (the m9 hole).
  // The send block is one element because the uniform counts say one per peer.
  const std::array<int, 1> one{{rank * 10}};
  if (rank == root) {
    mp1::gatherv(c, root, one, recv, mp1::blocks{uniform(size, 1), prefix(uniform(size, 1))});
    for (int peer = 0; peer < size; ++peer)
      BOOST_TEST(recv[static_cast<std::size_t>(peer)] == peer * 10);
  } else {
    mp1::gatherv(c, root, one, std::span<int>{}, mp1::blocks{});
  }
  barrier();
}

// Nonblocking copy: the count/displacement vectors are destroyed immediately after initiation;
// completion must use the plan's copies.  A stored `blocks` value is the same object the
// caller keeps, so the wrapper must never store its spans.
BOOST_AUTO_TEST_CASE(nonblocking_copies_descriptor) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  std::array<int, 1> send{{rank + 100}};
  std::vector<int> recv(static_cast<std::size_t>(size), -1);

  mp1::blocks stored;
  {
    std::vector<mp1::count_t> counts = uniform(size, 1);
    std::vector<mp1::count_t> displs = prefix(counts);
    stored = mp1::blocks{.counts = counts, .displacements = displs};
    stored = mp1::blocks{counts, displs};
    // Call while both vectors are alive, then destroy them before completion.
    mp1::request pending = rank == root
                               ? mp1::igatherv(c, root, send, recv, stored)
                               : mp1::igatherv(c, root, send, std::span<int>{}, mp1::blocks{});
    // The stored descriptor is the caller's; destroying the arrays must not affect the plan.
    counts.assign(static_cast<std::size_t>(size), 77);
    displs.assign(static_cast<std::size_t>(size), 77);
    pending.wait();
  }
  if (rank == root) {
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 100);
  }
  barrier();
}

// Persistent copy and restart: *_init copies the descriptor; rewriting the caller's arrays
// after init cannot change any cycle (MPI 4.1 §6.13: the arrays handed to MPI stay fixed).
BOOST_AUTO_TEST_CASE(persistent_copies_and_revalidates) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  std::array<int, 1> send{{1}};
  std::vector<int> recv(static_cast<std::size_t>(size), -1);
  const std::span<int> recv_span{recv};

  std::vector<mp1::count_t> counts = uniform(size, 1);
  std::vector<mp1::count_t> displs = prefix(counts);
  mp1::request persistent =
      rank == root ? mp1::gatherv_init(c, root, send, recv_span, mp1::blocks{counts, displs})
                   : mp1::gatherv_init(c, root, send, std::span<int>{}, mp1::blocks{});
  counts.assign(static_cast<std::size_t>(size), 9);
  displs.assign(static_cast<std::size_t>(size), 9);
  for (int cycle = 0; cycle < 2; ++cycle) {
    send[0] = 1 + cycle * 100;
    persistent.start();
    persistent.wait();
    if (rank == root) {
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 1 + cycle * 100);
    }
  }
  barrier();
}

// Lengths are exact, entries are non-negative, and every block must fit the caller's span.
BOOST_AUTO_TEST_CASE(rejection_matrix) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  std::vector<int> send(static_cast<std::size_t>(size), rank);
  std::vector<int> recv(static_cast<std::size_t>(size), 0);

  // Mismatched counts/displacements lengths.
  if (rank == root) {
    std::vector<mp1::count_t> counts = uniform(size, 1);
    std::vector<mp1::count_t> short_displs = prefix(counts);
    short_displs.pop_back();
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(c, root, std::span<const int>{send.data(), 1}, recv,
                   mp1::blocks{counts, short_displs});
    }));
  }
  barrier();

  // Length must equal the peer count, not merely match the other array.
  if (rank == root) {
    std::vector<mp1::count_t> short_counts(static_cast<std::size_t>(size - 1), 1);
    std::vector<mp1::count_t> short_displs(static_cast<std::size_t>(size - 1), 0);
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(c, root, std::span<const int>{send.data(), 1}, recv,
                   mp1::blocks{short_counts, short_displs});
    }));
  }
  barrier();

  // Negative count and negative displacement.
  if (rank == root) {
    std::vector<mp1::count_t> negative_counts = uniform(size, 1);
    negative_counts[0] = -1;
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(c, root, std::span<const int>{send.data(), 1}, recv,
                   mp1::blocks{negative_counts, prefix(uniform(size, 1))});
    }));
    std::vector<mp1::count_t> negative_displs = prefix(uniform(size, 1));
    negative_displs[0] = -1;
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(c, root, std::span<const int>{send.data(), 1}, recv,
                   mp1::blocks{uniform(size, 1), negative_displs});
    }));
  }
  barrier();

  // Sparse bound: the span must cover max(displacement + count).  The old
  // raw-pointer API could not check this; the dense descriptor now can.
  if (rank == root) {
    std::vector<mp1::count_t> counts = uniform(size, 2);
    std::vector<mp1::count_t> displs = prefix(counts);
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(c, root, std::span<const int>{send.data(), 2}, recv,
                   mp1::blocks{counts, displs});
    }));
  }
  barrier();

  // Zero blocks with a zero-length span are legal; a null pointer with a non-empty block is
  // not reached because the length/bound checks run first.
  {
    const std::vector<mp1::count_t> zero_counts = uniform(size, 0);
    const std::vector<mp1::count_t> zero_displs = prefix(zero_counts);
    mp1::gatherv(c, root, std::span<const int>{}, std::span<int>{},
                 mp1::blocks{zero_counts, zero_displs});
    mp1::allgatherv(c, std::span<const int>{}, std::span<int>{},
                    mp1::blocks{zero_counts, zero_displs});
    mp1::alltoallv(c, std::span<const int>{}, mp1::blocks{zero_counts, zero_displs},
                   std::span<int>{}, mp1::blocks{zero_counts, zero_displs});
  }
  barrier();
}

// MPI_PROC_NULL and MPI_ROOT on an intercommunicator: the ignored group members pass
// `blocks{}`; the descriptor is never validated there.
BOOST_AUTO_TEST_CASE(proc_null_and_zero) {
  const auto world = mp1::comm_world();
  if (world.size() < 2) {
    BOOST_TEST_MESSAGE("intercommunicator blocks test needs at least 2 ranks");
    return;
  }
  const int first = world.size() / 2;
  const int color = world.rank() < first ? 0 : 1;
  auto local = mp1::communicator::split(world, color, world.rank());
  const int remote_leader = color == 0 ? first : 0;
  auto inter = mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{921});
  const int local_rank = local.rank();
  const int remote_size = inter.remote_size();

  // Group 1 sends one int per rank; the group-0 root receives with a dense descriptor.
  {
    std::vector<mp1::count_t> counts = uniform(remote_size, 1);
    std::vector<mp1::count_t> displs = prefix(counts);
    std::vector<int> recv(static_cast<std::size_t>(remote_size), -1);
    const int send_value = 7000 + local_rank;
    if (color == 1) {
      mp1::gatherv(inter, 0, send_value, std::span<int>{}, mp1::blocks{});
    } else if (local_rank == 0) {
      mp1::gatherv(inter, MPI_ROOT, 0, recv, mp1::blocks{counts, displs});
      for (int i = 0; i < remote_size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 7000 + i);
    } else {
      mp1::gatherv(inter, MPI_PROC_NULL, 0, std::span<int>{}, mp1::blocks{});
    }
  }

  // Group 0 root sends one int per remote rank; the ignored root-group members pass blocks{}.
  {
    std::vector<mp1::count_t> counts = uniform(remote_size, 1);
    std::vector<mp1::count_t> displs = prefix(counts);
    std::vector<int> send(static_cast<std::size_t>(remote_size), -1);
    for (int i = 0; i < remote_size; ++i)
      send[static_cast<std::size_t>(i)] = 8000 + i;
    if (color == 0 and local_rank == 0) {
      mp1::scatterv(inter, MPI_ROOT, send, mp1::blocks{counts, displs}, std::span<int>{});
    } else if (color == 0) {
      mp1::scatterv(inter, MPI_PROC_NULL, std::span<const int>{}, mp1::blocks{},
                    std::span<int>{});
    } else {
      int recv = -1;
      mp1::scatterv(inter, 0, std::span<const int>{}, mp1::blocks{}, recv);
      BOOST_TEST(recv == 8000 + local_rank);
    }
  }

  inter.disconnect();
}

// The dense direct path pays no staging and no datatype duplication; the plan owns the wire
// arrays (two allocation-free small vectors up to the inline block count).
BOOST_AUTO_TEST_CASE(dense_direct_has_no_staging) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  std::vector<mp1::count_t> counts = uniform(size, 1);
  std::vector<mp1::count_t> displs = prefix(counts);
  std::vector<int> send(1, rank);
  std::vector<int> recv(static_cast<std::size_t>(size), -1);
#if MP1_BLOCKS_HAVE_WRAP
  if (mpi_wrap::supported()) {
    mpi_wrap::reset_counts();
    mp1::gatherv(c, root, send, recv, mp1::blocks{counts, displs});
    BOOST_TEST(mpi_wrap::count("MPI_Gatherv_c") == 1u);
    BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 0u);
    BOOST_TEST(mpi_wrap::count("MPI_Pack_c") == 0u);
    BOOST_TEST(mpi_wrap::count("MPI_Pack_size_c") == 0u);
  } else
#endif
  {
    mp1::gatherv(c, root, send, recv, mp1::blocks{counts, displs});
  }
  if (rank == root) {
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(i)] == i);
  }
  barrier();
}

BOOST_AUTO_TEST_SUITE_END()
