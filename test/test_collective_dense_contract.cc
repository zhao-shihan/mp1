// Dense variable collectives, typed descriptor borrows, reduce_scatter container forms and
// reduce_local.
//
// The test covers the four dense families (gatherv/scatterv/allgatherv/alltoallv) with the
// `blocks` descriptor in their blocking, nonblocking and persistent spellings, the
// count/displacement/bounds validation matrix (normal, zero total, negative, overflow, wrong
// length, restricted-side overlap, padded native type acceptance), typed descriptor borrows,
// the two-container and in-place reduce_scatter/reduce_scatter_block forms, and the
// reduce_local scalar/pointer/layout/container spellings.
//
// MPI 4.1 basis: §6.5–§6.8 (variable collectives), §6.9.7 (MPI_Reduce_local), §6.12
// (nonblocking/persistent collective lifetime).  Runs at 2 and 4 ranks; every expected
// rejection is a local-validation throw on every rank (or root-only with the peers skipping
// the call entirely), so no rank leaves a case while a peer is inside MPI.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_DENSE_WRAP 1
#else
#define MP1_DENSE_WRAP 0
#endif


namespace {

  // A reflected value whose native datatype is padded: sizeof == 16, type-map size == 12,
  // with an interior hole.  The typed wire accepts it in the dense counts spelling at any
  // count (MPI 4.1 5.1.11 repeats the datatype at its extent stride).
  struct padded_value {
    int first;
    double second;
  };

  template<class Function>
  bool throws_invalid_argument(Function&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class Function>
  bool throws_overflow_error(Function&& function) {
    try {
      function();
    } catch (const std::overflow_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  [[nodiscard]] std::vector<mp1::count_t> prefix_counts(int size, int base_count) {
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      counts[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(base_count + i);
    return counts;
  }

  [[nodiscard]] std::vector<mp1::count_t> prefix_displacements(
      const std::vector<mp1::count_t>& counts) {
    std::vector<mp1::count_t> displs(counts.size(), 0);
    mp1::count_t offset = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      displs[i] = offset;
      offset += counts[i];
    }
    return displs;
  }

  struct dense_fixture {
    dense_fixture() {
      mp1::init();
    }

    ~dense_fixture() {
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture cleanup must not throw
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

}  // namespace

MP1_REFLECTION(padded_value, first, second)


BOOST_TEST_GLOBAL_FIXTURE(dense_fixture);

BOOST_AUTO_TEST_SUITE(collective_dense_contract)

BOOST_AUTO_TEST_CASE(dense_gatherv_scatterv_normal) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const std::vector<mp1::count_t> counts = prefix_counts(size, 1);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);
  const int total = size * (size + 1) / 2;

  std::vector<int> send(static_cast<std::size_t>(rank + 1));
  for (int j = 0; j <= rank; ++j)
    send[static_cast<std::size_t>(j)] = rank * 100 + j;
  std::vector<int> recv(static_cast<std::size_t>(total), -1);

  mp1::gatherv(c, 0, send, recv, mp1::blocks{counts, displs});
  if (rank == 0) {
    for (int peer = 0; peer < size; ++peer) {
      for (int j = 0; j <= peer; ++j)
        BOOST_TEST(recv[static_cast<std::size_t>(displs[peer] + j)] == peer * 100 + j);
    }
  }

  // scatterv: the root sends peer p the p+1 elements it contributed above.
  std::vector<int> root_send(static_cast<std::size_t>(total), -1);
  if (rank == 0) {
    for (int peer = 0; peer < size; ++peer) {
      for (int j = 0; j <= peer; ++j)
        root_send[static_cast<std::size_t>(displs[peer] + j)] = peer * 100 + j;
    }
  }
  std::vector<int> scattered(static_cast<std::size_t>(rank + 1), -1);
  mp1::scatterv(c, 0, root_send, mp1::blocks{counts, displs}, scattered);
  for (int j = 0; j <= rank; ++j)
    BOOST_TEST(scattered[static_cast<std::size_t>(j)] == rank * 100 + j);

  // allgatherv: each rank contributes its own block; the receive side is the same shape.
  std::vector<int> all_recv(static_cast<std::size_t>(total), -1);
  mp1::allgatherv(c, send, all_recv, mp1::blocks{counts, displs});
  for (int peer = 0; peer < size; ++peer) {
    for (int j = 0; j <= peer; ++j)
      BOOST_TEST(all_recv[static_cast<std::size_t>(displs[peer] + j)] == peer * 100 + j);
  }

  // alltoallv: one element to every peer.
  const std::vector<mp1::count_t> one_per_peer(static_cast<std::size_t>(size), 1);
  const std::vector<mp1::count_t> all_to_all_displs = prefix_displacements(one_per_peer);
  std::vector<int> a2a_send(static_cast<std::size_t>(size));
  for (int peer = 0; peer < size; ++peer)
    a2a_send[static_cast<std::size_t>(peer)] = rank * 100 + peer;
  std::vector<int> a2a_recv(static_cast<std::size_t>(size), -1);
  mp1::alltoallv(c, a2a_send, mp1::blocks{one_per_peer, all_to_all_displs}, a2a_recv,
                 mp1::blocks{one_per_peer, all_to_all_displs});
  for (int peer = 0; peer < size; ++peer)
    BOOST_TEST(a2a_recv[static_cast<std::size_t>(peer)] == peer * 100 + rank);
}

BOOST_AUTO_TEST_CASE(dense_zero_total_and_nonblocking) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const std::vector<mp1::count_t> zero_counts(static_cast<std::size_t>(size), 0);
  const std::vector<mp1::count_t> zero_displs(static_cast<std::size_t>(size), 0);

  // Zero-total dense forms accept null buffers and still validate the array lengths.
  const std::span<const int> no_send{};
  const std::span<int> no_recv{};
  const mp1::blocks zero_blocks{zero_counts, zero_displs};
  mp1::gatherv(c, 0, no_send, no_recv, zero_blocks);
  mp1::scatterv(c, 0, no_send, zero_blocks, no_recv);
  mp1::allgatherv(c, no_send, no_recv, zero_blocks);
  mp1::alltoallv(c, no_send, zero_blocks, no_recv, zero_blocks);

  // Nonblocking dense forms use the same conversion but own the converted arrays.
  const std::vector<mp1::count_t> counts = prefix_counts(size, 1);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);
  const int total = size * (size + 1) / 2;
  std::vector<int> send(static_cast<std::size_t>(rank + 1), rank * 100);
  std::vector<int> recv(static_cast<std::size_t>(total), -1);
  mp1::request gather_request = mp1::igatherv(c, 0, send, recv, mp1::blocks{counts, displs});
  gather_request.wait();
  if (rank == 0) {
    for (int peer = 0; peer < size; ++peer)
      BOOST_TEST(recv[static_cast<std::size_t>(displs[peer])] == peer * 100);
  }

  std::vector<int> root_send(static_cast<std::size_t>(total), 0);
  std::vector<int> scattered(static_cast<std::size_t>(rank + 1), -1);
  mp1::request scatter_request =
      mp1::iscatterv(c, 0, root_send, mp1::blocks{counts, displs}, scattered);
  scatter_request.wait();

  std::vector<int> all_recv(static_cast<std::size_t>(total), -1);
  mp1::request allgather_request =
      mp1::iallgatherv(c, send, all_recv, mp1::blocks{counts, displs});
  allgather_request.wait();

  const std::vector<mp1::count_t> one_per_peer(static_cast<std::size_t>(size), 1);
  const std::vector<mp1::count_t> all_to_all_displs = prefix_displacements(one_per_peer);
  std::vector<int> a2a_send(static_cast<std::size_t>(size), rank);
  std::vector<int> a2a_recv(static_cast<std::size_t>(size), -1);
  mp1::request alltoall_request =
      mp1::ialltoallv(c, a2a_send, mp1::blocks{one_per_peer, all_to_all_displs}, a2a_recv,
                      mp1::blocks{one_per_peer, all_to_all_displs});
  alltoall_request.wait();
  for (int peer = 0; peer < size; ++peer)
    BOOST_TEST(a2a_recv[static_cast<std::size_t>(peer)] == peer);

#if MP1_DENSE_WRAP
  if (mpi_wrap::supported()) {
    // One native call per blocking dense family, and no datatype construction or packing.
    std::vector<int> counted_send(static_cast<std::size_t>(size), rank);
    std::vector<int> counted_recv(static_cast<std::size_t>(size * size), -1);
    mpi_wrap::reset_counts();
    mp1::allgatherv(c, std::span<const int>{counted_send.data(), 1}, counted_recv,
                    mp1::blocks{one_per_peer, all_to_all_displs});
    BOOST_TEST(mpi_wrap::count("MPI_Allgatherv_c") == 1u);
    BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 0u);
    BOOST_TEST(mpi_wrap::count("MPI_Pack_c") == 0u);
    BOOST_TEST(mpi_wrap::count("MPI_Pack_size_c") == 0u);

    mpi_wrap::reset_counts();
    mp1::alltoallv(c, counted_send, mp1::blocks{one_per_peer, all_to_all_displs}, counted_recv,
                   mp1::blocks{one_per_peer, all_to_all_displs});
    BOOST_TEST(mpi_wrap::count("MPI_Alltoallv_c") == 1u);
    BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 0u);
  }
#endif
}

BOOST_AUTO_TEST_CASE(dense_persistent_and_borrow_forms) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const std::vector<mp1::count_t> counts = prefix_counts(size, 1);
  const std::vector<mp1::count_t> displs = prefix_displacements(counts);
  const int total = size * (size + 1) / 2;

  std::vector<int> send(static_cast<std::size_t>(rank + 1), rank * 100);
  std::vector<int> recv(static_cast<std::size_t>(total), -1);
  const std::span<const int> send_span{send};
  const std::span<int> recv_span{recv};
  mp1::request persistent =
      mp1::gatherv_init(c, 0, send_span, recv_span, mp1::blocks{counts, displs});
  persistent.start();
  persistent.wait();
  if (rank == 0)
    BOOST_TEST(recv[0] == 0);

  const std::vector<mp1::count_t> one_per_peer(static_cast<std::size_t>(size), 1);
  const std::vector<mp1::count_t> all_to_all_displs = prefix_displacements(one_per_peer);
  std::vector<int> a2a_send(static_cast<std::size_t>(size), rank);
  std::vector<int> a2a_recv(static_cast<std::size_t>(size), -1);
  const std::span<const int> a2a_send_span{a2a_send};
  const std::span<int> a2a_recv_span{a2a_recv};
  mp1::request persistent_a2a =
      mp1::alltoallv_init(c, a2a_send_span, mp1::blocks{one_per_peer, all_to_all_displs},
                          a2a_recv_span, mp1::blocks{one_per_peer, all_to_all_displs});
  persistent_a2a.start();
  persistent_a2a.wait();
  for (int peer = 0; peer < size; ++peer)
    BOOST_TEST(a2a_recv[static_cast<std::size_t>(peer)] == peer);

  // Blocking borrows: make_layout_span selects the borrow sibling and produces the same data as
  // the owning const& form.  An lvalue owner still binds the const& overload.
  std::vector<mp1::layout<int>> layout_blocks;
  std::vector<mp1::count_t> layout_displs;
  mp1::count_t offset = 0;
  for (int peer = 0; peer < size; ++peer) {
    layout_blocks.push_back(mp1::layout<int>::contiguous(peer + 1));
    layout_displs.push_back(offset);
    offset += peer + 1;
  }
  std::vector<int> borrow_send(static_cast<std::size_t>(rank + 1), rank * 100);
  std::vector<int> borrow_recv(static_cast<std::size_t>(total), -1);
  mp1::layout<int> single_layout = mp1::layout<int>::contiguous(rank + 1);
  mp1::gatherv(c, 0, mp1::region{borrow_send.data(), single_layout}, borrow_recv, layout_blocks,
               layout_displs);
  if (rank == 0)
    BOOST_TEST(borrow_recv[0] == 0);

  // The single-block borrow form is used on a gatherv non-root send side.  Gatherv is one
  // collective for the whole communicator (MPI 4.1 6.5), so the root must issue the matching
  // root form in the same step: a root that skipped the call would leave every peer inside
  // MPI and shift every later collective.
  if (rank != 0) {
    mp1::gatherv(c, 0, std::span<const int>{borrow_send.data(), borrow_send.size()},
                 std::span<int>{}, mp1::blocks{});
  } else {
    mp1::gatherv(c, 0, mp1::region{borrow_send.data(), single_layout}, borrow_recv,
                 layout_blocks, layout_displs);
  }

  // alltoallv borrows both block arrays.
  std::vector<mp1::layout<int>> a2a_blocks(static_cast<std::size_t>(size),
                                           mp1::layout<int>::contiguous(1));
  std::vector<mp1::count_t> a2a_displs(static_cast<std::size_t>(size), 0);
  for (int peer = 0; peer < size; ++peer)
    a2a_displs[static_cast<std::size_t>(peer)] = peer;
  std::vector<int> borrow_a2a_send(static_cast<std::size_t>(size), rank * 100);
  std::vector<int> borrow_a2a_recv(static_cast<std::size_t>(size), -1);
  mp1::alltoallv(c, borrow_a2a_send, a2a_blocks, a2a_displs, borrow_a2a_recv, a2a_blocks,
                 a2a_displs);
  for (int peer = 0; peer < size; ++peer)
    BOOST_TEST(borrow_a2a_recv[static_cast<std::size_t>(peer)] == peer * 100);
}

BOOST_AUTO_TEST_CASE(dense_rejections) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  std::vector<int> send(static_cast<std::size_t>(size), 0);
  std::vector<int> recv(static_cast<std::size_t>(size * size), 0);

  const std::vector<mp1::count_t> valid_counts(static_cast<std::size_t>(size), 1);
  const std::vector<mp1::count_t> valid_displs = prefix_displacements(valid_counts);

  // Wrong array lengths are rejected before any MPI call (all ranks validate their receive
  // side, so the exception is raised everywhere).
  std::vector<mp1::count_t> short_counts(static_cast<std::size_t>(size - 1), 1);
  const std::span<const int> send_one{send.data(), 1};
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::allgatherv(c, send_one, recv, mp1::blocks{short_counts, valid_displs}); }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::alltoallv(c, send, mp1::blocks{valid_counts, valid_displs}, recv,
                   mp1::blocks{valid_counts, short_counts});
  }));

  // Negative counts and displacements.
  std::vector<mp1::count_t> negative_counts(static_cast<std::size_t>(size), 1);
  negative_counts[0] = -1;
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::allgatherv(c, send_one, recv, mp1::blocks{negative_counts, valid_displs}); }));
  std::vector<mp1::count_t> negative_displs(static_cast<std::size_t>(size), 0);
  negative_displs[0] = -1;
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::allgatherv(c, send_one, recv, mp1::blocks{valid_counts, negative_displs}); }));

  // There is deliberately no `count * sizeof(T)` overflow case any more: a dense count is a
  // wire element count (MPI 4.1 5.1.11), never a byte count, so a count_t that fits MPI_Count
  // is passed unchanged.  The conversion-width invariant is a static_assert in xfer_side.hpp.
  // The staged path still converts a displacement to a caller byte offset (d * extent), and
  // that product is checked before any MPI call.
  {
    const mp1::layout<int> holey = mp1::layout<int>::vector(2, 1, 2);
    std::vector<mp1::layout<int>> holey_blocks(static_cast<std::size_t>(size), holey);
    const std::vector<mp1::count_t> huge_displs(static_cast<std::size_t>(size),
                                                std::numeric_limits<mp1::count_t>::max() / 2);
    std::array<int, 3> holey_send{{1, 2, 3}};
    std::vector<int> holey_recv(static_cast<std::size_t>(4 * size), 0);
    // The receive plan is built (and the d * extent product checked) before the send block is
    // packed, so a rank that would overflow never touches an under-sized send buffer.
    BOOST_TEST(throws_overflow_error([&] {
      mp1::allgatherv(c, mp1::region{holey_send.data(), holey}, holey_recv, holey_blocks,
                      huge_displs);
    }));
  }

  // The dense path passes d_i through as an extent unit and MPI forms d_i * extent itself
  // (MPI 4.1 6.5/6.6/6.8), but the library still validates that implied byte address locally:
  // an unrepresentable displacement is a std::overflow_error, not a wild pointer handed to MPI.
  {
    const std::vector<mp1::count_t> huge_dense_displs(
        static_cast<std::size_t>(size), std::numeric_limits<mp1::count_t>::max() / 2);
    BOOST_TEST(throws_overflow_error([&] {
      mp1::allgatherv(c, send_one, recv, mp1::blocks{valid_counts, huge_dense_displs});
    }));
  }

  // Restricted-side overlap on the allgatherv receive map.
  std::vector<mp1::count_t> overlap_counts(static_cast<std::size_t>(size), 2);
  std::vector<mp1::count_t> overlap_displs(static_cast<std::size_t>(size), 0);
  for (int i = 0; i < size; ++i)
    overlap_displs[static_cast<std::size_t>(i)] = i;  // [0,8), [4,12), ...
  std::vector<int> overlap_recv(static_cast<std::size_t>(2 * size + 2), 0);
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::allgatherv(c, std::span<const int>{send.data(), 2}, overlap_recv,
                    mp1::blocks{overlap_counts, overlap_displs});
  }));

  // A padded native datatype is ACCEPTED in the dense spelling at any count: the typed wire
  // repeats native_datatype<T>() at its extent stride (MPI 4.1 5.1.11), which is exactly
  // sizeof(T), so the count is the exact image of the caller's T array.  `padded_value` has
  // an interior hole (size 12 < extent 16), the shape the old `packed`/size==extent gate
  // rejected; both members of every element must survive.
  {
    constexpr int elements_per_rank = 2;
    std::vector<mp1::count_t> padded_counts(static_cast<std::size_t>(size), elements_per_rank);
    const std::vector<mp1::count_t> padded_displs = prefix_displacements(padded_counts);
    std::vector<padded_value> padded_send(static_cast<std::size_t>(elements_per_rank));
    for (int j = 0; j < elements_per_rank; ++j) {
      padded_send[static_cast<std::size_t>(j)] =
          padded_value{rank * 100 + j, static_cast<double>(rank * 100 + j + 50)};
    }
    std::vector<padded_value> padded_recv(static_cast<std::size_t>(elements_per_rank * size),
                                          padded_value{-1, -1.0});
    mp1::allgatherv(c, padded_send, padded_recv, mp1::blocks{padded_counts, padded_displs});
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < elements_per_rank; ++j) {
        const padded_value& value =
            padded_recv[static_cast<std::size_t>(elements_per_rank * i + j)];
        BOOST_TEST(value.first == i * 100 + j);
        BOOST_TEST(value.second == i * 100 + j + 50);
      }
    }
  }

  // Root-only gatherv overlap: peers do not enter the call, then synchronize with a barrier.
  if (c.rank() == 0) {
    std::vector<mp1::count_t> root_counts(static_cast<std::size_t>(size), 2);
    std::vector<mp1::count_t> root_displs(static_cast<std::size_t>(size), 0);
    for (int i = 0; i < size; ++i)
      root_displs[static_cast<std::size_t>(i)] = i;
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(c, 0, std::span<const int>{send.data(), 2}, overlap_recv,
                   mp1::blocks{root_counts, root_displs});
    }));
  }
  mp1::barrier(c);
}

BOOST_AUTO_TEST_CASE(reduce_scatter_container_forms) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
  std::plus<int> plus;

  // Out-of-place, two containers with different extents.
  std::vector<int> send(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    send[static_cast<std::size_t>(i)] = rank * 100 + i;
  if (size == 2) {
    std::array<int, 2> array_send{send[0], send[1]};
    std::array<int, 1> array_recv{-1};
    mp1::reduce_scatter(c, plus, array_send, array_recv, counts);
    const int expected = 100 * (size - 1) * size / 2 + rank * size;
    BOOST_TEST(array_recv[0] == expected);
  } else {
    std::array<int, 4> array_send{send[0], send[1], send[2], send[3]};
    std::array<int, 1> array_recv{-1};
    mp1::reduce_scatter(c, plus, array_send, array_recv, counts);
    const int expected = 100 * (size - 1) * size / 2 + rank * size;
    BOOST_TEST(array_recv[0] == expected);
  }

  // In-place single container: every rank's vector holds its whole contribution.
  std::vector<int> in_place(static_cast<std::size_t>(size), rank + 1);
  mp1::reduce_scatter(c, plus, mp1::in_place, in_place, counts);
  // The local result block has recvcounts[rank] elements and starts at recvbuf[0]; with the
  // all-ones counts it is a single element.  Every rank contributes its whole vector, so the
  // result is sum(rank + 1).
  BOOST_TEST(in_place.size() == static_cast<std::size_t>(size));
  BOOST_TEST(in_place[0] == size * (size + 1) / 2);

  // Nonblocking container form.
  std::vector<int> async_recv(1, -1);
  mp1::request request = mp1::ireduce_scatter(c, plus, send, async_recv, counts);
  request.wait();
  const int expected_sum = 100 * size * (size - 1) / 2;
  BOOST_TEST(async_recv[0] == expected_sum + rank * size);

  // reduce_scatter_block container forms with a two-element block.
  const mp1::count_t block = 2;
  std::vector<int> block_send(static_cast<std::size_t>(block * size));
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < block; ++j)
      block_send[static_cast<std::size_t>(i * block + j)] = rank * 100 + i * block + j;
  std::vector<int> block_recv(static_cast<std::size_t>(block), -1);
  mp1::reduce_scatter_block(c, plus, block_send, block_recv, block);
  for (int j = 0; j < block; ++j) {
    int expected = 0;
    for (int i = 0; i < size; ++i)
      expected += i * 100 + rank * block + j;
    BOOST_TEST(block_recv[static_cast<std::size_t>(j)] == expected);
  }

  // In-place block form.
  std::vector<int> block_in_place(static_cast<std::size_t>(block * size), rank + 1);
  mp1::reduce_scatter_block(c, plus, mp1::in_place, block_in_place, block);
  BOOST_TEST(block_in_place[0] == size * (size + 1) / 2);

  // Exact-size validation and the zero-total form.
  std::vector<int> wrong_recv(static_cast<std::size_t>(size + 1), 0);
  BOOST_TEST(
      throws_invalid_argument([&] { mp1::reduce_scatter(c, plus, send, wrong_recv, counts); }));
  std::vector<int> wrong_block_recv(static_cast<std::size_t>(block + 1), 0);
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::reduce_scatter_block(c, plus, block_send, wrong_block_recv, block); }));

  const std::vector<mp1::count_t> zero_counts(static_cast<std::size_t>(size), 0);
  std::vector<int> zero_buffer;
  mp1::reduce_scatter(c, plus, zero_buffer, zero_buffer, zero_counts);
  mp1::reduce_scatter(c, plus, mp1::in_place, zero_buffer, zero_counts);
  mp1::reduce_scatter_block(c, plus, zero_buffer, zero_buffer, 0);
  mp1::reduce_scatter_block(c, plus, mp1::in_place, zero_buffer, 0);
}

BOOST_AUTO_TEST_CASE(reduce_local_forms) {
  std::plus<int> plus;

  int scalar_send = 3;
  int scalar_recv = 4;
  mp1::reduce_local(plus, scalar_send, scalar_recv);
  BOOST_TEST(scalar_recv == 7);

  int send[4] = {1, 2, 3, 4};
  int recv[4] = {10, 20, 30, 40};
  mp1::reduce_local(plus, std::span<int>{send}, std::span<int>{recv});
  BOOST_TEST(recv[0] == 11);
  BOOST_TEST(recv[1] == 22);
  BOOST_TEST(recv[2] == 33);
  BOOST_TEST(recv[3] == 44);

#if MP1_DENSE_WRAP
  if (mpi_wrap::supported()) {
    int counted_send[4] = {1, 1, 1, 1};
    int counted_recv[4] = {0, 0, 0, 0};
    mpi_wrap::reset_counts();
    mp1::reduce_local(plus, std::span<int>{counted_send}, std::span<int>{counted_recv});
    BOOST_TEST(mpi_wrap::count("MPI_Reduce_local_c") == 1u);
    BOOST_TEST(mpi_wrap::count("MPI_Pack_c") == 0u);
    BOOST_TEST(mpi_wrap::count("MPI_Pack_size_c") == 0u);
  }
#endif

  // Holey layout: one layout for both buffers; the untouched holes keep their values.
  mp1::layout<int> holey = mp1::layout<int>::vector(2, 1, 2);
  int holey_send[4] = {1, -1, 3, -1};
  int holey_recv[4] = {10, 20, 30, 40};
  mp1::reduce_local(plus, mp1::region<int>{holey_send, holey},
                    mp1::region<int>{holey_recv, holey});
  BOOST_TEST(holey_recv[0] == 11);
  BOOST_TEST(holey_recv[1] == 20);
  BOOST_TEST(holey_recv[2] == 33);
  BOOST_TEST(holey_recv[3] == 40);

  // Zero count with null buffers is legal and still issues one native call.
  mp1::reduce_local(plus, std::span<const int>{}, std::span<int>{});

  // Container form requires equal sizes.
  std::vector<int> container_send{1, 2, 3};
  std::vector<int> container_recv{10, 20, 30};
  mp1::reduce_local(plus, container_send, container_recv);
  BOOST_TEST(container_recv[0] == 11);
  BOOST_TEST(container_recv[1] == 22);
  BOOST_TEST(container_recv[2] == 33);

  std::vector<int> short_recv{0, 0};
  BOOST_TEST(
      throws_invalid_argument([&] { mp1::reduce_local(plus, container_send, short_recv); }));
}

BOOST_AUTO_TEST_SUITE_END()
