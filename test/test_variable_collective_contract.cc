// Variable-collective contract: descriptor lifetime (copy-on-need), displacement
// validation on the restricted side, intra self-block checks, intercommunicator asymmetry,
// collective status, std::byte layouts, and the move overloads.
//
// MPI 4.1 citations: 5.1.8 true extent, 5.1.11 type-map use/overlap, 6.5 Gatherv,
// 6.6 Scatterv, 6.7 Allgather(v) intercommunicator advice, 6.8 Alltoall(v) intercommunicator
// advice, 6.13 persistent requests, 3.7.5 wait/test status.

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#if defined(MP1_TEST_WRAP)
#include "support/mpi_wrap.hpp"
#define MP1_TEST_HAVE_MPI_WRAP 1
#else
#define MP1_TEST_HAVE_MPI_WRAP 0
#endif


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

  [[nodiscard]] mp1::layout<int> holey_two() {
    return mp1::layout<int>::vector(2, 1, 2);
  }

  [[nodiscard]] mp1::layout<int> duplicate_hole() {
    mp1::hindexed_parameter parameter;
    parameter.add(1, 0);
    parameter.add(1, 0);
    parameter.add(1, 8);
    return mp1::layout<int>::hindexed(parameter);
  }

  [[nodiscard]] std::vector<mp1::count_t> cumulative(int size, mp1::count_t stride) {
    std::vector<mp1::count_t> values(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      values[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * stride;
    return values;
  }

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);

// ---------------------------------------------------------------------------------------
// Receive layouts with a proven internal overlap: MPI 4.1 5.1.11 makes a datatype whose
// receive buffer is written more than once erroneous, so the wrapper rejects it locally
// before entering the collective.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(duplicate_hole_receive_layouts_are_rejected) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  const mp1::layout<int> send = mp1::layout<int>::contiguous(2);
  const std::array<int, 2> send_data{{rank * 10, rank * 10 + 1}};
  const mp1::layout<int> bad = duplicate_hole();
  std::vector<mp1::layout<int>> bad_layouts(static_cast<std::size_t>(size), bad);
  const std::vector<mp1::count_t> displs = cumulative(size, 3);

  // Allgatherv and alltoallv: every rank prepares the same erroneous receive side, so every
  // rank throws locally and no process enters MPI.
  std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::allgatherv(c, mp1::region{send_data.data(), send}, recv, bad_layouts, displs);
  }));
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::alltoallv(c, send_data, bad_layouts, displs, recv, bad_layouts, displs); }));

  // Gatherv: only the root's receive side is significant.  The root throws before MPI; the
  // other ranks do not call the collective at all (a barrier proves the job stays healthy).
  if (rank == root) {
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(c, root, mp1::region{send_data.data(), send}, recv, bad_layouts, displs);
    }));
  }
  world_barrier();

  // Scatterv: the root's send side is the read-restricted side (MPI 4.1 6.6).
  std::vector<mp1::layout<int>> bad_send_layouts(static_cast<std::size_t>(size), bad);
  if (rank == root) {
    std::vector<int> scatter_send(static_cast<std::size_t>(3 * size), rank);
    std::array<int, 2> local{-1, -1};
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::scatterv(c, root, scatter_send, bad_send_layouts, displs, local); }));
  }
  world_barrier();
}

// ---------------------------------------------------------------------------------------
// Copy-on-need lifetime: a staged nonblocking receive must survive destruction or reuse of
// the caller's layouts.  The request payload owns the descriptors completion reads.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(staged_nonblocking_receive_owns_layouts) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const mp1::layout<int> send = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> holey = holey_two();
  const std::vector<mp1::count_t> displs = cumulative(size, 3);
  const std::array<int, 3> send_data{{rank * 10, -1, rank * 10 + 1}};

  // Destroy the named layouts before completion.
  {
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    mp1::request r;
    {
      std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), holey);
      r = mp1::iallgatherv(c, mp1::region{send_data.data(), holey}, recv, layouts, displs);
    }  // layouts destroyed: the payload owns its copies
    r.wait();
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i * 10);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == i * 10 + 1);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 1)] == -1);
    }
  }
  world_barrier();

  // Reuse the same vector object (same element storage) before completion.
  {
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), holey);
    mp1::request r =
        mp1::iallgatherv(c, mp1::region{send_data.data(), holey}, recv, layouts, displs);
    layouts.assign(static_cast<std::size_t>(size), mp1::layout<int>::contiguous(1));
    r.wait();
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i * 10);
  }
  world_barrier();

  // The typed descriptors are copied at initiation: destroying the caller's array after the
  // call is safe (MPI 4.1 6.12).
  {
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), holey);
    mp1::request r =
        mp1::iallgatherv(c, mp1::region{send_data.data(), holey}, recv, layouts, displs);
    r.wait();
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i * 10);
  }
  world_barrier();

  // A single-layout staged receive (scatterv) follows the same rule: the layout object is
  // destroyed before completion and the payload still has its copy.
  if (size > 1) {
    const int scatter_root = 0;
    std::vector<int> send(static_cast<std::size_t>(3 * size), -1);
    for (int i = 0; i < size; ++i) {
      send[static_cast<std::size_t>(3 * i)] = 500 + i;
      send[static_cast<std::size_t>(3 * i + 2)] = 600 + i;
    }
    std::vector<mp1::layout<int>> send_layouts(static_cast<std::size_t>(size), holey);
    std::vector<int> recv(3, -1);
    mp1::request r;
    {
      std::optional<mp1::layout<int>> recv_layout;
      recv_layout.emplace(holey_two());
      r = mp1::iscatterv(c, scatter_root, send, send_layouts, displs,
                         mp1::region{recv.data(), *recv_layout});
    }  // recv_layout destroyed before completion; the payload owns its copy
    r.wait();
    BOOST_TEST(recv[0] == 500 + rank);
    BOOST_TEST(recv[1] == -1);
    BOOST_TEST(recv[2] == 600 + rank);
    world_barrier();
  }
}

// ---------------------------------------------------------------------------------------
// Displacements: the invented zero default is only accepted when the resulting restricted
// side really is disjoint.  Explicit arrays are authoritative; self-placing layouts remain a
// valid way to describe the placement with zero displacements.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(omitted_displacement_aliasing_is_rejected) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;

  std::vector<mp1::layout<int>> contiguous_layouts;
  for (int i = 0; i < size; ++i)
    contiguous_layouts.push_back(mp1::layout<int>::contiguous(2));
  std::vector<int> recv(static_cast<std::size_t>(2 * size), -1);
  const std::array<int, 2> send{{rank, rank + 100}};

  if (rank == root) {
    // Omitted displacements, contiguous blocks: every block would start at byte 0.
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::gatherv(c, root, send, recv, contiguous_layouts); }));
    // An explicit empty array is the same request, not an opt-out.
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::gatherv(c, root, send, recv, contiguous_layouts, std::span<const mp1::count_t>{});
    }));
    // Explicit zeros are authoritative and still overlap.
    const std::vector<mp1::count_t> zeros(static_cast<std::size_t>(size), 0);
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::gatherv(c, root, send, recv, contiguous_layouts, zeros); }));
  } else {
    // The root throws before entering MPI; the other ranks never call.
  }
  world_barrier();

  // Cumulative displacements are accepted and place each block exactly.
  {
    const std::vector<mp1::count_t> displs = cumulative(size, 2);
    if (rank == root) {
      mp1::gatherv(c, root, send, recv, contiguous_layouts, displs);
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(2 * i)] == i);
        BOOST_TEST(recv[static_cast<std::size_t>(2 * i + 1)] == i + 100);
      }
    } else {
      mp1::gatherv(c, root, send, std::span<int>{}, mp1::blocks{});
    }
    world_barrier();
  }

  // Self-placing layouts: the layout's own true lower bound places the block, so omitted
  // (all-zero) displacements are legal because the computed spans are disjoint.
  {
    std::vector<mp1::layout<int>> self_placed;
    for (int i = 0; i < size; ++i) {
      mp1::indexed_parameter parameter;
      parameter.add(2, static_cast<mp1::count_t>(2 * i));
      self_placed.push_back(mp1::layout<int>::indexed(parameter));
    }
    std::vector<int> placed(static_cast<std::size_t>(2 * size), -1);
    if (rank == root) {
      mp1::gatherv(c, root, send, placed, self_placed);
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(placed[static_cast<std::size_t>(2 * i)] == i);
        BOOST_TEST(placed[static_cast<std::size_t>(2 * i + 1)] == i + 100);
      }
    } else {
      mp1::gatherv(c, root, send, std::span<int>{}, mp1::blocks{});
    }
    world_barrier();
  }

  // Scatterv's root send is read-restricted: omitted displacements with contiguous blocks
  // alias every read and are rejected at the root.
  {
    std::vector<mp1::layout<int>> send_layouts(static_cast<std::size_t>(size),
                                               mp1::layout<int>::contiguous(1));
    if (rank == root) {
      std::vector<int> send_data(static_cast<std::size_t>(size), 7);
      std::array<int, 1> local{-1};
      BOOST_TEST(throws_invalid_argument([&] {
        mp1::scatterv(c, root, send_data, send_layouts, std::span<const mp1::count_t>{}, local);
      }));
    }
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(explicit_displacements_are_copied) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  const mp1::layout<int> block = mp1::layout<int>::contiguous(1);

  std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(size), block);
  std::vector<int> recv(static_cast<std::size_t>(size), -1);
  // A nonblocking collective keeps the send buffer address until completion, so the scalar
  // send is an lvalue; prvalue scalar sends are rejected at compile time.
  const int send_value = rank + 100;
  if (rank == root) {
    // A temporary displacement array is copied at call time and must be accepted.
    std::vector<mp1::count_t> values(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      values[static_cast<std::size_t>(i)] = i;
    mp1::request r = mp1::igatherv(c, root, send_value, recv, recv_layouts, values);
    r.wait();
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 100);
  } else {
    mp1::request r = mp1::igatherv(c, root, send_value, std::span<int>{}, mp1::blocks{});
    r.wait();
  }
  world_barrier();

  // A displacement temporary whose lifetime ends before completion is safe: the values are
  // copied into the plan at call time.
  {
    std::vector<int> local_recv(static_cast<std::size_t>(size), -1);
    const int second_send_value = rank + 200;
    mp1::request r;
    if (rank == root) {
      std::vector<mp1::count_t> values(static_cast<std::size_t>(size));
      for (int i = 0; i < size; ++i)
        values[static_cast<std::size_t>(i)] = i;
      r = mp1::igatherv(c, root, second_send_value, local_recv, recv_layouts, values);
      values.clear();
    } else {
      r = mp1::igatherv(c, root, second_send_value, std::span<int>{}, mp1::blocks{});
    }
    r.wait();
    if (rank == root) {
      for (int i = 0; i < size; ++i)
        BOOST_TEST(local_recv[static_cast<std::size_t>(i)] == i + 200);
    }
  }
  world_barrier();
}

// ---------------------------------------------------------------------------------------
// Intracommunicator self-block checks (MPI 4.1 6.5-6.8).  The local send block and the local
// receive block must agree; on an intercommunicator they need not (tested below).
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(intra_self_block_checks) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  const mp1::layout<int> two = mp1::layout<int>::contiguous(2);
  const mp1::layout<int> one = mp1::layout<int>::contiguous(1);
  const std::array<int, 2> send_data{{rank * 10, rank * 10 + 1}};

  // Allgatherv: sending two elements while the local receive block is one element.
  {
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(size), one);
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::allgatherv(c, mp1::region{send_data.data(), two}, recv, recv_layouts,
                      cumulative(size, 1));
    }));
  }
  world_barrier();

  // Alltoallv: process r's send block to itself and its receive block from itself differ.
  {
    std::vector<mp1::layout<int>> send_layouts(static_cast<std::size_t>(size), one);
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(size), one);
    send_layouts[static_cast<std::size_t>(rank)] = two;
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::alltoallv(c, send_data, send_layouts, cumulative(size, 1), recv, recv_layouts,
                     cumulative(size, 1));
    }));
  }
  world_barrier();

  // Gatherv: the root's send block and its own receive slot differ.  Only the root checks;
  // the other ranks do not enter the collective.
  {
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(size), one);
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    if (rank == root) {
      BOOST_TEST(throws_invalid_argument([&] {
        mp1::gatherv(c, root, mp1::region{send_data.data(), two}, recv, recv_layouts,
                     cumulative(size, 1));
      }));
    }
  }
  world_barrier();

  // Scatterv: the root's segment for itself and its local receive block differ.
  {
    std::vector<mp1::layout<int>> send_layouts(static_cast<std::size_t>(size), one);
    send_layouts[static_cast<std::size_t>(root)] = two;
    std::vector<int> send_data_local(static_cast<std::size_t>(size), 0);
    if (rank == root) {
      BOOST_TEST(throws_invalid_argument([&] {
        mp1::scatterv(c, root, send_data_local, send_layouts, cumulative(size, 1),
                      mp1::region{send_data_local.data(), one});
      }));
    }
  }
  world_barrier();

  // The non-root gatherv form has no receive side and must not inspect one.
  {
    std::array<int, 2> data{{rank, rank}};
    if (rank != root) {
      mp1::gatherv(c, root, data, std::span<int>{}, mp1::blocks{});
    } else {
      std::vector<int> recv(static_cast<std::size_t>(2 * size), -1);
      std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(size), two);
      mp1::gatherv(c, root, mp1::region{data.data(), two}, recv, recv_layouts,
                   cumulative(size, 2));
    }
  }
  world_barrier();
}

// ---------------------------------------------------------------------------------------
// Intercommunicator: local send and receive directions are in different groups and may
// differ in size (MPI 4.1 6.7/6.8 advice to users).  No local symmetry check is applied.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(intercommunicator_asymmetric_variable) {
  const auto world = mp1::comm_world();
  const int world_size = world.size();
  if (world_size < 2) {
    BOOST_TEST_MESSAGE("intercommunicator variable test needs at least 2 ranks");
    return;
  }
  const int first = world_size / 2;
  const int color = world.rank() < first ? 0 : 1;
  auto local = mp1::communicator::split(world, color, world.rank());
  const int remote_leader = color == 0 ? first : 0;
  auto inter = mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{881});
  const int local_rank = local.rank();
  const int remote_size = inter.remote_size();
  const int remote_rank = inter.rank();

  // Group 0 sends two ints per rank and receives one int per remote rank; group 1 the
  // reverse.  This is a valid asymmetric allgatherv exchange.
  {
    const mp1::layout<int> send_layout =
        color == 0 ? mp1::layout<int>::contiguous(2) : mp1::layout<int>::contiguous(1);
    const mp1::layout<int> recv_layout =
        color == 0 ? mp1::layout<int>::contiguous(1) : mp1::layout<int>::contiguous(2);
    const std::array<int, 2> send_data{{color == 0 ? 1000 + local_rank : 2000 + local_rank,
                                        color == 0 ? 1100 + local_rank : 0}};
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(remote_size),
                                               recv_layout);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(remote_size));
    const mp1::count_t recv_stride = color == 0 ? 1 : 2;
    const mp1::count_t send_stride = color == 0 ? 2 : 1;
    for (int i = 0; i < remote_size; ++i)
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * recv_stride;
    std::vector<int> recv(static_cast<std::size_t>(recv_stride * remote_size), -1);

    mp1::allgatherv(inter, mp1::region{send_data.data(), send_layout}, recv, recv_layouts,
                    displs);
    if (color == 0) {
      for (int i = 0; i < remote_size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == 2000 + i);
    } else {
      for (int i = 0; i < remote_size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(2 * i)] == 1000 + i);
        BOOST_TEST(recv[static_cast<std::size_t>(2 * i + 1)] == 1100 + i);
      }
    }

    // Alltoallv with the same asymmetry: group 0 sends two ints to every remote peer and
    // receives one; group 1 sends one and receives two.
    std::vector<mp1::layout<int>> send_layouts(static_cast<std::size_t>(remote_size),
                                               send_layout);
    std::vector<mp1::count_t> send_displs(static_cast<std::size_t>(remote_size));
    for (int i = 0; i < remote_size; ++i)
      send_displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * send_stride;
    std::vector<int> send(static_cast<std::size_t>(send_stride * remote_size), -1);
    for (int i = 0; i < remote_size; ++i) {
      if (color == 0) {
        send[static_cast<std::size_t>(2 * i)] = 3000 + local_rank * 10 + i;
        send[static_cast<std::size_t>(2 * i + 1)] = 3100 + local_rank * 10 + i;
      } else {
        send[static_cast<std::size_t>(i)] = 4000 + local_rank * 10 + i;
      }
    }
    std::vector<int> recv2(static_cast<std::size_t>(recv_stride * remote_size), -1);
    mp1::alltoallv(inter, send, send_layouts, send_displs, recv2, recv_layouts, displs);
    if (color == 0) {
      for (int i = 0; i < remote_size; ++i)
        BOOST_TEST(recv2[static_cast<std::size_t>(i)] == 4000 + i * 10 + local_rank);
    } else {
      for (int i = 0; i < remote_size; ++i) {
        BOOST_TEST(recv2[static_cast<std::size_t>(2 * i)] == 3000 + i * 10 + local_rank);
        BOOST_TEST(recv2[static_cast<std::size_t>(2 * i + 1)] == 3100 + i * 10 + local_rank);
      }
    }
    (void)remote_rank;
  }

  inter.disconnect();
}

// ---------------------------------------------------------------------------------------
// Zero-size blocks, std::byte layouts and collective completion status.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(zero_blocks_and_std_byte_layouts) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();

  // All-empty receive directions with null buffers and omitted displacements: no interval is
  // non-empty, so the conservative check accepts the zero default (MPI 4.1 5.1.8).
  {
    std::vector<mp1::layout<int>> empty_layouts(static_cast<std::size_t>(size),
                                                mp1::layout<int>::contiguous(0));
    mp1::allgatherv(c, std::span<const int>{}, std::span<int>{}, empty_layouts);
  }
  world_barrier();

  // Mixed zero and non-zero blocks: a rank sends zero elements exactly when every receiver
  // reserves an empty block for it, so the local self-block check stays consistent.
  {
    std::vector<mp1::layout<int>> layouts;
    std::vector<mp1::count_t> displs;
    for (int i = 0; i < size; ++i) {
      layouts.push_back(i % 2 == 0 ? mp1::layout<int>::contiguous(0)
                                   : mp1::layout<int>::contiguous(1));
      displs.push_back(i);
    }
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    int value = rank + 1;
    if (rank % 2 == 0)
      mp1::allgatherv(c, std::span<const int>{}, recv, layouts, displs);
    else
      mp1::allgatherv(c, std::span<const int>{&value, 1}, recv, layouts, displs);
    for (int i = 0; i < size; ++i) {
      if (i % 2 == 0)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == -1);
      else
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 1);
    }
  }
  world_barrier();

  // std::byte layouts + alltoallv are the documented type-erasure route (MPI_BYTE).
  {
    std::vector<mp1::layout<std::byte>> send_layouts(static_cast<std::size_t>(size),
                                                     mp1::layout<std::byte>::contiguous(1));
    std::vector<mp1::layout<std::byte>> recv_layouts(static_cast<std::size_t>(size),
                                                     mp1::layout<std::byte>::contiguous(1));
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      displs[static_cast<std::size_t>(i)] = i;
    std::vector<std::byte> send(static_cast<std::size_t>(size));
    std::vector<std::byte> recv(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = static_cast<std::byte>((rank + 1) * 10 + i);
    mp1::alltoallv(c, send, send_layouts, displs, recv, recv_layouts, displs);
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(static_cast<int>(recv[static_cast<std::size_t>(i)]) == (i + 1) * 10 + rank);
    }
  }
  world_barrier();

  // A collective completion exposes the normalized empty status: source/tag are not
  // meaningful and count accessors return 0 without a second MPI query (MPI 4.1 6.12).
  {
    const int root = 0;
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(size),
                                               mp1::layout<int>::contiguous(1));
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i)
      displs[static_cast<std::size_t>(i)] = i;
    mp1::request r = rank == root
                         ? mp1::igatherv(c, root, rank, recv, recv_layouts, displs)
                         : mp1::igatherv(c, root, rank, std::span<int>{}, mp1::blocks{});
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
    const mp1::status s = r.wait();
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
    BOOST_TEST(static_cast<int>(s.kind()) == static_cast<int>(mp1::status_kind::normalized));
    BOOST_TEST(s.source() == MPI_ANY_SOURCE);
    BOOST_TEST(s.tag().value == mp1::any_tag.value);
    BOOST_TEST(s.error() == MPI_SUCCESS);
    BOOST_TEST(s.get_count<int>() == 0);
    BOOST_TEST(s.get_elements<int>() == 0);
    BOOST_TEST(not s.is_cancelled());
#if MP1_TEST_HAVE_MPI_WRAP
    // Normalized status accessors make no MPI call.
    if (mpi_wrap::supported())
      BOOST_TEST(mpi_wrap::total() == 0);
#endif
    if (rank == root) {
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == i);
    }
  }
  world_barrier();
}

// ---------------------------------------------------------------------------------------
// The typed descriptors are borrowed spans; a request that outlives the call owns a duplicate
// of every staged receive layout (one MPI_Type_dup per block) and never stores the caller's
// array object.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(typed_descriptors_are_copied_at_initiation) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const mp1::layout<int> holey = holey_two();
  const std::vector<mp1::count_t> displs = cumulative(size, 3);
  const std::array<int, 3> send{{rank, -1, rank + 100}};
  std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);

  std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), holey);
#if MP1_TEST_HAVE_MPI_WRAP
  if (mpi_wrap::supported())
    mpi_wrap::reset_counts();
#endif
  mp1::request r = mp1::iallgatherv(c, mp1::region{send.data(), holey}, recv, layouts, displs);
  r.wait();
#if MP1_TEST_HAVE_MPI_WRAP
  // The staged receive needs one owned duplicate per block; the caller's array may die.
  if (mpi_wrap::supported())
    BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == static_cast<std::size_t>(size));
#endif
  for (int i = 0; i < size; ++i) {
    BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i);
    BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == i + 100);
  }
  world_barrier();
}

// ---------------------------------------------------------------------------------------
// Persistent variable plans own the descriptors they re-read at start/completion: the caller
// may destroy its layouts after *_init (MPI 4.1 6.13 requires the arrays handed to MPI to
// stay unmodified; mp1 owns its copies).  Two directions built from one array share one copy
// set, and a moved array transfers instead of duplicating.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(persistent_owns_layouts_and_dedups) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const mp1::layout<int> holey = holey_two();
  const std::vector<mp1::count_t> displs = cumulative(size, 3);
  const std::array<int, 3> send{{rank, -1, rank + 100}};

  // The layouts die immediately after *_init; start/wait still deliver the staged receive.
  {
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    const std::span<int> recv_span{recv};
    mp1::request r;
    {
      std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), holey);
      r = mp1::allgatherv_init(c, mp1::region{send.data(), holey}, recv_span, layouts, displs);
    }  // caller layouts destroyed
    r.start();
    r.wait();
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == i + 100);
    }
  }
  world_barrier();

  // The persistent plan owns one duplicate of every staged layout it re-reads at start: the
  // caller's receive array may die immediately after *_init (MPI 4.1 6.13).
  {
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    const std::span<int> recv_span{recv};
    std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), holey);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
    mp1::request r =
        mp1::allgatherv_init(c, mp1::region{send.data(), holey}, recv_span, layouts, displs);
#if MP1_TEST_HAVE_MPI_WRAP
    // One owned duplicate per staged block plus the single staged send layout: the caller's
    // region and array may both die after *_init (MPI 4.1 6.13).
    if (mpi_wrap::supported())
      BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == static_cast<std::size_t>(size + 1));
#endif
    r.start();
    r.wait();
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == i);
  }
  world_barrier();

  // Send and receive arrays that are the same object share one descriptor copy set: a staged
  // direction is re-packed at every persistent start, so it is copied once and the other
  // direction reuses that copy set.
  {
    std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), holey);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(size));
    std::vector<int> send_data(static_cast<std::size_t>(3 * size), -1);
    std::vector<int> recv(static_cast<std::size_t>(3 * size), -1);
    for (int i = 0; i < size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(3 * i);
      send_data[static_cast<std::size_t>(3 * i)] = 700 + rank * 10 + i;
      send_data[static_cast<std::size_t>(3 * i + 2)] = 900 + rank * 10 + i;
    }
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
    const std::span<const int> send_span{send_data};
    const std::span<int> recv_span{recv};
    mp1::request r =
        mp1::alltoallv_init(c, send_span, layouts, displs, recv_span, layouts, displs);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == static_cast<std::size_t>(size));
#endif
    r.start();
    r.wait();
    for (int i = 0; i < size; ++i) {
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i)] == 700 + i * 10 + rank);
      BOOST_TEST(recv[static_cast<std::size_t>(3 * i + 2)] == 900 + i * 10 + rank);
    }
  }
  world_barrier();

  // A dense persistent direction never reads its layout after *_init, so it must not pay
  // MPI_Type_dup at all.  Both directions are dense here.
  {
    std::vector<mp1::layout<int>> dense_layouts(static_cast<std::size_t>(size),
                                                mp1::layout<int>::contiguous(1));
    std::vector<mp1::count_t> one_displs(static_cast<std::size_t>(size));
    std::vector<int> send_data(static_cast<std::size_t>(size), -1);
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i) {
      one_displs[static_cast<std::size_t>(i)] = i;
      send_data[static_cast<std::size_t>(i)] = 300 + rank * 10 + i;
    }
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
    const std::span<const int> send_span{send_data};
    const std::span<int> recv_span{recv};
    mp1::request r = mp1::alltoallv_init(c, send_span, dense_layouts, one_displs, recv_span,
                                         dense_layouts, one_displs);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 0);
#endif
    r.start();
    r.wait();
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(i)] == 300 + i * 10 + rank);
  }
  world_barrier();
}

// ---------------------------------------------------------------------------------------
// Scalar forms of the variable families.  A scalar block is described by the base datatype of
// T and never has a caller layout (MPI 4.1 5.1.8: the scalar type map is the datatype's own
// map), so blocking, nonblocking and persistent forms must all accept it without looking for
// a layout.  The mixed root pointer+layout / non-root scalar gatherv_init is the
// natural MPI 4.1 6.5/6.13 spelling and must not throw on one rank while the root blocks.
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(scalar_variable_forms) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const mp1::layout<int> block = mp1::layout<int>::contiguous(1);
  std::vector<mp1::layout<int>> block_layouts(static_cast<std::size_t>(size), block);
  const std::vector<mp1::count_t> displs = cumulative(size, 1);

  // Blocking gatherv: scalar root send with a layout receive side, scalar non-root form.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    if (rank == 0) {
      int value = 1;
      mp1::gatherv(c, 0, value, recv, block_layouts, displs);
    } else {
      int value = rank + 1;
      mp1::gatherv(c, 0, value, std::span<int>{}, mp1::blocks{});
    }
    if (rank == 0)
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 1);
    world_barrier();
  }

  // Nonblocking gatherv: same scalar forms; the element direction owns no descriptor.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    int value = rank + 1;
    mp1::request r = rank == 0 ? mp1::igatherv(c, 0, value, recv, block_layouts, displs)
                               : mp1::igatherv(c, 0, value, std::span<int>{}, mp1::blocks{});
    r.wait();
    if (rank == 0)
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 1);
    world_barrier();
  }

  // Persistent gatherv, mixed spelling: the root uses a pointer+layout send block while the
  // non-root ranks use the scalar form.  Both are legal Gatherv forms (MPI 4.1 6.5); *_init
  // is collective (6.13), so a local throw on the non-root rank would deadlock the root.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    const std::span<int> recv_span{recv};
    std::array<int, 1> send{{1}};
    int scalar = rank + 1;
    mp1::request r = rank == 0
                         ? mp1::gatherv_init(c, 0, mp1::region{send.data(), block}, recv_span,
                                             block_layouts, displs)
                         : mp1::gatherv_init(c, 0, scalar, std::span<int>{}, mp1::blocks{});
    for (int cycle = 0; cycle < 2; ++cycle) {
      send[0] = 1 + cycle * 100;
      scalar = rank + 1 + cycle * 100;
      r.start();
      r.wait();
      if (rank == 0)
        for (int i = 0; i < size; ++i)
          BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 1 + cycle * 100);
    }
    world_barrier();
  }

  // Blocking and nonblocking allgatherv: scalar send, layout receive side.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    int value = rank + 1;
    mp1::allgatherv(c, value, recv, block_layouts, displs);
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 1);
    world_barrier();
  }
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    int value = rank + 1;
    mp1::request r = mp1::iallgatherv(c, value, recv, block_layouts, displs);
    r.wait();
    for (int i = 0; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 1);
    world_barrier();
  }

  // Persistent allgatherv scalar send: data changed between starts is used.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    int value = rank + 1;
    const std::span<int> recv_span{recv};
    mp1::request r = mp1::allgatherv_init(c, value, recv_span, block_layouts, displs);
    for (int cycle = 0; cycle < 2; ++cycle) {
      value = rank + 1 + cycle * 100;
      r.start();
      r.wait();
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 1 + cycle * 100);
    }
    world_barrier();
  }

  // Persistent move-layouts scalar send: a named scalar with a temporary layouts array must
  // select the layouts<T>&& overload without ambiguity and own the moved descriptors across
  // restarts (the scalar address itself is borrowed and must stay stable).
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    int value = rank + 1;
    std::vector<mp1::layout<int>> moved(static_cast<std::size_t>(size), block);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
    const std::span<int> recv_span{recv};
    mp1::request r = mp1::gatherv_init(c, 0, value, recv_span, moved, displs);
#if MP1_TEST_HAVE_MPI_WRAP
    // The receive layouts are the direct contiguous repetition: no staged metadata is read
    // after initiation, so no MPI_Type_dup is needed.
    if (mpi_wrap::supported())
      BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 0);
#endif
    for (int cycle = 0; cycle < 2; ++cycle) {
      value = rank + 1 + cycle * 100;
      r.start();
      r.wait();
      if (rank == 0)
        for (int i = 0; i < size; ++i)
          BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 1 + cycle * 100);
    }
    world_barrier();
  }
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    int value = rank + 1;
    std::vector<mp1::layout<int>> moved(static_cast<std::size_t>(size), block);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      mpi_wrap::reset_counts();
#endif
    const std::span<int> recv_span{recv};
    mp1::request r = mp1::allgatherv_init(c, value, recv_span, moved, displs);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      BOOST_TEST(mpi_wrap::count("MPI_Type_dup") == 0);
#endif
    for (int cycle = 0; cycle < 2; ++cycle) {
      value = rank + 1 + cycle * 100;
      r.start();
      r.wait();
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == i + 1 + cycle * 100);
    }
    world_barrier();
  }

  // Blocking and nonblocking scatterv: root pointer+layouts send, scalar receive.
  {
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = i + 1;
    int recv = -1;
    if (rank == 0)
      mp1::scatterv(c, 0, send, block_layouts, displs, mp1::in_place);
    else
      mp1::scatterv(c, 0, std::span<const int>{}, mp1::blocks{}, recv);
    if (rank != 0)
      BOOST_TEST(recv == rank + 1);
    world_barrier();
  }
  {
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    for (int i = 0; i < size; ++i)
      send[static_cast<std::size_t>(i)] = i + 1;
    int recv = -1;
    mp1::request r = rank == 0
                         ? mp1::iscatterv(c, 0, send, block_layouts, displs, mp1::in_place)
                         : mp1::iscatterv(c, 0, std::span<const int>{}, mp1::blocks{}, recv);
    r.wait();
    if (rank != 0)
      BOOST_TEST(recv == rank + 1);
    world_barrier();
  }

  // Persistent scatterv scalar receive: the scalar receive form.  Two cycles prove
  // the scalar receive plan keeps working after restart.
  {
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    const std::span<const int> send_span{send};
    int recv = -1;
    mp1::request r =
        rank == 0 ? mp1::scatterv_init(c, 0, send_span, block_layouts, displs, mp1::in_place)
                  : mp1::scatterv_init(c, 0, std::span<const int>{}, mp1::blocks{}, recv);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int i = 0; i < size; ++i)
        send[static_cast<std::size_t>(i)] = i + 1 + cycle * 100;
      r.start();
      r.wait();
      if (rank != 0)
        BOOST_TEST(recv == rank + 1 + cycle * 100);
    }
    world_barrier();
  }
}

// ---------------------------------------------------------------------------------------
// Reflected (holed) scalar receive direction: a reflected type's native datatype is not
// packed, so the scalar scatterv receive stages through MPI_Pack/MPI_Unpack with the base
// datatype.  The plan must not look for a caller layout (MPI 4.1 5.1.8/5.1.11).
// ---------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(reflected_scalar_scatterv) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const mp1::layout<tuple> block = mp1::layout<tuple>::contiguous(1);
  std::vector<mp1::layout<tuple>> layouts(static_cast<std::size_t>(size), block);
  const std::vector<mp1::count_t> displs = cumulative(size, 1);

  std::vector<tuple> send(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    send[static_cast<std::size_t>(i)] = tuple{10 + i, 20.0 + i};

  // Blocking scalar receive of the reflected type on every rank except the root.
  {
    tuple recv{};
    if (rank == 0)
      mp1::scatterv(c, 0, send, layouts, displs, mp1::in_place);
    else
      mp1::scatterv(c, 0, std::span<const tuple>{}, mp1::blocks{}, recv);
    if (rank != 0) {
      BOOST_TEST(recv.a == 10 + rank);
      BOOST_TEST(recv.b == 20.0 + rank);
    }
    world_barrier();
  }

  // Nonblocking scalar receive: the completion unpacks the staged reflected block.
  {
    tuple recv{};
    mp1::request r = rank == 0
                         ? mp1::iscatterv(c, 0, send, layouts, displs, mp1::in_place)
                         : mp1::iscatterv(c, 0, std::span<const tuple>{}, mp1::blocks{}, recv);
    r.wait();
    if (rank != 0) {
      BOOST_TEST(recv.a == 10 + rank);
      BOOST_TEST(recv.b == 20.0 + rank);
    }
    world_barrier();
  }

  // Persistent scalar receive: restart and completion both read only the base datatype.
  {
    tuple recv{};
    const std::span<const tuple> send_span{send};
    mp1::request r =
        rank == 0 ? mp1::scatterv_init(c, 0, send_span, layouts, displs, mp1::in_place)
                  : mp1::scatterv_init(c, 0, std::span<const tuple>{}, mp1::blocks{}, recv);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int i = 0; i < size; ++i)
        send[static_cast<std::size_t>(i)] = tuple{10 + i + cycle * 100, 20.0 + i + cycle * 100};
      r.start();
      r.wait();
      if (rank != 0) {
        BOOST_TEST(recv.a == 10 + rank + cycle * 100);
        BOOST_TEST(recv.b == 20.0 + rank + cycle * 100);
      }
    }
    world_barrier();
  }
}

// ---------------------------------------------------------------------------------------
// Non-dense self-placing subarray tiles (the matrix_gather shape).  MPI 4.1 §5.1.11 makes
// the type map authoritative for the locations a block addresses, so two tiles whose
// conservative true extents overlap but whose type maps are disjoint are a legal
// restricted side: Gatherv/Allgatherv/Alltoallv receives and Scatterv's root send must
// accept them with omitted and with explicit zero displacements.  A real type-map alias
// must still be rejected before MPI.  The tile grid is 2x3 ints per rank, so every tile is
// non-dense whenever the process count is at least 2 (the CI rank counts are 2 and 4).
// ---------------------------------------------------------------------------------------

namespace {

  struct tile_grid {
    int rows{1};
    int cols{1};
    int total_ints{0};
  };

  // `size` tiles of 2x3 ints cover a (2*rows)x(3*cols) int matrix.
  [[nodiscard]] tile_grid tiling_for(int size) {
    tile_grid grid;
    grid.cols = (size % 2 == 0) ? 2 : size;
    grid.rows = size / grid.cols;
    grid.total_ints = 6 * size;
    return grid;
  }

  [[nodiscard]] mp1::layout<int> tile_at(const tile_grid& grid, int row, int column) {
    mp1::subarray_parameter parameter;
    parameter.add(grid.rows * 2, 2, row * 2);
    parameter.add(grid.cols * 3, 3, column * 3);
    return mp1::layout<int>::subarray(parameter);
  }

  [[nodiscard]] mp1::layout<int> tile_layout(const tile_grid& grid, int index) {
    return tile_at(grid, index / grid.cols, index % grid.cols);
  }

  // A tile shifted two columns into its neighbour: its type map shares selected columns with
  // tile 0, so it is a genuine (non-identical) alias.
  [[nodiscard]] mp1::layout<int> shifted_tile(const tile_grid& grid) {
    mp1::subarray_parameter parameter;
    parameter.add(grid.rows * 2, 2, 0);
    parameter.add(grid.cols * 3, 3, 2);
    return mp1::layout<int>::subarray(parameter);
  }

  // Flat row-major index of the `slot`-th (0..5) element of tile `index`.
  [[nodiscard]] std::size_t tile_linear(const tile_grid& grid, int index, int slot) {
    const int row = index / grid.cols;
    const int column = index % grid.cols;
    const int slot_row = slot / 3;
    const int slot_column = slot % 3;
    return static_cast<std::size_t>((row * 2 + slot_row) * (grid.cols * 3) + column * 3 +
                                    slot_column);
  }

  [[nodiscard]] std::vector<mp1::layout<int>> tile_layouts(const tile_grid& grid, int size) {
    std::vector<mp1::layout<int>> layouts;
    for (int i = 0; i < size; ++i)
      layouts.push_back(tile_layout(grid, i));
    return layouts;
  }

}  // namespace

BOOST_AUTO_TEST_CASE(self_placing_subarray_layouts_are_disjoint) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  const tile_grid grid = tiling_for(size);
  const mp1::layout<int> block = mp1::layout<int>::contiguous(6);
  const std::vector<mp1::layout<int>> tiles = tile_layouts(grid, size);
  const std::vector<mp1::count_t> zeros(static_cast<std::size_t>(size), 0);
  std::array<int, 6> send{};
  for (int j = 0; j < 6; ++j)
    send[static_cast<std::size_t>(j)] = rank * 100 + j;

  // Gatherv: omitted displacements.
  {
    std::vector<int> recv(static_cast<std::size_t>(grid.total_ints), -1);
    mp1::gatherv(c, root, send, recv, tiles);
    if (rank == root) {
      for (int i = 0; i < size; ++i)
        for (int j = 0; j < 6; ++j)
          BOOST_TEST(recv[tile_linear(grid, i, j)] == i * 100 + j);
    }
  }
  world_barrier();

  // Gatherv: explicit all-zero displacements follow the same type-map rule.
  {
    std::vector<int> recv(static_cast<std::size_t>(grid.total_ints), -1);
    mp1::gatherv(c, root, send, recv, tiles, zeros);
    if (rank == root)
      BOOST_TEST(recv[tile_linear(grid, size - 1, 5)] == (size - 1) * 100 + 5);
  }
  world_barrier();

  // Scatterv: the root's self-placed send blocks are the read-restricted side.
  {
    std::vector<int> source(static_cast<std::size_t>(grid.total_ints), -1);
    if (rank == root) {
      for (int i = 0; i < size; ++i)
        for (int j = 0; j < 6; ++j)
          source[tile_linear(grid, i, j)] = i * 100 + j;
    }
    std::array<int, 6> local{};
    mp1::scatterv(c, root, source, tiles, std::span<const mp1::count_t>{}, local);
    for (int j = 0; j < 6; ++j)
      BOOST_TEST(local[static_cast<std::size_t>(j)] == rank * 100 + j);
  }
  world_barrier();

  // Allgatherv: every rank's receive side is self-placed.
  {
    std::vector<int> recv(static_cast<std::size_t>(grid.total_ints), -1);
    mp1::allgatherv(c, send, recv, tiles);
    for (int i = 0; i < size; ++i)
      for (int j = 0; j < 6; ++j)
        BOOST_TEST(recv[tile_linear(grid, i, j)] == i * 100 + j);
  }
  world_barrier();

  // Alltoallv: allocate distinct send blocks per peer, self-place the receive blocks.
  {
    std::vector<mp1::layout<int>> send_layouts;
    std::vector<int> source(static_cast<std::size_t>(6 * size), -1);
    for (int peer = 0; peer < size; ++peer) {
      send_layouts.push_back(mp1::layout<int>::contiguous(6));
      for (int j = 0; j < 6; ++j)
        source[static_cast<std::size_t>(6 * peer + j)] = rank * 100 + j;
    }
    std::vector<int> recv(static_cast<std::size_t>(grid.total_ints), -1);
    mp1::alltoallv(c, source, send_layouts, cumulative(size, 6), recv, tiles,
                   std::span<const mp1::count_t>{});
    for (int peer = 0; peer < size; ++peer)
      for (int j = 0; j < 6; ++j)
        BOOST_TEST(recv[tile_linear(grid, peer, j)] == peer * 100 + j);
  }
  world_barrier();

  // The send side of alltoallv is read-only: identical send layouts at displacement zero
  // alias on the wire and remain legal.
  {
    std::vector<mp1::layout<int>> send_layouts(static_cast<std::size_t>(size),
                                               mp1::layout<int>::contiguous(6));
    std::vector<int> recv(static_cast<std::size_t>(grid.total_ints), -1);
    mp1::alltoallv(c, send, send_layouts, std::span<const mp1::count_t>{}, recv, tiles,
                   std::span<const mp1::count_t>{});
    BOOST_TEST(recv[tile_linear(grid, 0, 0)] == 0);
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(non_dense_self_placing_maps_are_disjoint) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  const std::array<int, 2> send{{rank * 10, rank * 10 + 1}};
  const mp1::layout<int> send_layout = mp1::layout<int>::contiguous(rank == 0 ? 2 : 1);

  // Block 0 is a holey vector {0, 8}; block 1 is a single int at byte 4, inside block 0's
  // conservative true extent [0, 12); the other blocks sit after it.  Every map is disjoint
  // but the true extent of block 0 overlaps block 1, so only an exact type-map check accepts
  // this legal self-placement.
  std::vector<mp1::layout<int>> blocks;
  blocks.push_back(mp1::layout<int>::vector(2, 1, 2));
  mp1::hindexed_parameter second_parameter;
  second_parameter.add(1, 4);
  blocks.push_back(mp1::layout<int>::hindexed(second_parameter));
  for (int i = 2; i < size; ++i) {
    mp1::hindexed_parameter parameter;
    parameter.add(1, static_cast<MPI_Aint>(4 * i + 4));
    blocks.push_back(mp1::layout<int>::hindexed(parameter));
  }
  std::vector<int> recv(static_cast<std::size_t>(size + 2), -1);
  mp1::gatherv(c, root, mp1::region{send.data(), send_layout}, recv, blocks);
  if (rank == root) {
    BOOST_TEST(recv[0] == 0);
    BOOST_TEST(recv[2] == 1);
    if (size > 1)
      BOOST_TEST(recv[1] == 10);
    for (int i = 2; i < size; ++i)
      BOOST_TEST(recv[static_cast<std::size_t>(i + 1)] == i * 10);
  }
  world_barrier();

  // The same legal map with an explicit all-zero displacement array.
  {
    std::vector<int> explicit_recv(static_cast<std::size_t>(size + 2), -1);
    const std::vector<mp1::count_t> zeros(static_cast<std::size_t>(size), 0);
    mp1::gatherv(c, root, mp1::region{send.data(), send_layout}, explicit_recv, blocks, zeros);
    if (rank == root) {
      BOOST_TEST(explicit_recv[0] == 0);
      BOOST_TEST(explicit_recv[2] == 1);
    }
  }
  world_barrier();

  // A genuine type-map alias inside that family is still rejected.
  {
    std::vector<mp1::layout<int>> aliased;
    aliased.push_back(mp1::layout<int>::vector(2, 1, 2));
    mp1::hindexed_parameter first_parameter;
    first_parameter.add(1, 0);
    aliased.push_back(mp1::layout<int>::hindexed(first_parameter));
    for (int i = 2; i < size; ++i) {
      mp1::hindexed_parameter parameter;
      parameter.add(1, static_cast<MPI_Aint>(4 * i + 4));
      aliased.push_back(mp1::layout<int>::hindexed(parameter));
    }
    std::vector<int> reject_recv(static_cast<std::size_t>(size + 2), -1);
    if (rank == root) {
      BOOST_TEST(throws_invalid_argument([&] {
        mp1::gatherv(c, root, mp1::region{send.data(), send_layout}, reject_recv, aliased);
      }));
    }
  }
  world_barrier();
}

BOOST_AUTO_TEST_CASE(overlapping_subarray_layouts_are_rejected) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int root = 0;
  const tile_grid grid = tiling_for(size);
  const mp1::layout<int> block = mp1::layout<int>::contiguous(6);
  const std::array<int, 6> send{{rank, rank + 1, rank + 2, rank + 3, rank + 4, rank + 5}};
  const std::vector<mp1::count_t> zeros(static_cast<std::size_t>(size), 0);
  std::vector<int> recv(static_cast<std::size_t>(grid.total_ints), -1);

  // Every rank aims at the same tile: the same type-map entries are written `size` times.
  {
    std::vector<mp1::layout<int>> aliased;
    for (int i = 0; i < size; ++i)
      aliased.push_back(tile_layout(grid, 0));
    if (rank == root) {
      BOOST_TEST(throws_invalid_argument([&] { mp1::gatherv(c, root, send, recv, aliased); }));
      BOOST_TEST(
          throws_invalid_argument([&] { mp1::gatherv(c, root, send, recv, aliased, zeros); }));
    }
  }
  world_barrier();

  // Non-identical tiles that share selected columns still alias exactly.
  {
    std::vector<mp1::layout<int>> partial;
    for (int i = 0; i < size; ++i)
      partial.push_back(i == 1 ? shifted_tile(grid) : tile_layout(grid, i));
    if (rank == root) {
      BOOST_TEST(throws_invalid_argument([&] { mp1::gatherv(c, root, send, recv, partial); }));
    }
  }
  world_barrier();

  // Allgatherv and alltoallv check the same map on every rank, so every rank throws.
  {
    std::vector<mp1::layout<int>> aliased;
    for (int i = 0; i < size; ++i)
      aliased.push_back(tile_layout(grid, 0));
    BOOST_TEST(throws_invalid_argument([&] { mp1::allgatherv(c, send, recv, aliased); }));
    world_barrier();
    std::vector<mp1::layout<int>> send_layouts(static_cast<std::size_t>(size), block);
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::alltoallv(c, send, send_layouts, zeros, recv, aliased, zeros); }));
  }
  world_barrier();

  // Scatterv's root send is read-restricted, so the same alias is erroneous.
  {
    std::vector<int> source(static_cast<std::size_t>(grid.total_ints), 0);
    std::array<int, 6> local{};
    std::vector<mp1::layout<int>> aliased(static_cast<std::size_t>(size), tile_layout(grid, 0));
    if (rank == root) {
      BOOST_TEST(throws_invalid_argument([&] {
        mp1::scatterv(c, root, source, aliased, std::span<const mp1::count_t>{}, local);
      }));
    }
  }
  world_barrier();
}
