// Persistent collective tests: all 16 intracommunicator families,
// intercommunicators (section 6.2.2 families), in-place forms, dense and holey
// layouts, and repeated start/wait cycles (the request payload must be re-packed and
// re-primed on every start).
//
// The unit is registered at 2 and 4 ranks by the glob in test/CMakeLists.txt; the nightly
// extra-ranks configuration additionally runs it at 5 ranks so the intercommunicator half is
// exercised with an asymmetric 3+2 split.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>


namespace {

  void world_barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
  }

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

  int sum_of_ranks(int size) {
    return size * (size - 1) / 2;
  }

  // Two int elements at byte offsets 0 and 8 of a 12-byte extent: a holey layout that forces
  // the variable and reduction staging paths.
  mp1::layout<int> holey_two() {
    return mp1::layout<int>::vector(2, 1, 2);
  }

  // One holey_two() block per rank, as an aggregate layout over the whole buffer.
  mp1::layout<int> aggregate_holey(int blocks) {
    mp1::indexed_parameter parameter;
    for (int i = 0; i < blocks; ++i) {
      parameter.add(1, static_cast<mp1::count_t>(3 * i));
      parameter.add(1, static_cast<mp1::count_t>(3 * i + 2));
    }
    return mp1::layout<int>::indexed(parameter);
  }

  // Split MPI_COMM_WORLD by parity and create the intercommunicator between the halves.  All
  // ranks call this; the halves may differ in size (the 5-rank nightly configuration).
  mp1::communicator make_intercommunicator(const mp1::mpi_communicator& world, int tag) {
    const int my_group = world.rank() % 2;
    auto local = mp1::communicator::split(world, my_group);
    const int remote_leader = my_group == 0 ? 1 : 0;
    return mp1::communicator::intercomm_create(local, 0, world, remote_leader, mp1::tag{tag});
  }

  // Running count of distinct intercommunicator creation tags in this file.
  int next_inter_tag() {
    static int tag = 100;
    return tag++;
  }

}  // namespace


struct mp1_persistent_collective_fixture {
  mp1_persistent_collective_fixture() {
    mp1::init();
  }

  ~mp1_persistent_collective_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor must not throw
    } catch (...) {
      // Another owner may finalize; a fixture destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_persistent_collective_fixture);

BOOST_AUTO_TEST_SUITE(persistent_collective)

// ---------------------------------------------------------------------------------------------
// Barrier and broadcast
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(barrier_and_bcast) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();

  {
    mp1::request r = mp1::barrier_init(c);
    BOOST_TEST(r.wait().error() == MPI_SUCCESS);  // inactive wait: no MPI call
    for (int cycle = 0; cycle < 3; ++cycle) {
      r.start();
      BOOST_TEST(r.is_active());
      BOOST_TEST(r.wait().error() == MPI_SUCCESS);
      BOOST_TEST(not r.is_active());
    }
  }

  // Scalar broadcast: the root rewrites the value before every start, so a stale payload or
  // a stale request would be visible.
  {
    int value = -1;
    mp1::request r = mp1::bcast_init(c, 0, value);
    for (int cycle = 0; cycle < 3; ++cycle) {
      if (rank == 0)
        value = 100 + cycle;
      r.start();
      BOOST_TEST(r.wait().error() == MPI_SUCCESS);
      BOOST_TEST(value == 100 + cycle);
    }
  }

  // Holey layout broadcast: one typed datatype occurrence, owned by the request.
  {
    const mp1::layout<int> l = holey_two();
    std::array<int, 3> data{{-1, -1, -1}};
    mp1::region<int> region{data.data(), l};
    mp1::request r = mp1::bcast_init(c, 0, region);
    for (int cycle = 0; cycle < 2; ++cycle) {
      if (rank == 0) {
        data[0] = 40 + cycle;
        data[2] = 80 + cycle;
      }
      r.start();
      r.wait();
      BOOST_TEST(data[0] == 40 + cycle);
      BOOST_TEST(data[2] == 80 + cycle);
    }
  }

  // start_all / wait_all with a mixed set (the elements are prepared before MPI_Startall).
  {
    int first = -1;
    int second = -1;
    mp1::request a = mp1::bcast_init(c, 0, first);
    mp1::request b = mp1::bcast_init(c, 0, second);
    if (rank == 0) {
      first = 7;
      second = 9;
    }
    std::vector<mp1::request> no_requests;
    mp1::start_all(no_requests);  // empty: no-op
    std::vector<mp1::request> requests;
    requests.push_back(std::move(a));
    requests.push_back(std::move(b));
    mp1::start_all(requests);
    BOOST_TEST(requests[0].is_active());
    BOOST_TEST(requests[1].is_active());
    const std::vector<mp1::status> statuses = mp1::wait_all(requests);
    BOOST_TEST(statuses.size() == 2u);
    BOOST_TEST(first == 7);
    BOOST_TEST(second == 9);
    BOOST_TEST(not requests[0].is_active());
    BOOST_TEST(not requests[1].is_active());
    // The handles survive completion: a second start/wait cycle works.
    if (rank == 0) {
      first = 17;
      second = 19;
    }
    mp1::start_all(requests);
    (void)mp1::wait_all(requests);
    BOOST_TEST(first == 17);
    BOOST_TEST(second == 19);
  }
}

// ---------------------------------------------------------------------------------------------
// Gather and scatter
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(gather_and_scatter) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();

  // Root in-place gather; every non-root rank sends its own block.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    int send = -1;
    if (rank == 0) {
      mp1::request r = mp1::gather_init(c, 0, mp1::in_place, std::span<int>{recv});
      for (int cycle = 0; cycle < 2; ++cycle) {
        recv[0] = cycle;
        r.start();
        r.wait();
        for (int i = 0; i < size; ++i)
          BOOST_TEST(recv[static_cast<std::size_t>(i)] == i * 100 + cycle);
      }
    } else {
      mp1::request r = mp1::gather_init(c, 0, send, std::span<int>{});
      for (int cycle = 0; cycle < 2; ++cycle) {
        send = rank * 100 + cycle;
        r.start();
        r.wait();
      }
    }
  }

  // Layout gather with a holey single block per rank and an aggregate receive layout.
  {
    const mp1::layout<int> l = holey_two();
    const mp1::layout<int> aggregate = aggregate_holey(size);
    std::array<int, 3> send{{-1, -1, -1}};
    std::vector<int> recv(static_cast<std::size_t>(size * 3), -1);
    mp1::region<int> send_region{send.data(), l};
    mp1::region<int> recv_region{recv.data(), aggregate};
    mp1::request r;
    if (rank == 0)
      r = mp1::gather_init(c, 0, send_region, recv_region);
    else
      r = mp1::gather_init(c, 0, send_region, std::span<int>{});
    for (int cycle = 0; cycle < 2; ++cycle) {
      send[0] = rank * 10 + cycle;
      send[2] = rank * 10 + 5 + cycle;
      r.start();
      r.wait();
      if (rank == 0) {
        for (int i = 0; i < size; ++i) {
          BOOST_TEST(recv[static_cast<std::size_t>(i * 3)] == i * 10 + cycle);
          BOOST_TEST(recv[static_cast<std::size_t>(i * 3 + 2)] == i * 10 + 5 + cycle);
        }
      }
    }
  }

  // Scatter: the root sends one block per rank; non-root ranks receive one.
  {
    std::vector<int> send;
    if (rank == 0)
      send.resize(static_cast<std::size_t>(size));
    int recv = -1;
    mp1::request r;
    if (rank == 0)
      r = mp1::scatter_init(c, 0, std::span<const int>{send}, recv);
    else
      r = mp1::scatter_init(c, 0, std::span<const int>{}, recv);
    for (int cycle = 0; cycle < 2; ++cycle) {
      if (rank == 0)
        for (int i = 0; i < size; ++i)
          send[static_cast<std::size_t>(i)] = i * 10 + cycle;
      r.start();
      r.wait();
      BOOST_TEST(recv == rank * 10 + cycle);
    }
  }

  // Scatter with a holey per-rank block layout and an aggregate root send layout.
  {
    const mp1::layout<int> l = holey_two();
    const mp1::layout<int> aggregate = aggregate_holey(size);
    std::vector<int> send;
    if (rank == 0)
      send.assign(static_cast<std::size_t>(size * 3), -1);
    std::array<int, 3> recv{{-1, -1, -1}};
    mp1::region<int> send_region{send.data(), aggregate};
    mp1::region<int> recv_region{recv.data(), l};
    mp1::request r;
    if (rank == 0)
      r = mp1::scatter_init(c, 0, send_region, recv_region);
    else
      r = mp1::scatter_init(c, 0, std::span<const int>{}, recv_region);
    for (int cycle = 0; cycle < 2; ++cycle) {
      if (rank == 0) {
        for (int i = 0; i < size; ++i) {
          send[static_cast<std::size_t>(i * 3)] = i * 10 + cycle;
          send[static_cast<std::size_t>(i * 3 + 2)] = i * 10 + 9 + cycle;
        }
      }
      r.start();
      r.wait();
      if (rank != 0) {
        BOOST_TEST(recv[0] == rank * 10 + cycle);
        BOOST_TEST(recv[2] == rank * 10 + 9 + cycle);
      }
    }
  }

  // Scatter with the root as the receive target (in-place) and an aggregate send layout.
  {
    const mp1::layout<int> l = holey_two();
    const mp1::layout<int> aggregate = aggregate_holey(size);
    std::vector<int> send;
    if (rank == 0)
      send.assign(static_cast<std::size_t>(size * 3), -1);
    std::array<int, 3> recv{{-1, -1, -1}};
    mp1::region<int> send_region{send.data(), aggregate};
    mp1::region<int> recv_region{recv.data(), l};
    mp1::request r;
    if (rank == 0)
      r = mp1::scatter_init(c, 0, send_region, mp1::in_place);
    else
      r = mp1::scatter_init(c, 0, std::span<const int>{}, recv_region);
    for (int cycle = 0; cycle < 2; ++cycle) {
      if (rank == 0) {
        for (int i = 0; i < size; ++i) {
          send[static_cast<std::size_t>(i * 3)] = i * 10 + cycle;
          send[static_cast<std::size_t>(i * 3 + 2)] = i * 10 + 7 + cycle;
        }
      }
      r.start();
      r.wait();
      if (rank != 0) {
        BOOST_TEST(recv[0] == rank * 10 + cycle);
        BOOST_TEST(recv[2] == rank * 10 + 7 + cycle);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Allgather and alltoall
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(allgather_and_alltoall) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();

  // Scalar allgather.
  {
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    int send = -1;
    mp1::request r = mp1::allgather_init(c, send, std::span<int>{recv});
    for (int cycle = 0; cycle < 2; ++cycle) {
      send = rank * 10 + cycle;
      r.start();
      r.wait();
      for (int i = 0; i < size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == i * 10 + cycle);
    }
  }

  // In-place allgather: the local slot is the contribution, so a stale receive staging buffer
  // would corrupt every rank.
  {
    std::vector<int> buffer(static_cast<std::size_t>(size), -1);
    mp1::request r = mp1::allgather_init(c, mp1::in_place, std::span<int>{buffer});
    for (int cycle = 0; cycle < 2; ++cycle) {
      buffer[static_cast<std::size_t>(rank)] = rank * 10 + cycle;
      r.start();
      r.wait();
      for (int i = 0; i < size; ++i)
        BOOST_TEST(buffer[static_cast<std::size_t>(i)] == i * 10 + cycle);
    }
  }

  // Holey layout allgather: single send block per rank, aggregate receive layout.
  {
    const mp1::layout<int> l = holey_two();
    const mp1::layout<int> aggregate = aggregate_holey(size);
    std::array<int, 3> send{{-1, -1, -1}};
    std::vector<int> recv(static_cast<std::size_t>(size * 3), -1);
    mp1::region<int> send_region{send.data(), l};
    mp1::region<int> recv_region{recv.data(), aggregate};
    mp1::request r = mp1::allgather_init(c, send_region, recv_region);
    for (int cycle = 0; cycle < 2; ++cycle) {
      send[0] = rank * 10 + cycle;
      send[2] = rank * 10 + 3 + cycle;
      r.start();
      r.wait();
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(i * 3)] == i * 10 + cycle);
        BOOST_TEST(recv[static_cast<std::size_t>(i * 3 + 2)] == i * 10 + 3 + cycle);
      }
    }
  }

  // Flat alltoall.
  {
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    std::vector<int> recv(static_cast<std::size_t>(size), -1);
    mp1::request r = mp1::alltoall_init(c, std::span<int>{send}, std::span<int>{recv});
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j)
        send[static_cast<std::size_t>(j)] = rank * 100 + j + cycle * 1000;
      r.start();
      r.wait();
      for (int j = 0; j < size; ++j)
        BOOST_TEST(recv[static_cast<std::size_t>(j)] == j * 100 + rank + cycle * 1000);
    }
  }

  // In-place alltoall.
  {
    std::vector<int> buffer(static_cast<std::size_t>(size), -1);
    mp1::request r = mp1::alltoall_init(c, mp1::in_place, std::span<int>{buffer});
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j)
        buffer[static_cast<std::size_t>(j)] = rank * 100 + j + cycle * 1000;
      r.start();
      r.wait();
      for (int j = 0; j < size; ++j)
        BOOST_TEST(buffer[static_cast<std::size_t>(j)] == j * 100 + rank + cycle * 1000);
    }
  }

  // Holey layout alltoall: both directions are whole-buffer aggregate layouts.
  {
    const mp1::layout<int> aggregate = aggregate_holey(size);
    std::vector<int> send(static_cast<std::size_t>(size * 3), -1);
    std::vector<int> recv(static_cast<std::size_t>(size * 3), -1);
    mp1::region<int> send_region{send.data(), aggregate};
    mp1::region<int> recv_region{recv.data(), aggregate};
    mp1::request r = mp1::alltoall_init(c, send_region, recv_region);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j) {
        send[static_cast<std::size_t>(j * 3)] = rank * 100 + j * 10 + cycle * 1000;
        send[static_cast<std::size_t>(j * 3 + 2)] = rank * 100 + j * 10 + 1 + cycle * 1000;
      }
      r.start();
      r.wait();
      for (int j = 0; j < size; ++j) {
        BOOST_TEST(recv[static_cast<std::size_t>(j * 3)] == j * 100 + rank * 10 + cycle * 1000);
        BOOST_TEST(recv[static_cast<std::size_t>(j * 3 + 2)] ==
                   j * 100 + rank * 10 + 1 + cycle * 1000);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Variable collectives
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(variable_collectives) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const mp1::layout<int> l = holey_two();
  std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(size), l);
  std::vector<mp1::count_t> displs(static_cast<std::size_t>(size));
  for (int i = 0; i < size; ++i)
    displs[static_cast<std::size_t>(i)] = i * 3;

  // Gatherv with per-rank holey blocks and explicit element displacements.
  {
    std::array<int, 3> send{{-1, -1, -1}};
    std::vector<int> recv;
    if (rank == 0)
      recv.assign(static_cast<std::size_t>(size * 3), -1);
    const std::span<int> recv_span{recv};
    mp1::request r = rank == 0 ? mp1::gatherv_init(c, 0, mp1::region{send.data(), l}, recv_span,
                                                   layouts, displs)
                               : mp1::gatherv_init(c, 0, mp1::region{send.data(), l},
                                                   std::span<int>{}, mp1::blocks{});
    for (int cycle = 0; cycle < 2; ++cycle) {
      send[0] = rank * 10 + cycle;
      send[2] = rank * 10 + 2 + cycle;
      r.start();
      r.wait();
      if (rank == 0) {
        for (int i = 0; i < size; ++i) {
          BOOST_TEST(recv[static_cast<std::size_t>(i * 3)] == i * 10 + cycle);
          BOOST_TEST(recv[static_cast<std::size_t>(i * 3 + 2)] == i * 10 + 2 + cycle);
        }
      }
    }
  }

  // In-place gatherv: the root slot lives in the receive buffer.
  {
    std::vector<int> recv(static_cast<std::size_t>(size * 3), -1);
    const std::span<int> recv_span{recv};
    if (rank == 0) {
      mp1::request r = mp1::gatherv_init(c, 0, mp1::in_place, recv_span, layouts, displs);
      for (int cycle = 0; cycle < 2; ++cycle) {
        recv[0] = cycle;
        recv[2] = 2 + cycle;
        r.start();
        r.wait();
        for (int i = 0; i < size; ++i) {
          BOOST_TEST(recv[static_cast<std::size_t>(i * 3)] == i * 10 + cycle);
          BOOST_TEST(recv[static_cast<std::size_t>(i * 3 + 2)] == i * 10 + 2 + cycle);
        }
      }
    } else {
      std::array<int, 3> send_block{{-1, -1, -1}};
      mp1::request r = mp1::gatherv_init(c, 0, mp1::region{send_block.data(), l},
                                         std::span<int>{}, mp1::blocks{});
      for (int cycle = 0; cycle < 2; ++cycle) {
        send_block[0] = rank * 10 + cycle;
        send_block[2] = rank * 10 + 2 + cycle;
        r.start();
        r.wait();
      }
    }
  }

  // Scatterv with a holey block per rank.
  {
    std::vector<int> send(static_cast<std::size_t>(size * 3), -1);
    std::array<int, 3> recv{{-1, -1, -1}};
    const std::span<const int> send_span{send};
    mp1::request r = rank == 0
                         ? mp1::scatterv_init(c, 0, send_span, layouts, displs, mp1::in_place)
                         : mp1::scatterv_init(c, 0, std::span<const int>{}, mp1::blocks{},
                                              mp1::region{recv.data(), l});
    for (int cycle = 0; cycle < 2; ++cycle) {
      if (rank == 0) {
        for (int i = 0; i < size; ++i) {
          send[static_cast<std::size_t>(i * 3)] = i * 10 + cycle;
          send[static_cast<std::size_t>(i * 3 + 2)] = i * 10 + 4 + cycle;
        }
      }
      r.start();
      r.wait();
      if (rank != 0) {
        BOOST_TEST(recv[0] == rank * 10 + cycle);
        BOOST_TEST(recv[2] == rank * 10 + 4 + cycle);
      }
    }
  }

  // Allgatherv, including the in-place form.
  {
    std::array<int, 3> send{{-1, -1, -1}};
    std::vector<int> recv(static_cast<std::size_t>(size * 3), -1);
    const std::span<int> recv_span{recv};
    mp1::request r =
        mp1::allgatherv_init(c, mp1::region{send.data(), l}, recv_span, layouts, displs);
    for (int cycle = 0; cycle < 2; ++cycle) {
      send[0] = rank * 10 + cycle;
      send[2] = rank * 10 + 6 + cycle;
      r.start();
      r.wait();
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(i * 3)] == i * 10 + cycle);
        BOOST_TEST(recv[static_cast<std::size_t>(i * 3 + 2)] == i * 10 + 6 + cycle);
      }
    }
  }
  {
    std::vector<int> recv(static_cast<std::size_t>(size * 3), -1);
    const std::span<int> recv_span{recv};
    mp1::request r = mp1::allgatherv_init(c, mp1::in_place, recv_span, layouts, displs);
    for (int cycle = 0; cycle < 2; ++cycle) {
      recv[static_cast<std::size_t>(rank * 3)] = rank * 10 + cycle;
      recv[static_cast<std::size_t>(rank * 3 + 2)] = rank * 10 + 8 + cycle;
      r.start();
      r.wait();
      for (int i = 0; i < size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(i * 3)] == i * 10 + cycle);
        BOOST_TEST(recv[static_cast<std::size_t>(i * 3 + 2)] == i * 10 + 8 + cycle);
      }
    }
  }

  // Alltoallv, including the in-place form.
  {
    std::vector<int> send(static_cast<std::size_t>(size * 3), -1);
    std::vector<int> recv(static_cast<std::size_t>(size * 3), -1);
    const std::span<const int> send_span{send};
    const std::span<int> recv_span{recv};
    mp1::request r =
        mp1::alltoallv_init(c, send_span, layouts, displs, recv_span, layouts, displs);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j) {
        send[static_cast<std::size_t>(j * 3)] = rank * 100 + j * 10 + cycle * 1000;
        send[static_cast<std::size_t>(j * 3 + 2)] = rank * 100 + j * 10 + 4 + cycle * 1000;
      }
      r.start();
      r.wait();
      for (int j = 0; j < size; ++j) {
        BOOST_TEST(recv[static_cast<std::size_t>(j * 3)] == j * 100 + rank * 10 + cycle * 1000);
        BOOST_TEST(recv[static_cast<std::size_t>(j * 3 + 2)] ==
                   j * 100 + rank * 10 + 4 + cycle * 1000);
      }
    }
  }
  {
    std::vector<int> buffer(static_cast<std::size_t>(size * 3), -1);
    const std::span<int> buffer_span{buffer};
    mp1::request r =
        mp1::alltoallv_init(c, mp1::in_place, std::span<const mp1::layout<int>>{},
                            std::span<const mp1::count_t>{}, buffer_span, layouts, displs);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j) {
        buffer[static_cast<std::size_t>(j * 3)] = rank * 100 + j * 10 + cycle * 1000;
        buffer[static_cast<std::size_t>(j * 3 + 2)] = rank * 100 + j * 10 + 6 + cycle * 1000;
      }
      r.start();
      r.wait();
      for (int j = 0; j < size; ++j) {
        BOOST_TEST(buffer[static_cast<std::size_t>(j * 3)] ==
                   j * 100 + rank * 10 + cycle * 1000);
        BOOST_TEST(buffer[static_cast<std::size_t>(j * 3 + 2)] ==
                   j * 100 + rank * 10 + 6 + cycle * 1000);
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(reductions) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const auto plus = [](int a, int b) noexcept { return a + b; };
  const mp1::layout<int> l = holey_two();

  // reduce, flat and holey, plus in-place.
  {
    int send = -1;
    int recv = -1;
    mp1::request r;
    if (rank == 0)
      r = mp1::reduce_init(c, plus, 0, send, recv);
    else
      r = mp1::reduce_init(c, plus, 0, send, std::span<int>{});
    for (int cycle = 0; cycle < 2; ++cycle) {
      send = rank + 1 + cycle * 10;
      r.start();
      r.wait();
      if (rank == 0)
        BOOST_TEST(recv == size * (size + 1) / 2 + cycle * 10 * size);
    }
  }
  {
    std::array<int, 3> send{{-1, -1, -1}};
    std::array<int, 3> recv{{-1, -1, -1}};
    mp1::region<int> send_region{send.data(), l};
    mp1::region<int> recv_region{recv.data(), l};
    mp1::request r;
    if (rank == 0)
      r = mp1::reduce_init(c, plus, 0, send_region, recv_region);
    else
      r = mp1::reduce_init(c, plus, 0, send_region, std::span<int>{});
    for (int cycle = 0; cycle < 2; ++cycle) {
      send[0] = rank + 1;
      send[2] = rank + 2 + cycle;
      r.start();
      r.wait();
      if (rank == 0) {
        BOOST_TEST(recv[0] == size * (size + 1) / 2);
        BOOST_TEST(recv[2] == size * (size + 3) / 2 + cycle * size);
      }
    }
  }
  {
    int recv = -1;
    int send = -1;
    if (rank == 0) {
      mp1::request r = mp1::reduce_init(c, plus, 0, mp1::in_place, recv);
      for (int cycle = 0; cycle < 2; ++cycle) {
        recv = 1 + cycle * 10;
        r.start();
        r.wait();
        BOOST_TEST(recv == size * (size + 1) / 2 + cycle * 10 * size);
      }
    } else {
      mp1::request r = mp1::reduce_init(c, plus, 0, send, std::span<int>{});
      for (int cycle = 0; cycle < 2; ++cycle) {
        send = rank + 1 + cycle * 10;
        r.start();
        r.wait();
      }
    }
  }

  // allreduce, flat, holey and in-place.
  {
    int send = -1;
    int recv = -1;
    mp1::request r = mp1::allreduce_init(c, plus, send, recv);
    for (int cycle = 0; cycle < 2; ++cycle) {
      send = rank + 1 + cycle * 10;
      r.start();
      r.wait();
      BOOST_TEST(recv == size * (size + 1) / 2 + cycle * 10 * size);
    }
  }
  {
    std::array<int, 3> send{{-1, -1, -1}};
    std::array<int, 3> recv{{-1, -1, -1}};
    mp1::region<int> send_region{send.data(), l};
    mp1::region<int> recv_region{recv.data(), l};
    mp1::request r = mp1::allreduce_init(c, plus, send_region, recv_region);
    for (int cycle = 0; cycle < 2; ++cycle) {
      send[0] = rank + 1;
      send[2] = rank + 5 + cycle;
      r.start();
      r.wait();
      BOOST_TEST(recv[0] == size * (size + 1) / 2);
      BOOST_TEST(recv[2] == size * (size + 9) / 2 + cycle * size);
    }
  }
  {
    int recv = -1;
    mp1::request r = mp1::allreduce_init(c, plus, mp1::in_place, recv);
    for (int cycle = 0; cycle < 2; ++cycle) {
      recv = rank + 1 + cycle * 10;
      r.start();
      r.wait();
      BOOST_TEST(recv == size * (size + 1) / 2 + cycle * 10 * size);
    }
  }

  // scan and exscan, flat.
  {
    int send = -1;
    int recv = -1;
    mp1::request r = mp1::scan_init(c, plus, send, recv);
    for (int cycle = 0; cycle < 2; ++cycle) {
      send = rank + 1 + cycle * 10;
      r.start();
      r.wait();
      BOOST_TEST(recv == (rank + 1) * (rank + 2) / 2 + cycle * 10 * (rank + 1));
    }
  }
  {
    int send = -1;
    int recv = -1;
    mp1::request r = mp1::exscan_init(c, plus, send, recv);
    for (int cycle = 0; cycle < 2; ++cycle) {
      send = rank + 1 + cycle * 10;
      r.start();
      r.wait();
      if (rank > 0)
        BOOST_TEST(recv == rank * (rank + 1) / 2 + cycle * 10 * rank);
    }
  }

  // scan and exscan, holey and in-place.
  {
    std::array<int, 3> send{{-1, -1, -1}};
    std::array<int, 3> recv{{-1, -1, -1}};
    mp1::region<int> send_region{send.data(), l};
    mp1::region<int> recv_region{recv.data(), l};
    mp1::request r = mp1::scan_init(c, plus, send_region, recv_region);
    for (int cycle = 0; cycle < 2; ++cycle) {
      send[0] = rank + 1;
      send[2] = rank + 2;
      r.start();
      r.wait();
      BOOST_TEST(recv[0] == (rank + 1) * (rank + 2) / 2);
      BOOST_TEST(recv[2] == (rank + 1) * (rank + 4) / 2);
      (void)cycle;
    }
  }
  {
    std::array<int, 3> recv{{-1, -1, -1}};
    mp1::region<int> recv_region{recv.data(), l};
    mp1::request r = mp1::scan_init(c, plus, mp1::in_place, recv_region);
    for (int cycle = 0; cycle < 2; ++cycle) {
      recv[0] = rank + 1 + cycle * 10;
      recv[2] = rank + 2 + cycle * 10;
      r.start();
      r.wait();
      BOOST_TEST(recv[0] == (rank + 1) * (rank + 2) / 2 + cycle * 10 * (rank + 1));
      BOOST_TEST(recv[2] == (rank + 1) * (rank + 4) / 2 + cycle * 10 * (rank + 1));
    }
  }
  {
    std::array<int, 3> recv{{-1, -1, -1}};
    mp1::region<int> recv_region{recv.data(), l};
    mp1::request r = mp1::exscan_init(c, plus, mp1::in_place, recv_region);
    for (int cycle = 0; cycle < 2; ++cycle) {
      recv[0] = rank + 1 + cycle * 10;
      recv[2] = rank + 3 + cycle * 10;
      r.start();
      r.wait();
      if (rank > 0) {
        BOOST_TEST(recv[0] == rank * (rank + 1) / 2 + cycle * 10 * rank);
        BOOST_TEST(recv[2] == rank * (rank + 5) / 2 + cycle * 10 * rank);
      }
    }
  }

  // reduce_scatter: counts are base-element counts, the region layout describes the whole
  // buffer, and a non-flat layout is staged at initiation.
  {
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    int recv = -1;
    mp1::request r = mp1::reduce_scatter_init(c, plus, std::span<int>{send},
                                              std::span<int>{&recv, 1}, counts);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j)
        send[static_cast<std::size_t>(j)] = rank * 10 + j + cycle * 100;
      r.start();
      r.wait();
      BOOST_TEST(recv == 10 * sum_of_ranks(size) + rank * size + cycle * 100 * size);
    }
  }
  {
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> buffer(static_cast<std::size_t>(size), -1);
    mp1::request r =
        mp1::reduce_scatter_init(c, plus, mp1::in_place, std::span<int>{buffer}, counts);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j)
        buffer[static_cast<std::size_t>(j)] = rank * 10 + j + cycle * 100;
      r.start();
      r.wait();
      BOOST_TEST(buffer[0] == 10 * sum_of_ranks(size) + rank * size + cycle * 100 * size);
    }
  }
  {
    // One element per rank, each at element offset 2*e+1 of the whole send buffer; the local
    // receive layout maps its single element at offset 1.
    mp1::indexed_parameter send_parameter;
    for (int e = 0; e < size; ++e)
      send_parameter.add(1, static_cast<mp1::count_t>(2 * e + 1));
    const mp1::layout<int> send_layout = mp1::layout<int>::indexed(send_parameter);
    mp1::indexed_parameter recv_parameter;
    recv_parameter.add(1, 1);
    const mp1::layout<int> recv_layout = mp1::layout<int>::indexed(recv_parameter);
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> send(static_cast<std::size_t>(size * 2), -1);
    std::array<int, 2> recv{{-1, -1}};
    mp1::region<int> send_region{send.data(), send_layout};
    mp1::region<int> recv_region{recv.data(), recv_layout};
    mp1::request r = mp1::reduce_scatter_init(c, plus, send_region, recv_region, counts);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j) {
        send[static_cast<std::size_t>(2 * j + 1)] = rank * 10 + j + cycle * 100;
      }
      r.start();
      r.wait();
      BOOST_TEST(recv[1] == 10 * sum_of_ranks(size) + rank * size + cycle * 100 * size);
      BOOST_TEST(recv[0] == -1);
    }
  }

  // reduce_scatter_block: scalar per-rank element count, whole-buffer region layouts.
  {
    std::vector<int> send(static_cast<std::size_t>(size), -1);
    int recv = -1;
    mp1::request r = mp1::reduce_scatter_block_init(c, plus, std::span<int>{send},
                                                    std::span<int>{&recv, 1}, 1);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j)
        send[static_cast<std::size_t>(j)] = rank * 10 + j + cycle * 100;
      r.start();
      r.wait();
      BOOST_TEST(recv == 10 * sum_of_ranks(size) + rank * size + cycle * 100 * size);
    }
  }
  {
    std::vector<int> buffer(static_cast<std::size_t>(size), -1);
    mp1::request r =
        mp1::reduce_scatter_block_init(c, plus, mp1::in_place, std::span<int>{buffer}, 1);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j)
        buffer[static_cast<std::size_t>(j)] = rank * 10 + j + cycle * 100;
      r.start();
      r.wait();
      BOOST_TEST(buffer[0] == 10 * sum_of_ranks(size) + rank * size + cycle * 100 * size);
    }
  }
  {
    // Two elements per rank; the aggregate send layout uses two mapped positions per
    // three-element group and the local receive layout is holey_two().
    const mp1::layout<int> aggregate = aggregate_holey(size);
    std::vector<int> send(static_cast<std::size_t>(size * 3), -1);
    std::array<int, 3> recv{{-1, -1, -1}};
    mp1::region<int> send_region{send.data(), aggregate};
    mp1::region<int> recv_region{recv.data(), l};
    mp1::request r = mp1::reduce_scatter_block_init(c, plus, send_region, recv_region, 2);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < size; ++j) {
        send[static_cast<std::size_t>(j * 3)] = rank * 10 + j + cycle * 100;
        send[static_cast<std::size_t>(j * 3 + 2)] = rank * 100 + j + cycle * 100;
      }
      r.start();
      r.wait();
      BOOST_TEST(recv[0] == 10 * sum_of_ranks(size) + rank * size + cycle * 100 * size);
      BOOST_TEST(recv[2] == 100 * sum_of_ranks(size) + rank * size + cycle * 100 * size);
    }
  }
}

BOOST_AUTO_TEST_CASE(reductions_native_and_standard_operators) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  const int sum_expected = size * (size + 1) / 2;
  int product_expected = 1;
  for (int i = 1; i <= size; ++i)
    product_expected *= i;

  // reduce_init with a native MPI_Op.
  {
    int send = rank + 1;
    int recv = -1;
    mp1::request r;
    if (rank == 0)
      r = mp1::reduce_init(c, MPI_SUM, 0, send, recv);
    else
      r = mp1::reduce_init(c, MPI_SUM, 0, send, std::span<int>{});
    r.start();
    r.wait();
    if (rank == 0)
      BOOST_TEST(recv == sum_expected);
  }

  // allreduce_init with a standard functor (mapped to MPI_SUM).
  {
    int send = rank + 1;
    int recv = -1;
    mp1::request r = mp1::allreduce_init(c, std::plus<int>{}, send, recv);
    r.start();
    r.wait();
    BOOST_TEST(recv == sum_expected);
  }

  // allreduce_init with std::multiplies<int> (mapped to MPI_PROD).
  {
    int send = rank + 1;
    int recv = -1;
    mp1::request r = mp1::allreduce_init(c, std::multiplies<int>{}, send, recv);
    r.start();
    r.wait();
    BOOST_TEST(recv == product_expected);
  }

  // scan_init with a native MPI_Op.
  {
    int send = rank + 1;
    int recv = -1;
    mp1::request r = mp1::scan_init(c, MPI_SUM, send, recv);
    r.start();
    r.wait();
    BOOST_TEST(recv == (rank + 1) * (rank + 2) / 2);
  }

  // exscan_init with a built-in functor (mapped to MPI_MAX); rank 0's result is undefined.
  {
    int send = rank;
    int recv = -1;
    mp1::request r = mp1::exscan_init(c, mp1::max<int>{}, send, recv);
    r.start();
    r.wait();
    if (rank > 0)
      BOOST_TEST(recv == rank - 1);
  }

  // reduce_scatter_init with a standard functor.
  {
    const std::vector<mp1::count_t> counts(static_cast<std::size_t>(size), 1);
    std::vector<int> send(static_cast<std::size_t>(size), rank + 1);
    int recv = -1;
    mp1::request r = mp1::reduce_scatter_init(c, std::plus<int>{}, std::span<int>{send},
                                              std::span<int>{&recv, 1}, counts);
    r.start();
    r.wait();
    BOOST_TEST(recv == sum_expected);
  }

  // reduce_scatter_block_init with a native MPI_Op.
  {
    std::vector<int> send(static_cast<std::size_t>(size), rank + 1);
    int recv = -1;
    mp1::request r = mp1::reduce_scatter_block_init(c, MPI_SUM, std::span<int>{send},
                                                    std::span<int>{&recv, 1}, mp1::count_t{1});
    r.start();
    r.wait();
    BOOST_TEST(recv == sum_expected);
  }

  // The persistent init path uses the same local MPI_OP_NULL validation.
  {
    int send = rank;
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)mp1::reduce_init(c, MPI_OP_NULL, 0, send, std::span<int>{}); }));
  }
}

// ---------------------------------------------------------------------------------------------
// Intercommunicator: the section 6.2.2 families
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(intercommunicator_collectives) {
  const auto world = mp1::comm_world();
  const int group = world.rank() % 2;
  mp1::communicator inter = make_intercommunicator(world, next_inter_tag());
  BOOST_REQUIRE(inter.is_inter());
  const int local_size = inter.size();
  const int remote_size = inter.remote_size();
  const int local_rank = inter.rank();
  const auto plus = [](int a, int b) noexcept { return a + b; };

  // Barrier: both groups start and complete in step.
  {
    mp1::request r = mp1::barrier_init(inter);
    for (int cycle = 0; cycle < 2; ++cycle) {
      r.start();
      r.wait();
    }
  }

  // Broadcast from the root of group 0; the other group receives.
  {
    int value = -1;
    const int root = group == 0 ? (local_rank == 0 ? MPI_ROOT : MPI_PROC_NULL) : 0;
    mp1::request r = mp1::bcast_init(inter, root, value);
    for (int cycle = 0; cycle < 2; ++cycle) {
      if (group == 0 and local_rank == 0)
        value = 500 + cycle;
      r.start();
      r.wait();
      if (group == 1)
        BOOST_TEST(value == 500 + cycle);
    }
  }

  // Gather into the root of group 0: it receives one element per remote rank.  In-place is
  // not defined on intercommunicators, so the root also passes an (ignored) send value.
  {
    std::vector<int> recv;
    int send = -1;
    const int root = group == 0 ? (local_rank == 0 ? MPI_ROOT : MPI_PROC_NULL) : 0;
    if (group == 0 and local_rank == 0)
      recv.assign(static_cast<std::size_t>(remote_size), -1);
    mp1::request r;
    if (group == 0 and local_rank == 0)
      r = mp1::gather_init(inter, root, send, std::span<int>{recv});
    else
      r = mp1::gather_init(inter, root, send, std::span<int>{});
    for (int cycle = 0; cycle < 2; ++cycle) {
      if (group == 1)
        send = local_rank * 10 + cycle;
      r.start();
      r.wait();
      if (group == 0 and local_rank == 0)
        for (int i = 0; i < remote_size; ++i)
          BOOST_TEST(recv[static_cast<std::size_t>(i)] == i * 10 + cycle);
    }
  }

  // Allgather across the intercommunicator: each rank receives one element per remote rank.
  {
    std::vector<int> recv(static_cast<std::size_t>(remote_size), -1);
    int send = -1;
    mp1::request r = mp1::allgather_init(inter, send, std::span<int>{recv});
    for (int cycle = 0; cycle < 2; ++cycle) {
      send = local_rank * 10 + cycle;
      r.start();
      r.wait();
      for (int i = 0; i < remote_size; ++i)
        BOOST_TEST(recv[static_cast<std::size_t>(i)] == i * 10 + cycle);
    }
  }

  // Alltoall across the intercommunicator: each rank sends one element per remote rank.
  {
    std::vector<int> send(static_cast<std::size_t>(remote_size), -1);
    std::vector<int> recv(static_cast<std::size_t>(remote_size), -1);
    // The remote rank order is not derivable locally; fetch it with a blocking allgather.
    std::vector<int> remote_ranks(static_cast<std::size_t>(remote_size), -1);
    mp1::allgather(inter, local_rank, std::span<int>{remote_ranks});
    mp1::request r = mp1::alltoall_init(inter, std::span<int>{send}, std::span<int>{recv});
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < remote_size; ++j)
        send[static_cast<std::size_t>(j)] = local_rank * 100 + j + cycle * 1000;
      r.start();
      r.wait();
      for (int j = 0; j < remote_size; ++j)
        BOOST_TEST(recv[static_cast<std::size_t>(j)] ==
                   remote_ranks[static_cast<std::size_t>(j)] * 100 + local_rank + cycle * 1000);
    }
  }

  // Reduce and allreduce: compare against the blocking implementation on the same data.
  {
    int send = -1;
    int recv = -1;
    int expected = -1;
    const int root = group == 0 ? (local_rank == 0 ? MPI_ROOT : MPI_PROC_NULL) : 0;
    const bool is_root = group == 0 and local_rank == 0;
    mp1::request r;
    if (is_root)
      r = mp1::reduce_init(inter, plus, root, send, recv);
    else
      r = mp1::reduce_init(inter, plus, root, send, std::span<int>{});
    for (int cycle = 0; cycle < 2; ++cycle) {
      send = local_rank + 1 + cycle * 10;
      // Blocking reference: every rank calls it (MPI_Reduce is collective over the intercomm).
      if (is_root) {
        int blocking_recv = -1;
        mp1::reduce(inter, plus, root, send, blocking_recv);
        expected = blocking_recv;
      } else {
        mp1::reduce(inter, plus, root, send, std::span<int>{});
      }
      r.start();
      r.wait();
      if (is_root)
        BOOST_TEST(recv == expected);
    }
  }
  {
    int send = -1;
    int recv = -1;
    int expected = -1;
    mp1::request r = mp1::allreduce_init(inter, plus, send, recv);
    for (int cycle = 0; cycle < 2; ++cycle) {
      send = local_rank + 1 + cycle * 10;
      mp1::allreduce(inter, plus, send, expected);
      r.start();
      r.wait();
      BOOST_TEST(recv == expected);
    }
  }

  // reduce_scatter: both groups provide the same total count, but the per-rank counts may
  // differ (especially in the 5-rank 3+2 nightly split).
  {
    const int total = std::max(local_size, remote_size);
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(local_size), 1);
    counts[0] = 1 + (total - local_size);
    std::vector<int> send(static_cast<std::size_t>(total), -1);
    std::vector<int> recv(
        static_cast<std::size_t>(counts[static_cast<std::size_t>(local_rank)]), -1);
    std::vector<int> expected(recv.size(), -1);
    mp1::request r = mp1::reduce_scatter_init(inter, plus, std::span<int>{send},
                                              std::span<int>{recv}, counts);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < total; ++j)
        send[static_cast<std::size_t>(j)] = local_rank * 10 + j + cycle * 100;
      mp1::reduce_scatter(inter, plus, std::span<int>{send}, std::span<int>{expected}, counts);
      r.start();
      r.wait();
      BOOST_TEST(recv == expected);
    }
  }

  // reduce_scatter_block needs equal group sizes (recvcount * group size must match).
  if (local_size == remote_size) {
    std::vector<int> send(static_cast<std::size_t>(local_size), -1);
    int recv = -1;
    int expected = -1;
    mp1::request r = mp1::reduce_scatter_block_init(inter, plus, std::span<int>{send},
                                                    std::span<int>{&recv, 1}, 1);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < local_size; ++j)
        send[static_cast<std::size_t>(j)] = local_rank * 10 + j + cycle * 100;
      mp1::reduce_scatter_block(inter, plus, std::span<int>{send}, std::span<int>{&expected, 1},
                                1);
      r.start();
      r.wait();
      BOOST_TEST(recv == expected);
    }
  }
}

BOOST_AUTO_TEST_CASE(intercommunicator_variable_and_scatter) {
  const auto world = mp1::comm_world();
  const int group = world.rank() % 2;
  mp1::communicator inter = make_intercommunicator(world, next_inter_tag());
  BOOST_REQUIRE(inter.is_inter());
  const int local_size = inter.size();
  const int remote_size = inter.remote_size();
  const int local_rank = inter.rank();
  (void)local_size;
  const mp1::layout<int> l = holey_two();
  std::vector<mp1::layout<int>> remote_layouts(static_cast<std::size_t>(remote_size), l);
  std::vector<mp1::count_t> remote_displs(static_cast<std::size_t>(remote_size));
  for (int i = 0; i < remote_size; ++i)
    remote_displs[static_cast<std::size_t>(i)] = i * 3;

  // Scatter: group-0 root sends one flat element per remote rank.
  {
    const int root = group == 0 ? (local_rank == 0 ? MPI_ROOT : MPI_PROC_NULL) : 0;
    std::vector<int> send;
    if (group == 0 and local_rank == 0)
      send.assign(static_cast<std::size_t>(remote_size), -1);
    int recv = -1;
    int ignored = 0;
    mp1::request r;
    if (group == 0 and local_rank == 0)
      r = mp1::scatter_init(inter, root, std::span<const int>{send}, ignored);
    else if (group == 0)
      r = mp1::scatter_init(inter, root, std::span<const int>{}, ignored);
    else
      r = mp1::scatter_init(inter, root, std::span<const int>{}, recv);
    for (int cycle = 0; cycle < 2; ++cycle) {
      if (group == 0 and local_rank == 0)
        for (int i = 0; i < remote_size; ++i)
          send[static_cast<std::size_t>(i)] = 2000 + i + cycle * 100;
      r.start();
      r.wait();
      if (group == 1)
        BOOST_TEST(recv == 2000 + local_rank + cycle * 100);
    }
  }

  // Gatherv: group-1 ranks send a holey block; the group-0 root receives one per remote rank.
  {
    const int root = group == 0 ? (local_rank == 0 ? MPI_ROOT : MPI_PROC_NULL) : 0;
    std::array<int, 3> send{{-1, -1, -1}};
    std::vector<int> recv;
    if (group == 0 and local_rank == 0)
      recv.assign(static_cast<std::size_t>(remote_size * 3), -1);
    int ignored = -1;
    std::vector<mp1::layout<int>> empty_layouts;
    std::vector<mp1::count_t> empty_displs;
    const std::span<int> recv_span{recv};
    mp1::request r =
        group == 1
            ? mp1::gatherv_init(inter, root, mp1::region{send.data(), l}, std::span<int>{},
                                mp1::blocks{})
            : (local_rank == 0 ? mp1::gatherv_init(inter, root, ignored, recv_span,
                                                   remote_layouts, remote_displs)
                               : mp1::gatherv_init(inter, root, ignored, std::span<int>{},
                                                   std::span<const mp1::layout<int>>{},
                                                   std::span<const mp1::count_t>{}));
    for (int cycle = 0; cycle < 2; ++cycle) {
      if (group == 1) {
        send[0] = 3000 + local_rank + cycle * 100;
        send[2] = 3100 + local_rank + cycle * 100;
      }
      r.start();
      r.wait();
      if (group == 0 and local_rank == 0) {
        for (int i = 0; i < remote_size; ++i) {
          BOOST_TEST(recv[static_cast<std::size_t>(i * 3)] == 3000 + i + cycle * 100);
          BOOST_TEST(recv[static_cast<std::size_t>(i * 3 + 2)] == 3100 + i + cycle * 100);
        }
      }
    }
  }

  // Scatterv: group-0 root sends a holey block to every group-1 rank.
  {
    const int root = group == 0 ? (local_rank == 0 ? MPI_ROOT : MPI_PROC_NULL) : 0;
    const std::vector<mp1::layout<int>> send_layouts(static_cast<std::size_t>(remote_size), l);
    std::vector<int> send;
    if (group == 0 and local_rank == 0)
      send.assign(static_cast<std::size_t>(remote_size * 3), -1);
    std::array<int, 3> recv{{-1, -1, -1}};
    int ignored = 0;
    std::vector<mp1::layout<int>> empty_layouts;
    std::vector<mp1::count_t> empty_displs;
    const std::span<const int> send_span{send};
    mp1::request r =
        group == 1
            ? mp1::scatterv_init(inter, root, std::span<const int>{}, mp1::blocks{},
                                 mp1::region{recv.data(), l})
            : (local_rank == 0 ? mp1::scatterv_init(inter, root, send_span, send_layouts,
                                                    remote_displs, ignored)
                               : mp1::scatterv_init(inter, root, std::span<const int>{},
                                                    std::span<const mp1::layout<int>>{},
                                                    std::span<const mp1::count_t>{}, ignored));
    for (int cycle = 0; cycle < 2; ++cycle) {
      if (group == 0 and local_rank == 0) {
        for (int i = 0; i < remote_size; ++i) {
          send[static_cast<std::size_t>(i * 3)] = 4000 + i + cycle * 100;
          send[static_cast<std::size_t>(i * 3 + 2)] = 4100 + i + cycle * 100;
        }
      }
      r.start();
      r.wait();
      if (group == 1) {
        BOOST_TEST(recv[0] == 4000 + local_rank + cycle * 100);
        BOOST_TEST(recv[2] == 4100 + local_rank + cycle * 100);
      }
    }
  }

  // Allgatherv: each group receives the remote group's holey contributions.
  {
    std::array<int, 3> send{{-1, -1, -1}};
    std::vector<int> recv(static_cast<std::size_t>(remote_size * 3), -1);
    const std::span<int> recv_span{recv};
    mp1::request r = mp1::allgatherv_init(inter, mp1::region{send.data(), l}, recv_span,
                                          remote_layouts, remote_displs);
    for (int cycle = 0; cycle < 2; ++cycle) {
      send[0] = 5000 + local_rank + cycle * 100;
      send[2] = 5100 + local_rank + cycle * 100;
      r.start();
      r.wait();
      for (int i = 0; i < remote_size; ++i) {
        BOOST_TEST(recv[static_cast<std::size_t>(i * 3)] == 5000 + i + cycle * 100);
        BOOST_TEST(recv[static_cast<std::size_t>(i * 3 + 2)] == 5100 + i + cycle * 100);
      }
    }
  }

  // Alltoallv: one holey block per remote rank in each direction.
  {
    const std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(remote_size), l);
    std::vector<int> send(static_cast<std::size_t>(remote_size * 3), -1);
    std::vector<int> recv(static_cast<std::size_t>(remote_size * 3), -1);
    const std::span<const int> send_span{send};
    const std::span<int> recv_span{recv};
    mp1::request r = mp1::alltoallv_init(inter, send_span, layouts, remote_displs, recv_span,
                                         layouts, remote_displs);
    for (int cycle = 0; cycle < 2; ++cycle) {
      for (int j = 0; j < remote_size; ++j) {
        send[static_cast<std::size_t>(j * 3)] = 6000 + local_rank * 100 + j + cycle * 1000;
        send[static_cast<std::size_t>(j * 3 + 2)] = 6100 + local_rank * 100 + j + cycle * 1000;
      }
      r.start();
      r.wait();
      for (int j = 0; j < remote_size; ++j) {
        BOOST_TEST(recv[static_cast<std::size_t>(j * 3)] ==
                   6000 + j * 100 + local_rank + cycle * 1000);
        BOOST_TEST(recv[static_cast<std::size_t>(j * 3 + 2)] ==
                   6100 + j * 100 + local_rank + cycle * 1000);
      }
    }
  }
  world_barrier();
}

// ---------------------------------------------------------------------------------------------
// Local validation and the info argument
// ---------------------------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(local_validation) {
  const auto c = mp1::comm_world();
  const int rank = c.rank();
  int value = 0;

  // Invalid root / negative count / null buffer are rejected before any MPI collective, so
  // only the offending rank needs to call them.
  if (rank == 0) {
    BOOST_TEST(throws_invalid_argument([&] { (void)mp1::bcast_init(c, c.size(), value); }));
    BOOST_TEST(throws_invalid_argument([&] { (void)mp1::bcast_init(c, -1, value); }));
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)mp1::bcast_init(c, 0, mp1::span_from(&value, -1)); }));
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)mp1::bcast_init(c, 0, std::span<int>{static_cast<int*>(nullptr), 1}); }));
    BOOST_TEST(throws_invalid_argument([&] { (void)mp1::barrier_init(mp1::comm_null()); }));
  }
  world_barrier();

  // The init info argument is passed through; a null info and a populated info both work.
  {
    // A benign application hint: the library passes the info handle through unchanged.
    // NOTE: Open MPI 6.0.0rc2 rejects MPI_Info_set of the MPI 4.1 key
    // mpi_assert_strict_persistent_collective_ordering on a generic info object
    // (MPI_ERR_INFO_KEY with MPI_ERRORS_ARE_FATAL), so the test cannot create that exact info
    // here.
    mp1::info hints;
    hints.set("mp1_test_hint", "1");
    int root_value = 0;
    mp1::request r = mp1::bcast_init(c, 0, root_value, hints);
    if (rank == 0)
      root_value = 77;
    r.start();
    r.wait();
    BOOST_TEST(root_value == 77);
  }
}

BOOST_AUTO_TEST_SUITE_END()
