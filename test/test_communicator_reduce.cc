// Reduce tests: free-function API with a non-zero root, root in-place, non-root form,
// buffer/container/layout forms, staged derived layouts, and blocking/nonblocking requests.

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

  struct add {
    int operator()(const int& left, const int& right) const noexcept {
      return left + right;
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


BOOST_AUTO_TEST_CASE(reduce_scalar_root_forms) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int root = size - 1;
  const int expected = rank_sum(size);
  const std::plus<int> plus{};

  // Root form on every rank: non-root receive buffers are ignored and stay untouched.
  {
    int recv = -12345;
    mp1::reduce(comm, plus, root, rank, recv);
    if (rank == root)
      BOOST_TEST(recv == expected);
    else
      BOOST_TEST(recv == -12345);
  }

  // Non-root form on the non-root ranks.
  {
    if (rank != root)
      mp1::reduce(comm, plus, root, rank, std::span<int>{});
    else {
      int recv = -1;
      mp1::reduce(comm, plus, root, rank, recv);
      BOOST_TEST(recv == expected);
    }
  }

  // Root in-place form: the root's contribution is already in its receive buffer.
  {
    int recv = rank;
    if (rank == root) {
      mp1::reduce(comm, plus, root, mp1::in_place, recv);
      BOOST_TEST(recv == expected);
    } else {
      mp1::reduce(comm, plus, root, rank, std::span<int>{});
    }
  }

  // Non-root form is invalid on the root rank (thrown before MPI).  No rank enters MPI in this
  // block, so all ranks stay in step for the next collective.
  if (rank == root) {
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::reduce(comm, plus, root, rank, std::span<int>{}); }));
  }

  // A different non-zero root exercises the root-role resolution again.
  {
    const int other_root = size > 1 ? 1 : 0;
    int recv = -1;
    if (rank == other_root) {
      mp1::reduce(comm, plus, other_root, rank, recv);
      BOOST_TEST(recv == expected);
    } else {
      mp1::reduce(comm, plus, other_root, rank, std::span<int>{});
    }
  }

  // User-defined stateless functor.
  {
    const add int_add{};
    int recv = -1;
    if (rank == root) {
      mp1::reduce(comm, int_add, root, rank, recv);
      BOOST_TEST(recv == expected);
    } else {
      mp1::reduce(comm, int_add, root, rank, std::span<int>{});
    }
  }

  // Native predefined MPI_Op.
  {
    int recv = -1;
    if (rank == root) {
      mp1::reduce(comm, MPI_SUM, root, rank, recv);
      BOOST_TEST(recv == expected);
    } else {
      mp1::reduce(comm, MPI_SUM, root, rank, std::span<int>{});
    }
  }

  // Standard-library functors: std::plus<int> maps to MPI_SUM, mp1::max<int> to MPI_MAX.
  {
    int recv = -1;
    if (rank == root) {
      mp1::reduce(comm, std::plus<int>{}, root, rank, recv);
      BOOST_TEST(recv == expected);
    } else {
      mp1::reduce(comm, std::plus<int>{}, root, rank, std::span<int>{});
    }
  }
  {
    int recv = -1;
    if (rank == root) {
      mp1::reduce(comm, mp1::max<int>{}, root, rank, recv);
      BOOST_TEST(recv == size - 1);
    } else {
      mp1::reduce(comm, mp1::max<int>{}, root, rank, std::span<int>{});
    }
  }

  // Local adapter validation: no rank enters MPI, so the collective stays in step.
  {
    int recv = -1;
    BOOST_TEST(
        throws_invalid_argument([&] { mp1::reduce(comm, MPI_OP_NULL, root, rank, recv); }));
  }
}


BOOST_AUTO_TEST_CASE(reduce_buffers_containers_and_padded) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int root = size - 1;
  const int expected = rank_sum(size);
  const std::plus<int> plus{};

  // Buffer out-of-place form with a non-zero root.
  {
    const std::vector<int> send{rank, 2 * rank};
    std::vector<int> recv(send.size(), -1);
    mp1::reduce(comm, plus, root, std::span<const int>{send}, std::span<int>{recv});
    if (rank == root) {
      BOOST_TEST(recv[0] == expected);
      BOOST_TEST(recv[1] == 2 * expected);
    }
  }

  // Buffer in-place form (root only) and buffer non-root form.
  {
    std::vector<int> buffer{rank, 2 * rank};
    if (rank == root) {
      mp1::reduce(comm, plus, root, mp1::in_place, std::span<int>{buffer});
      BOOST_TEST(buffer[0] == expected);
      BOOST_TEST(buffer[1] == 2 * expected);
    } else {
      mp1::reduce(comm, plus, root, std::span<int>{buffer}, std::span<int>{});
    }
  }

  // Container forms.
  {
    const std::vector<int> send{rank, 2 * rank};
    std::vector<int> recv(send.size(), -1);
    if (rank == root) {
      mp1::reduce(comm, plus, root, send, recv);
      BOOST_TEST(recv[0] == expected);
      BOOST_TEST(recv[1] == 2 * expected);
    } else {
      mp1::reduce(comm, plus, root, send, std::span<int>{});
    }
  }
  {
    std::vector<int> buffer{rank, 2 * rank};
    if (rank == root) {
      mp1::reduce(comm, plus, root, mp1::in_place, buffer);
      BOOST_TEST(buffer[0] == expected);
      BOOST_TEST(buffer[1] == 2 * expected);
    } else {
      mp1::reduce(comm, plus, root, buffer, std::span<int>{});
    }
  }

  // Padded element type through the scalar root forms.
  {
    const padded_value send{rank, static_cast<double>(rank) / 2.0};
    padded_value recv{-1, -1.0};
    if (rank == root) {
      mp1::reduce(comm, padded_add{}, root, send, recv);
      BOOST_TEST(recv.first == expected);
      BOOST_TEST(recv.second == 0.5 * static_cast<double>(expected));
    } else {
      mp1::reduce(comm, padded_add{}, root, send, std::span<padded_value>{});
    }
  }
}


BOOST_AUTO_TEST_CASE(reduce_layouts) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int root = size - 1;
  const int expected = rank_sum(size);
  const std::plus<int> plus{};

  // Dense layout on the root (flat base-T path) and the non-root layout form elsewhere.
  const mp1::layout<int> dense = mp1::layout<int>::contiguous(3);
  {
    const std::vector<int> send{rank, rank + 1, rank + 2};
    std::vector<int> recv(send.size(), -1);
    if (rank == root) {
      mp1::reduce(comm, plus, root, mp1::region<const int>{send.data(), dense},
                  mp1::region<int>{recv.data(), dense});
      for (std::size_t j = 0; j < recv.size(); ++j)
        BOOST_TEST(recv[j] == expected + size * static_cast<int>(j));
    } else {
      mp1::reduce(comm, plus, root, mp1::region<const int>{send.data(), dense},
                  std::span<int>{});
    }
  }

  // Derived layout with a hole; the root receives through the staging path and only the
  // mapped elements are unpacked.
  const mp1::layout<int> holey = mp1::layout<int>::vector(2, 1, 2);
  {
    std::vector<int> send(3, -7);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    if (rank == root) {
      mp1::reduce(comm, plus, root, mp1::region<int>{send.data(), holey},
                  mp1::region<int>{recv.data(), holey});
      BOOST_TEST(recv[0] == expected);
      BOOST_TEST(recv[1] == -1);
      BOOST_TEST(recv[2] == expected + 10 * size);
    } else {
      mp1::reduce(comm, plus, root, mp1::region<int>{send.data(), holey}, std::span<int>{});
    }
  }

  // Root in-place layout forms (flat and staged).
  {
    std::vector<int> buffer{rank, rank + 1, rank + 2};
    if (rank == root) {
      mp1::reduce(comm, plus, root, mp1::in_place, mp1::region<int>{buffer.data(), dense});
      for (std::size_t j = 0; j < buffer.size(); ++j)
        BOOST_TEST(buffer[j] == expected + size * static_cast<int>(j));
    } else {
      mp1::reduce(comm, plus, root, mp1::region<int>{buffer.data(), dense}, std::span<int>{});
    }
  }
  {
    std::vector<int> buffer(3, -7);
    buffer[0] = rank;
    buffer[2] = rank + 10;
    if (rank == root) {
      mp1::reduce(comm, plus, root, mp1::in_place, mp1::region<int>{buffer.data(), holey});
      BOOST_TEST(buffer[0] == expected);
      BOOST_TEST(buffer[1] == -7);
      BOOST_TEST(buffer[2] == expected + 10 * size);
    } else {
      mp1::reduce(comm, plus, root, mp1::region<int>{buffer.data(), holey}, std::span<int>{});
    }
  }
}


BOOST_AUTO_TEST_CASE(reduce_nonblocking) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int root = size - 1;
  const int expected = rank_sum(size);
  const std::plus<int> plus{};

  // Scalar request, out-of-place on the root and non-root form elsewhere.
  {
    int recv = -1;
    if (rank == root) {
      mp1::request request = mp1::ireduce(comm, plus, root, rank, recv);
      request.wait();
      BOOST_TEST(recv == expected);
    } else {
      mp1::request request = mp1::ireduce(comm, plus, root, rank, std::span<int>{});
      request.wait();
    }
  }

  // Native MPI_Op and standard functor through the nonblocking path.
  {
    int recv = -1;
    if (rank == root) {
      mp1::request request = mp1::ireduce(comm, MPI_SUM, root, rank, recv);
      request.wait();
      BOOST_TEST(recv == expected);
    } else {
      mp1::request request = mp1::ireduce(comm, MPI_SUM, root, rank, std::span<int>{});
      request.wait();
    }
  }
  {
    int recv = -1;
    if (rank == root) {
      mp1::request request = mp1::ireduce(comm, std::plus<int>{}, root, rank, recv);
      request.wait();
      BOOST_TEST(recv == expected);
    } else {
      mp1::request request = mp1::ireduce(comm, std::plus<int>{}, root, rank, std::span<int>{});
      request.wait();
    }
  }

  // Root in-place request.
  {
    int recv = rank;
    if (rank == root) {
      mp1::request request = mp1::ireduce(comm, plus, root, mp1::in_place, recv);
      const mp1::status status = request.wait();
      BOOST_TEST(status.error() == MPI_SUCCESS);
      BOOST_TEST(recv == expected);
    } else {
      mp1::request request = mp1::ireduce(comm, plus, root, rank, std::span<int>{});
      request.wait();
    }
  }

  // Layout request with a derived layout; the layout outlives the request.
  {
    const mp1::layout<int> holey = mp1::layout<int>::vector(2, 1, 2);
    std::vector<int> send(3, -7);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    if (rank == root) {
      mp1::region<int> send_region{send.data(), holey};
      mp1::region<int> recv_region{recv.data(), holey};
      mp1::request request = mp1::ireduce(comm, plus, root, send_region, recv_region);
      request.wait();
      BOOST_TEST(recv[0] == expected);
      BOOST_TEST(recv[1] == -1);
      BOOST_TEST(recv[2] == expected + 10 * size);
    } else {
      mp1::request request = mp1::ireduce(
          comm, plus, root, mp1::region<int>{send.data(), holey}, std::span<int>{});
      request.wait();
    }
  }

  // Container request on the root and container non-root form elsewhere.
  {
    const std::vector<int> send{rank, 2 * rank};
    std::vector<int> recv(send.size(), -1);
    if (rank == root) {
      mp1::request request = mp1::ireduce(comm, plus, root, send, recv);
      request.wait();
      BOOST_TEST(recv[0] == expected);
      BOOST_TEST(recv[1] == 2 * expected);
    } else {
      mp1::request request = mp1::ireduce(comm, plus, root, send, std::span<int>{});
      request.wait();
    }
  }
}
