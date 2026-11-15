// Exclusive-scan tests: free-function API, scalar/buffer/container/layout forms, in-place
// forms, padded element type and nonblocking requests.  Rank 0's receive buffer is never
// written (mp1 chooses "unchanged" for the result MPI leaves undefined).

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
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

  // Exclusive prefix of the rank values 0..rank-1; rank 0 is empty (zero).
  [[nodiscard]] int exclusive_prefix(int rank) {
    return rank * (rank - 1) / 2;
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);


BOOST_AUTO_TEST_CASE(exscan_scalar_and_operators) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int expected = exclusive_prefix(rank);
  const std::plus<int> plus{};

  // Out-of-place scalar: rank 0's sentinel receive buffer is unchanged.
  int recv = 777;
  mp1::exscan(comm, plus, rank, recv);
  if (rank == 0)
    BOOST_TEST(recv == 777);
  else
    BOOST_TEST(recv == expected);

  // Temporary built-in functor and a user-defined stateless functor.
  recv = 777;
  mp1::exscan(comm, std::plus<int>{}, rank, recv);
  if (rank == 0)
    BOOST_TEST(recv == 777);
  else
    BOOST_TEST(recv == expected);

  recv = 777;
  const add int_add{};
  mp1::exscan(comm, int_add, rank, recv);
  if (rank == 0)
    BOOST_TEST(recv == 777);
  else
    BOOST_TEST(recv == expected);

  // Native predefined MPI_Op and standard-library functors.
  recv = 777;
  mp1::exscan(comm, MPI_SUM, rank, recv);
  if (rank == 0)
    BOOST_TEST(recv == 777);
  else
    BOOST_TEST(recv == expected);

  recv = 777;
  mp1::exscan(comm, std::plus<int>{}, rank, recv);
  if (rank == 0)
    BOOST_TEST(recv == 777);
  else
    BOOST_TEST(recv == expected);

  recv = 777;
  mp1::exscan(comm, std::plus<void>{}, rank, recv);
  if (rank == 0)
    BOOST_TEST(recv == 777);
  else
    BOOST_TEST(recv == expected);

  recv = 777;
  mp1::exscan(comm, mp1::max<int>{}, rank, recv);
  if (rank == 0)
    BOOST_TEST(recv == 777);
  else
    BOOST_TEST(recv == rank - 1);

  // In-place scalar: the input is the buffer value 100 + rank, so the exclusive prefix is
  // 100 * rank + exclusive_prefix(rank); rank 0 keeps its buffer.
  int in_place = 100 + rank;
  mp1::exscan(comm, plus, mp1::in_place, in_place);
  if (rank == 0)
    BOOST_TEST(in_place == 100);
  else
    BOOST_TEST(in_place == 100 * rank + expected);

  // Padded element type through the scalar form.
  const padded_value send{rank, 0.5 * rank};
  padded_value padded_recv{-1, -1.0};
  if (rank == 0) {
    padded_recv = padded_value{777, 7.5};
    mp1::exscan(comm, padded_add{}, send, padded_recv);
    BOOST_TEST(padded_recv.first == 777);
    BOOST_TEST(padded_recv.second == 7.5);
  } else {
    mp1::exscan(comm, padded_add{}, send, padded_recv);
    BOOST_TEST(padded_recv.first == expected);
    BOOST_TEST(padded_recv.second == 0.5 * static_cast<double>(expected));
  }
}


BOOST_AUTO_TEST_CASE(exscan_buffers_and_containers) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int expected = exclusive_prefix(rank);
  const std::plus<int> plus{};

  // Buffer form: element j is the exclusive prefix of (i + j) for i = 0..rank-1.
  {
    const std::vector<int> send{rank, rank + 1};
    std::vector<int> recv(send.size(), -1);
    mp1::exscan(comm, plus, std::span<const int>{send}, std::span<int>{recv});
    if (rank == 0) {
      BOOST_TEST(recv[0] == -1);
      BOOST_TEST(recv[1] == -1);
    } else {
      BOOST_TEST(recv[0] == expected);
      BOOST_TEST(recv[1] == expected + rank);
    }
  }

  // Buffer in-place form; rank 0 keeps every element of its buffer.
  {
    std::vector<int> buffer{100 + rank, 200 + rank};
    mp1::exscan(comm, plus, mp1::in_place, std::span<int>{buffer});
    if (rank == 0) {
      BOOST_TEST(buffer[0] == 100);
      BOOST_TEST(buffer[1] == 200);
    } else {
      BOOST_TEST(buffer[0] == 100 * rank + expected);
      BOOST_TEST(buffer[1] == 200 * rank + expected);
    }
  }

  // Contiguous-container forms.
  {
    const std::vector<int> send{rank, 2 * rank};
    std::vector<int> recv(send.size(), -1);
    if (rank == 0) {
      mp1::exscan(comm, plus, send, recv);
      BOOST_TEST(recv[0] == -1);
      BOOST_TEST(recv[1] == -1);
    } else {
      mp1::exscan(comm, plus, send, recv);
      BOOST_TEST(recv[0] == expected);
      BOOST_TEST(recv[1] == 2 * expected);
    }
  }
  {
    std::vector<int> buffer{rank, 2 * rank};
    mp1::exscan(comm, plus, mp1::in_place, buffer);
    if (rank == 0) {
      BOOST_TEST(buffer[0] == 0);
      BOOST_TEST(buffer[1] == 0);
    } else {
      BOOST_TEST(buffer[0] == expected);
      BOOST_TEST(buffer[1] == 2 * expected);
    }
  }
}


BOOST_AUTO_TEST_CASE(exscan_layouts) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int expected = exclusive_prefix(rank);
  const std::plus<int> plus{};

  // Dense contiguous layout.
  const mp1::layout<int> dense = mp1::layout<int>::contiguous(3);
  {
    const std::vector<int> send{rank, rank + 1, rank + 2};
    std::vector<int> recv(send.size(), -1);
    mp1::exscan(comm, plus, mp1::region<const int>{send.data(), dense},
                mp1::region<int>{recv.data(), dense});
    if (rank == 0) {
      for (const int value : recv)
        BOOST_TEST(value == -1);
    } else {
      for (std::size_t j = 0; j < recv.size(); ++j)
        BOOST_TEST(recv[j] == expected + static_cast<int>(j) * rank);
    }
  }

  // Derived layout with a hole; rank 0's mapped elements and every hole stay untouched.
  const mp1::layout<int> holey = mp1::layout<int>::vector(2, 1, 2);
  {
    std::vector<int> send(3, -7);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    mp1::exscan(comm, plus, mp1::region<int>{send.data(), holey},
                mp1::region<int>{recv.data(), holey});
    if (rank == 0) {
      BOOST_TEST(recv[0] == -1);
      BOOST_TEST(recv[1] == -1);
      BOOST_TEST(recv[2] == -1);
    } else {
      BOOST_TEST(recv[0] == expected);
      BOOST_TEST(recv[1] == -1);
      BOOST_TEST(recv[2] == expected + 10 * rank);
    }
  }

  // Padded element type with a multi-element contiguous layout.
  const mp1::layout<padded_value> padded = mp1::layout<padded_value>::contiguous(2);
  {
    const std::vector<padded_value> send{padded_value{rank, 0.5 * rank},
                                         padded_value{rank + 1, 0.5 * rank + 1.0}};
    std::vector<padded_value> recv(send.size(), padded_value{-1, -1.0});
    mp1::exscan(comm, padded_add{}, mp1::region<const padded_value>{send.data(), padded},
                mp1::region<padded_value>{recv.data(), padded});
    if (rank == 0) {
      BOOST_TEST(recv[0].first == -1);
      BOOST_TEST(recv[1].first == -1);
    } else {
      BOOST_TEST(recv[0].first == expected);
      BOOST_TEST(recv[0].second == 0.5 * static_cast<double>(expected));
      BOOST_TEST(recv[1].first == expected + rank);
      BOOST_TEST(recv[1].second == 0.5 * static_cast<double>(expected) + rank);
    }
  }

  // In-place layout forms (dense and staged); rank 0 keeps its buffer.
  {
    std::vector<int> buffer{rank, rank + 1, rank + 2};
    mp1::exscan(comm, plus, mp1::in_place, mp1::region<int>{buffer.data(), dense});
    if (rank == 0) {
      for (std::size_t j = 0; j < buffer.size(); ++j)
        BOOST_TEST(buffer[j] == static_cast<int>(j));
    } else {
      for (std::size_t j = 0; j < buffer.size(); ++j)
        BOOST_TEST(buffer[j] == expected + static_cast<int>(j) * rank);
    }
  }
  {
    std::vector<int> buffer(3, -7);
    buffer[0] = rank;
    buffer[2] = rank + 10;
    mp1::exscan(comm, plus, mp1::in_place, mp1::region<int>{buffer.data(), holey});
    if (rank == 0) {
      BOOST_TEST(buffer[0] == 0);
      BOOST_TEST(buffer[1] == -7);
      BOOST_TEST(buffer[2] == 10);
    } else {
      BOOST_TEST(buffer[0] == expected);
      BOOST_TEST(buffer[1] == -7);
      BOOST_TEST(buffer[2] == expected + 10 * rank);
    }
  }
}


BOOST_AUTO_TEST_CASE(exscan_nonblocking) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int expected = exclusive_prefix(rank);
  const std::plus<int> plus{};

  // Scalar request/status path.
  {
    int recv = 777;
    mp1::request request = mp1::iexscan(comm, plus, rank, recv);
    const mp1::status status = request.wait();
    BOOST_TEST(status.error() == MPI_SUCCESS);
    if (rank == 0)
      BOOST_TEST(recv == 777);
    else
      BOOST_TEST(recv == expected);
  }

  // Native MPI_Op and standard functor through the nonblocking path.
  {
    int recv = 777;
    mp1::request request = mp1::iexscan(comm, MPI_SUM, rank, recv);
    request.wait();
    if (rank == 0)
      BOOST_TEST(recv == 777);
    else
      BOOST_TEST(recv == expected);
  }
  {
    int recv = 777;
    mp1::request request = mp1::iexscan(comm, std::plus<int>{}, rank, recv);
    request.wait();
    if (rank == 0)
      BOOST_TEST(recv == 777);
    else
      BOOST_TEST(recv == expected);
  }

  // Layout request; the caller's layout outlives the request.
  {
    const mp1::layout<int> holey = mp1::layout<int>::vector(2, 1, 2);
    std::vector<int> send(3, -7);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    mp1::region<int> send_region{send.data(), holey};
    mp1::region<int> recv_region{recv.data(), holey};
    mp1::request request = mp1::iexscan(comm, plus, send_region, recv_region);
    request.wait();
    if (rank == 0) {
      BOOST_TEST(recv[0] == -1);
      BOOST_TEST(recv[1] == -1);
      BOOST_TEST(recv[2] == -1);
    } else {
      BOOST_TEST(recv[0] == expected);
      BOOST_TEST(recv[1] == -1);
      BOOST_TEST(recv[2] == expected + 10 * rank);
    }
  }

  // Container and in-place request forms.
  {
    const std::vector<int> send{rank, 2 * rank};
    std::vector<int> recv(send.size(), -1);
    mp1::request request = mp1::iexscan(comm, plus, send, recv);
    request.wait();
    if (rank == 0) {
      BOOST_TEST(recv[0] == -1);
      BOOST_TEST(recv[1] == -1);
    } else {
      BOOST_TEST(recv[0] == expected);
      BOOST_TEST(recv[1] == 2 * expected);
    }
  }
  {
    std::vector<int> buffer{rank, 2 * rank};
    mp1::request request = mp1::iexscan(comm, plus, mp1::in_place, buffer);
    request.wait();
    if (rank == 0) {
      BOOST_TEST(buffer[0] == 0);
      BOOST_TEST(buffer[1] == 0);
    } else {
      BOOST_TEST(buffer[0] == expected);
      BOOST_TEST(buffer[1] == 2 * expected);
    }
  }
}
