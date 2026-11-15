// Inclusive-scan tests: free-function API, scalar/buffer/container/layout forms, in-place
// forms, built-in plus and a user-defined stateless functor, padded element type and
// nonblocking requests.

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

  // Inclusive prefix of the rank values 0..rank.
  [[nodiscard]] int inclusive_prefix(int rank) {
    return rank * (rank + 1) / 2;
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);


BOOST_AUTO_TEST_CASE(scan_scalar_and_operators) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int expected = inclusive_prefix(rank);
  const std::plus<int> plus{};

  // Named lvalue functor, temporary built-in functor and a user-defined stateless functor.
  int recv = -1;
  mp1::scan(comm, plus, rank, recv);
  BOOST_TEST(recv == expected);

  recv = -1;
  mp1::scan(comm, std::plus<int>{}, rank, recv);
  BOOST_TEST(recv == expected);

  recv = -1;
  const add int_add{};
  mp1::scan(comm, int_add, rank, recv);
  BOOST_TEST(recv == expected);

  // Native predefined MPI_Op, std::plus<int>, mp1::max<int> and std::plus<void>.
  recv = -1;
  mp1::scan(comm, MPI_SUM, rank, recv);
  BOOST_TEST(recv == expected);

  recv = -1;
  mp1::scan(comm, std::plus<int>{}, rank, recv);
  BOOST_TEST(recv == expected);

  recv = -1;
  mp1::scan(comm, std::plus<void>{}, rank, recv);
  BOOST_TEST(recv == expected);

  recv = -1;
  mp1::scan(comm, mp1::max<int>{}, rank, recv);
  BOOST_TEST(recv == rank);

  double double_recv = -1.0;
  mp1::scan(comm, std::plus<double>{}, 0.5 * rank, double_recv);
  BOOST_TEST(double_recv == 0.5 * static_cast<double>(expected));

  // Rank 0 receives its own contribution, and a constant input scans to (rank + 1) copies.
  int constant_recv = -1;
  mp1::scan(comm, plus, 42, constant_recv);
  BOOST_TEST(constant_recv == 42 * (rank + 1));

  // Padded element type through the scalar form.
  const padded_value send{rank, 0.5 * rank};
  padded_value padded_recv{-1, -1.0};
  mp1::scan(comm, padded_add{}, send, padded_recv);
  BOOST_TEST(padded_recv.first == expected);
  BOOST_TEST(padded_recv.second == 0.5 * static_cast<double>(expected));
}


BOOST_AUTO_TEST_CASE(scan_buffers_and_containers) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int expected = inclusive_prefix(rank);
  const std::plus<int> plus{};

  // Buffer form: element j is the prefix of (i + j) for i = 0..rank.
  {
    const std::vector<int> send{rank, rank + 1, rank + 2};
    std::vector<int> recv(send.size(), -1);
    mp1::scan(comm, plus, std::span<const int>{send}, std::span<int>{recv});
    for (std::size_t j = 0; j < recv.size(); ++j) {
      BOOST_TEST(recv[j] == expected + static_cast<int>(j) * (rank + 1));
    }
  }

  // Contiguous-container form.
  {
    const std::vector<int> send{rank, 2 * rank};
    std::vector<int> recv(send.size(), -1);
    mp1::scan(comm, plus, send, recv);
    BOOST_TEST(recv[0] == expected);
    BOOST_TEST(recv[1] == 2 * expected);
  }

  // Buffer in-place form.
  {
    std::vector<int> buffer{rank, rank + 1, rank + 2};
    mp1::scan(comm, plus, mp1::in_place, std::span<int>{buffer});
    for (std::size_t j = 0; j < buffer.size(); ++j) {
      BOOST_TEST(buffer[j] == expected + static_cast<int>(j) * (rank + 1));
    }
  }

  // Container in-place form.
  {
    std::vector<int> buffer{rank, 2 * rank};
    mp1::scan(comm, plus, mp1::in_place, buffer);
    BOOST_TEST(buffer[0] == expected);
    BOOST_TEST(buffer[1] == 2 * expected);
  }
}


BOOST_AUTO_TEST_CASE(scan_layouts) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int expected = inclusive_prefix(rank);
  const std::plus<int> plus{};

  // Dense contiguous layout.
  const mp1::layout<int> dense = mp1::layout<int>::contiguous(3);
  {
    const std::vector<int> send{rank, rank + 1, rank + 2};
    std::vector<int> recv(send.size(), -1);
    mp1::scan(comm, plus, mp1::region<const int>{send.data(), dense},
              mp1::region<int>{recv.data(), dense});
    for (std::size_t j = 0; j < recv.size(); ++j) {
      BOOST_TEST(recv[j] == expected + static_cast<int>(j) * (rank + 1));
    }
  }

  // Derived layout with a hole: only elements 0 and 2 are scanned and written.
  const mp1::layout<int> holey = mp1::layout<int>::vector(2, 1, 2);
  {
    std::vector<int> send(3, -7);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    mp1::scan(comm, plus, mp1::region<int>{send.data(), holey},
              mp1::region<int>{recv.data(), holey});
    BOOST_TEST(recv[0] == expected);
    BOOST_TEST(recv[1] == -1);
    BOOST_TEST(recv[2] == expected + 10 * (rank + 1));
  }

  // Padded element type with a multi-element contiguous layout.
  const mp1::layout<padded_value> padded = mp1::layout<padded_value>::contiguous(2);
  {
    const std::vector<padded_value> send{padded_value{rank, 0.5 * rank},
                                         padded_value{rank + 1, 0.5 * rank + 1.0}};
    std::vector<padded_value> recv(send.size(), padded_value{-1, -1.0});
    mp1::scan(comm, padded_add{}, mp1::region<const padded_value>{send.data(), padded},
              mp1::region<padded_value>{recv.data(), padded});
    BOOST_TEST(recv[0].first == expected);
    BOOST_TEST(recv[0].second == 0.5 * static_cast<double>(expected));
    BOOST_TEST(recv[1].first == expected + (rank + 1));
    BOOST_TEST(recv[1].second == 0.5 * static_cast<double>(expected) + (rank + 1));
  }

  // In-place layout forms (dense and staged).
  {
    std::vector<int> buffer{rank, rank + 1, rank + 2};
    mp1::scan(comm, plus, mp1::in_place, mp1::region<int>{buffer.data(), dense});
    for (std::size_t j = 0; j < buffer.size(); ++j) {
      BOOST_TEST(buffer[j] == expected + static_cast<int>(j) * (rank + 1));
    }
  }
  {
    std::vector<int> buffer(3, -7);
    buffer[0] = rank;
    buffer[2] = rank + 10;
    mp1::scan(comm, plus, mp1::in_place, mp1::region<int>{buffer.data(), holey});
    BOOST_TEST(buffer[0] == expected);
    BOOST_TEST(buffer[1] == -7);
    BOOST_TEST(buffer[2] == expected + 10 * (rank + 1));
  }
}


BOOST_AUTO_TEST_CASE(scan_nonblocking) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int expected = inclusive_prefix(rank);
  const std::plus<int> plus{};

  // Scalar request/status path.
  {
    int recv = -1;
    mp1::request request = mp1::iscan(comm, plus, rank, recv);
    const mp1::status status = request.wait();
    BOOST_TEST(status.error() == MPI_SUCCESS);
    BOOST_TEST(recv == expected);
  }

  // Native MPI_Op and standard functor through the nonblocking path.
  {
    int recv = -1;
    mp1::request request = mp1::iscan(comm, MPI_SUM, rank, recv);
    request.wait();
    BOOST_TEST(recv == expected);
  }
  {
    int recv = -1;
    mp1::request request = mp1::iscan(comm, std::plus<int>{}, rank, recv);
    request.wait();
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
    mp1::request request = mp1::iscan(comm, plus, send_region, recv_region);
    request.wait();
    BOOST_TEST(recv[0] == expected);
    BOOST_TEST(recv[1] == -1);
    BOOST_TEST(recv[2] == expected + 10 * (rank + 1));
  }

  // Container and in-place request forms.
  {
    const std::vector<int> send{rank, 2 * rank};
    std::vector<int> recv(send.size(), -1);
    mp1::request request = mp1::iscan(comm, plus, send, recv);
    request.wait();
    BOOST_TEST(recv[0] == expected);
    BOOST_TEST(recv[1] == 2 * expected);
  }
  {
    std::vector<int> buffer{rank, 2 * rank};
    mp1::request request = mp1::iscan(comm, plus, mp1::in_place, buffer);
    request.wait();
    BOOST_TEST(buffer[0] == expected);
    BOOST_TEST(buffer[1] == 2 * expected);
  }
}
