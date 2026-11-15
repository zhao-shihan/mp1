// Allreduce tests: free-function API, scalar/buffer/container/layout forms, in-place forms,
// built-in plus and a user-defined stateless functor, and a padded reflected element type.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>


// Padding matters: sizeof(padded_value) == 16, the reflected datatype size is 12, so a
// contiguous layout of more than one element is flat only through the base extent rule.
struct padded_value {
  int first{0};
  double second{0.0};
};

MP1_REFLECTION(padded_value, first, second)

// op_traits classifies standard-library functors but keeps non-commutative operations and
// user-defined element types (including std::plus<std::string>) non-commutative; complex
// addition and multiplication are the explicit standard-library exception.
static_assert(mp1::op_traits<std::plus<int>>::is_commutative);
static_assert(mp1::op_traits<std::multiplies<int>>::is_commutative);
static_assert(mp1::op_traits<std::bit_and<unsigned>>::is_commutative);
static_assert(mp1::op_traits<std::plus<int>>::is_commutative);
static_assert(mp1::op_traits<std::multiplies<int>>::is_commutative);
static_assert(mp1::op_traits<std::bit_xor<unsigned>>::is_commutative);
static_assert(mp1::op_traits<std::logical_or<int>>::is_commutative);
static_assert(mp1::op_traits<std::equal_to<int>>::is_commutative);
static_assert(mp1::op_traits<std::not_equal_to<int>>::is_commutative);
static_assert(not mp1::op_traits<std::equal_to<std::string>>::is_commutative);
static_assert(not mp1::op_traits<std::minus<int>>::is_commutative);
static_assert(not mp1::op_traits<std::divides<int>>::is_commutative);
static_assert(not mp1::op_traits<std::plus<std::string>>::is_commutative);
static_assert(not mp1::op_traits<std::multiplies<std::string>>::is_commutative);

// Transparent standard functors are classified from the reduction element type.
static_assert(mp1::detail::is_commutative_reduction_v<int, std::plus<void>>);
static_assert(mp1::detail::is_commutative_reduction_v<double, std::multiplies<void>>);
static_assert(not mp1::detail::is_commutative_reduction_v<std::string, std::plus<void>>);
static_assert(not mp1::detail::is_commutative_reduction_v<int, std::minus<void>>);
static_assert(mp1::detail::is_commutative_reduction_v<int, std::equal_to<void>>);

// The predefined mapping is deliberately type-safe and conservative.
static_assert(mp1::detail::is_predefined_reduction_functor_v<int, std::plus<int>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<double, std::plus<double>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<int, std::multiplies<int>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<unsigned, std::bit_and<unsigned>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<int, mp1::max<int>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<double, mp1::max<double>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<double, std::plus<int>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<bool, std::plus<bool>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<std::string, std::plus<void>>);
// Logical predefined ops are never mapped in collectives: MPI 4.1 §6.9.2 does not specify a
// numeric result for the C-integer logical reductions, so the stateless callback remains the
// single path for every logical functor (including bool).
static_assert(not mp1::detail::is_predefined_reduction_functor_v<int, std::logical_and<int>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<int, std::logical_or<int>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<bool, std::logical_and<bool>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<bool, std::logical_or<bool>>);

// RMA has no callback fallback.  The §6.9.2 (Predefined Reduction Operations) table maps
// floating-point max/min and complex sum/product there, std::byte bitwise ops to
// MPI_BAND/BOR/BXOR, and bool logical ops to MPI_LAND/LOR/LXOR; C-integer logical functors
// stay rejected because they have no exact predefined mapping.
static_assert(mp1::detail::is_predefined_rma_functor_v<double, mp1::max<double>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<std::complex<double>,
                                                       std::plus<std::complex<double>>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<std::complex<float>,
                                                       std::multiplies<std::complex<float>>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<std::byte, std::bit_or<std::byte>>);
static_assert(not mp1::detail::is_predefined_rma_functor_v<double, std::plus<int>>);
static_assert(not mp1::detail::is_predefined_rma_functor_v<std::string, std::plus<void>>);
static_assert(not mp1::detail::is_predefined_rma_functor_v<int, std::logical_and<int>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<bool, std::logical_and<bool>>);
static_assert(not mp1::detail::is_predefined_rma_functor_v<std::complex<double>,
                                                           mp1::max<std::complex<double>>>);


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
        // Never throw from a fixture destructor; the harness may already own teardown.
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

  struct tagged_add : mp1::commutative_op {
    int operator()(const int& left, const int& right) const noexcept {
      return left + right;
    }
  };

  static_assert(not mp1::op_traits<add>::is_commutative);
  static_assert(mp1::op_traits<tagged_add>::is_commutative);

  [[nodiscard]] int rank_sum(int size) {
    return size * (size - 1) / 2;
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);


BOOST_AUTO_TEST_CASE(allreduce_scalar_and_operators) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = rank_sum(size);
  const std::plus<int> plus{};

  // Named lvalue functor, temporary built-in functor and a user-defined stateless functor.
  int recv = -1;
  mp1::allreduce(comm, plus, rank, recv);
  BOOST_TEST(recv == expected);

  recv = -1;
  mp1::allreduce(comm, std::plus<int>{}, rank, recv);
  BOOST_TEST(recv == expected);

  recv = -1;
  const add int_add{};
  mp1::allreduce(comm, int_add, rank, recv);
  BOOST_TEST(recv == expected);

  // User opt-in through the commutative_op tag remains available.
  recv = -1;
  mp1::allreduce(comm, tagged_add{}, rank, recv);
  BOOST_TEST(recv == expected);

  // Native predefined MPI_Op and standard-library functors.
  recv = -1;
  mp1::allreduce(comm, MPI_SUM, rank, recv);
  BOOST_TEST(recv == expected);

  recv = -1;
  mp1::allreduce(comm, std::plus<int>{}, rank, recv);
  BOOST_TEST(recv == expected);

  recv = -1;
  mp1::allreduce(comm, std::plus<void>{}, rank, recv);
  BOOST_TEST(recv == expected);

  recv = 1;
  mp1::allreduce(comm, std::multiplies<int>{}, rank + 1, recv);
  int factorial = 1;
  for (int i = 2; i <= size; ++i)
    factorial *= i;
  BOOST_TEST(recv == factorial);

  recv = -1;
  mp1::allreduce(comm, mp1::max<int>{}, rank, recv);
  BOOST_TEST(recv == size - 1);

  // The adapter must select the predefined handles, not just produce the right result.
  BOOST_TEST((mp1::detail::predefined_mpi_op<int, std::plus<int>>() == MPI_SUM));
  BOOST_TEST((mp1::detail::predefined_mpi_op<int, std::plus<int>>() == MPI_SUM));
  BOOST_TEST((mp1::detail::predefined_mpi_op<double, std::multiplies<double>>() == MPI_PROD));
  BOOST_TEST((mp1::detail::predefined_mpi_op<int, mp1::max<int>>() == MPI_MAX));
  BOOST_TEST((mp1::detail::predefined_mpi_op<unsigned, std::bit_xor<unsigned>>() == MPI_BXOR));
  BOOST_TEST((mp1::detail::predefined_mpi_op<double, mp1::min<double>>() == MPI_MIN));

  double double_recv = -1.0;
  mp1::allreduce(comm, std::plus<double>{}, 0.5 * rank, double_recv);
  BOOST_TEST(double_recv == 0.5 * static_cast<double>(expected));

  // Padded element type through the scalar form: the staging path must move whole T values.
  const padded_value send{rank, static_cast<double>(rank) / 2.0};
  padded_value padded_recv{-1, -1.0};
  mp1::allreduce(comm, padded_add{}, send, padded_recv);
  BOOST_TEST(padded_recv.first == expected);
  BOOST_TEST(padded_recv.second == 0.5 * static_cast<double>(expected));
}


BOOST_AUTO_TEST_CASE(allreduce_logical_operator_results) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int size = comm.size();

  // Logical functors normalize to 0/1.  Mapping MPI_INT to MPI_LAND/MPI_LOR can return the
  // input value instead of a normalized result, so these use the callback path.
  int recv = -1;
  mp1::allreduce(comm, std::logical_and<int>{}, 3, recv);
  BOOST_TEST(recv == 1);

  recv = -1;
  mp1::allreduce(comm, std::logical_or<int>{}, 3, recv);
  BOOST_TEST(recv == 1);

  recv = -1;
  mp1::allreduce(comm, mp1::logical_xor<int>{}, 3, recv);
  BOOST_TEST(recv == (size % 2));
}

BOOST_AUTO_TEST_CASE(allreduce_buffers_and_containers) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = rank_sum(size);
  const std::plus<int> plus{};

  // Buffer form: element j of the result is sum over ranks of (rank + j).
  {
    const std::vector<int> send{rank, rank + 1, rank + 2};
    std::vector<int> recv(send.size(), -1);
    mp1::allreduce(comm, plus, std::span<const int>{send}, std::span<int>{recv});
    for (std::size_t j = 0; j < recv.size(); ++j) {
      BOOST_TEST(recv[j] == expected + size * static_cast<int>(j));
    }
  }

  // Contiguous-container form.
  {
    const std::vector<int> send{rank, 2 * rank};
    std::vector<int> recv(send.size(), -1);
    mp1::allreduce(comm, plus, send, recv);
    BOOST_TEST(recv[0] == expected);
    BOOST_TEST(recv[1] == 2 * expected);
  }

  // Buffer in-place form (send is replaced by the reduced vector).
  {
    std::vector<int> buffer{rank, 2 * rank, 3 * rank};
    mp1::allreduce(comm, plus, mp1::in_place, std::span<int>{buffer});
    BOOST_TEST(buffer[0] == expected);
    BOOST_TEST(buffer[1] == 2 * expected);
    BOOST_TEST(buffer[2] == 3 * expected);
  }

  // Container in-place form.
  {
    std::vector<int> buffer{rank, 2 * rank};
    mp1::allreduce(comm, plus, mp1::in_place, buffer);
    BOOST_TEST(buffer[0] == expected);
    BOOST_TEST(buffer[1] == 2 * expected);
  }

  // A zero-length reduction still enters MPI with count 0 and null buffers.
  {
    int dummy = 0;
    const mp1::layout<int> empty = mp1::layout<int>::contiguous(0);
    mp1::allreduce(comm, plus, std::span<const int>{}, std::span<int>{&dummy, 0});
    mp1::allreduce(comm, plus, mp1::region<const int>{nullptr, empty},
                   mp1::region<int>{&dummy, empty});
  }
}


BOOST_AUTO_TEST_CASE(allreduce_layouts) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = rank_sum(size);
  const std::plus<int> plus{};

  // Dense contiguous layout: flat fast path over the base T.
  const mp1::layout<int> dense = mp1::layout<int>::contiguous(3);
  {
    const std::vector<int> send{rank, rank + 1, rank + 2};
    std::vector<int> recv(send.size(), -1);
    mp1::allreduce(comm, plus, mp1::region<const int>{send.data(), dense},
                   mp1::region<int>{recv.data(), dense});
    for (std::size_t j = 0; j < recv.size(); ++j) {
      BOOST_TEST(recv[j] == expected + size * static_cast<int>(j));
    }
  }

  // Derived layout with a hole: two elements at element offsets 0 and 2.  The reduction
  // stages both directions, so only the mapped elements are written.
  const mp1::layout<int> holey = mp1::layout<int>::vector(2, 1, 2);
  {
    std::vector<int> send(3, -7);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    mp1::allreduce(comm, plus, mp1::region<int>{send.data(), holey},
                   mp1::region<int>{recv.data(), holey});
    BOOST_TEST(recv[0] == expected);
    BOOST_TEST(recv[1] == -1);
    BOOST_TEST(recv[2] == expected + 10 * size);
  }

  // Indexed layout: one element at 0 and two elements at 2, i.e. a different derived shape.
  mp1::indexed_parameter indexed_parameter;
  indexed_parameter.add(1, 0);
  indexed_parameter.add(2, 2);
  const mp1::layout<int> indexed = mp1::layout<int>::indexed(indexed_parameter);
  {
    std::vector<int> send(4, -7);
    send[0] = rank;
    send[2] = rank + 1;
    send[3] = rank + 2;
    std::vector<int> recv(4, -1);
    mp1::allreduce(comm, plus, mp1::region<int>{send.data(), indexed},
                   mp1::region<int>{recv.data(), indexed});
    BOOST_TEST(recv[0] == expected);
    BOOST_TEST(recv[1] == -1);
    BOOST_TEST(recv[2] == expected + size);
    BOOST_TEST(recv[3] == expected + 2 * size);
  }

  // Padded element type with a multi-element contiguous layout (flat base-T path).
  const mp1::layout<padded_value> padded = mp1::layout<padded_value>::contiguous(2);
  {
    const std::vector<padded_value> send{padded_value{rank, 0.5 * rank},
                                         padded_value{rank + 1, 0.5 * rank + 1.0}};
    std::vector<padded_value> recv(send.size(), padded_value{-1, -1.0});
    mp1::allreduce(comm, padded_add{}, mp1::region<const padded_value>{send.data(), padded},
                   mp1::region<padded_value>{recv.data(), padded});
    BOOST_TEST(recv[0].first == expected);
    BOOST_TEST(recv[0].second == 0.5 * static_cast<double>(expected));
    BOOST_TEST(recv[1].first == expected + size);
    BOOST_TEST(recv[1].second == 0.5 * static_cast<double>(expected) + size);
  }

  // In-place dense and staged layout forms.
  {
    std::vector<int> buffer{rank, rank + 1, rank + 2};
    mp1::allreduce(comm, plus, mp1::in_place, mp1::region<int>{buffer.data(), dense});
    for (std::size_t j = 0; j < buffer.size(); ++j) {
      BOOST_TEST(buffer[j] == expected + size * static_cast<int>(j));
    }
  }
  {
    std::vector<int> buffer(3, -7);
    buffer[0] = rank;
    buffer[2] = rank + 10;
    mp1::allreduce(comm, plus, mp1::in_place, mp1::region<int>{buffer.data(), holey});
    BOOST_TEST(buffer[0] == expected);
    BOOST_TEST(buffer[1] == -7);
    BOOST_TEST(buffer[2] == expected + 10 * size);
  }
}


BOOST_AUTO_TEST_CASE(allreduce_nonblocking) {
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = rank_sum(size);
  const std::plus<int> plus{};

  // Scalar request/status path.
  {
    int recv = -1;
    mp1::request request = mp1::iallreduce(comm, plus, rank, recv);
    const mp1::status status = request.wait();
    BOOST_TEST(status.error() == MPI_SUCCESS);
    BOOST_TEST(recv == expected);
  }

  // Native MPI_Op and standard functor through the nonblocking path.
  {
    int recv = -1;
    mp1::request request = mp1::iallreduce(comm, MPI_SUM, rank, recv);
    request.wait();
    BOOST_TEST(recv == expected);
  }
  {
    int recv = -1;
    mp1::request request = mp1::iallreduce(comm, std::plus<int>{}, rank, recv);
    request.wait();
    BOOST_TEST(recv == expected);
  }

  // Layout request: the caller's layout must stay alive until wait().
  {
    const mp1::layout<int> holey = mp1::layout<int>::vector(2, 1, 2);
    std::vector<int> send(3, -7);
    send[0] = rank;
    send[2] = rank + 10;
    std::vector<int> recv(3, -1);
    mp1::region<int> send_region{send.data(), holey};
    mp1::region<int> recv_region{recv.data(), holey};
    mp1::request request = mp1::iallreduce(comm, plus, send_region, recv_region);
    request.wait();
    BOOST_TEST(recv[0] == expected);
    BOOST_TEST(recv[1] == -1);
    BOOST_TEST(recv[2] == expected + 10 * size);
  }

  // Container and in-place request forms.
  {
    const std::vector<int> send{rank, 2 * rank};
    std::vector<int> recv(send.size(), -1);
    mp1::request request = mp1::iallreduce(comm, plus, send, recv);
    request.wait();
    BOOST_TEST(recv[0] == expected);
    BOOST_TEST(recv[1] == 2 * expected);
  }
  {
    std::vector<int> buffer{rank, 2 * rank};
    mp1::request request = mp1::iallreduce(comm, plus, mp1::in_place, buffer);
    request.wait();
    BOOST_TEST(buffer[0] == expected);
    BOOST_TEST(buffer[1] == 2 * expected);
  }
}
