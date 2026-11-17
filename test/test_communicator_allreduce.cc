// Allreduce tests: free-function API, scalar/buffer/container/layout forms, in-place forms,
// built-in plus and a user-defined stateless functor, and a padded reflected element type.

#include "mp1/mp1.hpp"

#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <functional>
#include <stdexcept>
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

// A custom trait may name a peer's datatype for the data wire.  MPI_INTEGER is outside the
// C++ reduction categories, so the typed predefined mapping does not apply to this type.
struct allreduce_fortran_int {
  int value{0};
};

[[nodiscard]] constexpr allreduce_fortran_int operator+(
    const allreduce_fortran_int& left, const allreduce_fortran_int& right) noexcept {
  return allreduce_fortran_int{left.value + right.value};
}

MP1_DATATYPE_TRAITS(allreduce_fortran_int, MPI_INTEGER)

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

// The predefined mapping is the MPI 4.1 §6.9.2 row for the C++ datatype category: C integer,
// floating and std::complex for SUM/PROD; C integer and floating for MAX/MIN; C integer and
// std::byte for bitwise; C integer and bool for the logical operations.  A pair outside the
// row uses the callback.
static_assert(mp1::detail::is_predefined_reduction_functor_v<int, std::plus<int>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<double, std::plus<double>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<std::complex<double>,
                                                             std::plus<std::complex<double>>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<int, std::multiplies<int>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<unsigned, std::bit_and<unsigned>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<int, mp1::max<int>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<double, mp1::max<double>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<int, std::logical_and<int>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<bool, std::logical_and<bool>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<double, std::plus<int>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<bool, std::plus<bool>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<std::string, std::plus<void>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<char, mp1::max<char>>);
static_assert(
    not mp1::detail::is_predefined_reduction_functor_v<double, std::logical_and<double>>);

// RMA uses the same C++ typed row.  Floating max/min, complex sum/product, std::byte bitwise
// and the C-integer/Logical logical operations map to the predefined handles.
static_assert(mp1::detail::is_predefined_rma_functor_v<double, mp1::max<double>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<std::complex<double>,
                                                       std::plus<std::complex<double>>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<std::complex<float>,
                                                       std::multiplies<std::complex<float>>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<std::byte, std::bit_or<std::byte>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<int, std::logical_and<int>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<int, mp1::logical_xor<>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<bool, std::logical_and<bool>>);
static_assert(not mp1::detail::is_predefined_rma_functor_v<double, std::plus<int>>);
static_assert(not mp1::detail::is_predefined_rma_functor_v<std::string, std::plus<void>>);
static_assert(not mp1::detail::is_predefined_rma_functor_v<std::complex<double>,
                                                           mp1::max<std::complex<double>>>);


// An enum element type borrows its underlying predefined datatype and stays on the callback
// path, so it locks the transparent logical_xor normalization: the 0/1 result must convert
// back to the enum type exactly like the typed spelling.
enum class parity_flag : unsigned char { off = 0, on = 1 };

static_assert(std::is_same_v<decltype(mp1::logical_xor<>{}(parity_flag::on, parity_flag::off)),
                             parity_flag>);

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

  // The transparent spellings deduce their operand: integral max/min and floating-point
  // max/min map to the predefined handles.
  recv = -1;
  mp1::allreduce(comm, mp1::max{}, rank, recv);
  BOOST_TEST(recv == size - 1);

  recv = -1;
  mp1::allreduce(comm, mp1::min{}, rank, recv);
  BOOST_TEST(recv == 0);

  double transparent_recv = -1.0;
  mp1::allreduce(comm, mp1::max{}, 0.5 * static_cast<double>(rank), transparent_recv);
  BOOST_TEST(transparent_recv == 0.5 * static_cast<double>(size - 1));

  // Character and Logical are outside MPI_MAX's categories: the transparent object stays on
  // the callback and the C++ max semantics apply.
  char char_send = static_cast<char>('a' + rank);
  char char_recv = '\0';
  mp1::allreduce(comm, mp1::max{}, char_send, char_recv);
  BOOST_TEST(char_recv == static_cast<char>('a' + size - 1));

  const bool bool_send = rank == 0;
  bool bool_recv = false;
  mp1::allreduce(comm, mp1::max{}, bool_send, bool_recv);
  BOOST_TEST(bool_recv);

  // A custom trait mapped to a peer's MPI_INTEGER wire is outside the typed predefined
  // mapping; the raw MPI_Op and the managed callback both use native_datatype<T>().
  static_assert(
      not mp1::detail::is_predefined_reduction_functor_v<allreduce_fortran_int,
                                                         std::plus<allreduce_fortran_int>>);
  // Open MPI built without Fortran support exposes MPI_INTEGER with zero extent; the
  // native_datatype<T>() validator rejects the trait before the reduction call.  That is a
  // fail-closed rejection, not a usable wire type in this configuration.
  MPI_Aint fortran_lb = 0;
  MPI_Aint fortran_extent = 0;
  BOOST_REQUIRE(MPI_Type_get_extent(MPI_INTEGER, &fortran_lb, &fortran_extent) == MPI_SUCCESS);
  if (fortran_extent == 0) {
    BOOST_TEST_MESSAGE(
        "allreduce_fortran_int data movement skipped: MPI_INTEGER has zero extent on this "
        "C-only provider; native_datatype validation rejects the trait first");
    allreduce_fortran_int raw_in{rank + 1};
    allreduce_fortran_int raw_out{};
    BOOST_CHECK_THROW(mp1::allreduce(comm, MPI_SUM, raw_in, raw_out), std::invalid_argument);
    allreduce_fortran_int callback_in{rank + 1};
    allreduce_fortran_int callback_out{};
    BOOST_CHECK_THROW(
        mp1::allreduce(comm, std::plus<allreduce_fortran_int>{}, callback_in, callback_out),
        std::invalid_argument);
  } else {
    const int fortran_expected = size * (size + 1) / 2;
    allreduce_fortran_int raw_in{rank + 1};
    allreduce_fortran_int raw_out{};
    mp1::allreduce(comm, MPI_SUM, raw_in, raw_out);
    BOOST_TEST(raw_out.value == fortran_expected);

    allreduce_fortran_int callback_in{rank + 1};
    allreduce_fortran_int callback_out{};
    mp1::allreduce(comm, std::plus<allreduce_fortran_int>{}, callback_in, callback_out);
    BOOST_TEST(callback_out.value == fortran_expected);
  }

  // The adapter must select the predefined handles, not just produce the right result.
  BOOST_TEST((mp1::detail::predefined_mpi_op<int, std::plus<int>>() == MPI_SUM));
  BOOST_TEST((mp1::detail::predefined_mpi_op<int, std::plus<int>>() == MPI_SUM));
  BOOST_TEST((mp1::detail::predefined_mpi_op<double, std::multiplies<double>>() == MPI_PROD));
  BOOST_TEST((mp1::detail::predefined_mpi_op<int, mp1::max<int>>() == MPI_MAX));
  BOOST_TEST((mp1::detail::predefined_mpi_op<int, mp1::max<>>() == MPI_MAX));
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
  const int rank = comm.rank();
  const int size = comm.size();

  // C-integer logical reductions use MPI_LAND/MPI_LOR/MPI_LXOR (MPI 4.1 §6.9.2).  The integer
  // result value is MPI's, so compare the functor path with the raw handle instead of assuming
  // a 0/1 normalization.
  int recv = -1;
  int raw = -1;
  mp1::allreduce(comm, std::logical_and<int>{}, 3, recv);
  mp1::allreduce(comm, MPI_LAND, 3, raw);
  BOOST_TEST(recv == raw);

  recv = -1;
  raw = -1;
  mp1::allreduce(comm, std::logical_or<int>{}, 3, recv);
  mp1::allreduce(comm, MPI_LOR, 3, raw);
  BOOST_TEST(recv == raw);

  recv = -1;
  raw = -1;
  mp1::allreduce(comm, mp1::logical_xor<int>{}, 3, recv);
  mp1::allreduce(comm, MPI_LXOR, 3, raw);
  BOOST_TEST(recv == raw);

  // The transparent spelling maps to the same predefined operation.
  recv = -1;
  raw = -1;
  mp1::allreduce(comm, mp1::logical_xor{}, 3, recv);
  mp1::allreduce(comm, MPI_LXOR, 3, raw);
  BOOST_TEST(recv == raw);

  // bool is the Logical category: MPI_LXOR over MPI_CXX_BOOL is exactly the boolean xor.
  bool bool_recv = false;
  mp1::allreduce(comm, mp1::logical_xor{}, (rank % 2) == 1, bool_recv);
  BOOST_TEST(bool_recv == ((size / 2) % 2 == 1));

  // A floating-point element is not a logical category, so it stays on the callback and keeps
  // the C++ 0/1 normalization.
  double double_recv = -1.0;
  mp1::allreduce(comm, mp1::logical_xor{}, 3.0, double_recv);
  BOOST_TEST(double_recv == static_cast<double>(size % 2));

  // The same operation over an enum element type keeps the enum type (not bool) as the
  // reduction result and uses the underlying predefined datatype.
  parity_flag flag_recv = parity_flag::off;
  mp1::allreduce(comm, mp1::logical_xor{}, parity_flag::on, flag_recv);
  BOOST_TEST(static_cast<int>(flag_recv) == (size % 2));
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
