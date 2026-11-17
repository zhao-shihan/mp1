// Built-in reduction functors opt into commutativity only for the exact
// operation/type pairs whose C++ operation is commutative.  The truth table is checked at
// compile time; the MPI_Op_create commute argument is captured at run time through the
// shared mpi_wrap support shim.

#include "mp1/mp1.hpp"
#include "mpi_wrap.hpp"

#include <boost/test/unit_test.hpp>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>


namespace {

  struct user_number {
    int value{0};
  };

  [[nodiscard]] constexpr user_number operator+(const user_number& left,
                                                const user_number& right) noexcept {
    return user_number{left.value + right.value};
  }

  struct user_add {
    user_number operator()(const user_number& left, const user_number& right) const noexcept {
      return left + right;
    }
  };

  struct tagged_add : mp1::commutative_op {
    int operator()(int left, int right) const noexcept {
      return left + right;
    }
  };

  struct plain_add {
    int operator()(int left, int right) const noexcept {
      return left + right;
    }
  };

}  // namespace

MP1_REFLECTION(user_number, value)

// ---- Compile-time truth table -------------------------------------------------------------

// The retained mp1 functors do not inherit the blanket tag.  The seven std:: duplicates
// (plus, multiplies, bit_and, bit_or, bit_xor, logical_and, logical_or) are no longer part of
// the mp1 surface; their std:: equivalents are checked below.
static_assert(not std::is_base_of_v<mp1::commutative_op, mp1::max<int>>);
static_assert(not std::is_base_of_v<mp1::commutative_op, mp1::min<int>>);
static_assert(not std::is_base_of_v<mp1::commutative_op, mp1::logical_xor<int>>);

// max/min: integral only.  The C++ expression is not commutative on floating-point values
// under the callback (x < y) ? y : x once an operand is NaN or a signed zero is involved; a
// user < is not assumed commutative.  The mapped C integer/floating categories never create a
// callback, so this trait is consulted only for the callback-only categories (Character,
// Logical, Byte, enums, user types).
static_assert(mp1::op_traits<mp1::max<int>>::is_commutative);
static_assert(mp1::op_traits<mp1::min<long long>>::is_commutative);
static_assert(not mp1::op_traits<mp1::max<double>>::is_commutative);
static_assert(not mp1::op_traits<mp1::max<float>>::is_commutative);
static_assert(not mp1::op_traits<mp1::min<double>>::is_commutative);
static_assert(not mp1::op_traits<mp1::max<user_number>>::is_commutative);
static_assert(not mp1::op_traits<mp1::max<std::byte>>::is_commutative);

// plus/multiplies: built-in arithmetic only, plus the explicit std::complex
// specializations that preserve the public truth table of the removed mp1 functors.
static_assert(mp1::op_traits<std::plus<int>>::is_commutative);
static_assert(mp1::op_traits<std::plus<double>>::is_commutative);
static_assert(mp1::op_traits<std::multiplies<unsigned>>::is_commutative);
static_assert(not mp1::op_traits<std::plus<user_number>>::is_commutative);
static_assert(not mp1::op_traits<std::multiplies<user_number>>::is_commutative);
static_assert(not mp1::op_traits<std::plus<std::string>>::is_commutative);
static_assert(mp1::op_traits<std::plus<std::complex<double>>>::is_commutative);
static_assert(mp1::op_traits<std::multiplies<std::complex<float>>>::is_commutative);
static_assert(mp1::op_traits<std::plus<std::complex<long double>>>::is_commutative);

// bit_*: integral or std::byte, matching the truth table those operations had before the
// mp1 spellings were removed.
static_assert(mp1::op_traits<std::bit_and<unsigned>>::is_commutative);
static_assert(mp1::op_traits<std::bit_or<std::uint64_t>>::is_commutative);
static_assert(mp1::op_traits<std::bit_xor<std::byte>>::is_commutative);
static_assert(not mp1::op_traits<std::bit_and<double>>::is_commutative);
static_assert(not mp1::op_traits<std::bit_or<user_number>>::is_commutative);

// logical_*: arithmetic only.
static_assert(mp1::op_traits<std::logical_and<bool>>::is_commutative);
static_assert(mp1::op_traits<std::logical_or<int>>::is_commutative);
static_assert(mp1::op_traits<mp1::logical_xor<double>>::is_commutative);
static_assert(not mp1::op_traits<std::logical_and<user_number>>::is_commutative);

// The defaulted void spellings are transparent: op_traits<F> alone cannot know the operand,
// so the call-site predicate classifies them from the reduction element type by forwarding to
// the typed op_traits<max<T>> etc. rule.  The truth table therefore does not depend on the
// spelling, and a user specialization of op_traits<max<T>> also governs mp1::max<>.  As with
// the typed spelling, the value is consumed only when the (functor, category) pair is outside
// the predefined table and a callback is created.
static_assert(not mp1::op_traits<mp1::max<>>::is_commutative);
static_assert(not mp1::op_traits<mp1::min<>>::is_commutative);
static_assert(not mp1::op_traits<mp1::logical_xor<>>::is_commutative);
static_assert(mp1::detail::is_commutative_reduction_v<int, mp1::max<>>);
static_assert(mp1::detail::is_commutative_reduction_v<long long, mp1::min<>>);
static_assert(mp1::detail::is_commutative_reduction_v<char, mp1::max<>>);
static_assert(not mp1::detail::is_commutative_reduction_v<double, mp1::max<>>);
static_assert(not mp1::detail::is_commutative_reduction_v<float, mp1::min<>>);
static_assert(not mp1::detail::is_commutative_reduction_v<user_number, mp1::max<>>);
static_assert(mp1::detail::is_commutative_reduction_v<double, mp1::logical_xor<>>);
static_assert(not mp1::detail::is_commutative_reduction_v<std::string, mp1::logical_xor<>>);

// User opt-in tag and explicit op_traits specialization remain available.
static_assert(mp1::op_traits<tagged_add>::is_commutative);
static_assert(not mp1::op_traits<plain_add>::is_commutative);

// The disjunction used by MPI_Op_create keeps the transparent std functors in place.
static_assert(mp1::detail::is_commutative_reduction_v<int, std::plus<int>>);
static_assert(mp1::detail::is_commutative_reduction_v<int, std::plus<void>>);
static_assert(mp1::detail::is_commutative_reduction_v<int, std::logical_and<int>>);

// The transparent std spellings forward to the typed op_traits row, so the explicit complex
// plus/multiplies row and the integral-or-std::byte bitwise row apply to the void spelling
// exactly as to the typed one; the truth table has one source.
static_assert(mp1::detail::is_commutative_reduction_v<std::complex<double>, std::plus<void>>);
static_assert(
    mp1::detail::is_commutative_reduction_v<std::complex<float>, std::multiplies<void>>);
static_assert(mp1::detail::is_commutative_reduction_v<std::byte, std::bit_and<void>>);
static_assert(mp1::detail::is_commutative_reduction_v<std::byte, std::bit_or<void>>);
static_assert(mp1::detail::is_commutative_reduction_v<std::byte, std::bit_xor<void>>);
static_assert(not mp1::detail::is_commutative_reduction_v<user_number, std::plus<void>>);
static_assert(not mp1::detail::is_commutative_reduction_v<std::string, std::plus<void>>);
static_assert(mp1::detail::is_commutative_reduction_v<std::complex<double>,
                                                      std::plus<std::complex<double>>>);
static_assert(not mp1::detail::is_commutative_reduction_v<double, mp1::max<double>>);
static_assert(not mp1::detail::is_commutative_reduction_v<user_number, std::plus<user_number>>);
static_assert(mp1::detail::is_commutative_reduction_v<int, tagged_add>);

// The mapping is the MPI §6.9.2 row for the operation and the C++ datatype category; it is
// independent of the trait.  Floating max/min, complex sum/product and the C-integer/Logical
// logical functors use the predefined operation; Character/Byte max/min, floating-point
// logical functors and user types stay on the callback.
static_assert(mp1::detail::is_predefined_reduction_functor_v<int, mp1::max<int>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<double, mp1::max<double>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<float, mp1::min<float>>);
// The transparent spelling is classified from the element type through the same gate.
static_assert(mp1::detail::is_predefined_reduction_functor_v<int, mp1::max<>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<double, mp1::max<>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<double, mp1::max<>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<bool, mp1::logical_xor<>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<int, mp1::logical_xor<>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<double, std::plus<double>>);
static_assert(
    not mp1::detail::is_predefined_reduction_functor_v<user_number, std::plus<user_number>>);
// Complex SUM/PROD maps in both collectives and RMA (MPI §6.9.2); the C++ datatype is
// MPI_CXX_DOUBLE_COMPLEX.
static_assert(mp1::detail::is_predefined_reduction_functor_v<std::complex<double>,
                                                             std::plus<std::complex<double>>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<std::complex<double>,
                                                       std::plus<std::complex<double>>>);
// The callback-only categories stay on the callback: Character is not a SUM/MAX category.
static_assert(not mp1::detail::is_predefined_reduction_functor_v<char, mp1::max<char>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<char, std::plus<char>>);

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

  [[nodiscard]] int op_create_calls() {
    return static_cast<int>(mpi_wrap::count("MPI_Op_create"));
  }

  [[nodiscard]] int last_commute() {
    return mpi_wrap::last_commute("MPI_Op_create");
  }

  void reset_op_create_capture() {
    mpi_wrap::reset_counts();
  }

}  // namespace

BOOST_TEST_GLOBAL_FIXTURE(mpi_fixture);

BOOST_AUTO_TEST_SUITE(operator_commutativity_test)

BOOST_AUTO_TEST_CASE(floating_max_maps_to_mpi_max) {
  if (not mpi_wrap::supported())
    return;
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();

  // MPI 4.1 §6.9.2 defines MPI_MAX for floating-point datatypes, so the wrapper uses the
  // predefined operation and passes MPI's result semantics through.  The C++ callback and its
  // commute=0 argument are not used for this category at all.
  static_assert(mp1::detail::is_predefined_reduction_functor_v<double, mp1::max<double>>);
  static_assert(mp1::detail::is_predefined_reduction_functor_v<double, mp1::max<>>);
  BOOST_TEST((mp1::detail::predefined_mpi_op<double, mp1::max<double>>() == MPI_MAX));
  BOOST_TEST((mp1::detail::predefined_mpi_op<double, mp1::min<double>>() == MPI_MIN));

  const double send = 1.0 + static_cast<double>(rank);
  reset_op_create_capture();
  double via_functor = -1.0;
  mp1::allreduce(comm, mp1::max<double>{}, send, via_functor);
  BOOST_TEST(op_create_calls() == 0);
  BOOST_TEST(via_functor == 1.0 + static_cast<double>(size - 1));

  // The transparent spelling maps to the same predefined operation.
  reset_op_create_capture();
  double via_transparent = -1.0;
  mp1::allreduce(comm, mp1::max{}, send, via_transparent);
  BOOST_TEST(op_create_calls() == 0);
  BOOST_TEST(via_transparent == via_functor);
}

BOOST_AUTO_TEST_CASE(user_type_plus_is_created_non_commutative) {
  if (not mpi_wrap::supported())
    return;
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size - 1) / 2;

  // std::plus<user_number> uses the callback (the native datatype is derived, so there is
  // no predefined mapping) and the operation is not assumed commutative.
  user_number recv{};
  reset_op_create_capture();
  mp1::allreduce(comm, std::plus<user_number>{}, user_number{rank}, recv);
  BOOST_TEST(recv.value == expected);
  BOOST_TEST(op_create_calls() == 1);
  BOOST_TEST(last_commute() == 0);

  // The (T, F) cache is process-lifetime: the same std functor creates nothing again.
  user_number cached{};
  reset_op_create_capture();
  mp1::allreduce(comm, std::plus<user_number>{}, user_number{rank}, cached);
  BOOST_TEST(cached.value == expected);
  BOOST_TEST(op_create_calls() == 0);

  // A plain user functor is not commutative unless it opts in.
  int int_recv = -1;
  reset_op_create_capture();
  mp1::allreduce(comm, plain_add{}, rank, int_recv);
  BOOST_TEST(int_recv == expected);
  BOOST_TEST(op_create_calls() == 1);
  BOOST_TEST(last_commute() == 0);
}

BOOST_AUTO_TEST_CASE(user_opt_in_tag_is_created_commutative) {
  if (not mpi_wrap::supported())
    return;
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size - 1) / 2;

  int recv = -1;
  reset_op_create_capture();
  mp1::allreduce(comm, tagged_add{}, rank, recv);
  BOOST_TEST(recv == expected);
  BOOST_TEST(op_create_calls() == 1);
  BOOST_TEST(last_commute() == 1);

  int commute = -1;
  BOOST_REQUIRE(MPI_Op_commutative(mp1::detail::mpi_op<int, tagged_add>(tagged_add{}),
                                   &commute) == MPI_SUCCESS);
  BOOST_TEST(commute == 1);
}

BOOST_AUTO_TEST_CASE(predefined_operations_never_create_a_callback_op) {
  if (not mpi_wrap::supported())
    return;
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  const int expected = size * (size - 1) / 2;

  int recv = -1;
  reset_op_create_capture();
  mp1::allreduce(comm, std::plus<int>{}, rank, recv);
  BOOST_TEST(recv == expected);
  BOOST_TEST(op_create_calls() == 0);

  recv = -1;
  mp1::allreduce(comm, mp1::max<int>{}, rank, recv);
  BOOST_TEST(recv == size - 1);
  BOOST_TEST(op_create_calls() == 0);

  // The transparent spelling maps to the same predefined handle from the element type.
  recv = -1;
  mp1::allreduce(comm, mp1::max{}, rank, recv);
  BOOST_TEST(recv == size - 1);
  BOOST_TEST(op_create_calls() == 0);
  BOOST_TEST((mp1::detail::predefined_mpi_op<int, mp1::max<>>() == MPI_MAX));

  double double_recv = -1.0;
  mp1::allreduce(comm, std::plus<double>{}, static_cast<double>(rank), double_recv);
  BOOST_TEST(double_recv == 0.5 * static_cast<double>(size * (size - 1)));
  BOOST_TEST(op_create_calls() == 0);

  // Floating MAX, complex SUM/PROD and the C-integer/Logical logical operations are all in
  // the MPI §6.9.2 row, so they use the predefined handles as well.
  double floating_max = -1.0;
  reset_op_create_capture();
  mp1::allreduce(comm, mp1::max<double>{}, 1.0 + static_cast<double>(rank), floating_max);
  BOOST_TEST(op_create_calls() == 0);
  BOOST_TEST(floating_max == 1.0 + static_cast<double>(size - 1));

  std::complex<double> complex_sum{};
  reset_op_create_capture();
  mp1::allreduce(comm, std::plus<std::complex<double>>{}, std::complex<double>{1.0, 2.0},
                 complex_sum);
  BOOST_TEST(op_create_calls() == 0);
  BOOST_TEST(complex_sum == std::complex<double>(static_cast<double>(size), 2.0 * size));

  reset_op_create_capture();
  int logical_and_value = 0;
  mp1::allreduce(comm, std::logical_and<int>{}, 3, logical_and_value);
  BOOST_TEST(op_create_calls() == 0);
  int logical_and_raw = 0;
  mp1::allreduce(comm, MPI_LAND, 3, logical_and_raw);
  BOOST_TEST(logical_and_value == logical_and_raw);

  BOOST_TEST((mp1::detail::predefined_mpi_op<int, std::plus<int>>() == MPI_SUM));
  BOOST_TEST((mp1::detail::predefined_mpi_op<int, mp1::max<int>>() == MPI_MAX));
  BOOST_TEST((mp1::detail::predefined_mpi_op<double, std::plus<double>>() == MPI_SUM));
  BOOST_TEST((mp1::detail::is_predefined_reduction_functor_v<double, mp1::max<double>>));
  BOOST_TEST((mp1::detail::is_predefined_reduction_functor_v<std::complex<double>,
                                                             std::plus<std::complex<double>>>));
  BOOST_TEST((mp1::detail::is_predefined_reduction_functor_v<bool, std::logical_and<bool>>));
  BOOST_TEST((mp1::detail::predefined_mpi_op<std::byte, std::bit_or<std::byte>>() == MPI_BOR));
}

BOOST_AUTO_TEST_SUITE_END()
