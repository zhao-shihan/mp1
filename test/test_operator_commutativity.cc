// Built-in reduction functors opt into commutativity only for the exact
// operation/type pairs whose C++ operation is commutative.  The truth table is checked at
// compile time; the MPI_Op_create commute argument is captured at run time through the
// shared mpi_wrap support shim.

#include "mp1/mp1.hpp"
#include "mpi_wrap.hpp"

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
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

// max/min: integral only.  Floating-point max/min is not commutative under the callback
// (x < y) ? y : x once an operand is NaN; a user < is not assumed commutative.
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

// User opt-in tag and explicit op_traits specialization remain available.
static_assert(mp1::op_traits<tagged_add>::is_commutative);
static_assert(not mp1::op_traits<plain_add>::is_commutative);

// The disjunction used by MPI_Op_create keeps the transparent std functors in place.
static_assert(mp1::detail::is_commutative_reduction_v<int, std::plus<int>>);
static_assert(mp1::detail::is_commutative_reduction_v<int, std::plus<void>>);
static_assert(mp1::detail::is_commutative_reduction_v<int, std::logical_and<int>>);
static_assert(mp1::detail::is_commutative_reduction_v<std::complex<double>,
                                                      std::plus<std::complex<double>>>);
static_assert(not mp1::detail::is_commutative_reduction_v<double, mp1::max<double>>);
static_assert(not mp1::detail::is_commutative_reduction_v<user_number, std::plus<user_number>>);
static_assert(mp1::detail::is_commutative_reduction_v<int, tagged_add>);

// The predefined mapping is independent of the trait: floating max/min still use the
// callback, integer max and floating plus still map to the predefined handles.
static_assert(mp1::detail::is_predefined_reduction_functor_v<int, mp1::max<int>>);
static_assert(not mp1::detail::is_predefined_reduction_functor_v<double, mp1::max<double>>);
static_assert(mp1::detail::is_predefined_reduction_functor_v<double, std::plus<double>>);
static_assert(
    not mp1::detail::is_predefined_reduction_functor_v<user_number, std::plus<user_number>>);
// Complex SUM/PROD is mapped only in RMA.  The collective path would use the managed
// callback (the nothrow gate still rejects std::complex today), so the trait assertions
// above preserve the public commutativity truth table rather than a live callback shortcut.
static_assert(not mp1::detail::is_predefined_reduction_functor_v<
              std::complex<double>, std::plus<std::complex<double>>>);
static_assert(mp1::detail::is_predefined_rma_functor_v<std::complex<double>,
                                                       std::plus<std::complex<double>>>);

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

BOOST_AUTO_TEST_CASE(floating_max_is_created_non_commutative) {
  if (not mpi_wrap::supported())
    return;
  const mp1::mpi_communicator comm = mp1::comm_world();
  const int rank = comm.rank();
  const int size = comm.size();
  (void)size;

  // Rank 0 contributes NaN: with commute = 0 MPI fixes the evaluation order to ascending rank
  // order, so (NaN, value) -> NaN propagates through the callback.  A false commute = 1 could
  // reorder the operands and lose the NaN.
  const double send =
      rank == 0 ? std::numeric_limits<double>::quiet_NaN() : 1.0 + static_cast<double>(rank);
  double recv = -1.0;

  reset_op_create_capture();
  mp1::allreduce(comm, mp1::max<double>{}, send, recv);
  BOOST_TEST(op_create_calls() == 1);
  BOOST_TEST(last_commute() == 0);
  BOOST_TEST(std::isnan(recv));

  // The created MPI_Op reports the same commute flag to MPI.
  int commute = -1;
  BOOST_REQUIRE(
      MPI_Op_commutative(mp1::detail::mpi_op<double, mp1::max<double>>(mp1::max<double>{}),
                         &commute) == MPI_SUCCESS);
  BOOST_TEST(commute == 0);

  // The (T, F) cache is process-lifetime: the second reduction creates nothing.
  reset_op_create_capture();
  recv = -1.0;
  mp1::allreduce(comm, mp1::max<double>{}, send, recv);
  BOOST_TEST(op_create_calls() == 0);
  BOOST_TEST(std::isnan(recv));
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

  double double_recv = -1.0;
  mp1::allreduce(comm, std::plus<double>{}, static_cast<double>(rank), double_recv);
  BOOST_TEST(double_recv == 0.5 * static_cast<double>(size * (size - 1)));
  BOOST_TEST(op_create_calls() == 0);

  BOOST_TEST((mp1::detail::predefined_mpi_op<int, std::plus<int>>() == MPI_SUM));
  BOOST_TEST((mp1::detail::predefined_mpi_op<int, mp1::max<int>>() == MPI_MAX));
  BOOST_TEST((mp1::detail::predefined_mpi_op<double, std::plus<double>>() == MPI_SUM));
  BOOST_TEST((not mp1::detail::is_predefined_reduction_functor_v<double, mp1::max<double>>));
  BOOST_TEST((mp1::detail::predefined_mpi_op<std::byte, std::bit_or<std::byte>>() == MPI_BOR));
}

BOOST_AUTO_TEST_SUITE_END()
