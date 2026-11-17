// Positive compile check of the transparent mp1 reduction functor spellings: the element-type
// parameter defaults to void, so mp1::max{}/min{}/logical_xor{} deduce their operands and work
// with the reduction families and with the RMA predefined_op conversion.

#include "mp1/mp1.hpp"

#include <complex>
#include <functional>
#include <type_traits>

// A user opt-in specialization on a transparent type itself is preserved by the
// is_commutative_reduction_v disjunction.
namespace mp1 {

  template<>
  struct op_traits<mp1::max<>> {
    static constexpr bool is_commutative = true;
  };

  template<>
  struct op_traits<std::plus<void>> {
    static constexpr bool is_commutative = true;
  };

}  // namespace mp1

static_assert(mp1::detail::is_commutative_reduction_v<double, mp1::max<>>);
static_assert(mp1::detail::is_commutative_reduction_v<double, std::plus<void>>);

static_assert(std::is_same_v<mp1::max<>, mp1::max<void>>);
static_assert(std::is_same_v<mp1::min<>, mp1::min<void>>);
static_assert(std::is_same_v<mp1::logical_xor<>, mp1::logical_xor<void>>);

// The transparent operator() returns a value of the common operand type.
static_assert(mp1::max{}(3, 5) == 5);
static_assert(mp1::min{}(3, 5) == 3);
static_assert(mp1::logical_xor{}(true, false));
static_assert(not mp1::logical_xor{}(true, true));
static_assert(std::is_same_v<decltype(mp1::max{}(1, 2L)), long>);
static_assert(std::is_same_v<decltype(mp1::max{}(1, 2)), int>);
static_assert(std::is_same_v<decltype(mp1::logical_xor{}(true, false)), bool>);
static_assert(std::is_same_v<decltype(mp1::logical_xor{}(1, 2)), int>);

// The transparent logical_xor normalizes to the common operand type, so an enum element type
// stays usable exactly as with the typed spelling.
enum class parity_flag : unsigned char { off = 0, on = 1 };
static_assert(std::is_same_v<decltype(mp1::logical_xor{}(parity_flag::on, parity_flag::off)),
                             parity_flag>);

int main() {
  int value = 0;
  mp1::allreduce(mp1::comm_world(), mp1::max{}, 1, value);
  mp1::allreduce(mp1::comm_world(), mp1::min{}, 1, value);
  mp1::allreduce(mp1::comm_world(), mp1::logical_xor{}, 1, value);

  // Floating-point max/min and complex sum/product are predefined §6.9.2 pairs.
  double double_value = 0.0;
  mp1::allreduce(mp1::comm_world(), mp1::max{}, 1.0, double_value);
  std::complex<double> complex_value{};
  mp1::allreduce(mp1::comm_world(), std::plus<std::complex<double>>{}, complex_value,
                 complex_value);

  // C-integer and bool logical operations use the predefined handles.
  bool bool_value = false;
  mp1::allreduce(mp1::comm_world(), mp1::logical_xor{}, true, bool_value);

  // RMA: the same object converts to the typed predefined_op<U> from the element type.
  const mp1::mpi_window w;
  const double origin = 0.0;
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, mp1::max{});

  // Transparent logical_xor maps for bool (MPI_CXX_BOOL is the Logical category).
  const bool flag = false;
  mp1::accumulate(w, mp1::span_from(&flag, 1), 0, 0, 1, mp1::logical_xor{});
  return value + static_cast<int>(double_value);
}
