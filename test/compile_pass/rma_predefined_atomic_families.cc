// Positive control for the RMA type/op boundary compile-fail probes: a predefined C integer,
// Logical and Byte datatype is accepted by compare_and_swap, and complex SUM/PROD, bool
// logical operations and std::byte bitwise operations are accepted by the accumulating
// families (MPI 4.1 §12.3.4 and §6.9.2 Predefined Reduction Operations).

#include "mp1/mp1.hpp"

#include <complex>
#include <cstddef>
#include <functional>

int main() {
  const mp1::mpi_window w;
  int i = 0;
  int result = 0;
  mp1::compare_and_swap(w, &i, &i, &result, 0, 0);

  bool flag = false;
  mp1::compare_and_swap(w, &flag, &flag, &flag, 0, 0);

  std::byte byte{};
  mp1::compare_and_swap(w, &byte, &byte, &byte, 0, 0);

  unsigned long long wide = 0;
  mp1::compare_and_swap(w, &wide, &wide, &wide, 0, 0);

  std::complex<double> complex_origin{}, complex_result{};
  mp1::accumulate(w, mp1::span_from(&complex_origin, 1), 0, 0, 1,
                  std::plus<std::complex<double>>{});
  mp1::fetch_and_op(w, &complex_origin, &complex_result, 0, 0, MPI_SUM);
  mp1::get_accumulate(w, mp1::span_from(&complex_origin, 1), mp1::span_from(&complex_result, 1),
                      0, 0, 1, MPI_PROD);

  bool logical = false;
  mp1::accumulate(w, mp1::span_from(&logical, 1), 0, 0, 1, std::logical_and<bool>{});

  mp1::accumulate(w, mp1::span_from(&byte, 1), 0, 0, 1, std::bit_or<std::byte>{});

  char character = 'a';
  mp1::accumulate(w, mp1::span_from(&character, 1), 0, 0, 1, MPI_REPLACE);
  return 0;
}
