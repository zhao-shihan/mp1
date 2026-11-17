// Negative probe: MPI_Compare_and_swap requires a predefined C++ datatype in the C integer,
// Logical or Byte category (MPI 4.1 §12.3.4 Compare and Swap).  double is Floating point,
// so it must be a compile-time error, not an Open MPI
// MPI_SUCCESS with an undefined result or a MPICH abort.

#include "mp1/mp1.hpp"

int main() {
  const mp1::mpi_window w;
  const double origin = 0.0;
  const double compare = 0.0;
  double result = 0.0;
  mp1::compare_and_swap(w, &origin, &compare, &result, 0, 0);
  return 0;
}
