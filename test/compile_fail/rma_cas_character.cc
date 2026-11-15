// Negative probe: character datatypes (MPI_CHAR/MPI_WCHAR) are predefined but are not in any
// of the MPI_Compare_and_swap categories (MPI 4.1 §12.3.4 Compare and Swap: C integer,
// Fortran integer, Logical, Multi-language, Byte) and cannot be used in reductions (§6.9.3
// Signed Characters and Reductions).  The typed boundary must reject them at compile time.

#include "mp1/mp1.hpp"

int main() {
  const mp1::mpi_window w;
  const char origin = 'a';
  const char compare = 'b';
  char result = 'c';
  mp1::compare_and_swap(w, &origin, &compare, &result, 0, 0);
  return 0;
}
