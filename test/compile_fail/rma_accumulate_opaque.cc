// Negative probe: a custom non-predefined datatype whose basic component cannot be derived
// from the C++ type is opaque.  Without MP1_DATATYPE_BASIC the accumulating families are a
// compile-time error (there is no general runtime classifier and no raw-datatype escape in
// 1.0; use MP1_DATATYPE_BASIC(T,U), a reflected aggregate, or native MPI).

#include "mp1/mp1.hpp"

struct opaque_value {
  int value;
};

MP1_DATATYPE_TRAITS_PACKED(opaque_value, MPI_INT)

int main() {
  const mp1::mpi_window w;
  opaque_value origin{};
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_SUM);
  return 0;
}
