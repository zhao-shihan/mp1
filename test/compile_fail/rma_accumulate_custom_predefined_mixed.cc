// Negative probe: a custom predefined trait is its own (opaque) basic component, so combining
// it with a different C++ type that happens to map to the same MPI handle cannot be proven
// homogeneous at compile time: pair<custom,int> is unknown, not int (exact limitation; use a
// reflected int struct or a plain int pair instead).

#include "mp1/mp1.hpp"

struct custom_int {
  int value;
};

MP1_DATATYPE_TRAITS(custom_int, MPI_INT)

struct custom_and_int {
  custom_int custom;
  int plain;
};

MP1_REFLECTION(custom_and_int, custom, plain)

int main() {
  const mp1::mpi_window w;
  custom_and_int origin{};
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_SUM);
  return 0;
}
