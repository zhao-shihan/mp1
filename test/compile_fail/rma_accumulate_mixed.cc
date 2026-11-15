// Negative probe: MPI_Accumulate requires a predefined datatype or a derived datatype whose
// basic components all have the same predefined datatype (MPI 4.1 §12.3.4 Accumulate).  A
// reflected struct with an int and a double violates the "same predefined datatype" rule, so
// the basic-component trait must reject it at compile time.  (Previously this was caught by
// the is_predefined_datatype_v<T> static_assert; the message now names the mixed components.)

#include "mp1/mp1.hpp"

struct mixed_value {
  int count;
  double scale;
};

MP1_REFLECTION(mixed_value, count, scale)

int main() {
  const mp1::mpi_window w;
  mixed_value origin{1, 0.5};
  mp1::accumulate(w, mp1::span_from(&origin, mp1::count_t{1}), 0, 0, mp1::count_t{1}, MPI_SUM);
  return 0;
}
