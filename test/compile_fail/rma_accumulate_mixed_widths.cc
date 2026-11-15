// Negative probe: two different C integer widths are different predefined basic types, so a
// reflected {uint32_t; uint8_t;} struct is mixed (MPI 4.1 §12.3.4; §6.9.2 names distinct
// C integer datatypes).

#include "mp1/mp1.hpp"

#include <cstdint>

struct mixed_widths {
  std::uint32_t wide;
  std::uint8_t narrow;
};

MP1_REFLECTION(mixed_widths, wide, narrow)

int main() {
  const mp1::mpi_window w;
  mixed_widths origin{};
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_SUM);
  return 0;
}
