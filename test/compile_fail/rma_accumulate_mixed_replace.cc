// Negative probe: MPI_REPLACE is legal for a predefined or homogeneous-derived datatype, but
// the §12.3.4 homogeneity rule still applies (MPI 4.1 §12.3.4; the historical Annex B
// "predefined only" sentence is the withdrawn MPI 2.1 rule).  A heterogeneous derived type
// must be a compile-time error and must never reach MPI, where MPICH aborts in
// MPIR_DATATYPE_IS_PREDEFINED and Open MPI returns MPI_ERR_TYPE.

#include "mp1/mp1.hpp"

struct mixed_replace_value {
  int count;
  double scale;
};

MP1_REFLECTION(mixed_replace_value, count, scale)

int main() {
  const mp1::mpi_window w;
  mixed_replace_value origin{1, 0.5};
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_REPLACE);
  return 0;
}
