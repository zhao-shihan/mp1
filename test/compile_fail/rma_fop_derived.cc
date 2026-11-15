// Negative probe: MPI_Fetch_and_op requires a predefined datatype (MPI 4.1 §12.3.4 Fetch and
// Op).  A reflected aggregate's native datatype is a derived datatype, so the call must be
// rejected at compile time (MPICH aborts "Datatype must be predefined"; Open MPI silently
// performs an undefined partial update).

#include "mp1/mp1.hpp"

int main() {
  const mp1::mpi_window w;
  std::pair<int, int> origin{3, 4};
  std::pair<int, int> result{0, 0};
  mp1::fetch_and_op(w, &origin, &result, 0, 0, MPI_SUM);
  return 0;
}
