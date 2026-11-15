// Negative probe: a writable result position takes span<T> or region<T>; span<const T> and
// region<const T> are read-only and must not be accepted by get/rget/get_accumulate/
// rget_accumulate.  The get result is written by MPI (MPI 4.1 §12.3.2).  Both result
// spellings are present, so the probe pins the span and the region form in one translation
// unit; both declarations are ill-formed for the same reason (no viable overload).

#include "mp1/mp1.hpp"

#include <span>

int main() {
  const mp1::mpi_window w;
  const int result[4] = {0, 0, 0, 0};
  mp1::get(w, std::span<const int>{result, 4}, 0, 0, 4);
  mp1::layout<int> map = mp1::layout<int>::contiguous(4);
  mp1::get(w, mp1::region<const int>{result, map}, 0, 0, 4);
  return 0;
}
