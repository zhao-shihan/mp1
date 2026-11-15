// Negative probe: the pointer+layout origin spelling is deleted.  A layout origin is a
// region<T> (data pointer + layout reference), so the span/region adapter can derive
// map().size() for the size check.

#include "mp1/mp1.hpp"

int main() {
  const mp1::mpi_window w;
  int origin[4] = {1, 2, 3, 4};
  const mp1::layout<int> origin_layout = mp1::layout<int>::contiguous(4);
  mp1::put(w, origin, origin_layout, 0, 0, 4);
  return 0;
}
