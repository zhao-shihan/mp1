// Negative probe: the raw pointer+count data-movement spelling is deleted.  A pointer
// carries no size and the size is the fact the size rule checks, so put/get/accumulate/r*
// accept a std::span or a
// region<T> only.  Use mp1::span_from(pointer, count) or std::span{pointer, count}.

#include "mp1/mp1.hpp"

int main() {
  const mp1::mpi_window w;
  int origin[4] = {1, 2, 3, 4};
  mp1::put(w, origin, 0, 0, 4);
  return 0;
}
