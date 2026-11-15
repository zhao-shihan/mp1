// Negative probe: MPI RMA only permits predefined operators (MPI 4.1 §12.3.4: "User-defined
// operators cannot be used").  mp1 maps built-in reduction functors to predefined MPI_Op
// values, but an arbitrary user functor has no such mapping and must be rejected at compile
// time instead of being turned into a user-defined MPI_Op that MPI would reject at run time.

#include "mp1/mp1.hpp"

namespace {

  struct user_add {
    int operator()(int left, int right) const {
      return left + right;
    }
  };

}  // namespace

int main() {
  const mp1::mpi_window w;
  const int origin = 0;
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, user_add{});
  return 0;
}
