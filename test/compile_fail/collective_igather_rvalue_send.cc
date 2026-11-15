// Compile-fail: a nonblocking collective borrows the send buffer address until completion
// (MPI 4.1 §6.12), so the forwarding-reference send position requires `borrowable<S&&>` and
// an rvalue container has no stable address.
#include "mp1/mp1.hpp"

#include <span>
#include <vector>

void probe(const mp1::mpi_communicator& c) {
  std::vector<int> recv(2, 0);
  (void)mp1::igather(c, 0, std::vector<int>(2, 0), std::span<int>{recv});
}
