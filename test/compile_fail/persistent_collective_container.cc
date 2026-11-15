// Compile-fail: persistent collectives are raw forms only (scalar, pointer+count, layout).
// There is no container overload: the plan captures the caller buffer for every restart, and
// a container's storage may reallocate between starts (MPI 4.1 §6.13).
#include "mp1/impl/collective.hpp"

#include <functional>
#include <vector>

void probe(const mp1::mpi_communicator& c) {
  std::vector<int> send(2, 0);
  std::vector<int> recv(2, 0);
  std::plus<int> plus;
  (void)mp1::reduce_init(c, plus, 0, send, recv);
}
