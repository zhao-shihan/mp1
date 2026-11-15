// Compile-fail: the in-place nonblocking scatter borrows the caller's container storage until
// completion (MPI 4.1 §3.7.2), so an rvalue container has no stable address.
#include "mp1/impl/collective.hpp"

#include <vector>

void probe(const mp1::mpi_communicator& c) {
  (void)mp1::iscatter(c, 0, std::vector<int>(4), mp1::in_place);
}
