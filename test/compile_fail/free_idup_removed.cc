// idup is a static member of communicator, not a free function.

#include "mp1/mp1.hpp"


void idup(const mp1::mpi_communicator& comm) {
  (void)mp1::idup(comm);
}
