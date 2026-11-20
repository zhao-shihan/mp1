// MPI_ERRHANDLER_NULL is the null handle, not a valid handler (MPI 4.1 §9.3.1);
// mp1::error_handler models only the predefined handlers.

#include "mp1/mp1.hpp"


void create(const mp1::group& g) {
  (void)mp1::communicator::create_from_group(g, "mp1-probe", mp1::info{}, MPI_ERRHANDLER_NULL);
}
