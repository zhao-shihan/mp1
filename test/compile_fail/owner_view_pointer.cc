// Negative probe: an owner's address must not convert to a view pointer.  A view pointer
// would permit deleting through a non-owning base (an ownership mistake);
// with composition an owner converts only to a view value, never to a view pointer.
//
// MPI 4.1 §2.5.1 (Opaque Objects): only the owner may deallocate the opaque object.  The
// positive control is test/compile_pass/owner_view_borrow.cc.

#include "mp1/mp1.hpp"

mp1::mpi_communicator* communicator_view_pointer(mp1::communicator& owner) {
  return &owner;
}

mp1::mpi_window* window_view_pointer(mp1::window& owner) {
  return &owner;
}
