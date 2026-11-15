// Negative probe: an owning communicator must not expose a mutable view base.  Binding a
// non-const mpi_communicator& to an owner would let `base = comm_world()` overwrite the owner's
// handle; with composition the owner converts only to a view value, so a mutable lvalue
// reference cannot bind to it.
//
// MPI 4.1 §2.5.1 (Opaque Objects) and §7.4.3 (Communicator Destructors): the owner must remain
// the only code that can replace or free the handle.  The positive control is
// test/compile_pass/owner_view_borrow.cc.

#include "mp1/mp1.hpp"

void overwrite_owner(mp1::communicator& owner) {
  mp1::mpi_communicator& view = owner;
  view = mp1::comm_world();
}
