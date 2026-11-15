// Positive control for the owner/view compile-fail probes: an owner converts to a view by
// value and its view() accessor returns the borrowed view by value, so all documented call
// shapes compile.  The conversion is also usable where a const view reference is expected
// because the temporary view binds to it.

#include "mp1/mp1.hpp"

void take_communicator(const mp1::mpi_communicator&);

void take_window(const mp1::mpi_window&);

void borrow(const mp1::communicator& owner, const mp1::window& window_owner) {
  take_communicator(owner);
  take_window(window_owner);
  const mp1::mpi_communicator view = owner.view();
  const mp1::mpi_window window_view = window_owner.view();
  (void)view;
  (void)window_view;
}
