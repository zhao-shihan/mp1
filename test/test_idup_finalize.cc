// Post-finalize ownership of an MPI_Comm_idup result and of an active idup request.
//
// One process: create a completed idup request (result_pending) and an active idup request,
// finalize MPI, then destroy both.  The destructors must not call MPI; the retained
// communicator and the request handle are abandoned and counted instead.

#include "mp1/mp1.hpp"

#include <cstdlib>
#include <optional>


namespace {

  int failures = 0;

  void check(bool condition) {
    if (not condition)
      ++failures;
  }

}  // namespace


int main(int, char**) {
  mp1::init();
  if (mp1::comm_world().size() != 1)
    return 2;

  mp1::finalize_at_exit();  // ensure a clean exit even if an early return happens below

  const auto self = mp1::comm_self();
  std::optional<mp1::request> pending;
  std::optional<mp1::request> active;

  pending.emplace(mp1::communicator::idup(self));
  (void)pending->wait();
  check(pending->has_result());
  active.emplace(mp1::communicator::idup(self));
  check(active->is_active());

  const std::size_t abandoned_before = mp1::detail::abandoned_handle_count();
  mp1::finalize();
  check(mp1::finalized());
  check(not mp1::available());

  // result_pending after finalize: the duplicate is abandoned and counted; no MPI_Comm_free.
  pending.reset();
  const std::size_t abandoned_after_pending = mp1::detail::abandoned_handle_count();
  check(abandoned_after_pending == abandoned_before + 1);

  // Active after finalize: the request handle is abandoned and counted; the possibly written
  // output slot is abandoned with the payload rather than freed.
  active.reset();
  check(mp1::detail::abandoned_handle_count() > abandoned_after_pending);

  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
