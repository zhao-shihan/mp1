// Positive control for compile_fail/invalid_send_mode.cc: every in-range send
// mode reaches the native dispatch without tripping the out-of-range sentinel.
#include "mp1/impl/p2p.hpp"

int main() {
  using mp1::detail::send_mode;
  (void)mp1::detail::send_mode_name<send_mode::standard>();
  (void)mp1::detail::isend_mode_name<send_mode::buffered>();
  (void)mp1::detail::send_init_mode_name<send_mode::synchronous>();

  const mp1::detail::buffer_view buffer{nullptr, 0, MPI_INT};
  mp1::detail::native_send<send_mode::ready>(buffer, 0, 0, MPI_COMM_WORLD);
  mp1::request isend_handle =
      mp1::detail::native_isend<send_mode::standard>(buffer, 0, 0, MPI_COMM_WORLD, nullptr);
  (void)isend_handle;
  mp1::request send_init_handle =
      mp1::detail::native_send_init<send_mode::buffered>(buffer, 0, 0, MPI_COMM_WORLD);
  (void)send_init_handle;
  return 0;
}
