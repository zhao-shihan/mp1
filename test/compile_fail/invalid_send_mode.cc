// Compile-fail: an out-of-range detail::send_mode must be rejected by every dispatch
// static_assert, not silently treated as send_mode::ready.
#include "mp1/impl/p2p.hpp"

int main() {
  constexpr auto invalid = static_cast<mp1::detail::send_mode>(99);
  (void)mp1::detail::send_mode_name<invalid>();
  (void)mp1::detail::isend_mode_name<invalid>();
  (void)mp1::detail::send_init_mode_name<invalid>();
  (void)&mp1::detail::native_send<invalid>;
  (void)&mp1::detail::native_isend<invalid>;
  (void)&mp1::detail::native_send_init<invalid>;
}
