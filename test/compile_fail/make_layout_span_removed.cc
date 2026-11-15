// Compile-fail: `mp1::make_layout_span` is not part of the API; a layout vector converts
// to the typed descriptor's `std::span<const layout<T>>` directly.
#include "mp1/mp1.hpp"

void probe(const std::vector<mp1::layout<int>>& map) {
  (void)mp1::make_layout_span(map);
}
