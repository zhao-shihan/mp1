// Compile-fail: the transitional `mp1::layouts<T>` alias is not part of the API; the typed
// descriptor is `std::span<const layout<T>>`.
#include "mp1/mp1.hpp"

void probe(const mp1::layouts<int>& values) {
  (void)values;
}
