// Compile-fail: the transitional `mp1::displacements` alias is not part of the API; typed
// element displacements are `std::span<const count_t>`.
#include "mp1/mp1.hpp"

void probe(const mp1::displacements& values) {
  (void)values;
}
