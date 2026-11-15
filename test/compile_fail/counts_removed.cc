// Compile-fail: the transitional `mp1::counts` alias is not part of the API.  The dense
// descriptor is `mp1::blocks`.
#include "mp1/mp1.hpp"

void probe(const mp1::counts& values) {
  (void)values;
}
