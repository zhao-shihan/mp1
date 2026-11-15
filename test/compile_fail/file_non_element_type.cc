// Compile-fail: `template<element T>` makes a non-element type a constraint failure rather
// than an in-body static_assert.  `opaque` has no MPI datatype mapping.
#include "mp1/impl/file.hpp"

struct opaque {};

void probe(const mp1::file& f) {
  (void)f.get_type_extent<opaque>();
}
