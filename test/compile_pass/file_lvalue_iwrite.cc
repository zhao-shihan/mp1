// Positive control for the file rvalue probes: the same nonblocking and split-begin calls
// accept a named lvalue scalar, whose address stays stable until the request completes.

#include "mp1/mp1.hpp"

void write_lvalues(mp1::file& f, int value) {
  (void)f.iwrite(value);
  f.write_all_begin(value);
}
