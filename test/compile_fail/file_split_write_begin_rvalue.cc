// Negative probe: a split-collective write begin records the caller's buffer address until
// the matching end (MPI 4.1 §14.4.5), so a temporary scalar would leave a dangling buffer.
// The write_*_begin value form is a forwarding reference constrained by borrowable<T&&>;
// an rvalue scalar is rejected with a concept diagnostic.  The blocking write* family is
// unaffected.
//
// The positive control is test/compile_pass/file_lvalue_iwrite.cc.

#include "mp1/mp1.hpp"

void begin_temporary(mp1::file& f) {
  f.write_all_begin(42);
}
