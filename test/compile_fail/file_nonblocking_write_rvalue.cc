// Negative probe: a nonblocking file write keeps the caller's buffer address until the
// request completes (MPI 4.1 §14.4.1), so a temporary scalar would leave a dangling buffer.
// The value form is a forwarding reference constrained by borrowable<T&&>; an rvalue
// scalar selects that constraint and is rejected with a concept diagnostic.  The blocking
// write* family still accepts rvalues because it completes before returning.
//
// The positive control is test/compile_pass/file_lvalue_iwrite.cc.

#include "mp1/mp1.hpp"

void write_temporary(mp1::file& f) {
  (void)f.iwrite(42);
}
