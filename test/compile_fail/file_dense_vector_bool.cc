// Negative probe: std::vector<bool> is the proxy-reference specialization and has no
// contiguous bool storage, so the dense container overloads are disabled for it.  The call
// selects the scalar overload and fails the element-datatype check instead of silently
// treating the proxy storage as bools.
//
// The positive control is test/compile_pass/file_dense_nonblocking_lvalues.cc.

#include "mp1/mp1.hpp"

#include <vector>

void read_vector_bool(mp1::file& f) {
  std::vector<bool> values{true, false};
  (void)f.iread_at(0, values);
}
