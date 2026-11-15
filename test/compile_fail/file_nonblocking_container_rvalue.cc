// Negative probe: a nonblocking file write borrows the container's contiguous storage until
// the request completes (MPI 4.1 §14.4.1), so a temporary container would leave a dangling
// data() address.  The buffer form is a forwarding reference that adds borrowable<R&&>
// and rejects an rvalue container with a concept diagnostic; the blocking write* family
// still accepts rvalues because it completes before returning.
//
// The positive control is test/compile_pass/file_dense_nonblocking_lvalues.cc.

#include "mp1/mp1.hpp"

#include <vector>

void write_temporary(mp1::file& f) {
  (void)f.iwrite(std::vector<int>{1, 2, 3});
}
