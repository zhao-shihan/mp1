// Negative probe: file dense access passes one (pointer, count) pair to MPI, so the
// write/iwrite/split-begin range forms require element_buffer (contiguous and sized).  A
// non-contiguous sequence such as std::list is rejected at overload resolution; it must not
// match the range overload and then fail deep inside <ranges> (std::ranges::data is not
// defined for a list).
//
// The positive control is test/compile_pass/file_dense_nonblocking_lvalues.cc.

#include "mp1/mp1.hpp"

#include <list>

void write_sequence(mp1::file& f) {
  std::list<int> values{1, 2, 3};
  (void)f.write(values);
  (void)f.write_at(0, values);
  (void)f.write_at_all_begin(0, values);
}
