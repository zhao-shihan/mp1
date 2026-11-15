// Positive control for compile_fail/file_non_element_type.cc: the element-constrained file
// forms still compile for a type with an MPI datatype mapping.
#include "mp1/impl/file.hpp"

void control(mp1::file& f) {
  mp1::layout<int> l = mp1::layout<int>::contiguous(1);
  (void)f.get_type_extent<int>();
  (void)f.get_type_extent(l);
  f.set_view(0, l);
}
