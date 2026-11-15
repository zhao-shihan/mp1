// Compile-fail: a contiguous container whose element has no MPI datatype mapping is rejected
// at the container boundary with the actionable message. The element_buffer concept already
// excludes it; a same-shape diagnostic overload names the element dependency, so the user
// does not see only "no matching function".
#include "mp1/mp1.hpp"

#include <vector>

namespace {

  struct no_datatype {
    int value = 0;
  };

}  // namespace

void probe(const mp1::mpi_communicator& c, int dest) {
  std::vector<no_datatype> buffer(2);
  (void)mp1::send(c, buffer, dest, mp1::tag{0});
}
