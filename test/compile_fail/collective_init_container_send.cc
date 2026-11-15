// Compile-fail: a persistent collective accepts fixed buffers only (span/array/C array/
// region/scalar).  A resizable container has no per-start data()/size() revalidation, so its
// send position is rejected by `!resizable_element_buffer`.
#include "mp1/mp1.hpp"

#include <array>
#include <span>
#include <vector>

void probe(const mp1::mpi_communicator& c) {
  std::vector<int> send(2, 0);
  std::array<int, 2> recv{0, 0};
  (void)mp1::gather_init(c, 0, send, std::span<int>{recv});
}
