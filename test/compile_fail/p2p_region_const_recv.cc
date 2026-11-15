#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: region<const T> is send-only; a receive region must be mutable (§2.2/§2.5).
void probe(const mp1::mpi_communicator& c) {
  int data = 0;
  mp1::layout<int> map = mp1::layout<int>::contiguous(1);
  (void)mp1::recv(c, mp1::region<const int>{&data, map}, 0);
}
