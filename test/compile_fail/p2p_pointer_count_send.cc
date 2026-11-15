#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: the pointer+count send form is deleted; span_from is the checked
// pointer+count spelling.
void probe(const mp1::mpi_communicator& c) {
  int data[4] = {0, 0, 0, 0};
  mp1::send(c, &data[0], 4, mp1::tag{0});
}
