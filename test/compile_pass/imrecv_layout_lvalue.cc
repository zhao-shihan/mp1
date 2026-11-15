// Positive control for the typed matched receive: a named layout in a region.
#include "mp1/impl/p2p.hpp"

int main() {
  int data = 0;
  mp1::message msg;
  mp1::layout<int> operation_layout = mp1::layout<int>::contiguous(1);
  mp1::request r = mp1::imrecv(mp1::region{&data, operation_layout}, msg);
  (void)r;
  return 0;
}
