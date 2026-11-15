// Compile-fail: inter_communicator and tag_t are removed names.
#include "mp1/impl/communicator.hpp"

int main() {
  mp1::inter_communicator inter;
  mp1::tag_t tag;
  (void)inter;
  (void)tag;
}
