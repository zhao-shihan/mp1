#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: matched receives take `message&` only, so an rvalue message in the one-shot
// spelling cannot bind (§1.4).  Name the message object.  The expectation also pins the
// overload reason (the message argument), so a wrong-arity call cannot satisfy the probe.
void probe(const mp1::mpi_communicator& c) {
  int value = 0;
  (void)mp1::mrecv(value, mp1::mprobe(c));
}
