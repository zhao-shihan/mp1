#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: an associative container is keyed, not a flat element sequence; it is
// excluded on send and receive.  Materialize to std::vector<std::pair<K,V>>.
void probe(const mp1::mpi_communicator& c) {
  std::map<int, int> entries{{1, 2}};
  mp1::send(c, entries, 0);
}
