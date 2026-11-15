#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

// Compile-fail: a string literal is a built-in character array; it would send its NUL
// Use std::string_view or std::string.
void probe(const mp1::mpi_communicator& c) {
  mp1::send(c, "abc", 0);
}
