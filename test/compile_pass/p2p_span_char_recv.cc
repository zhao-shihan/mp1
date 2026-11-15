#include "mp1/impl/p2p.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <vector>

void control(const mp1::mpi_communicator& c) {
  std::array<char, 4> buffer{};
  (void)mp1::recv(c, std::span(buffer), 0);
}
