// Compile-fail: the container send form of isendrecv requires a named lvalue container; the
// request borrows its storage until completion (MPI 4.1 §3.7.2).
#include "mp1/impl/p2p.hpp"

#include <vector>

void probe(const mp1::mpi_communicator& c) {
  std::vector<int> recv(1, 0);
  (void)mp1::isendrecv(c, std::vector<int>{1}, 0, mp1::tag{0}, recv, 1, mp1::tag{0});
}
