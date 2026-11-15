// Positive control for the combined replace region form.
#include "mp1/impl/p2p.hpp"

int main() {
  const mp1::mpi_communicator c;
  int data = 0;
  mp1::layout<int> operation_layout = mp1::layout<int>::contiguous(1);
  mp1::request r = mp1::isendrecv_replace(c, mp1::region{&data, operation_layout}, 1, mp1::tag{0}, 0, mp1::tag{0});
  (void)r;
  return 0;
}
