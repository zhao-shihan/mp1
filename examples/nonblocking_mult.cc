#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>
#include <vector>


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  double x{1.23456 + comm_world.rank()};
  auto r_send{mp1::isend(comm_world, x, 0)};  // nonblocking send to rank 0
  // rank 0 receives data from all ranks
  if (comm_world.rank() == 0) {
    std::vector<double> v(comm_world.size());
    std::vector<mp1::request> r_pool;
    for (int i{0}; i < comm_world.size(); ++i)
      r_pool.push_back(mp1::irecv(comm_world, v[i], i));
    mp1::wait_all(r_pool);  // wait to finish all receive operations
    for (int i{0}; i < comm_world.size(); ++i)
      std::cout << i << '\t' << v[i] << '\n';
  }
  r_send.wait();  // wait to finish send operation
  return EXIT_SUCCESS;
}
