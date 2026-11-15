#include <cstdlib>
#include <functional>
#include <iostream>
#include <mp1/mp1.hpp>


int main() {
  mp1::environment env;
  // check communicator properties of comm_world
  const auto comm_world{mp1::comm_world()};
  std::cout << "comm_world  rank: " << comm_world.rank() << "\tsize: " << comm_world.size()
            << std::endl;
  mp1::barrier(comm_world);
  // split comm_world into 3 disjoint communicators
  // and carry out some collective communication
  auto comm_3{mp1::communicator::split(comm_world, comm_world.rank() % 3)};
  int key;
  if (comm_3.rank() == 0)
    key = comm_world.rank() % 3;
  mp1::bcast(comm_3, 0, key);
  std::cout << "comm_3     rank: " << comm_3.rank() << "\tsize: " << comm_3.size()
            << "\tkey: " << key << std::endl;
  mp1::barrier(comm_world);
  // split comm_world into a communicator which contains all processes
  // except rank 0 of comm_world and carry out some collective communication
  auto comm_without_0{
      mp1::communicator::split(comm_world, comm_world.rank() == 0 ? MPI_UNDEFINED : 1)};
  if (comm_world.rank() != 0) {
    double data{1};
    mp1::allreduce(comm_without_0, std::plus<double>{}, mp1::in_place, data);
    std::cout << "sum: " << data << std::endl;
  }
  mp1::barrier(comm_world);
  std::cout << "comm_world  rank: " << comm_world.rank()
            << "\tcomm valid: " << (comm_without_0.is_valid() ? "yes" : "no") << std::endl;
  return EXIT_SUCCESS;
}
