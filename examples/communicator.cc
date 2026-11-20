#include <array>
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

  // A group built from MPI-shaped (first, last, stride) triplets, then a communicator created
  // from that group.  create_from_group is collective over the group's members only and needs
  // both an info (here the valid no-hints spelling) and a predefined error handler.
  auto world_group = mp1::get_group(comm_world);
  std::array<mp1::rank_range, 1> every_rank{mp1::rank_range{0, comm_world.size() - 1, 1}};
  auto range_group = mp1::group_range_incl(world_group, every_rank);
  auto from_group = mp1::communicator::create_from_group(
      range_group, "mp1-example-communicator", mp1::info{}, mp1::error_handler::returns);
  std::cout << "from_group  rank: " << from_group.rank() << "\tsize: " << from_group.size()
            << std::endl;

  // MPI_Comm_idup is nonblocking: the duplicate is available only after completion, then
  // take_communicator transfers ownership out of the request.
  mp1::request idup = mp1::communicator::idup(from_group);
  (void)idup.wait();
  auto idup_comm = mp1::take_communicator(idup);
  std::cout << "idup        rank: " << idup_comm.rank() << "\tsize: " << idup_comm.size()
            << std::endl;
  idup_comm.free();
  from_group.free();

  return EXIT_SUCCESS;
}
