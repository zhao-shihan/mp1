#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>
#include <string>


int main() {
  mp1::environment env;
  // get a reference to communicator "world"
  const auto comm_world{mp1::comm_world()};
  // each process prints a message containing the processor name, the rank
  // in communicator world and the size of communicator world
  // output may depend on MPI implementation
  std::cout << "Hello world! I am running on \"" << mp1::processor_name() << "\". My rank is "
            << comm_world.rank() << " out of " << comm_world.size() << " processes.\n";
  // if there are two or more processes send a message from process 0 to process 1
  if (comm_world.size() >= 2) {
    if (comm_world.rank() == 0) {
      std::string message{"Hello world!"};
      mp1::send(comm_world, message, 1);  // send message to rank 1
    } else if (comm_world.rank() == 1) {
      std::string message;
      mp1::recv(comm_world, message, 0);  // receive message from rank 0
      std::cout << "got: \"" << message << "\"\n";
    }
  }
  return EXIT_SUCCESS;
}
