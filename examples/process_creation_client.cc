#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>


int main(int argc, char* argv[]) {
  // spawned processes start without MPI initialized; the environment guard owns the
  // init/finalize pair and the parent communicator disconnects before finalize
  mp1::environment env;

  // get the parent inter-communicator; it is owned by the returned value and is
  // disconnected when it goes out of scope
  auto inter_comm{mp1::parent()};
  std::cout << "Hello world! I am running on \"" << mp1::processor_name() << "\". My rank is "
            << inter_comm.rank() << " out of " << inter_comm.size() << " processes.\n";
  std::cout << "commandline arguments: ";
  for (int i{0}; i < argc; ++i)
    std::cout << argv[i] << ' ';
  std::cout << std::endl;
  double message;
  mp1::bcast(inter_comm, 0, message);
  std::cout << "got: " << message << '\n';
  return EXIT_SUCCESS;
}
