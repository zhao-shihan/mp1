#include <cstdlib>
#include <mp1/mp1.hpp>
#include <string>
#include <vector>


int main() {
  mp1::environment env;
  using namespace std::string_literals;
  // get a reference to communicator "world"
  const auto comm_world{mp1::comm_world()};
  // spawn one new process per command line
  mp1::info info;
  info.set("host", "localhost");
  mp1::spawn_multiple_options spawns;
  spawns.add({"./process_creation_client"s, "arg1"s}, 1, info)
      .add({"./process_creation_client"s, "arg1"s, "arg2"s}, 1, info)
      .add({"./process_creation_client"s, "arg1"s, "arg2"s, "arg3"s}, 1, info);
  auto inter_comm{mp1::spawn_multiple(comm_world, 0, spawns)};
  // broadcast a message to the created processes
  double message;
  if (comm_world.rank() == 0) {
    // root rank
    message = 1.23;
    mp1::bcast(inter_comm, MPI_ROOT, message);
  } else
    // non-root ranks
    mp1::bcast(inter_comm, MPI_PROC_NULL, message);
  // disconnect is collective over the intercommunicator; the spawned client disconnects
  // its parent intercommunicator before finalizing, so the parent must do the same
  inter_comm.disconnect();

  return EXIT_SUCCESS;
}
