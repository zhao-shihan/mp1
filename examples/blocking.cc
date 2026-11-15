#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  // run the program with two or more processes
  if (comm_world.size() < 2)
    return EXIT_FAILURE;
  // process 0 sends
  enum class tag_enum : short { tag = 1 };
  if (comm_world.rank() == 0) {
    // see MPI Standard for the semantics of standard send, buffered send,
    // synchronous send and ready send
    double x{1.23456};
    // send x to rank 1 via standard send
    mp1::send(comm_world, x, 1, mp1::tag{static_cast<int>(tag_enum::tag)});
    ++x;
    {
      // create a buffer for buffered send,
      // memory will be freed on leaving the scope
      const int size{static_cast<int>(sizeof(x))};
      mp1::bsend_buffer buff{size};
      mp1::bsend(comm_world, x, 1);  // send x to rank 1 via buffered send
    }
    ++x;
    mp1::ssend(comm_world, x, 1);  // send x to rank 1 via synchronous send
    ++x;
    mp1::rsend(comm_world, x, 1);  // send x to rank 1 via ready send
  }
  // process 1 receives
  if (comm_world.rank() == 1) {
    double x;
    // receive x from rank 0
    mp1::recv(comm_world, x, 0, mp1::tag{static_cast<int>(tag_enum::tag)});
    std::cout << "x = " << x << '\n';
    mp1::recv(comm_world, x, 0);  // receive x from rank 0
    std::cout << "x = " << x << '\n';
    mp1::recv(comm_world, x, 0);  // receive x from rank 0
    std::cout << "x = " << x << '\n';
    mp1::recv(comm_world, x, 0);  // receive x from rank 0
    std::cout << "x = " << x << '\n';
  }
  return EXIT_SUCCESS;
}
