#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>
#include <numeric>
#include <vector>


template<typename I>
void print_range(const char* const str, I i_1, I i_2) {
  std::cout << str;
  while (i_1 != i_2) {
    std::cout << (*i_1);
    ++i_1;
    std::cout << ((i_1 != i_2) ? ' ' : '\n');
  }
}


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  // run the program with two or more processes
  if (comm_world.size() < 2)
    return EXIT_FAILURE;
  if (comm_world.rank() == 0) {
    // send a message of n elements to rank 1
    enum class tag { send = 29 };
    const int n{12};
    std::vector<int> v(n);
    const auto l{mp1::layout<int>::contiguous(n)};
    std::iota(v.begin(), v.end(), 0);
    mp1::send(comm_world, mp1::region{v.data(), l}, 1, mp1::tag{static_cast<int>(tag::send)});
  }
  if (comm_world.rank() == 1) {
    // receive a message of an a priori unknown number of elements from rank 0
    // first probe for a message from some arbitrary rank with any tag
    const mp1::status s{mp1::probe(comm_world, MPI_ANY_SOURCE, mp1::any_tag)};
    // decode the number of elements, the source and the tag
    const auto n{s.get_count<int>()};
    const int source{s.source()};
    const int tag{s.tag().value};
    std::cerr << "source : " << s.source() << '\n'
              << "tag    : " << s.tag().value << '\n'
              << "error  : " << s.error() << '\n'
              << "count  : " << n << '\n';
    // reserve sufficient amount of memory to receive the message
    std::vector<int> v(static_cast<std::size_t>(n));
    const auto l{mp1::layout<int>::contiguous(n)};
    // finally, receive the message
    mp1::recv(comm_world, mp1::region{v.data(), l}, source, mp1::tag{tag});
    print_range("v = ", v.begin(), v.end());
  }
  return EXIT_SUCCESS;
}
