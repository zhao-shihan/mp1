#include <array>
#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>
#include <numeric>
#include <utility>
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

  // test layout for a piece of contiguous memory
  if (comm_world.rank() == 0) {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 1);                    // fill vector with some data
    const auto l{mp1::layout<int>::contiguous(10)};      // layout with 10 elements
    mp1::send(comm_world, mp1::region{v.data(), l}, 1);  // send data to rank 1
  }
  if (comm_world.rank() == 1) {
    std::vector<int> v(20, 0);
    const auto l{mp1::layout<int>::contiguous(10)};      // layout with 10 elements
    mp1::recv(comm_world, mp1::region{v.data(), l}, 0);  // receive data from rank 0
    print_range("v = ", v.begin(), v.end());
  }

  // test layout for a piece of contiguous memory
  // a contiguous layout and a vector layout with one block are almost equivalent
  // contiguous layout has some additional internal bookkeeping
  if (comm_world.rank() == 0) {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 1);                    // fill vector with some data
    const auto l{mp1::layout<int>::contiguous(10)};      // layout with 10 elements
    mp1::send(comm_world, mp1::region{v.data(), l}, 1);  // send data to rank 1
  }
  if (comm_world.rank() == 1) {
    std::vector<int> v(20, 0);
    // vector layout with 10 elements
    const auto l{mp1::layout<int>::vector(
        mp1::vector_options<int>{.count = 1, .blocklength = 10, .stride = 10})};
    mp1::recv(comm_world, mp1::region{v.data(), l}, 0);  // receive data from rank 0
    print_range("v = ", v.begin(), v.end());
  }

  // test layout for pieces of contiguous memory (equally spaced blocks of constant size)
  // layouts on sending and receiving side may differ but must be compatible
  if (comm_world.rank() == 0) {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 1);                    // fill vector with some data
    const auto l{mp1::layout<int>::contiguous(3 * 4)};   // layout with 12 elements
    mp1::send(comm_world, mp1::region{v.data(), l}, 1);  // send data to rank 1
  }
  if (comm_world.rank() == 1) {
    std::vector<int> v(20, 0);
    // 3 blocks of length 4 with a block spacing of 6
    const auto l{mp1::layout<int>::vector(
        mp1::vector_options<int>{.count = 3, .blocklength = 4, .stride = 6})};
    mp1::recv(comm_world, mp1::region{v.data(), l}, 0);  // receive data from rank 0
    print_range("v = ", v.begin(), v.end());
  }

  // test layout for a sequence of blocks of memory of varying block length
  // layouts on sending and receiving side may differ but must be compatible
  if (comm_world.rank() == 0) {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 1);                       // fill vector with some data
    const auto l{mp1::layout<int>::contiguous(3 + 4 + 2)};  // layout with 9 elements
    mp1::send(comm_world, mp1::region{v.data(), l}, 1);     // send data to rank 1
  }
  if (comm_world.rank() == 1) {
    std::vector<int> v(20, 0);
    // 1st block of length 3 at displacement 1, 2nd block of length 4 at
    // displacement 8, 3rd block of length 2 at displacement 16
    const std::array entries{std::pair<mp1::count_t, mp1::count_t>{3, 1},
                             std::pair<mp1::count_t, mp1::count_t>{4, 8},
                             std::pair<mp1::count_t, mp1::count_t>{2, 16}};
    const auto l{mp1::layout<int>::indexed(mp1::indexed_parameter{entries})};
    mp1::recv(comm_world, mp1::region{v.data(), l}, 0);  // receive data from rank 0
    print_range("v = ", v.begin(), v.end());
  }

  // test layout for a sequence of blocks of memory of constant block length
  // layouts on sending and receiving side may differ but must be compatible
  if (comm_world.rank() == 0) {
    std::vector<int> v(20);
    std::iota(v.begin(), v.end(), 1);                    // fill vector with some data
    const auto l{mp1::layout<int>::contiguous(3 * 3)};   // layout with 9 elements
    mp1::send(comm_world, mp1::region{v.data(), l}, 1);  // send data to rank 1
  }
  if (comm_world.rank() == 1) {
    std::vector<int> v(20, 0);
    const std::array<mp1::count_t, 3> displacements{1, 8, 12};
    const auto l{
        mp1::layout<int>::indexed_block(3, mp1::indexed_block_parameter{displacements})};
    mp1::recv(comm_world, mp1::region{v.data(), l}, 0);  // receive data from rank 0
    print_range("v = ", v.begin(), v.end());
  }

  // test layouts of layouts
  // the layouts on sending and receiving side may differ but must be compatible
  if (comm_world.rank() == 0) {
    std::vector<int> v(3 * 3 * 4);
    std::iota(v.begin(), v.end(), 1);                       // fill vector with some data
    const auto l{mp1::layout<int>::contiguous(3 * 3 * 4)};  // layout with 36 elements
    mp1::send(comm_world, mp1::region{v.data(), l}, 1);     // send data to rank 1
  }
  if (comm_world.rank() == 1) {
    std::vector<int> v(15 * 4, 0);
    // layout consists of 3 blocks of length 3 (3 * 3 = 9 ints in total)
    const std::array<mp1::count_t, 3> displacements{1, 8, 12};
    auto l1{mp1::layout<int>::indexed_block(3, mp1::indexed_block_parameter{displacements})};
    // the layout l1 starts with a hole, by default this hole is ignored when combining
    // several versions of l1, thus we explicitly set its lower bound to 0 and its
    // extent to 15 (= start of last block plus block length)
    l1.byte_resize(0, 15 * static_cast<mp1::count_t>(sizeof(int)));
    // concatenate 4 indexed layouts, the resulting layout holds 3 * 3 * 4 ints
    const auto l2{mp1::layout<int>::contiguous(4, l1)};   // contiguous layout of l1
    mp1::recv(comm_world, mp1::region{v.data(), l2}, 0);  // receive data from rank 0
    print_range("v = ", v.begin(), v.end());
  }

  return EXIT_SUCCESS;
}
