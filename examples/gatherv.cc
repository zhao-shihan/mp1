#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>
#include <numeric>
#include <span>
#include <vector>


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  const auto c_rank{comm_world.rank()};
  const auto c_size{comm_world.size()};
  // fill vector with C_rank+1 elements, each having the value C_rank+1
  std::vector<int> x(c_rank + 1, c_rank + 1);
  const auto l{mp1::layout<int>::contiguous(c_rank + 1)};
  // root rank will send and receive in gather operation
  if (c_rank == 0) {
    // messages of varying size will be received; each rank contributes i + 1
    // elements, placed contiguously in the receive buffer
    std::vector<mp1::count_t> counts(static_cast<std::size_t>(c_size));
    for (int i{0}; i < c_size; ++i)
      counts[static_cast<std::size_t>(i)] = i + 1;
    std::vector<mp1::count_t> displacements(static_cast<std::size_t>(c_size), 0);
    std::partial_sum(counts.begin(), counts.end() - 1, displacements.begin() + 1);
    std::vector<int> y((c_size * c_size + c_size) / 2);  // receive buffer
    mp1::gatherv(comm_world, 0, mp1::region{x.data(), l}, y,
                 mp1::blocks{.counts = counts, .displacements = displacements});
    // print data
    for (const auto& f : y)
      std::cout << f << '\n';
  } else
    // non-root ranks just send
    mp1::gatherv(comm_world, 0, mp1::region{x.data(), l}, std::span<int>{}, mp1::blocks{});
  return EXIT_SUCCESS;
}
