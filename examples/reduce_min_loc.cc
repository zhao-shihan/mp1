#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mp1/mp1.hpp>
#include <random>
#include <span>
#include <vector>


// data type to store the data and the position of the global minimum; a
// reflected aggregate gives the reduction a native datatype with named fields
struct pair_t {
  double value;
  int rank;
};

MP1_REFLECTION(pair_t, value, rank)


bool operator<(const pair_t& a, const pair_t& b) {
  if (a.value < b.value)
    return true;
  if (b.value < a.value)
    return false;
  return a.rank < b.rank;
}


struct pair_min {
  pair_t operator()(const pair_t& a, const pair_t& b) const noexcept {
    return (b < a) ? b : a;
  }
};


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  // generate data
  std::mt19937_64 g(std::time(nullptr) * comm_world.rank());  // random seed
  std::uniform_real_distribution<> uniform;
  const int n{8};
  // populate vector with random data
  std::vector<pair_t> v(n);
  std::generate(v.begin(), v.end(), [&comm_world, &g, &uniform]() {
    return pair_t{uniform(g), comm_world.rank()};
  });
  // calculate minimum and its location and send result to rank root
  const int root{0};
  const auto l{mp1::layout<pair_t>::contiguous(n)};
  if (comm_world.rank() == root) {
    std::vector<pair_t> result(n);
    // calculate minimum
    mp1::reduce(comm_world, pair_min{}, root, v, result);
    // display data from all ranks
    std::cout << "arguments:\n";
    for (int r{0}; r < comm_world.size(); ++r) {
      if (r > 0)
        mp1::recv(comm_world, mp1::region{v.data(), l}, r);
      for (auto i : v)
        std::cout << std::fixed << std::setprecision(5) << i.value << ' ' << i.rank << '\t';
      std::cout << '\n';
    }
    // display results of global reduction
    std::cout << "\nresults:\n";
    for (const pair_t& i : result)
      std::cout << std::fixed << std::setprecision(5) << i.value << ' ' << i.rank << '\t';
    std::cout << '\n';
  } else {
    // calculate minimum and its location and send result to rank 0
    mp1::reduce(comm_world, pair_min{}, root, v, std::span<pair_t>{});
    // send data to rank 0 for display
    mp1::send(comm_world, mp1::region{v.data(), l}, root);
  }
  return EXIT_SUCCESS;
}
