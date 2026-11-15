#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mp1/mp1.hpp>
#include <random>
#include <span>
#include <vector>


// calculate the least common multiple of two arguments
template<typename T>
class lcm {
  // helper: calculate greatest common divisor
  [[nodiscard]] T gcd(T a, T b) const {
    constexpr T zero{};
    if (a < zero)
      a = -a;
    if (b < zero)
      b = -b;
    while (b > zero) {
      const T t{a % b};
      a = b;
      b = t;
    }
    return a;
  }

public:
  T operator()(T a, T b) const noexcept {
    constexpr T zero{};
    const T t{(a / gcd(a, b)) * b};
    if (t < zero)
      return -t;
    return t;
  }
};


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  // generate data
  std::mt19937_64 g(std::time(nullptr) * comm_world.rank());  // random seed
  std::uniform_int_distribution uniform{1, 12};
  const int n{8};
  // populate vector with random data
  std::vector<int> v(n);
  std::generate(v.begin(), v.end(), [&g, &uniform]() { return uniform(g); });
  // calculate the least common multiple and send result to rank 0
  const auto l{mp1::layout<int>::contiguous(n)};
  if (comm_world.rank() == 0) {
    std::vector<int> result(n);
    // calculate the least common multiple
    mp1::reduce(comm_world, lcm<int>{}, 0, v, result);
    // to check the result display data from all ranks
    std::cout << "Arguments:\n";
    for (int r{0}; r < comm_world.size(); ++r) {
      if (r > 0)
        mp1::recv(comm_world, mp1::region{v.data(), l}, r);
      for (auto i : v)
        std::cout << i << '\t';
      std::cout << '\n';
    }
    // display results of global reduction
    std::cout << "\nResults:\n";
    for (auto i : result)
      std::cout << i << '\t';
    std::cout << '\n';
  } else {
    // calculate the least common multiple
    mp1::reduce(comm_world, lcm<int>{}, 0, v, std::span<int>{});
    // send data to rank 0 for display
    mp1::send(comm_world, mp1::region{v.data(), l}, 0);
  }
  return EXIT_SUCCESS;
}
