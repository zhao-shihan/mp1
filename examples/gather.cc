#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>
#include <span>
#include <vector>


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  const auto c_rank{comm_world.rank()};
  const auto c_size{comm_world.size()};
  // gather a single int from all ranks to rank root=0
  {
    const int root{0};
    int x{c_rank + 1};
    if (c_rank == root) {
      std::vector<int> y(c_size);
      mp1::gather(comm_world, root, x, y);
      for (int i{0}; i < c_size; ++i)
        std::cout << y[i] << ' ';
      std::cout << "\n";
    } else
      mp1::gather(comm_world, root, x, std::span<int>{});
  }
  // gather a single int from all ranks to rank root=0
  // root and non-root rank use different function overloads of gather
  {
    const int root{0};
    int x{-(c_rank + 1)};
    if (c_rank == root) {
      std::vector<int> y(c_size);
      mp1::gather(comm_world, root, x, y);
      for (int i{0}; i < c_size; ++i)
        std::cout << y[i] << ' ';
      std::cout << "\n";
    } else
      mp1::gather(comm_world, root, x, std::span<int>{});
  }
  // gather several ints from all ranks to rank root=0
  {
    const int root{0}, n{3};
    std::vector<int> x(n, c_rank + 1);
    if (c_rank == root) {
      std::vector<int> y(n * c_size);
      mp1::gather(comm_world, root, x, y);
      for (int i{0}; i < c_size * n; ++i)
        std::cout << y[i] << ' ';
      std::cout << "\n";
    } else
      mp1::gather(comm_world, root, x, std::span<int>{});
  }
  return EXIT_SUCCESS;
}
