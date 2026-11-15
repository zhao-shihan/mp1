#include <cstdlib>
#include <functional>
#include <iostream>
#include <mp1/mp1.hpp>
#include <numeric>
#include <span>
#include <vector>


struct my_array {
  static constexpr int n_0{3};
  static constexpr int n_1{4};
  double data[n_0][n_1]{};
  // overload operators for some syntactical sugar
  const double& operator()(int i_0, int i_1) const {
    return data[i_0][i_1];
  }
  double& operator()(int i_0, int i_1) {
    return data[i_0][i_1];
  }
};


// use reflection macro to make the struct compatible with mp1
MP1_REFLECTION(my_array, data)


// overload plus operator
my_array operator+(const my_array& a, const my_array& b) noexcept {
  my_array res;
  for (int i_1{0}; i_1 < my_array::n_1; ++i_1)
    for (int i_0{0}; i_0 < my_array::n_0; ++i_0)
      res(i_0, i_1) = a(i_0, i_1) + b(i_0, i_1);
  return res;
}


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  int root{0};

  // synchronize processes via barrier
  mp1::barrier(comm_world);
  std::cout << mp1::processor_name() << " has passed barrier\n";
  mp1::barrier(comm_world);
  double x{0};
  if (comm_world.rank() == root)
    x = 10;

  // broadcast x to all from root rank
  mp1::bcast(comm_world, root, x);
  std::cout << "x = " << x << '\n';

  // collect data from all ranks via gather to root rank
  x = comm_world.rank() + 1;
  if (comm_world.rank() == root) {
    std::vector<double> v(comm_world.size());  // receive buffer
    mp1::gather(comm_world, root, x, v);
    std::cout << "v = ";
    for (const auto& item : v)
      std::cout << item << ' ';
    std::cout << '\n';
  } else
    mp1::gather(comm_world, root, x, std::span<double>{});

  // send data to all ranks via scatter from root rank
  double y{0};
  if (comm_world.rank() == root) {
    std::vector<double> v(comm_world.size());  // send buffer
    std::iota(v.begin(), v.end(), 1);          // populate send buffer
    mp1::scatter(comm_world, root, std::span<const double>{v}, y);
  } else
    mp1::scatter(comm_world, root, std::span<const double>{}, y);
  std::cout << "y = " << y << '\n';

  // reduce/sum all values of x on all nodes and send global result to root
  if (comm_world.rank() == root) {
    mp1::reduce(comm_world, std::plus<double>{}, root, x, y);
    std::cout << "sum after reduce " << y << '\n';
  } else
    mp1::reduce(comm_world, std::plus<double>{}, root, x, std::span<double>{});

  // reduce/multiply all values of x on all nodes and send global result to all
  mp1::allreduce(comm_world, std::multiplies<double>{}, x, y);
  std::cout << "sum after allreduce " << y << '\n';

  // reduce a C-style array using a contiguous layout
  {
    const int n_0{3}, n_1{4};
    using array_type = double[n_0][n_1];
    array_type array;
    for (int i_1{0}; i_1 < n_1; ++i_1)
      for (int i_0{0}; i_0 < n_0; ++i_0)
        array[i_0][i_1] = i_0 + 100 * i_1;
    const auto l{mp1::layout<double>::contiguous(n_1 * n_0)};
    mp1::allreduce(comm_world, std::plus<double>{}, mp1::in_place,
                   mp1::region<double>{&array[0][0], l});
    if (comm_world.rank() == 0) {
      std::cout << "array after allreduce\n";
      for (int i_1{0}; i_1 < n_1; ++i_1) {
        for (int i_0{0}; i_0 < n_0; ++i_0)
          std::cout << array[i_0][i_1] << '\t';
        std::cout << '\n';
      }
    }
  }

  // reduce a wrapped array
  {
    my_array array;
    for (int i_1{0}; i_1 < my_array::n_1; ++i_1)
      for (int i_0{0}; i_0 < my_array::n_0; ++i_0)
        array(i_0, i_1) = i_0 + 100 * i_1;
    mp1::allreduce(comm_world, std::plus<my_array>{}, mp1::in_place, array);
    if (comm_world.rank() == 0) {
      std::cout << "array after allreduce\n";
      for (int i_1{0}; i_1 < my_array::n_1; ++i_1) {
        for (int i_0{0}; i_0 < my_array::n_0; ++i_0)
          std::cout << array(i_0, i_1) << '\t';
        std::cout << '\n';
      }
    }
  }

  return EXIT_SUCCESS;
}
