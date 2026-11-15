#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <list>
#include <mp1/mp1.hpp>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  // run the program with two or more processes
  if (comm_world.size() < 2)
    comm_world.abort(EXIT_FAILURE);
  // send / receive a contiguous container: the range form is zero-copy and the receive is a
  // fixed span over the caller's elements
  {
    const int n{10};
    std::vector<double> l(n);
    if (comm_world.rank() == 0) {
      std::iota(begin(l), end(l), 1);
      mp1::send(comm_world, l, 1);
    }
    if (comm_world.rank() == 1) {
      mp1::recv(comm_world, std::span(l), 0);
      std::for_each(begin(l), end(l), [](double x) { std::cout << x << '\n'; });
    }
  }
  // std::ranges::subrange over contiguous iterators is the range spelling of an iterator pair
  {
    const int n{10};
    std::vector<double> l(n);
    if (comm_world.rank() == 0) {
      std::iota(begin(l), end(l), 1);
      mp1::send(comm_world, std::ranges::subrange(begin(l), end(l)), 1);
    }
    if (comm_world.rank() == 1) {
      mp1::recv(comm_world, std::span(l), 0);
      std::for_each(begin(l), end(l), [](double x) { std::cout << x << '\n'; });
    }
  }
  // a non-contiguous sequence is send-only (the wire form is staged); the receive target is a
  // contiguous container.  Receiving into a std::list is a compile-time error.
  {
    const int n{10};
    std::list<double> l(n);
    if (comm_world.rank() == 0) {
      std::iota(begin(l), end(l), 1);
      mp1::send(comm_world, l, 1);
    }
    if (comm_world.rank() == 1) {
      std::vector<double> received;
      mp1::recv(comm_world, received, 0);
      std::for_each(begin(received), end(received), [](double x) { std::cout << x << '\n'; });
    }
  }
  return EXIT_SUCCESS;
}
