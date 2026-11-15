#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <mp1/mp1.hpp>
#include <string>
#include <utility>
#include <vector>


// print elements of a pair
template<typename ch, typename tr, typename T1, typename T2>
std::basic_ostream<ch, tr>& operator<<(std::basic_ostream<ch, tr>& out,
                                       const std::pair<T1, T2>& p) {
  return out << '(' << p.first << ',' << p.second << ')';
}


// print all elements of a container
template<typename ch, typename tr, typename C>
std::basic_ostream<ch, tr>& print_container(std::basic_ostream<ch, tr>& out, const C& c) {
  out << '(';
  for (auto i{std::begin(c)}; i != std::end(c); ++i) {
    out << (*i);
    if (std::next(i) != std::end(c))
      out << ',';
  }
  return out << ')';
}


// print all elements of a vector
template<typename ch, typename tr, typename T, typename A>
std::basic_ostream<ch, tr>& operator<<(std::basic_ostream<ch, tr>& out,
                                       const std::vector<T, A>& v) {
  return print_container(out, v);
}


// print all elements of a list
template<typename ch, typename tr, typename T, typename A>
std::basic_ostream<ch, tr>& operator<<(std::basic_ostream<ch, tr>& out,
                                       const std::list<T, A>& l) {
  return print_container(out, l);
}


// print all elements of a map
template<typename ch, typename tr, typename K, typename C, typename A>
std::basic_ostream<ch, tr>& operator<<(std::basic_ostream<ch, tr>& out,
                                       const std::map<K, C, A>& m) {
  return print_container(out, m);
}


// std::pair has a built-in mp1 datatype mapping, so a key/value entry can be
// sent directly; a std::map is not a p2p container, so receive staging uses a
// vector of entries.
using entry = std::pair<int, double>;


// send an STL container
template<typename T>
void send(const mp1::mpi_communicator& comm, const T& x) {
  mp1::send(comm, x, 1);
}


// send an STL container
template<typename T>
void isend(const mp1::mpi_communicator& comm, const T& x) {
  auto r{mp1::isend(comm, x, 1)};
  r.wait();
}


// receive an STL container; the blocking convenience form resizes it to the message
template<typename T>
void recv(const mp1::mpi_communicator& comm) {
  T x;
  const auto s{mp1::recv(comm, x, 0)};
  std::cout << "x = " << x << " with " << s.template get_count<typename T::value_type>()
            << " elements\n";
}


// receive a nonblocking contiguous container; the capacity form resizes on completion
template<typename T>
void irecv_resizable(const mp1::mpi_communicator& comm) {
  constexpr mp1::capacity cap{32};
  T x;
  auto r{mp1::irecv(comm, x, cap, 0)};
  const auto s{r.wait()};
  std::cout << "x = " << x << " with " << s.template get_count<typename T::value_type>()
            << " elements\n";
}


// receive a map: associative containers are not p2p containers in mp1, so stage entries
void recv_map(const mp1::mpi_communicator& comm) {
  std::vector<entry> staging;
  const auto s{mp1::recv(comm, staging, 0)};
  std::map<int, double> x;
  for (const entry& e : staging)
    x.emplace(e.first, e.second);
  std::cout << "x = " << x << " with " << s.get_count<entry>() << " elements\n";
}


// nonblocking variant of recv_map
void irecv_map(const mp1::mpi_communicator& comm) {
  constexpr mp1::capacity cap{32};
  std::vector<entry> staging;
  auto r{mp1::irecv(comm, staging, cap, 0)};
  const auto s{r.wait()};
  std::map<int, double> x;
  for (const entry& e : staging)
    x.emplace(e.first, e.second);
  std::cout << "x = " << x << " with " << s.get_count<entry>() << " elements\n";
}


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  // run the program with two or more processes
  if (comm_world.size() < 2)
    comm_world.abort(EXIT_FAILURE);
  // process 0 sends
  if (comm_world.rank() == 0) {
    std::string t_1{"Hello World!"};
    send(comm_world, t_1);
    isend(comm_world, t_1);
    std::vector<int> t_2{0, 1, 2, 3, 4, 5, 6, 77, 42};
    send(comm_world, t_2);
    isend(comm_world, t_2);
    std::vector<entry> t_3{{0, 0.0}, {1, 0.1}, {2, 0.2}, {3, 0.3}, {4, 0.4}};
    send(comm_world, t_3);
    isend(comm_world, t_3);
    // std::vector<bool> is not a supported container (proxy references); use 0/1 ints
    std::vector<int> t_4{0, 1, 0, 1, 1};
    send(comm_world, t_4);
    isend(comm_world, t_4);
    // std::valarray is not a contiguous range form; use std::vector for the same data
    std::vector<double> t_5{1, 2, 3, 4, 42, 4 * std::atan(1.0)};
    send(comm_world, t_5);
    isend(comm_world, t_5);
  }
  // process 1 receives
  if (comm_world.rank() == 1) {
    recv<std::string>(comm_world);
    irecv_resizable<std::string>(comm_world);
    // std::list is send-only (a staged send); the receive target is a contiguous container
    recv<std::vector<int>>(comm_world);
    irecv_resizable<std::vector<int>>(comm_world);
    recv_map(comm_world);
    irecv_map(comm_world);
    recv<std::vector<int>>(comm_world);
    irecv_resizable<std::vector<int>>(comm_world);
    recv<std::vector<double>>(comm_world);
    irecv_resizable<std::vector<double>>(comm_world);
  }
  return EXIT_SUCCESS;
}
