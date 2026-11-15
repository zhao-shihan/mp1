#include <algorithm>
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
  const int n{12};
  std::vector<int> v_1(n), v_2(n), v_3(n), v_4(n);
  const auto l{mp1::layout<int>::contiguous(n)};
  // process 0 sends
  if (comm_world.rank() == 0) {
    // see MPI Standard for the semantics of standard send, buffered send,
    // synchronous send and ready send
    double x{1.23456};
    // send x to rank 1 via standard send
    mp1::request r{mp1::isend(comm_world, x, 1)};
    r.wait();  // wait until send has finished
    ++x;
    {
      // create a buffer for buffered send,
      // memory will be freed on leaving the scope
      const int size{static_cast<int>(sizeof(x))};
      mp1::bsend_buffer buff(size);
      r = mp1::ibsend(comm_world, x, 1);  // send x to rank 1 via buffered send
      r.wait();                           // wait until send has finished
    }
    ++x;
    r = mp1::issend(comm_world, x, 1);  // send x to rank 1 via synchronous send
    r.wait();                           // wait until send has finished
    ++x;
    r = mp1::irsend(comm_world, x, 1);  // send x to rank 1 via ready send
    r.wait();                           // wait until send has finished
    std::iota(v_1.begin(), v_1.end(), 0);
    std::iota(v_2.begin(), v_2.end(), 1);
    std::iota(v_3.begin(), v_3.end(), 2);
    std::iota(v_4.begin(), v_4.end(), 3);
    {
      // create a buffer for buffered send,
      // memory will be freed on leaving the scope
      const int size{static_cast<int>(l.byte_size())};
      mp1::bsend_buffer buff(size);
      std::vector<mp1::request> r;
      r.push_back(
          mp1::isend(comm_world, mp1::region{v_1.data(), l}, 1));  // send v1 via standard send
      r.push_back(
          mp1::ibsend(comm_world, mp1::region{v_2.data(), l}, 1));  // send v2 via buffered send
      r.push_back(mp1::issend(comm_world, mp1::region{v_3.data(), l},
                              1));  // send v3 via synchronous send
      r.push_back(
          mp1::irsend(comm_world, mp1::region{v_4.data(), l}, 1));  // send v4 via ready send
      mp1::wait_all(r);  // wait until all sends have finished
    }
    {
      std::vector<mp1::request> r;
      const int size{static_cast<int>(l.byte_size())};
      mp1::bsend_buffer buff(size);
      r.push_back(
          mp1::isend(comm_world, mp1::region{v_1.data(), l}, 1));  // send v1 via standard send
      r.push_back(
          mp1::ibsend(comm_world, mp1::region{v_2.data(), l}, 1));  // send v2 via buffered send
      r.push_back(mp1::issend(comm_world, mp1::region{v_3.data(), l},
                              1));  // send v3 via synchronous send
      r.push_back(
          mp1::irsend(comm_world, mp1::region{v_4.data(), l}, 1));  // send v4 via ready send
      while (true) {
        const auto finished{mp1::wait_some(r)};  // wait until one or more sends have finished
        if (finished.empty())                    // there have been no pending sends
          break;
        // print indices of finished sends
        std::cout << "send finished : ";
        std::for_each(finished.begin(), finished.end(),
                      [](const mp1::completion& c) { std::cout << c.first << ' '; });
        std::cout << "\n";
      }
    }
  }
  // process 1 receives
  if (comm_world.rank() == 1) {
    double x;
    mp1::request r{mp1::irecv(comm_world, x, 0)};  // receive x from rank 0
    r.wait();                                      // wait until receive has finished
    std::cout << "x = " << x << '\n';
    r = mp1::irecv(comm_world, x, 0);  // receive x from rank 0
    r.wait();                          // wait until receive has finished
    std::cout << "x = " << x << '\n';
    r = mp1::irecv(comm_world, x, 0);  // receive x from rank 0
    r.wait();                          // wait until receive has finished
    std::cout << "x = " << x << '\n';
    r = mp1::irecv(comm_world, x, 0);  // receive x from rank 0
    r.wait();                          // wait until receive has finished
    std::cout << "x = " << x << '\n';
    {
      std::vector<mp1::request> r;
      r.push_back(
          mp1::irecv(comm_world, mp1::region{v_1.data(), l}, 0));  // receive v1 from rank 0
      r.push_back(
          mp1::irecv(comm_world, mp1::region{v_2.data(), l}, 0));  // receive v2 from rank 0
      r.push_back(
          mp1::irecv(comm_world, mp1::region{v_3.data(), l}, 0));  // receive v3 from rank 0
      r.push_back(
          mp1::irecv(comm_world, mp1::region{v_4.data(), l}, 0));  // receive v4 from rank 0
      mp1::wait_all(r);  // wait until all receives have finished
      print_range("v = ", v_1.begin(), v_1.end());
      print_range("v = ", v_2.begin(), v_2.end());
      print_range("v = ", v_3.begin(), v_3.end());
      print_range("v = ", v_4.begin(), v_4.end());
    }
    {
      std::vector<mp1::request> r;
      r.push_back(
          mp1::irecv(comm_world, mp1::region{v_1.data(), l}, 0));  // receive v1 from rank 0
      r.push_back(
          mp1::irecv(comm_world, mp1::region{v_2.data(), l}, 0));  // receive v2 from rank 0
      r.push_back(
          mp1::irecv(comm_world, mp1::region{v_3.data(), l}, 0));  // receive v3 from rank 0
      r.push_back(
          mp1::irecv(comm_world, mp1::region{v_4.data(), l}, 0));  // receive v4 from rank 0
      while (true) {
        const auto finished{
            mp1::wait_some(r)};  // wait until one or more receives have finished
        if (finished.empty())    // there have been no pending receives
          break;
        // print indices of finished receives
        std::cout << "recv finished : ";
        std::for_each(finished.begin(), finished.end(),
                      [](const mp1::completion& c) { std::cout << c.first << ' '; });
        std::cout << '\n';
      }
      print_range("v = ", v_1.begin(), v_1.end());
      print_range("v = ", v_2.begin(), v_2.end());
      print_range("v = ", v_3.begin(), v_3.end());
      print_range("v = ", v_4.begin(), v_4.end());
    }
  }
  return EXIT_SUCCESS;
}
