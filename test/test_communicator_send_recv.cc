// Point-to-point send/recv tests: every supported scalar and container form for the
// standard, buffered, synchronous and ready modes, plus iterator-range forms.

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


namespace {

  template<typename T>
  void send_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    mp1::send(comm, data, dest);
  }

  template<typename T>
  void bsend_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    mp1::bsend(comm, data, dest);
  }

  template<typename T>
  void ssend_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    mp1::ssend(comm, data, dest);
  }

  template<typename T>
  void rsend_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    mp1::rsend(comm, data, dest);
  }

  // Receive into a value: the scalar/value policy, the fixed policy for a std::array, and the
  // auto-size policy for a resizable contiguous container.  A non-contiguous sequence is
  // send-only, so its receive target is a vector built from the same elements.
  template<typename T>
  T recv_value(const mp1::mpi_communicator& comm) {
    if constexpr (mp1::recv_sink<T&>) {
      T data_r;
      (void)mp1::recv(comm, data_r, 0);
      return data_r;
    } else {
      std::vector<typename T::value_type> data_r;
      (void)mp1::recv(comm, data_r, 0);
      return T(data_r.begin(), data_r.end());
    }
  }

  template<typename T>
  int bsend_buffer_size(const T& data) {
    return payload_size_bytes(data);
  }

}  // namespace


bool send_recv_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    mp1::send(comm_world, std::span<int>{}, 1);
  if (comm_world.rank() == 1)
    (void)mp1::recv(comm_world, std::span<int>{}, 0);
  return true;
}


template<typename T>
bool send_recv_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    send_value(comm_world, data, 1);
  if (comm_world.rank() == 1) {
    const T data_r = recv_value<T>(comm_world);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool send_recv_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    if constexpr (std::ranges::contiguous_range<T>)
      mp1::send(comm_world, std::ranges::subrange{std::begin(data), std::end(data)}, 1);
    else
      mp1::send(comm_world, data, 1);  // non-contiguous: the container range is staged
  }
  if (comm_world.rank() == 1) {
    std::vector<typename T::value_type> data_r(data.size());
    (void)mp1::recv(comm_world, std::span(data_r), 0);
    return std::equal(data_r.begin(), data_r.end(), data.begin(), data.end());
  }
  return true;
}


bool bsend_recv_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::bsend_buffer buff(mp1::bsend_overhead);
    mp1::bsend(comm_world, std::span<int>{}, 1);
  }
  if (comm_world.rank() == 1)
    (void)mp1::recv(comm_world, std::span<int>{}, 0);
  return true;
}


template<typename T>
bool bsend_recv_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::bsend_buffer buff(bsend_buffer_size(data));
    bsend_value(comm_world, data, 1);
  }
  if (comm_world.rank() == 1) {
    const T data_r = recv_value<T>(comm_world);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool bsend_recv_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::bsend_buffer buff(bsend_buffer_size(data));
    if constexpr (std::ranges::contiguous_range<T>)
      mp1::bsend(comm_world, std::ranges::subrange{std::begin(data), std::end(data)}, 1);
    else
      mp1::bsend(comm_world, data, 1);
  }
  if (comm_world.rank() == 1) {
    std::vector<typename T::value_type> data_r(data.size());
    (void)mp1::recv(comm_world, std::span(data_r), 0);
    return std::equal(data_r.begin(), data_r.end(), data.begin(), data.end());
  }
  return true;
}


bool ssend_recv_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    mp1::ssend(comm_world, std::span<int>{}, 1);
  if (comm_world.rank() == 1)
    (void)mp1::recv(comm_world, std::span<int>{}, 0);
  return true;
}


template<typename T>
bool ssend_recv_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    ssend_value(comm_world, data, 1);
  if (comm_world.rank() == 1) {
    const T data_r = recv_value<T>(comm_world);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool ssend_recv_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    if constexpr (std::ranges::contiguous_range<T>)
      mp1::ssend(comm_world, std::ranges::subrange{std::begin(data), std::end(data)}, 1);
    else
      mp1::ssend(comm_world, data, 1);
  }
  if (comm_world.rank() == 1) {
    std::vector<typename T::value_type> data_r(data.size());
    (void)mp1::recv(comm_world, std::span(data_r), 0);
    return std::equal(data_r.begin(), data_r.end(), data.begin(), data.end());
  }
  return true;
}


bool rsend_recv_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::barrier(comm_world);
    mp1::rsend(comm_world, std::span<int>{}, 1);
  } else if (comm_world.rank() == 1) {
    auto r = mp1::irecv(comm_world, std::span<int>{}, 0);
    mp1::barrier(comm_world);
    r.wait();
  } else
    mp1::barrier(comm_world);
  return true;
}


template<typename T>
bool rsend_recv_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::barrier(comm_world);
    rsend_value(comm_world, data, 1);
  } else if (comm_world.rank() == 1) {
    // The receive must be posted before the ready send enters MPI.
    if constexpr (has_begin_end<T>() and has_size<T>()) {
      std::vector<typename T::value_type> data_r(data.size());
      auto r = mp1::irecv(comm_world, std::span(data_r), 0);
      mp1::barrier(comm_world);
      r.wait();
      return std::equal(std::begin(data_r), std::end(data_r), std::begin(data));
    } else {
      T data_r;
      auto r = mp1::irecv(comm_world, data_r, 0);
      mp1::barrier(comm_world);
      r.wait();
      return data_r == data;
    }
  } else
    mp1::barrier(comm_world);
  return true;
}


template<typename T>
bool rsend_recv_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::barrier(comm_world);
    mp1::rsend(comm_world, data, 1);
  } else if (comm_world.rank() == 1) {
    std::vector<typename T::value_type> data_r(data.size());
    auto r = mp1::irecv(comm_world, std::span(data_r), 0);
    mp1::barrier(comm_world);
    r.wait();
    return std::equal(std::begin(data_r), std::end(data_r), std::begin(data));
  } else
    mp1::barrier(comm_world);
  return true;
}


BOOST_AUTO_TEST_CASE(send_recv) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(send_recv_test());
  // integer types
  BOOST_TEST(send_recv_test(std::byte(77)));
  BOOST_TEST(send_recv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(send_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(send_recv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(send_recv_test(static_cast<char16_t>('A')));
  BOOST_TEST(send_recv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(send_recv_test(static_cast<float>(3.14)));
  BOOST_TEST(send_recv_test(static_cast<double>(3.14)));
  BOOST_TEST(send_recv_test(static_cast<long double>(3.14)));
  BOOST_TEST(send_recv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(send_recv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(send_recv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(send_recv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(send_recv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(send_recv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(send_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(send_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // MP1_REFLECTION_TEMPLATE types
  BOOST_TEST(send_recv_test(wrapper<int>{42}));
  BOOST_TEST(send_recv_test(wrapper<double>{3.14}));
  BOOST_TEST(send_recv_test(pair_like<int, double>{1, 2.3}));
  // strings and STL containers
  BOOST_TEST(send_recv_test(std::string{"Hello World"}));
  BOOST_TEST(send_recv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(send_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_recv_test(std::list<int>{1, 2, 3, 4, 5}));
  // std::set is excluded (keyed, not flat); see the compile-fail probes.
  // iterators
  BOOST_TEST(send_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(send_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  // std::set is excluded (keyed, not flat); see the compile-fail probes.
}


BOOST_AUTO_TEST_CASE(bsend_recv) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(bsend_recv_test());
  // integer types
  BOOST_TEST(bsend_recv_test(std::byte(77)));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(bsend_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(bsend_recv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(bsend_recv_test(static_cast<char16_t>('A')));
  BOOST_TEST(bsend_recv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(bsend_recv_test(static_cast<float>(3.14)));
  BOOST_TEST(bsend_recv_test(static_cast<double>(3.14)));
  BOOST_TEST(bsend_recv_test(static_cast<long double>(3.14)));
  BOOST_TEST(bsend_recv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(bsend_recv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(bsend_recv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(bsend_recv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(bsend_recv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(bsend_recv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(bsend_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(bsend_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // MP1_REFLECTION_TEMPLATE types
  BOOST_TEST(bsend_recv_test(wrapper<int>{42}));
  BOOST_TEST(bsend_recv_test(wrapper<double>{3.14}));
  BOOST_TEST(bsend_recv_test(pair_like<int, double>{1, 2.3}));
  // strings and STL containers
  BOOST_TEST(bsend_recv_test(std::string{"Hello World"}));
  BOOST_TEST(bsend_recv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(bsend_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_recv_test(std::list<int>{1, 2, 3, 4, 5}));
  // std::set is excluded (keyed, not flat); see the compile-fail probes.
  // iterators
  BOOST_TEST(bsend_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(bsend_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  // std::set is excluded (keyed, not flat); see the compile-fail probes.
}


BOOST_AUTO_TEST_CASE(ssend_recv) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(ssend_recv_test());
  // integer types
  BOOST_TEST(ssend_recv_test(std::byte(77)));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(ssend_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(ssend_recv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(ssend_recv_test(static_cast<char16_t>('A')));
  BOOST_TEST(ssend_recv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(ssend_recv_test(static_cast<float>(3.14)));
  BOOST_TEST(ssend_recv_test(static_cast<double>(3.14)));
  BOOST_TEST(ssend_recv_test(static_cast<long double>(3.14)));
  BOOST_TEST(ssend_recv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(ssend_recv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(ssend_recv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(ssend_recv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(ssend_recv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(ssend_recv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(ssend_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(ssend_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // MP1_REFLECTION_TEMPLATE types
  BOOST_TEST(ssend_recv_test(wrapper<int>{42}));
  BOOST_TEST(ssend_recv_test(wrapper<double>{3.14}));
  BOOST_TEST(ssend_recv_test(pair_like<int, double>{1, 2.3}));
  // strings and STL containers
  BOOST_TEST(ssend_recv_test(std::string{"Hello World"}));
  BOOST_TEST(ssend_recv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(ssend_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_recv_test(std::list<int>{1, 2, 3, 4, 5}));
  // std::set is excluded (keyed, not flat); see the compile-fail probes.
  // iterators
  BOOST_TEST(ssend_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(ssend_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  // std::set is excluded (keyed, not flat); see the compile-fail probes.
}


BOOST_AUTO_TEST_CASE(rsend_recv) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(rsend_recv_test());
  // integer types
  BOOST_TEST(rsend_recv_test(std::byte(77)));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(rsend_recv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(rsend_recv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(rsend_recv_test(static_cast<char16_t>('A')));
  BOOST_TEST(rsend_recv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(rsend_recv_test(static_cast<float>(3.14)));
  BOOST_TEST(rsend_recv_test(static_cast<double>(3.14)));
  BOOST_TEST(rsend_recv_test(static_cast<long double>(3.14)));
  BOOST_TEST(rsend_recv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(rsend_recv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(rsend_recv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(rsend_recv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(rsend_recv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(rsend_recv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(rsend_recv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(rsend_recv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // MP1_REFLECTION_TEMPLATE types
  BOOST_TEST(rsend_recv_test(wrapper<int>{42}));
  BOOST_TEST(rsend_recv_test(wrapper<double>{3.14}));
  BOOST_TEST(rsend_recv_test(pair_like<int, double>{1, 2.3}));
  // strings and STL containers
  BOOST_TEST(rsend_recv_test(std::string{"Hello World"}));
  BOOST_TEST(rsend_recv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(rsend_recv_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_recv_test(std::list<int>{1, 2, 3, 4, 5}));
  // std::set is excluded (keyed, not flat); see the compile-fail probes.
  // iterators
  BOOST_TEST(rsend_recv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_recv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(rsend_recv_iter_test(std::list<int>{1, 2, 3, 4, 5}));
  // std::set is excluded (keyed, not flat); see the compile-fail probes.
}
