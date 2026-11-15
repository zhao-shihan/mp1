// Probe/iprobe tests: blocking and nonblocking probe of scalar and container messages, status
// counts, and the iterator-range receive path.

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

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

  // Receive the probed message with the form matching T: the fixed policy for std::array,
  // the auto-size policy for a resizable contiguous container, and the value policy.
  template<typename T>
  void recv_probed(const mp1::mpi_communicator& comm, T& data_r, int count) {
    if constexpr (has_resize<T>()) {
      data_r.resize(static_cast<std::size_t>(count));
      (void)mp1::recv(comm, std::span(data_r), 0);
    } else {
      (void)count;
      (void)mp1::recv(comm, data_r, 0);
    }
  }

}  // namespace


template<typename T>
bool probe_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    send_value(comm_world, data, 1);
  if (comm_world.rank() == 1) {
    mp1::status s{mp1::probe(comm_world, 0)};
    if (s.source() != 0)
      return false;
    if constexpr (has_size<T>::value) {
      if (s.get_count<typename T::value_type>() != static_cast<mp1::count_t>(data.size()))
        return false;
    } else {
      if (s.get_count<T>() != 1)
        return false;
    }
    T data_r{};
    if constexpr (has_size<T>::value)
      recv_probed(comm_world, data_r, static_cast<int>(data.size()));
    else
      recv_probed(comm_world, data_r, 1);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool probe_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    send_value(comm_world, data, 1);
  if (comm_world.rank() == 1) {
    mp1::status s{mp1::probe(comm_world, 0)};
    if (s.source() != 0)
      return false;
    const mp1::count_t count{s.get_count<typename T::value_type>()};
    if (count != static_cast<mp1::count_t>(data.size()))
      return false;
    std::vector<typename T::value_type> data_r(static_cast<std::size_t>(count));
    (void)mp1::recv(comm_world, std::span(data_r), 0);
    return std::equal(data_r.begin(), data_r.end(), data.begin(), data.end());
  }
  return true;
}


template<typename T>
bool iprobe_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    send_value(comm_world, data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto s{mp1::iprobe(comm_world, 0)};
      if (s) {
        if (s->source() != 0)
          return false;
        if constexpr (has_size<T>::value) {
          if (s->template get_count<typename T::value_type>() !=
              static_cast<mp1::count_t>(data.size()))
            return false;
        } else {
          if (s->template get_count<T>() != 1)
            return false;
        }
        T data_r{};
        if constexpr (has_size<T>::value)
          recv_probed(comm_world, data_r, static_cast<int>(data.size()));
        else
          recv_probed(comm_world, data_r, 1);
        return data_r == data;
      }
    }
  }
  return true;
}


template<typename T>
bool iprobe_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    send_value(comm_world, data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto s{mp1::iprobe(comm_world, 0)};
      if (s) {
        if (s->source() != 0)
          return false;
        const mp1::count_t count{s->template get_count<typename T::value_type>()};
        if (count != static_cast<mp1::count_t>(data.size()))
          return false;
        std::vector<typename T::value_type> data_r(static_cast<std::size_t>(count));
        auto request{mp1::irecv(comm_world, std::span(data_r), 0)};
        request.wait();
        return std::equal(data_r.begin(), data_r.end(), data.begin(), data.end());
      }
    }
  }
  return true;
}


BOOST_AUTO_TEST_CASE(probe) {
  if (not mp1::initialized())
    mp1::init();

  // integer types
  BOOST_TEST(probe_test(std::byte(77)));
  BOOST_TEST(probe_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(probe_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(probe_test(static_cast<wchar_t>('A')));
  BOOST_TEST(probe_test(static_cast<char16_t>('A')));
  BOOST_TEST(probe_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(probe_test(static_cast<float>(3.14)));
  BOOST_TEST(probe_test(static_cast<double>(3.14)));
  BOOST_TEST(probe_test(static_cast<long double>(3.14)));
  BOOST_TEST(probe_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(probe_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(probe_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(probe_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(probe_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(probe_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(probe_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(probe_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(probe_test(std::string{"Hello World"}));
  BOOST_TEST(probe_test(std::wstring{L"Hello World"}));
  BOOST_TEST(probe_test(std::vector<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(probe_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(probe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
}


BOOST_AUTO_TEST_CASE(iprobe) {
  if (not mp1::initialized())
    mp1::init();

  // integer types
  BOOST_TEST(iprobe_test(std::byte(77)));
  BOOST_TEST(iprobe_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(iprobe_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(iprobe_test(static_cast<wchar_t>('A')));
  BOOST_TEST(iprobe_test(static_cast<char16_t>('A')));
  BOOST_TEST(iprobe_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(iprobe_test(static_cast<float>(3.14)));
  BOOST_TEST(iprobe_test(static_cast<double>(3.14)));
  BOOST_TEST(iprobe_test(static_cast<long double>(3.14)));
  BOOST_TEST(iprobe_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(iprobe_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(iprobe_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(iprobe_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(iprobe_test(my_enum::val));
  // pairs, tuples, arrays
  BOOST_TEST(iprobe_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(iprobe_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(iprobe_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(iprobe_test(std::string{"Hello World"}));
  BOOST_TEST(iprobe_test(std::wstring{L"Hello World"}));
  BOOST_TEST(iprobe_test(std::vector<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(iprobe_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(iprobe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
}
