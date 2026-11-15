// Matched probe/receive tests: mprobe/improbe return the message handle plus status, and
// mrecv/imrecv consume it through the scalar, pointer and iterator forms.

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


namespace {

  template<typename T>
  void send_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    mp1::send(comm, data, dest);
  }

  // mrecv: 1 value / 2 fixed array / 4 auto-size.
  template<typename T>
  mp1::status mrecv_value(T& data_r, mp1::message& message) {
    return mp1::mrecv(data_r, message);
  }

  // imrecv has no auto-size policy (4): a resizable container uses the bounded policy with
  // the probed count as the bound.
  template<typename T>
  mp1::request imrecv_value(T& data_r, mp1::message& message, mp1::count_t count) {
    if constexpr (mp1::resizable_element_buffer<T>)
      return mp1::imrecv(data_r, mp1::capacity{count}, message);
    else {
      (void)count;
      return mp1::imrecv(data_r, message);
    }
  }

}  // namespace


template<typename T>
bool mprobe_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    send_value(comm_world, data, 1);
  if (comm_world.rank() == 1) {
    auto probed{mp1::mprobe(comm_world, 0)};
    if (probed.probed_status().source() != 0)
      return false;
    if constexpr (has_size<T>::value) {
      if (probed.probed_status().get_count<typename T::value_type>() !=
          static_cast<mp1::count_t>(data.size()))
        return false;
    } else {
      if (probed.probed_status().get_count<T>() != 1)
        return false;
    }
    T data_r{};
    if constexpr (has_size<T>::value) {
      if constexpr (has_resize<T>())
        data_r.resize(data.size());
    }
    mrecv_value(data_r, probed);
    return data_r == data;
  }
  return true;
}


template<typename T>
bool mprobe_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    send_value(comm_world, data, 1);
  if (comm_world.rank() == 1) {
    auto probed{mp1::mprobe(comm_world, 0)};
    if (probed.probed_status().source() != 0)
      return false;
    const mp1::count_t count{probed.probed_status().get_count<typename T::value_type>()};
    if (count != static_cast<mp1::count_t>(data.size()))
      return false;
    std::vector<typename T::value_type> data_r(static_cast<std::size_t>(count));
    (void)mp1::mrecv(std::span(data_r), probed);
    return std::equal(data_r.begin(), data_r.end(), data.begin(), data.end());
  }
  return true;
}


template<typename T>
bool improbe_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    send_value(comm_world, data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto res{mp1::improbe(comm_world, 0)};
      if (res) {
        auto probed{std::move(res).value()};
        if (probed.probed_status().source() != 0)
          return false;
        if constexpr (has_size<T>::value) {
          if (probed.probed_status().get_count<typename T::value_type>() !=
              static_cast<mp1::count_t>(data.size()))
            return false;
        } else {
          if (probed.probed_status().get_count<T>() != 1)
            return false;
        }
        T data_r{};
        mp1::count_t probed_count{1};
        if constexpr (has_size<T>::value) {
          probed_count = static_cast<mp1::count_t>(data.size());
          if constexpr (has_resize<T>())
            data_r.resize(data.size());
        }
        auto request{imrecv_value(data_r, probed, probed_count)};
        request.wait();
        return data_r == data;
      }
    }
  }
  return true;
}


template<typename T>
bool improbe_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0)
    send_value(comm_world, data, 1);
  if (comm_world.rank() == 1) {
    while (true) {
      auto res{mp1::improbe(comm_world, 0)};
      if (res) {
        auto probed{std::move(res).value()};
        if (probed.probed_status().source() != 0)
          return false;
        const mp1::count_t count{probed.probed_status().get_count<typename T::value_type>()};
        if (count != static_cast<mp1::count_t>(data.size()))
          return false;
        std::vector<typename T::value_type> data_r(static_cast<std::size_t>(count));
        auto request{mp1::imrecv(std::span(data_r), probed)};
        request.wait();
        return std::equal(data_r.begin(), data_r.end(), data.begin(), data.end());
      }
    }
  }
  return true;
}


BOOST_AUTO_TEST_CASE(mprobe) {
  if (not mp1::initialized())
    mp1::init();

  // integer types
  BOOST_TEST(mprobe_test(std::byte(77)));
  BOOST_TEST(mprobe_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(mprobe_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(mprobe_test(static_cast<wchar_t>('A')));
  BOOST_TEST(mprobe_test(static_cast<char16_t>('A')));
  BOOST_TEST(mprobe_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(mprobe_test(static_cast<float>(3.14)));
  BOOST_TEST(mprobe_test(static_cast<double>(3.14)));
  BOOST_TEST(mprobe_test(static_cast<long double>(3.14)));
  BOOST_TEST(mprobe_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(mprobe_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(mprobe_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(mprobe_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(mprobe_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(mprobe_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(mprobe_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(mprobe_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(mprobe_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(mprobe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(mprobe_iter_test(std::list<int>{1, 2, 3, 4, 5}));
}


BOOST_AUTO_TEST_CASE(improbe) {
  if (not mp1::initialized())
    mp1::init();

  // integer types
  BOOST_TEST(improbe_test(std::byte(77)));
  BOOST_TEST(improbe_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(improbe_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(improbe_test(static_cast<wchar_t>('A')));
  BOOST_TEST(improbe_test(static_cast<char16_t>('A')));
  BOOST_TEST(improbe_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(improbe_test(static_cast<float>(3.14)));
  BOOST_TEST(improbe_test(static_cast<double>(3.14)));
  BOOST_TEST(improbe_test(static_cast<long double>(3.14)));
  BOOST_TEST(improbe_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(improbe_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(improbe_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(improbe_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(improbe_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(improbe_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(improbe_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(improbe_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(improbe_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(improbe_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(improbe_iter_test(std::list<int>{1, 2, 3, 4, 5}));
}
