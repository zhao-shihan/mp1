// Combined send-receive tests: sendrecv and sendrecv_replace for every supported scalar type,
// the pointer+count form (the replacement for the removed iterator form), and a layout form.

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <complex>
#include <cstddef>
#include <limits>
#include <list>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


namespace {

  // One declaration per family: the value and range forms are parameter alternatives.
  template<typename T>
  mp1::status sendrecv_value(const mp1::mpi_communicator& comm, const T& send, int dest,
                             int send_tag, T& recv, int source, int recv_tag) {
    return mp1::sendrecv(comm, send, dest, mp1::tag{send_tag}, recv, source, mp1::tag{recv_tag});
  }

  template<typename T>
  mp1::status sendrecv_replace_value(const mp1::mpi_communicator& comm, T& data, int dest,
                                     int send_tag, int source, int recv_tag) {
    return mp1::sendrecv_replace(comm, data, dest, mp1::tag{send_tag}, source, mp1::tag{recv_tag});
  }

}  // namespace


template<typename T>
bool sendrecv_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  const int rank{comm_world.rank()};
  const int size{comm_world.size()};
  T data_r;
  sendrecv_value(comm_world, data, (rank + 1) % size, 0, data_r, (rank - 1 + size) % size, 0);
  return data_r == data;
}


template<typename T>
bool sendrecv_buffer_test(const T& data) {
  static_assert(has_size<T>());
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  using value_type = typename T::value_type;
  std::vector<value_type> send_data(std::begin(data), std::end(data));
  std::vector<value_type> data_r(send_data.size());
  if (comm_world.rank() == 0) {
    mp1::sendrecv(comm_world, std::span(send_data), 1, mp1::tag{4}, std::span(data_r), 1, mp1::tag{4});
  }
  if (comm_world.rank() == 1) {
    mp1::sendrecv(comm_world, std::span(send_data), 0, mp1::tag{4}, std::span(data_r), 0, mp1::tag{4});
  }
  return comm_world.rank() > 1 or data_r == send_data;
}


template<typename T>
bool sendrecv_layout_test() {
  const auto comm_world = mp1::comm_world();
  const int rank{comm_world.rank()};
  const int size{comm_world.size()};
  constexpr int count = 4;
  // Every other element: slots 0 and 2 of a 4-element buffer.
  auto l = mp1::layout<T>::vector(2, 1, 2);
  std::vector<T> send_data(count, static_cast<T>(rank));
  std::vector<T> recv_data(count);
  mp1::sendrecv(comm_world, mp1::region{send_data.data(), l}, (rank + 1) % size, mp1::tag{0},
                mp1::region{recv_data.data(), l}, (rank - 1 + size) % size, mp1::tag{0});
  const T expected = static_cast<T>((rank + size - 1) % size);
  return recv_data[0] == expected and recv_data[1] == T{} and recv_data[2] == expected and
         recv_data[3] == T{};
}


template<typename T>
struct data_type_helper;

template<typename T, std::size_t n>
struct data_type_helper<std::array<T, n>> {
  static std::array<T, n> get(int val) {
    std::array<T, n> res;
    std::fill(std::begin(res), std::end(res), val);
    return res;
  }
};


template<typename T1, typename T2>
struct data_type_helper<std::pair<T1, T2>> {
  static std::pair<T1, T2> get(int val) {
    return std::make_pair<T1, T2>(T1(val), T2(val));
  }
};


template<typename... Ts>
struct data_type_helper<std::tuple<Ts...>> {
  static std::tuple<Ts...> get(int val) {
    return std::make_tuple<Ts...>(Ts(val)...);
  }
};


template<typename T>
struct data_type_helper<std::vector<T>> {
  static std::vector<T> get(int val) {
    return {T(val), T(val), T(val), T(val), T(val)};
  }
};


template<typename T>
struct data_type_helper<std::list<T>> {
  static std::list<T> get(int val) {
    return {T(val), T(val), T(val), T(val), T(val)};
  }
};


template<typename T>
bool sendrecv_replace_test() {
  const auto comm_world = mp1::comm_world();
  const int rank{comm_world.rank()};
  int size{comm_world.size()};
  T x, expected;
  if constexpr (std::is_constructible_v<T, int> or std::is_enum_v<T>) {
    x = T(rank);
    expected = T((rank + size - 1) % size);
  } else {
    x = data_type_helper<T>::get(rank);
    expected = data_type_helper<T>::get((rank + size - 1) % size);
  }
  sendrecv_replace_value(comm_world, x, (rank + 1) % size, 0, (rank - 1 + size) % size, 0);
  return x == expected;
}


// The removed iterator form is replaced by a pointer+count round trip over a vector view.
template<typename T>
bool sendrecv_replace_buffer_test() {
  const auto comm_world = mp1::comm_world();
  const int rank{comm_world.rank()};
  int size{comm_world.size()};
  const auto initial = data_type_helper<T>::get(rank);
  const auto final_value = data_type_helper<T>::get((rank + size - 1) % size);
  std::vector<typename T::value_type> x(std::begin(initial), std::end(initial));
  std::vector<typename T::value_type> expected(std::begin(final_value), std::end(final_value));
  mp1::sendrecv_replace(comm_world, x, (rank + 1) % size, mp1::tag{0}, (rank - 1 + size) % size, mp1::tag{0});
  return x == expected;
}


template<typename T>
bool sendrecv_replace_layout_test() {
  const auto comm_world = mp1::comm_world();
  const int rank{comm_world.rank()};
  const int size{comm_world.size()};
  constexpr int count = 4;
  auto l = mp1::layout<T>::vector(2, 1, 2);
  std::vector<T> data(count, static_cast<T>(rank));
  mp1::sendrecv_replace(comm_world, mp1::region{data.data(), l}, (rank + 1) % size, mp1::tag{0},
                        (rank - 1 + size) % size, mp1::tag{0});
  const T expected = static_cast<T>((rank + size - 1) % size);
  // sendrecv_replace only overwrites the selected slots; the unselected ones keep this rank's
  // own value.
  return data[0] == expected and data[1] == static_cast<T>(rank) and data[2] == expected and
         data[3] == static_cast<T>(rank);
}


BOOST_AUTO_TEST_CASE(sendrecv) {
  if (not mp1::initialized())
    mp1::init();

  // integer types
  BOOST_TEST(sendrecv_test(std::byte(77)));
  BOOST_TEST(sendrecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(sendrecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(sendrecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(sendrecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(sendrecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(sendrecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(sendrecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(sendrecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(sendrecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(sendrecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(sendrecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(sendrecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(sendrecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(sendrecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(sendrecv_test(static_cast<float>(3.14)));
  BOOST_TEST(sendrecv_test(static_cast<double>(3.14)));
  BOOST_TEST(sendrecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(sendrecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(sendrecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(sendrecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(sendrecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(sendrecv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(sendrecv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(sendrecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(sendrecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // pointer+count and layout forms
  BOOST_TEST(sendrecv_buffer_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(sendrecv_buffer_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(sendrecv_buffer_test(std::list<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(sendrecv_buffer_test(std::set<int>{1, 2, 3, 4, 5}));
  BOOST_TEST(sendrecv_layout_test<int>());
}


BOOST_AUTO_TEST_CASE(sendrecv_replace) {
  if (not mp1::initialized())
    mp1::init();

  // integer types
  BOOST_TEST(sendrecv_replace_test<std::byte>());
  BOOST_TEST(sendrecv_replace_test<char>());
  BOOST_TEST(sendrecv_replace_test<signed char>());
  BOOST_TEST(sendrecv_replace_test<unsigned char>());
  BOOST_TEST(sendrecv_replace_test<signed short>());
  BOOST_TEST(sendrecv_replace_test<unsigned short>());
  BOOST_TEST(sendrecv_replace_test<signed int>());
  BOOST_TEST(sendrecv_replace_test<unsigned int>());
  BOOST_TEST(sendrecv_replace_test<signed long>());
  BOOST_TEST(sendrecv_replace_test<unsigned long>());
  BOOST_TEST(sendrecv_replace_test<signed long long>());
  BOOST_TEST(sendrecv_replace_test<unsigned long long>());
  // character types
  BOOST_TEST(sendrecv_replace_test<wchar_t>());
  BOOST_TEST(sendrecv_replace_test<char16_t>());
  BOOST_TEST(sendrecv_replace_test<char32_t>());
  // floating point number types
  BOOST_TEST(sendrecv_replace_test<float>());
  BOOST_TEST(sendrecv_replace_test<double>());
  BOOST_TEST(sendrecv_replace_test<long double>());
  BOOST_TEST(sendrecv_replace_test<std::complex<float>>());
  BOOST_TEST(sendrecv_replace_test<std::complex<double>>());
  BOOST_TEST(sendrecv_replace_test<std::complex<long double>>());
  // logical type
  BOOST_TEST(sendrecv_replace_test<bool>());
  // enums
  enum class my_enum : int;
  BOOST_TEST(sendrecv_replace_test<my_enum>());
  // pairs, tuples and arrays
  using std_pair = std::pair<int, double>;
  BOOST_TEST(sendrecv_replace_test<std_pair>());
  using std_tuple = std::tuple<int, double, bool>;
  BOOST_TEST(sendrecv_replace_test<std_tuple>());
  using array = std::array<int, 5>;
  BOOST_TEST(sendrecv_replace_test<array>());
  // pointer+count forms (the removed iterator forms)
  BOOST_TEST(sendrecv_replace_buffer_test<array>());
  using vector = std::vector<int>;
  BOOST_TEST(sendrecv_replace_buffer_test<vector>());
  using list = std::list<int>;
  BOOST_TEST(sendrecv_replace_buffer_test<list>());
  // layout form
  BOOST_TEST(sendrecv_replace_layout_test<int>());
}
