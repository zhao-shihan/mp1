// Nonblocking point-to-point tests: isend/ibsend/issend/irsend with every supported scalar,
// container and iterator-range send form, and matching irecv forms with request::test().

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
  mp1::request isend_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    if constexpr (has_size<T>::value)
      return mp1::isend(comm, data, dest);
    else
      return mp1::isend(comm, data, dest);
  }

  template<typename T>
  mp1::request ibsend_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    if constexpr (has_size<T>::value)
      return mp1::ibsend(comm, data, dest);
    else
      return mp1::ibsend(comm, data, dest);
  }

  template<typename T>
  mp1::request issend_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    if constexpr (has_size<T>::value)
      return mp1::issend(comm, data, dest);
    else
      return mp1::issend(comm, data, dest);
  }

  template<typename T>
  mp1::request irsend_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    if constexpr (has_size<T>::value)
      return mp1::irsend(comm, data, dest);
    else
      return mp1::irsend(comm, data, dest);
  }

  template<typename T>
  void prepare_recv(T& data_r, std::size_t n) {
    if constexpr (has_resize<T>())
      data_r.resize(n);
    else
      (void)data_r, (void)n;
  }

  // Nonblocking receive: the bounded policy (4) for a resizable container and the fixed
  // policy (2) for std::array.  A sequence container is send-only; its staged send is
  // exercised by isend_sequence_test().
  template<typename T>
  mp1::request post_container_recv(const mp1::mpi_communicator& comm, T& data_r,
                                   std::size_t n) {
    if constexpr (has_resize<T>())
      return mp1::irecv(comm, data_r, mp1::capacity{static_cast<mp1::count_t>(n)}, 0);
    else {
      (void)n;
      return mp1::irecv(comm, data_r, 0);
    }
  }

  template<typename T>
  mp1::request post_scalar_recv(const mp1::mpi_communicator& comm, T& data_r) {
    return mp1::irecv(comm, data_r, 0);
  }

}  // namespace


bool isend_irecv_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{mp1::isend(comm_world, std::span<int>{}, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    auto r{mp1::irecv(comm_world, std::span<int>{}, 0)};
    while (not r.test()) {
    }
  }
  return true;
}


template<typename T>
bool isend_irecv_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{isend_value(comm_world, data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (has_size<T>::value) {
      prepare_recv(data_r, data.size());
      auto r = post_container_recv(comm_world, data_r, data.size());
      while (not r.test()) {
      }
    } else {
      auto r = post_scalar_recv(comm_world, data_r);
      while (not r.test()) {
      }
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool isend_irecv_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r = [&] {
      if constexpr (std::ranges::contiguous_range<T>)
        return mp1::isend(comm_world, std::ranges::subrange{std::begin(data), std::end(data)},
                          1);
      else
        return mp1::isend(comm_world, data, 1);
    }();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    prepare_recv(data_r, data.size());
    auto r = post_container_recv(comm_world, data_r, data.size());
    while (not r.test()) {
    }
    return data_r == data;
  }
  return true;
}


bool ibsend_irecv_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::bsend_buffer buff(mp1::bsend_overhead);
    auto r{mp1::ibsend(comm_world, std::span<int>{}, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    auto r{mp1::irecv(comm_world, std::span<int>{}, 0)};
    while (not r.test()) {
    }
  }
  return true;
}


template<typename T>
bool ibsend_irecv_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::bsend_buffer buff(payload_size_bytes(data));
    auto r{ibsend_value(comm_world, data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (has_size<T>::value) {
      prepare_recv(data_r, data.size());
      auto r = post_container_recv(comm_world, data_r, data.size());
      while (not r.test()) {
      }
    } else {
      auto r = post_scalar_recv(comm_world, data_r);
      while (not r.test()) {
      }
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool ibsend_irecv_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::bsend_buffer buff(payload_size_bytes(data));
    auto r = [&] {
      if constexpr (std::ranges::contiguous_range<T>)
        return mp1::ibsend(comm_world, std::ranges::subrange{std::begin(data), std::end(data)},
                           1);
      else
        return mp1::ibsend(comm_world, data, 1);
    }();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    prepare_recv(data_r, data.size());
    auto r = post_container_recv(comm_world, data_r, data.size());
    while (not r.test()) {
    }
    return data_r == data;
  }
  return true;
}


bool issend_irecv_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{mp1::issend(comm_world, std::span<int>{}, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    auto r{mp1::irecv(comm_world, std::span<int>{}, 0)};
    while (not r.test()) {
    }
  }
  return true;
}


template<typename T>
bool issend_irecv_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{issend_value(comm_world, data, 1)};
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (has_size<T>::value) {
      prepare_recv(data_r, data.size());
      auto r = post_container_recv(comm_world, data_r, data.size());
      while (not r.test()) {
      }
    } else {
      auto r = post_scalar_recv(comm_world, data_r);
      while (not r.test()) {
      }
    }
    return data_r == data;
  }
  return true;
}


template<typename T>
bool issend_irecv_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r = [&] {
      if constexpr (std::ranges::contiguous_range<T>)
        return mp1::issend(comm_world, std::ranges::subrange{std::begin(data), std::end(data)},
                           1);
      else
        return mp1::issend(comm_world, data, 1);
    }();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    T data_r;
    prepare_recv(data_r, data.size());
    auto r = post_container_recv(comm_world, data_r, data.size());
    while (not r.test()) {
    }
    return data_r == data;
  }
  return true;
}


bool irsend_irecv_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::barrier(comm_world);
    auto r{mp1::irsend(comm_world, std::span<int>{}, 1)};
    r.wait();
  } else if (comm_world.rank() == 1) {
    auto r{mp1::irecv(comm_world, std::span<int>{}, 0)};
    mp1::barrier(comm_world);
    while (not r.test()) {
    }
  } else
    mp1::barrier(comm_world);
  return true;
}


template<typename T>
bool irsend_irecv_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::barrier(comm_world);
    auto r{irsend_value(comm_world, data, 1)};
    r.wait();
  } else if (comm_world.rank() == 1) {
    T data_r;
    if constexpr (has_size<T>::value) {
      prepare_recv(data_r, data.size());
      auto r = post_container_recv(comm_world, data_r, data.size());
      mp1::barrier(comm_world);
      while (not r.test()) {
      }
    } else {
      auto r = post_scalar_recv(comm_world, data_r);
      mp1::barrier(comm_world);
      while (not r.test()) {
      }
    }
    return data_r == data;
  } else
    mp1::barrier(comm_world);
  return true;
}


template<typename T>
bool irsend_irecv_iter_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::barrier(comm_world);
    auto r = [&] {
      if constexpr (std::ranges::contiguous_range<T>)
        return mp1::irsend(comm_world, std::ranges::subrange{std::begin(data), std::end(data)},
                           1);
      else
        return mp1::irsend(comm_world, data, 1);
    }();
    while (not r.test()) {
    }
  } else if (comm_world.rank() == 1) {
    T data_r;
    prepare_recv(data_r, data.size());
    auto r = post_container_recv(comm_world, data_r, data.size());
    mp1::barrier(comm_world);
    while (not r.test()) {
    }
    return data_r == data;
  } else
    mp1::barrier(comm_world);
  return true;
}


BOOST_AUTO_TEST_CASE(isend_irecv) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(isend_irecv_test());
  // integer types
  BOOST_TEST(isend_irecv_test(std::byte(77)));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(isend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(isend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(isend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(isend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(isend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(isend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(isend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(isend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(isend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(isend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(isend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(isend_irecv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(isend_irecv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(isend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(isend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(isend_irecv_test(std::string{"Hello World"}));
  BOOST_TEST(isend_irecv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(isend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(isend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(isend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
}


BOOST_AUTO_TEST_CASE(ibsend_irecv) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(ibsend_irecv_test());
  // integer types
  BOOST_TEST(ibsend_irecv_test(std::byte(77)));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(ibsend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(ibsend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(ibsend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(ibsend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(ibsend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(ibsend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(ibsend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(ibsend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(ibsend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(ibsend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(ibsend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(ibsend_irecv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(ibsend_irecv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(ibsend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(ibsend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(ibsend_irecv_test(std::string{"Hello World"}));
  BOOST_TEST(ibsend_irecv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(ibsend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(ibsend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(ibsend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
}


BOOST_AUTO_TEST_CASE(issend_irecv) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(issend_irecv_test());
  // integer types
  BOOST_TEST(issend_irecv_test(std::byte(77)));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(issend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(issend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(issend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(issend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(issend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(issend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(issend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(issend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(issend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(issend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(issend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(issend_irecv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(issend_irecv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(issend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(issend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(issend_irecv_test(std::string{"Hello World"}));
  BOOST_TEST(issend_irecv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(issend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(issend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(issend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
}


BOOST_AUTO_TEST_CASE(irsend_irecv) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(irsend_irecv_test());
  // integer types
  BOOST_TEST(irsend_irecv_test(std::byte(77)));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(irsend_irecv_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(irsend_irecv_test(static_cast<wchar_t>('A')));
  BOOST_TEST(irsend_irecv_test(static_cast<char16_t>('A')));
  BOOST_TEST(irsend_irecv_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(irsend_irecv_test(static_cast<float>(3.14)));
  BOOST_TEST(irsend_irecv_test(static_cast<double>(3.14)));
  BOOST_TEST(irsend_irecv_test(static_cast<long double>(3.14)));
  BOOST_TEST(irsend_irecv_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(irsend_irecv_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(irsend_irecv_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(irsend_irecv_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(irsend_irecv_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(irsend_irecv_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(irsend_irecv_test(std::tuple<int, double, bool>{1, 2.3, true}));
  BOOST_TEST(irsend_irecv_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  // strings and STL containers
  BOOST_TEST(irsend_irecv_test(std::string{"Hello World"}));
  BOOST_TEST(irsend_irecv_test(std::wstring{L"Hello World"}));
  BOOST_TEST(irsend_irecv_test(std::vector<int>{1, 2, 3, 4, 5}));
  // iterators
  BOOST_TEST(irsend_irecv_iter_test(std::array<int, 5>{1, 2, 3, 4, 5}));
  BOOST_TEST(irsend_irecv_iter_test(std::vector<int>{1, 2, 3, 4, 5}));
}
