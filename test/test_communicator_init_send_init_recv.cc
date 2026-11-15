// Persistent point-to-point tests: send_init/recv_init and the buffered, synchronous and ready
// variants for every supported scalar type, plus pointer+count and pointer+layout forms (the
// container and iterator *_init forms do not exist in the new API) and request state checks.

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>


namespace {

  template<typename T>
  mp1::request send_init_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    return mp1::send_init(comm, data, dest);
  }

  template<typename T>
  mp1::request bsend_init_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    return mp1::bsend_init(comm, data, dest);
  }

  template<typename T>
  mp1::request ssend_init_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    return mp1::ssend_init(comm, data, dest);
  }

  template<typename T>
  mp1::request rsend_init_value(const mp1::mpi_communicator& comm, const T& data, int dest) {
    return mp1::rsend_init(comm, data, dest);
  }

  template<typename T>
  mp1::request recv_init_value(const mp1::mpi_communicator& comm, T& data, int source) {
    return mp1::recv_init(comm, data, source);
  }

  template<class F>
  bool throws_logic_error(F&& f) {
    try {
      f();
    } catch (const std::logic_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class F>
  bool throws_invalid_argument(F&& f) {
    try {
      f();
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  }

}  // namespace


bool send_init_recv_init_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{mp1::send_init(comm_world, std::span<int>{}, 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    auto r{mp1::recv_init(comm_world, std::span<int>{}, 0)};
    r.start();
    while (not r.test()) {
    }
    r.start();
    while (not r.test()) {
    }
  }
  return true;
}


template<typename T>
bool send_init_recv_init_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{send_init_value(comm_world, data, 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    auto r{recv_init_value(comm_world, data_r, 0)};
    r.start();
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    r.start();
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    return ok;
  }
  return true;
}


// Pointer+count form (the replacement for the removed container/iterator *_init forms).
template<typename T>
bool buffer_send_init_recv_init_test(const std::vector<T>& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{mp1::send_init(comm_world, std::span(data), 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    std::vector<T> data_r(data.size());
    auto r{mp1::recv_init(comm_world, std::span(data_r), 0)};
    r.start();
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    r.start();
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    return ok;
  }
  return true;
}


// Pointer+layout form: a strided layout selects every other element.
template<typename T>
bool layout_send_init_recv_init_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  constexpr int count = 3;
  auto l = mp1::layout<T>::vector(2, 1, 2);  // elements 0 and 2
  std::vector<T> data(count);
  for (int i{0}; i < count; ++i)
    data[static_cast<std::size_t>(i)] = static_cast<T>(i + 1);
  if (comm_world.rank() == 0) {
    auto r{mp1::send_init(comm_world, mp1::region{data.data(), l}, 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    std::vector<T> data_r(count);
    auto r{mp1::recv_init(comm_world, mp1::region{data_r.data(), l}, 0)};
    r.start();
    while (not r.test()) {
    }
    ok = ok and data_r[0] == data[0] and data_r[1] == T{} and data_r[2] == data[2];
    r.start();
    while (not r.test()) {
    }
    ok = ok and data_r[0] == data[0] and data_r[2] == data[2];
    return ok;
  }
  return true;
}


bool bsend_init_recv_init_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::bsend_buffer buff(mp1::bsend_overhead);
    auto r{mp1::bsend_init(comm_world, std::span<int>{}, 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    auto r{mp1::recv_init(comm_world, std::span<int>{}, 0)};
    r.start();
    while (not r.test()) {
    }
    r.start();
    while (not r.test()) {
    }
  }
  return true;
}


template<typename T>
bool bsend_init_recv_init_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    mp1::bsend_buffer buff(payload_size_bytes(data));
    auto r{bsend_init_value(comm_world, data, 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    auto r{recv_init_value(comm_world, data_r, 0)};
    r.start();
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    r.start();
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    return ok;
  }
  return true;
}


bool ssend_init_recv_init_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{mp1::ssend_init(comm_world, std::span<int>{}, 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    auto r{mp1::recv_init(comm_world, std::span<int>{}, 0)};
    r.start();
    while (not r.test()) {
    }
    r.start();
    while (not r.test()) {
    }
  }
  return true;
}


template<typename T>
bool ssend_init_recv_init_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{ssend_init_value(comm_world, data, 1)};
    r.start();
    r.wait();
    r.start();
    r.wait();
  }
  if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    auto r{recv_init_value(comm_world, data_r, 0)};
    r.start();
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    r.start();
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    return ok;
  }
  return true;
}


bool rsend_init_recv_init_test() {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{mp1::rsend_init(comm_world, std::span<int>{}, 1)};
    mp1::barrier(comm_world);
    r.start();
    r.wait();
    mp1::barrier(comm_world);
    r.start();
    r.wait();
  } else if (comm_world.rank() == 1) {
    auto r{mp1::recv_init(comm_world, std::span<int>{}, 0)};
    r.start();
    mp1::barrier(comm_world);
    while (not r.test()) {
    }
    r.start();
    mp1::barrier(comm_world);
    while (not r.test()) {
    }
  } else {
    mp1::barrier(comm_world);
    mp1::barrier(comm_world);
  }
  return true;
}


template<typename T>
bool rsend_init_recv_init_test(const T& data) {
  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return false;
  if (comm_world.rank() == 0) {
    auto r{rsend_init_value(comm_world, data, 1)};
    mp1::barrier(comm_world);
    r.start();
    r.wait();
    mp1::barrier(comm_world);
    r.start();
    r.wait();
  } else if (comm_world.rank() == 1) {
    bool ok{true};
    T data_r;
    auto r{recv_init_value(comm_world, data_r, 0)};
    r.start();
    mp1::barrier(comm_world);
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    r.start();
    mp1::barrier(comm_world);
    while (not r.test()) {
    }
    ok = ok and data_r == data;
    return ok;
  } else {
    mp1::barrier(comm_world);
    mp1::barrier(comm_world);
  }
  return true;
}


BOOST_AUTO_TEST_CASE(persistent_request_state) {
  if (not mp1::initialized())
    mp1::init();

  const auto comm_world = mp1::comm_world();
  if (comm_world.size() < 2)
    return;

  if (comm_world.rank() == 0) {
    int value = 42;
    auto send = mp1::send_init(comm_world, value, 1);
    BOOST_TEST(send.is_valid());
    BOOST_TEST(not send.is_active());
    BOOST_TEST(send.is_persistent());
    // Waiting an inactive persistent request returns the empty status without MPI.
    BOOST_TEST(not send.wait().is_cancelled());
    // start() makes the request active; a second start() is an error.
    send.start();
    BOOST_TEST(send.is_active());
    BOOST_TEST(throws_logic_error([&] { send.start(); }));
    send.wait();
    BOOST_TEST(send.is_valid());
    BOOST_TEST(not send.is_active());
    // A non-persistent request rejects start().
    auto plain = mp1::isend(comm_world, std::span<int>{}, 1);
    BOOST_TEST(throws_invalid_argument([&] { plain.start(); }));
    plain.wait();
  } else if (comm_world.rank() == 1) {
    int value = 0;
    auto recv = mp1::recv_init(comm_world, value, 0);
    recv.start();
    BOOST_TEST(recv.is_active());
    recv.wait();
    BOOST_TEST(recv.is_valid());
    BOOST_TEST(not recv.is_active());
    recv.start();
    recv.wait();
  }
}


BOOST_AUTO_TEST_CASE(send_init_recv_init) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(send_init_recv_init_test());
  // integer types
  BOOST_TEST(send_init_recv_init_test(std::byte(77)));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(send_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(send_init_recv_init_test(static_cast<wchar_t>('A')));
  BOOST_TEST(send_init_recv_init_test(static_cast<char16_t>('A')));
  BOOST_TEST(send_init_recv_init_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(send_init_recv_init_test(static_cast<float>(3.14)));
  BOOST_TEST(send_init_recv_init_test(static_cast<double>(3.14)));
  BOOST_TEST(send_init_recv_init_test(static_cast<long double>(3.14)));
  BOOST_TEST(send_init_recv_init_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(send_init_recv_init_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(send_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(send_init_recv_init_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(send_init_recv_init_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(send_init_recv_init_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(send_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
  // pointer+count and pointer+layout (the removed container/iterator *_init forms)
  BOOST_TEST(buffer_send_init_recv_init_test(std::vector<int>{1, 2, 3, 4, 5}));
  BOOST_TEST((buffer_send_init_recv_init_test(
      std::vector<std::pair<int, double>>{{1, 2.3}, {3, 4.5}})));
  BOOST_TEST(layout_send_init_recv_init_test<int>());
}


BOOST_AUTO_TEST_CASE(bsend_init_recv_init) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(bsend_init_recv_init_test());
  // integer types
  BOOST_TEST(bsend_init_recv_init_test(std::byte(77)));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(bsend_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(bsend_init_recv_init_test(static_cast<wchar_t>('A')));
  BOOST_TEST(bsend_init_recv_init_test(static_cast<char16_t>('A')));
  BOOST_TEST(bsend_init_recv_init_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(bsend_init_recv_init_test(static_cast<float>(3.14)));
  BOOST_TEST(bsend_init_recv_init_test(static_cast<double>(3.14)));
  BOOST_TEST(bsend_init_recv_init_test(static_cast<long double>(3.14)));
  BOOST_TEST(bsend_init_recv_init_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(bsend_init_recv_init_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(bsend_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(bsend_init_recv_init_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(bsend_init_recv_init_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(bsend_init_recv_init_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(bsend_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
}


BOOST_AUTO_TEST_CASE(ssend_init_recv_init) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(ssend_init_recv_init_test());
  // integer types
  BOOST_TEST(ssend_init_recv_init_test(std::byte(77)));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(ssend_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(ssend_init_recv_init_test(static_cast<wchar_t>('A')));
  BOOST_TEST(ssend_init_recv_init_test(static_cast<char16_t>('A')));
  BOOST_TEST(ssend_init_recv_init_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(ssend_init_recv_init_test(static_cast<float>(3.14)));
  BOOST_TEST(ssend_init_recv_init_test(static_cast<double>(3.14)));
  BOOST_TEST(ssend_init_recv_init_test(static_cast<long double>(3.14)));
  BOOST_TEST(ssend_init_recv_init_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(ssend_init_recv_init_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(ssend_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(ssend_init_recv_init_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(ssend_init_recv_init_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(ssend_init_recv_init_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(ssend_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
}


BOOST_AUTO_TEST_CASE(rsend_init_recv_init) {
  if (not mp1::initialized())
    mp1::init();

  // empty message
  BOOST_TEST(rsend_init_recv_init_test());
  // integer types
  BOOST_TEST(rsend_init_recv_init_test(std::byte(77)));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<char>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<signed char>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<unsigned char>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<signed short>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<unsigned short>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<signed int>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<unsigned int>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<signed long>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<unsigned long>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<signed long long>::max() - 1));
  BOOST_TEST(rsend_init_recv_init_test(std::numeric_limits<unsigned long long>::max() - 1));
  // character types
  BOOST_TEST(rsend_init_recv_init_test(static_cast<wchar_t>('A')));
  BOOST_TEST(rsend_init_recv_init_test(static_cast<char16_t>('A')));
  BOOST_TEST(rsend_init_recv_init_test(static_cast<char32_t>('A')));
  // floating point number types
  BOOST_TEST(rsend_init_recv_init_test(static_cast<float>(3.14)));
  BOOST_TEST(rsend_init_recv_init_test(static_cast<double>(3.14)));
  BOOST_TEST(rsend_init_recv_init_test(static_cast<long double>(3.14)));
  BOOST_TEST(rsend_init_recv_init_test(std::complex<float>(3.14, 2.72)));
  BOOST_TEST(rsend_init_recv_init_test(std::complex<double>(3.14, 2.72)));
  BOOST_TEST(rsend_init_recv_init_test(std::complex<long double>(3.14, 2.72)));
  // logical type
  BOOST_TEST(rsend_init_recv_init_test(true));
  // enums
  enum class my_enum : int { val = std::numeric_limits<int>::max() - 1 };
  BOOST_TEST(rsend_init_recv_init_test(my_enum::val));
  // pairs, tuples and arrays
  BOOST_TEST(rsend_init_recv_init_test(std::pair<int, double>{1, 2.3}));
  BOOST_TEST(rsend_init_recv_init_test(std::tuple<int, double, bool>{1, 2.3, true}));
}
