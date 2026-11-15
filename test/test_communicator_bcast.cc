// Broadcast tests: scalar and contiguous-container forms, blocking and nonblocking, plus a
// non-contiguous layout payload.

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <vector>


template<typename T>
bool bcast_test(const T& val) {
  const auto comm_world = mp1::comm_world();
  T x{};
  if (comm_world.rank() == 0)
    x = val;
  mp1::bcast(comm_world, 0, x);
  return x == val;
}


template<typename T>
bool bcast_test(const std::vector<T>& send, const std::vector<T>& expected,
                const mp1::layout<T>& l) {
  const auto comm_world = mp1::comm_world();
  std::vector<T> x(send.size(), {});
  if (comm_world.rank() == 0)
    x = send;
  mp1::bcast(comm_world, 0, mp1::region<int>{x.data(), l});
  if (comm_world.rank() == 0)
    return x == send;
  else
    return x == expected;
}


template<typename T>
bool ibcast_test(const T& val) {
  const auto comm_world = mp1::comm_world();
  T x{};
  if (comm_world.rank() == 0)
    x = val;
  auto r{mp1::ibcast(comm_world, 0, x)};
  r.wait();
  return x == val;
}


template<typename T>
bool ibcast_test(const std::vector<T>& send, const std::vector<T>& expected,
                 const mp1::layout<T>& l) {
  const auto comm_world = mp1::comm_world();
  std::vector<T> x(send.size(), {});
  if (comm_world.rank() == 0)
    x = send;
  auto r{mp1::ibcast(comm_world, 0, mp1::region<int>{x.data(), l})};
  r.wait();
  if (comm_world.rank() == 0)
    return x == send;
  else
    return x == expected;
}


BOOST_AUTO_TEST_CASE(bcast) {
  if (not mp1::initialized())
    mp1::init();

  mp1::indexed_parameter parameter;
  parameter.add(2, 1);
  parameter.add(1, 4);
  BOOST_TEST(bcast_test(1.0));
  BOOST_TEST(bcast_test(std::array{1, 2, 3, 4}));
  BOOST_TEST(bcast_test(std::vector{1, 2, 3, 4, 5, 6}, std::vector{0, 2, 3, 0, 5, 0},
                        mp1::layout<int>::indexed(parameter)));

  BOOST_TEST(ibcast_test(1.0));
  BOOST_TEST(ibcast_test(std::array{1, 2, 3, 4}));
  BOOST_TEST(ibcast_test(std::vector{1, 2, 3, 4, 5, 6}, std::vector{0, 2, 3, 0, 5, 0},
                         mp1::layout<int>::indexed(parameter)));
}
