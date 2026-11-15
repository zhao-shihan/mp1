// Allgather tests: scalar, contiguous-container and array-as-container forms; blocking and
// nonblocking.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <vector>


template<typename T>
bool allgather_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  std::vector<T> v(comm_world.size());
  mp1::allgather(comm_world, val, v);
  return std::all_of(v.begin(), v.end(), [&val](const auto& x) { return x == val; });
}


template<typename T>
bool allgather_array_test(const std::array<T, 4>& val) {
  const auto comm_world{mp1::comm_world()};
  const std::vector<T> send(val.begin(), val.end());
  std::vector<T> v(comm_world.size() * send.size());
  mp1::allgather(comm_world, send, v);
  return std::equal(v.begin(), v.begin() + send.size(), send.begin()) and
         std::equal(v.rbegin(), v.rbegin() + send.size(), send.rbegin());
}


template<typename T>
bool iallgather_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  std::vector<T> v(comm_world.size());
  auto r{mp1::iallgather(comm_world, val, v)};
  r.wait();
  return std::all_of(v.begin(), v.end(), [&val](const auto& x) { return x == val; });
}


template<typename T>
bool iallgather_array_test(const std::array<T, 4>& val) {
  const auto comm_world{mp1::comm_world()};
  const std::vector<T> send(val.begin(), val.end());
  std::vector<T> v(comm_world.size() * send.size());
  auto r{mp1::iallgather(comm_world, send, v)};
  r.wait();
  return std::equal(v.begin(), v.begin() + send.size(), send.begin()) and
         std::equal(v.rbegin(), v.rbegin() + send.size(), send.rbegin());
}


BOOST_AUTO_TEST_CASE(allgather) {
  if (not mp1::initialized())
    mp1::init();

  BOOST_TEST(allgather_test(1.0));
  BOOST_TEST(allgather_array_test(std::array{1, 2, 3, 4}));

  BOOST_TEST(iallgather_test(1.0));
  BOOST_TEST(iallgather_array_test(std::array{1, 2, 3, 4}));
}
