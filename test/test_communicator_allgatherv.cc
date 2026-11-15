// Allgatherv tests: indexed (non-contiguous) send layouts, indexed/contiguous receive
// layouts, with and without explicit element displacements, blocking and nonblocking.

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include <numeric>
#include <tuple>
#include <vector>


template<typename T>
bool allgatherv_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v1(N);
  std::vector<T> v2(N);
  std::iota(begin(v1), end(v1), val);
  std::vector<mp1::layout<T>> l;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    mp1::indexed_parameter parameter;
    parameter.add(i + 1, offset);
    l.push_back(mp1::layout<T>::indexed(parameter));
    offset += i + 1;
  }
  mp1::allgatherv(comm_world, mp1::region{v1.data(), l[comm_world.rank()]}, v2, l);
  return v1 == v2;
}


template<typename T>
bool allgatherv_contiguous_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v1(N);
  std::vector<T> v2(N);
  std::iota(begin(v1), end(v1), val);
  std::vector<mp1::layout<T>> l;
  std::vector<mp1::count_t> displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    l.push_back(mp1::layout<T>::contiguous(i + 1));
    displacements.push_back(offset);
    offset += i + 1;
  }
  const auto rank{comm_world.rank()};
  mp1::allgatherv(comm_world, mp1::region{v1.data() + displacements[rank], l[rank]}, v2, l,
                  displacements);
  return v1 == v2;
}


template<typename T>
bool iallgatherv_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v1(N);
  std::vector<T> v2(N);
  std::iota(begin(v1), end(v1), val);
  std::vector<mp1::layout<T>> l;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    mp1::indexed_parameter parameter;
    parameter.add(i + 1, offset);
    l.push_back(mp1::layout<T>::indexed(parameter));
    offset += i + 1;
  }
  auto r{mp1::iallgatherv(comm_world, mp1::region{v1.data(), l[comm_world.rank()]}, v2, l)};
  r.wait();
  return v1 == v2;
}


template<typename T>
bool iallgatherv_contiguous_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v1(N);
  std::vector<T> v2(N);
  std::iota(begin(v1), end(v1), val);
  std::vector<mp1::layout<T>> l;
  std::vector<mp1::count_t> displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    l.push_back(mp1::layout<T>::contiguous(i + 1));
    displacements.push_back(offset);
    offset += i + 1;
  }
  const auto rank{comm_world.rank()};
  auto r{mp1::iallgatherv(comm_world, mp1::region{v1.data() + displacements[rank], l[rank]}, v2,
                          l, displacements)};
  r.wait();
  return v1 == v2;
}


BOOST_AUTO_TEST_CASE(allgatherv) {
  if (not mp1::initialized())
    mp1::init();

  BOOST_TEST(allgatherv_test(1.0));
  BOOST_TEST(allgatherv_test(tuple{1, 2.0}));

  BOOST_TEST(allgatherv_contiguous_test(1.0));
  BOOST_TEST(allgatherv_contiguous_test(tuple{1, 2.0}));

  BOOST_TEST(iallgatherv_test(1.0));
  BOOST_TEST(iallgatherv_test(tuple{1, 2.0}));

  BOOST_TEST(iallgatherv_contiguous_test(1.0));
  BOOST_TEST(iallgatherv_contiguous_test(tuple{1, 2.0}));
}
