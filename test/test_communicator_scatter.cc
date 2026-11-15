// Scatter tests: root/non-root scalar forms, layout forms, and contiguous-container forms;
// blocking and nonblocking.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <span>
#include <vector>

// One three-element block (element offsets 1, 2, 4) repeated with a six-element stride, as an
// aggregate layout over the whole send buffer.  This is the target region spelling of the
// former per-block layout repeated by MPI.
template<typename T>
[[nodiscard]] mp1::layout<T> aggregate_holey_layout(int peers) {
  mp1::indexed_parameter parameter;
  for (int i = 0; i < peers; ++i) {
    parameter.add(1, static_cast<mp1::count_t>(6 * i + 1));
    parameter.add(1, static_cast<mp1::count_t>(6 * i + 2));
    parameter.add(1, static_cast<mp1::count_t>(6 * i + 4));
  }
  return mp1::layout<T>::indexed(parameter);
}


template<typename T>
bool scatter_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  T recv;
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size(), val);
    mp1::scatter(comm_world, 0, std::span<const T>{v}, recv);
  } else {
    mp1::scatter(comm_world, 0, std::span<const T>{}, recv);
  }
  return recv == val;
}


template<typename T>
bool scatter_array_test(const std::array<T, 4>& val) {
  const auto comm_world{mp1::comm_world()};
  // std::array is a contiguous container in the new API: scatter one size()-element block to
  // every rank.
  std::vector<T> recv(val.size());
  if (comm_world.rank() == 0) {
    std::vector<T> send;
    for (int i{0}; i < comm_world.size(); ++i)
      send.insert(send.end(), val.begin(), val.end());
    mp1::scatter(comm_world, 0, send, recv);
  } else {
    mp1::scatter(comm_world, 0, std::span<const T>{}, recv);
  }
  return std::equal(recv.begin(), recv.end(), val.begin(), val.end());
}


template<typename T>
bool scatter_test(const std::vector<T>& send, const std::vector<T>& expected,
                  const mp1::layout<T>& l) {
  const auto comm_world{mp1::comm_world()};
  std::vector<T> recv(send.size());
  if (comm_world.rank() == 0) {
    std::vector<T> v_send;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(send.begin(), send.end(), std::back_inserter(v_send));
    const mp1::layout<T> aggregate = aggregate_holey_layout<T>(comm_world.size());
    mp1::region<T> send_region{v_send.data(), aggregate};
    mp1::region<T> recv_region{recv.data(), l};
    mp1::scatter(comm_world, 0, send_region, recv_region);
  } else {
    mp1::region<T> recv_region{recv.data(), l};
    mp1::scatter(comm_world, 0, std::span<const T>{}, recv_region);
  }
  return recv == expected;
}


template<typename T>
bool iscatter_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  T recv;
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size(), val);
    auto r{mp1::iscatter(comm_world, 0, std::span<const T>{v}, recv)};
    r.wait();
  } else {
    auto r{mp1::iscatter(comm_world, 0, std::span<const T>{}, recv)};
    r.wait();
  }
  return recv == val;
}


template<typename T>
bool iscatter_array_test(const std::array<T, 4>& val) {
  const auto comm_world{mp1::comm_world()};
  std::vector<T> recv(val.size());
  if (comm_world.rank() == 0) {
    std::vector<T> send;
    for (int i{0}; i < comm_world.size(); ++i)
      send.insert(send.end(), val.begin(), val.end());
    auto r{mp1::iscatter(comm_world, 0, send, recv)};
    r.wait();
  } else {
    auto r{mp1::iscatter(comm_world, 0, std::span<const T>{}, recv)};
    r.wait();
  }
  return std::equal(recv.begin(), recv.end(), val.begin(), val.end());
}


template<typename T>
bool iscatter_test(const std::vector<T>& send, const std::vector<T>& expected,
                   const mp1::layout<T>& l) {
  const auto comm_world{mp1::comm_world()};
  std::vector<T> recv(send.size());
  if (comm_world.rank() == 0) {
    std::vector<T> v_send;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(send.begin(), send.end(), std::back_inserter(v_send));
    const mp1::layout<T> aggregate = aggregate_holey_layout<T>(comm_world.size());
    mp1::region<T> send_region{v_send.data(), aggregate};
    mp1::region<T> recv_region{recv.data(), l};
    auto r{mp1::iscatter(comm_world, 0, send_region, recv_region)};
    r.wait();
  } else {
    mp1::region<T> recv_region{recv.data(), l};
    auto r{mp1::iscatter(comm_world, 0, std::span<const T>{}, recv_region)};
    r.wait();
  }
  return recv == expected;
}


BOOST_AUTO_TEST_CASE(scatter) {
  if (not mp1::initialized())
    mp1::init();

  BOOST_TEST(scatter_test(1.0));
  BOOST_TEST(scatter_array_test(std::array{1, 2, 3, 4}));
  {
    const std::vector send{1, 2, 3, 4, 5, 6};
    const std::vector expected{0, 2, 3, 0, 5, 0};
    mp1::indexed_parameter parameter;
    parameter.add(2, 1);
    parameter.add(1, 4);
    auto l = mp1::layout<int>::indexed(parameter);
    BOOST_TEST(scatter_test(send, expected, l));
  }

  BOOST_TEST(iscatter_test(1.0));
  BOOST_TEST(iscatter_array_test(std::array{1, 2, 3, 4}));
  {
    const std::vector send{1, 2, 3, 4, 5, 6};
    const std::vector expected{0, 2, 3, 0, 5, 0};
    mp1::indexed_parameter parameter;
    parameter.add(2, 1);
    parameter.add(1, 4);
    auto l = mp1::layout<int>::indexed(parameter);
    BOOST_TEST(iscatter_test(send, expected, l));
  }
}
