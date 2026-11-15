// Gather tests: root/non-root scalar forms, layout forms, and contiguous-container forms;
// blocking and nonblocking.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <iterator>
#include <span>
#include <vector>

// One three-element block (element offsets 1, 2, 4) repeated with a six-element stride, as an
// aggregate layout over the whole receive buffer.  This is the target region spelling of the
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
bool gather_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size());
    mp1::gather(comm_world, 0, val, v);
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      v_expected.push_back(val);
    return v == v_expected;
  } else {
    mp1::gather(comm_world, 0, val, std::span<T>{});
    return true;
  }
}


template<typename T>
bool gather_array_test(const std::array<T, 4>& val) {
  const auto comm_world{mp1::comm_world()};
  // std::array is a contiguous container in the new API: gather all elements of the array
  // from every rank into a receive vector of size() blocks.
  const std::vector<T> send(val.begin(), val.end());
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size() * send.size());
    mp1::gather(comm_world, 0, send, v);
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(send.begin(), send.end(), std::back_inserter(v_expected));
    return v == v_expected;
  } else {
    mp1::gather(comm_world, 0, send, std::span<T>{});
    return true;
  }
}


template<typename T>
bool gather_test(const std::vector<T>& send, const std::vector<T>& expected,
                 const mp1::layout<T>& l) {
  const auto comm_world{mp1::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(static_cast<std::size_t>(6 * comm_world.size()));
    const mp1::layout<T> aggregate = aggregate_holey_layout<T>(comm_world.size());
    mp1::region<T> send_region{const_cast<T*>(send.data()), l};
    mp1::region<T> recv_region{v.data(), aggregate};
    mp1::gather(comm_world, 0, send_region, recv_region);
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(expected.begin(), expected.end(), std::back_inserter(v_expected));
    return v == v_expected;
  } else {
    mp1::gather(comm_world, 0, mp1::region<T>{const_cast<T*>(send.data()), l}, std::span<T>{});
    return true;
  }
}


template<typename T>
bool igather_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size());
    auto r{mp1::igather(comm_world, 0, val, v)};
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      v_expected.push_back(val);
    r.wait();
    return v == v_expected;
  } else {
    auto r{mp1::igather(comm_world, 0, val, std::span<T>{})};
    r.wait();
    return true;
  }
}


template<typename T>
bool igather_array_test(const std::array<T, 4>& val) {
  const auto comm_world{mp1::comm_world()};
  const std::vector<T> send(val.begin(), val.end());
  if (comm_world.rank() == 0) {
    std::vector<T> v(comm_world.size() * send.size());
    auto r{mp1::igather(comm_world, 0, send, v)};
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(send.begin(), send.end(), std::back_inserter(v_expected));
    r.wait();
    return v == v_expected;
  } else {
    auto r{mp1::igather(comm_world, 0, send, std::span<T>{})};
    r.wait();
    return true;
  }
}


template<typename T>
bool igather_test(const std::vector<T>& send, const std::vector<T>& expected,
                  const mp1::layout<T>& l) {
  const auto comm_world{mp1::comm_world()};
  if (comm_world.rank() == 0) {
    std::vector<T> v(static_cast<std::size_t>(6 * comm_world.size()));
    const mp1::layout<T> aggregate = aggregate_holey_layout<T>(comm_world.size());
    mp1::region<T> send_region{const_cast<T*>(send.data()), l};
    mp1::region<T> recv_region{v.data(), aggregate};
    auto r{mp1::igather(comm_world, 0, send_region, recv_region)};
    std::vector<T> v_expected;
    for (int i{0}; i < comm_world.size(); ++i)
      std::copy(expected.begin(), expected.end(), std::back_inserter(v_expected));
    r.wait();
    return v == v_expected;
  } else {
    auto r{mp1::igather(comm_world, 0, mp1::region<T>{const_cast<T*>(send.data()), l},
                        std::span<T>{})};
    r.wait();
    return true;
  }
}


BOOST_AUTO_TEST_CASE(gather) {
  if (not mp1::initialized())
    mp1::init();

  BOOST_TEST(gather_test(1.0));
  BOOST_TEST(gather_array_test(std::array{1, 2, 3, 4}));
  {
    const std::vector send{1, 2, 3, 4, 5, 6};
    const std::vector expected{0, 2, 3, 0, 5, 0};
    mp1::indexed_parameter parameter;
    parameter.add(2, 1);
    parameter.add(1, 4);
    auto l = mp1::layout<int>::indexed(parameter);
    BOOST_TEST(gather_test(send, expected, l));
  }

  BOOST_TEST(igather_test(1.0));
  BOOST_TEST(igather_array_test(std::array{1, 2, 3, 4}));
  {
    const std::vector send{1, 2, 3, 4, 5, 6};
    const std::vector expected{0, 2, 3, 0, 5, 0};
    mp1::indexed_parameter parameter;
    parameter.add(2, 1);
    parameter.add(1, 4);
    auto l = mp1::layout<int>::indexed(parameter);
    BOOST_TEST(igather_test(send, expected, l));
  }
}
