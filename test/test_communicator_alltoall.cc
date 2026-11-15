// All-to-all tests: scalar pointer forms, layout forms, in-place forms, and the
// array-as-contiguous-container form; blocking and nonblocking.

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <algorithm>
#include <array>
#include <boost/test/unit_test.hpp>
#include <span>
#include <vector>

// A two-element block (element offsets 0 and 2) repeated with a three-element stride, as an
// aggregate all-to-all send layout.  This is the target region spelling of the former
// per-block layout repeated by MPI.
template<typename T>
[[nodiscard]] mp1::layout<T> alltoall_send_layout(int peers) {
  mp1::indexed_parameter parameter;
  for (int i = 0; i < peers; ++i) {
    parameter.add(1, static_cast<mp1::count_t>(3 * i));
    parameter.add(1, static_cast<mp1::count_t>(3 * i + 2));
  }
  return mp1::layout<T>::indexed(parameter);
}


template<typename T>
bool alltoall_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(comm_world.size(), send_val);
  std::vector<T> recv_data(comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  mp1::alltoall(comm_world, std::span<T>{send_data}, std::span<T>{recv_data});
  return recv_data == expected;
}


template<typename T>
bool alltoall_array_test(const std::array<T, 4>& val) {
  const auto comm_world{mp1::comm_world()};
  // One 4-element block is sent to every peer through the contiguous-container form.
  std::vector<T> send_data;
  for (int i{0}; i < comm_world.size(); ++i)
    send_data.insert(send_data.end(), val.begin(), val.end());
  std::vector<T> recv_data(send_data.size());
  mp1::alltoall(comm_world, send_data, recv_data);
  for (int i{0}; i < comm_world.size(); ++i) {
    if (not std::equal(recv_data.begin() + static_cast<std::ptrdiff_t>(i) * val.size(),
                       recv_data.begin() + static_cast<std::ptrdiff_t>(i + 1) * val.size(),
                       val.begin()))
      return false;
  }
  return true;
}


template<typename T>
bool alltoall_layout_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(3 * comm_world.size(), send_val);
  std::vector<T> recv_data(2 * comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    expected.push_back(expected_val);
    ++expected_val;
  }
  mp1::indexed_parameter send_parameter;
  send_parameter.add(1, 0);
  send_parameter.add(1, 2);
  auto sendl = mp1::layout<T>::indexed(send_parameter);
  auto recvl = mp1::layout<T>::contiguous(2);
  const mp1::layout<T> send_aggregate = alltoall_send_layout<T>(comm_world.size());
  const mp1::layout<T> recv_aggregate = mp1::layout<T>::contiguous(2 * comm_world.size());
  mp1::region<T> send_region{send_data.data(), send_aggregate};
  mp1::region<T> recv_region{recv_data.data(), recv_aggregate};
  mp1::alltoall(comm_world, send_region, recv_region);
  return recv_data == expected;
}


template<typename T>
bool alltoall_inplace_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> sendrecv_data(comm_world.size(), send_val);
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  mp1::alltoall(comm_world, mp1::in_place, std::span<T>{sendrecv_data});
  return sendrecv_data == expected;
}


template<typename T>
bool ialltoall_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(comm_world.size(), send_val);
  std::vector<T> recv_data(comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  auto r{mp1::ialltoall(comm_world, std::span<T>{send_data}, std::span<T>{recv_data})};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool ialltoall_array_test(const std::array<T, 4>& val) {
  const auto comm_world{mp1::comm_world()};
  std::vector<T> send_data;
  for (int i{0}; i < comm_world.size(); ++i)
    send_data.insert(send_data.end(), val.begin(), val.end());
  std::vector<T> recv_data(send_data.size());
  auto r{mp1::ialltoall(comm_world, send_data, recv_data)};
  r.wait();
  for (int i{0}; i < comm_world.size(); ++i) {
    if (not std::equal(recv_data.begin() + static_cast<std::ptrdiff_t>(i) * val.size(),
                       recv_data.begin() + static_cast<std::ptrdiff_t>(i + 1) * val.size(),
                       val.begin()))
      return false;
  }
  return true;
}


template<typename T>
bool ialltoall_layout_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> send_data(3 * comm_world.size(), send_val);
  std::vector<T> recv_data(2 * comm_world.size());
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    expected.push_back(expected_val);
    ++expected_val;
  }
  mp1::indexed_parameter send_parameter;
  send_parameter.add(1, 0);
  send_parameter.add(1, 2);
  auto sendl = mp1::layout<T>::indexed(send_parameter);
  auto recvl = mp1::layout<T>::contiguous(2);
  const mp1::layout<T> send_aggregate = alltoall_send_layout<T>(comm_world.size());
  const mp1::layout<T> recv_aggregate = mp1::layout<T>::contiguous(2 * comm_world.size());
  mp1::region<T> send_region{send_data.data(), send_aggregate};
  mp1::region<T> recv_region{recv_data.data(), recv_aggregate};
  auto r{mp1::ialltoall(comm_world, send_region, recv_region)};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool ialltoall_inplace_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  T send_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  std::vector<T> sendrecv_data(comm_world.size(), send_val);
  std::vector<T> expected;
  T expected_val{val};
  for (int i{0}; i < comm_world.size(); ++i) {
    expected.push_back(expected_val);
    ++expected_val;
  }
  auto r{mp1::ialltoall(comm_world, mp1::in_place, std::span<T>{sendrecv_data})};
  r.wait();
  return sendrecv_data == expected;
}


BOOST_AUTO_TEST_CASE(alltoall) {
  if (not mp1::initialized())
    mp1::init();

  BOOST_TEST(alltoall_test(1.0));
  BOOST_TEST(alltoall_test(tuple{1, 2.0}));

  BOOST_TEST(alltoall_array_test(std::array{1, 2, 3, 4}));

  BOOST_TEST(alltoall_layout_test(1.0));
  BOOST_TEST(alltoall_layout_test(tuple{1, 2.0}));

  BOOST_TEST(alltoall_inplace_test(1.0));
  BOOST_TEST(alltoall_inplace_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoall_test(1.0));
  BOOST_TEST(ialltoall_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoall_array_test(std::array{1, 2, 3, 4}));

  BOOST_TEST(ialltoall_layout_test(1.0));
  BOOST_TEST(ialltoall_layout_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoall_inplace_test(1.0));
  BOOST_TEST(ialltoall_inplace_test(tuple{1, 2.0}));
}
