// Alltoallv tests: indexed (non-contiguous) layouts, contiguous layouts with explicit element
// displacements, and in-place forms; blocking and nonblocking.

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include <vector>


template<typename T>
bool alltoallv_with_displacements_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};  // elements to send to each process
  const int N_recv{(N_processes * N_processes + N_processes) / 2};
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  std::vector<mp1::layout<T>> sendls;
  std::vector<mp1::layout<T>> recvls;
  std::vector<mp1::count_t> senddispls;
  std::vector<mp1::count_t> recvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    sendls.push_back(mp1::layout<T>::contiguous(N_send));
    senddispls.push_back(j * N_send);
    recvls.push_back(mp1::layout<T>::contiguous(j + 1));
    recvdispls.push_back((j * j + j) / 2);
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  mp1::alltoallv(comm_world, send_data, sendls, senddispls, recv_data, recvls, recvdispls);
  return recv_data == expected;
}


template<typename T>
bool alltoallv_with_displacements_contiguous_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};
  const int N_recv{(N_processes * N_processes + N_processes) / 2};
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  std::vector<mp1::layout<T>> sendls;
  std::vector<mp1::layout<T>> recvls;
  std::vector<mp1::count_t> senddispls;
  std::vector<mp1::count_t> recvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    sendls.push_back(mp1::layout<T>::contiguous(N_send));
    senddispls.push_back(j * N_send);
    recvls.push_back(mp1::layout<T>::contiguous(j + 1));
    recvdispls.push_back((j * j + j) / 2);
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  mp1::alltoallv(comm_world, send_data, sendls, senddispls, recv_data, recvls, recvdispls);
  return recv_data == expected;
}


template<typename T>
bool alltoallv_without_displacements_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};
  const int N_recv{(N_processes * N_processes + N_processes) / 2};
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  std::vector<mp1::layout<T>> sendls;
  std::vector<mp1::layout<T>> recvls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    mp1::indexed_parameter send_parameter;
    send_parameter.add(N_send, j * N_send);
    sendls.push_back(mp1::layout<T>::indexed(send_parameter));
    mp1::indexed_parameter recv_parameter;
    recv_parameter.add(j + 1, (j * j + j) / 2);
    recvls.push_back(mp1::layout<T>::indexed(recv_parameter));
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  mp1::alltoallv(comm_world, send_data, sendls, std::span<const mp1::count_t>{}, recv_data,
                 recvls);
  return recv_data == expected;
}


template<typename T>
bool ialltoallv_with_displacements_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};
  const int N_recv{(N_processes * N_processes + N_processes) / 2};
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  std::vector<mp1::layout<T>> sendls;
  std::vector<mp1::layout<T>> recvls;
  std::vector<mp1::count_t> senddispls;
  std::vector<mp1::count_t> recvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    sendls.push_back(mp1::layout<T>::contiguous(N_send));
    senddispls.push_back(j * N_send);
    recvls.push_back(mp1::layout<T>::contiguous(j + 1));
    recvdispls.push_back((j * j + j) / 2);
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  auto r{mp1::ialltoallv(comm_world, send_data, sendls, senddispls, recv_data, recvls,
                         recvdispls)};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool ialltoallv_with_displacements_contiguous_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};
  const int N_recv{(N_processes * N_processes + N_processes) / 2};
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  std::vector<mp1::layout<T>> sendls;
  std::vector<mp1::layout<T>> recvls;
  std::vector<mp1::count_t> senddispls;
  std::vector<mp1::count_t> recvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    sendls.push_back(mp1::layout<T>::contiguous(N_send));
    senddispls.push_back(j * N_send);
    recvls.push_back(mp1::layout<T>::contiguous(j + 1));
    recvdispls.push_back((j * j + j) / 2);
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  auto r{mp1::ialltoallv(comm_world, send_data, sendls, senddispls, recv_data, recvls,
                         recvdispls)};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool ialltoallv_without_displacements_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N_processes{comm_world.size()};
  const int N_send{comm_world.rank() + 1};
  const int N_recv{(N_processes * N_processes + N_processes) / 2};
  std::vector<T> send_data;
  std::vector<T> recv_data(N_recv);
  std::vector<T> expected;
  std::vector<mp1::layout<T>> sendls;
  std::vector<mp1::layout<T>> recvls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++expected_val;
  for (int j{0}; j < N_processes; ++j) {
    for (int i{0}; i < N_send; ++i)
      send_data.push_back(send_val);
    ++send_val;
    mp1::indexed_parameter send_parameter;
    send_parameter.add(N_send, j * N_send);
    sendls.push_back(mp1::layout<T>::indexed(send_parameter));
    mp1::indexed_parameter recv_parameter;
    recv_parameter.add(j + 1, (j * j + j) / 2);
    recvls.push_back(mp1::layout<T>::indexed(recv_parameter));
    for (int i{0}; i < j + 1; ++i)
      expected.push_back(expected_val);
  }
  auto r{mp1::ialltoallv(comm_world, send_data, sendls, std::span<const mp1::count_t>{},
                         recv_data, recvls)};
  r.wait();
  return recv_data == expected;
}


template<typename T>
bool alltoallv_in_place_with_displacements_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N_processes{comm_world.size()};
  std::vector<T> sendrecv_data;
  std::vector<T> expected;
  std::vector<mp1::layout<T>> sendrecvls;
  std::vector<mp1::count_t> sendrecvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  int displ{0};
  for (int j{0}; j < N_processes; ++j) {
    const int N_sendrecv{j + comm_world.rank() + 1};  // must be symmetric in j and rank
    for (int i{0}; i < N_sendrecv; ++i) {
      sendrecv_data.push_back(send_val);
      expected.push_back(expected_val);
    }
    sendrecvls.push_back(mp1::layout<T>::contiguous(N_sendrecv));
    sendrecvdispls.push_back(displ);
    displ += N_sendrecv;
    ++expected_val;
  }
  mp1::alltoallv(comm_world, mp1::in_place, std::span<const mp1::layout<T>>{},
                 std::span<const mp1::count_t>{}, sendrecv_data, sendrecvls, sendrecvdispls);
  return sendrecv_data == expected;
}


template<typename T>
bool alltoallv_in_place_without_displacements_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N_processes{comm_world.size()};
  std::vector<T> sendrecv_data;
  std::vector<T> expected;
  std::vector<mp1::layout<T>> sendrecvls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  int displ{0};
  for (int j{0}; j < N_processes; ++j) {
    const int N_sendrecv{j + comm_world.rank() + 1};  // must be symmetric in j and rank
    for (int i{0}; i < N_sendrecv; ++i) {
      sendrecv_data.push_back(send_val);
      expected.push_back(expected_val);
    }
    mp1::indexed_parameter sendrecv_parameter;
    sendrecv_parameter.add(N_sendrecv, displ);
    sendrecvls.push_back(mp1::layout<T>::indexed(sendrecv_parameter));
    displ += N_sendrecv;
    ++expected_val;
  }
  mp1::alltoallv(comm_world, mp1::in_place, std::span<const mp1::layout<T>>{},
                 std::span<const mp1::count_t>{}, sendrecv_data, sendrecvls);
  return sendrecv_data == expected;
}


template<typename T>
bool ialltoallv_in_place_with_displacements_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N_processes{comm_world.size()};
  std::vector<T> sendrecv_data;
  std::vector<T> expected;
  std::vector<mp1::layout<T>> sendrecvls;
  std::vector<mp1::count_t> sendrecvdispls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  int displ{0};
  for (int j{0}; j < N_processes; ++j) {
    const int N_sendrecv{j + comm_world.rank() + 1};
    for (int i{0}; i < N_sendrecv; ++i) {
      sendrecv_data.push_back(send_val);
      expected.push_back(expected_val);
    }
    sendrecvls.push_back(mp1::layout<T>::contiguous(N_sendrecv));
    sendrecvdispls.push_back(displ);
    displ += N_sendrecv;
    ++expected_val;
  }
  auto r{mp1::ialltoallv(comm_world, mp1::in_place, std::span<const mp1::layout<T>>{},
                         std::span<const mp1::count_t>{}, sendrecv_data, sendrecvls,
                         sendrecvdispls)};
  r.wait();
  return sendrecv_data == expected;
}


template<typename T>
bool ialltoallv_in_place_without_displacements_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N_processes{comm_world.size()};
  std::vector<T> sendrecv_data;
  std::vector<T> expected;
  std::vector<mp1::layout<T>> sendrecvls;
  T send_val{val};
  T expected_val{val};
  for (int i{0}; i < comm_world.rank(); ++i)
    ++send_val;
  int displ{0};
  for (int j{0}; j < N_processes; ++j) {
    const int N_sendrecv{j + comm_world.rank() + 1};
    for (int i{0}; i < N_sendrecv; ++i) {
      sendrecv_data.push_back(send_val);
      expected.push_back(expected_val);
    }
    mp1::indexed_parameter sendrecv_parameter;
    sendrecv_parameter.add(N_sendrecv, displ);
    sendrecvls.push_back(mp1::layout<T>::indexed(sendrecv_parameter));
    displ += N_sendrecv;
    ++expected_val;
  }
  auto r{mp1::ialltoallv(comm_world, mp1::in_place, std::span<const mp1::layout<T>>{},
                         std::span<const mp1::count_t>{}, sendrecv_data, sendrecvls)};
  r.wait();
  return sendrecv_data == expected;
}


BOOST_AUTO_TEST_CASE(alltoallv) {
  if (not mp1::initialized())
    mp1::init();

  BOOST_TEST(alltoallv_with_displacements_test(1.0));
  BOOST_TEST(alltoallv_with_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(alltoallv_with_displacements_contiguous_test(1.0));
  BOOST_TEST(alltoallv_with_displacements_contiguous_test(tuple{1, 2.0}));

  BOOST_TEST(alltoallv_without_displacements_test(1.0));
  BOOST_TEST(alltoallv_without_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoallv_with_displacements_test(1.0));
  BOOST_TEST(ialltoallv_with_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoallv_with_displacements_contiguous_test(1.0));
  BOOST_TEST(ialltoallv_with_displacements_contiguous_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoallv_without_displacements_test(1.0));
  BOOST_TEST(ialltoallv_without_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(alltoallv_in_place_with_displacements_test(1.0));
  BOOST_TEST(alltoallv_in_place_with_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(alltoallv_in_place_without_displacements_test(1.0));
  BOOST_TEST(alltoallv_in_place_without_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoallv_in_place_with_displacements_test(1.0));
  BOOST_TEST(ialltoallv_in_place_with_displacements_test(tuple{1, 2.0}));

  BOOST_TEST(ialltoallv_in_place_without_displacements_test(1.0));
  BOOST_TEST(ialltoallv_in_place_without_displacements_test(tuple{1, 2.0}));
}
