// Gatherv tests: indexed (non-contiguous) and contiguous layouts, with and without explicit
// element displacements, root and non-root forms, blocking and nonblocking.

#include "mp1/mp1.hpp"
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include <numeric>
#include <tuple>
#include <vector>


template<use_non_root_overload variant, typename T>
bool gatherv_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  std::vector<mp1::layout<T>> layouts;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    mp1::indexed_parameter parameter;
    parameter.add(i + 1, offset);
    layouts.push_back(mp1::layout<T>::indexed(parameter));
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const auto layout = mp1::layout<T>::contiguous(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0)
      mp1::gatherv(comm_world, 0, mp1::region{v_send.data(), layout}, v_gather, layouts);
    else
      mp1::gatherv(comm_world, 0, mp1::region{v_send.data(), layout}, std::span<T>{},
                   mp1::blocks{});
  } else {
    mp1::gatherv(comm_world, 0, mp1::region{v_send.data(), layout}, v_gather, layouts);
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool gatherv_contiguous_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  std::vector<mp1::layout<T>> layouts;
  std::vector<mp1::count_t> displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mp1::layout<T>::contiguous(i + 1));
    displacements.push_back(offset);
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const auto layout = mp1::layout<T>::contiguous(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0)
      mp1::gatherv(comm_world, 0, v_send, v_gather, layouts, displacements);
    else
      mp1::gatherv(comm_world, 0, mp1::region{v_send.data(), layout}, std::span<T>{},
                   mp1::blocks{});
  } else {
    mp1::gatherv(comm_world, 0, v_send, v_gather, layouts, displacements);
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool igatherv_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  std::vector<mp1::layout<T>> layouts;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    mp1::indexed_parameter parameter;
    parameter.add(i + 1, offset);
    layouts.push_back(mp1::layout<T>::indexed(parameter));
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const auto layout = mp1::layout<T>::contiguous(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0) {
      auto r{
          mp1::igatherv(comm_world, 0, mp1::region{v_send.data(), layout}, v_gather, layouts)};
      r.wait();
    } else {
      auto r{mp1::igatherv(comm_world, 0, mp1::region{v_send.data(), layout}, std::span<T>{},
                           mp1::blocks{})};
      r.wait();
    }
  } else {
    auto r{mp1::igatherv(comm_world, 0, mp1::region{v_send.data(), layout}, v_gather, layouts)};
    r.wait();
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


template<use_non_root_overload variant, typename T>
bool igatherv_contiguous_test(const T& val) {
  const auto comm_world{mp1::comm_world()};
  const int N{(comm_world.size() * comm_world.size() + comm_world.size()) / 2};
  std::vector<T> v_gather(N);
  std::vector<T> v_send(comm_world.rank() + 1);
  std::vector<T> v_expected(N);
  std::iota(begin(v_expected), end(v_expected), val);
  std::vector<mp1::layout<T>> layouts;
  std::vector<mp1::count_t> displacements;
  for (int i{0}, i_end{comm_world.size()}, offset{0}; i < i_end; ++i) {
    layouts.push_back(mp1::layout<T>::contiguous(i + 1));
    displacements.push_back(offset);
    offset += i + 1;
  }
  T t_val{val};
  for (int i{0}, i_end{(comm_world.rank() * comm_world.rank() + comm_world.rank()) / 2};
       i < i_end; ++i)
    ++t_val;
  std::iota(begin(v_send), end(v_send), t_val);
  const auto layout = mp1::layout<T>::contiguous(comm_world.rank() + 1);
  if constexpr (variant == use_non_root_overload::yes) {
    if (comm_world.rank() == 0) {
      auto r{mp1::igatherv(comm_world, 0, v_send, v_gather, layouts, displacements)};
      r.wait();
    } else {
      auto r{mp1::igatherv(comm_world, 0, mp1::region{v_send.data(), layout}, std::span<T>{},
                           mp1::blocks{})};
      r.wait();
    }
  } else {
    auto r{mp1::igatherv(comm_world, 0, v_send, v_gather, layouts, displacements)};
    r.wait();
  }
  return comm_world.rank() == 0 ? v_gather == v_expected : true;
}


BOOST_AUTO_TEST_CASE(gatherv) {
  if (not mp1::initialized())
    mp1::init();

  BOOST_TEST(gatherv_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(gatherv_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(gatherv_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(gatherv_test<use_non_root_overload::yes>(tuple{1, 2.0}));

  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(gatherv_contiguous_test<use_non_root_overload::yes>(tuple{1, 2.0}));

  BOOST_TEST(igatherv_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(igatherv_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(igatherv_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(igatherv_test<use_non_root_overload::yes>(tuple{1, 2.0}));

  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::no>(1.0));
  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::no>(tuple{1, 2.0}));

  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::yes>(1.0));
  BOOST_TEST(igatherv_contiguous_test<use_non_root_overload::yes>(tuple{1, 2.0}));
}
