#if !(defined MPLR_TEST_HELPER_HPP)
#define MPLR_TEST_HELPER_HPP

#include "mplr/mplr.hpp"

#include <type_traits>


template<typename, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<decltype(&T::size)>> : std::true_type {};


template<typename, typename = void>
struct has_resize : std::false_type {};

template<typename T>
struct has_resize<T, std::void_t<decltype(T().resize(1))>> : std::true_type {};


template<typename, typename = void>
struct has_begin_end : std::false_type {};

template<typename T>
struct has_begin_end<
    T, std::void_t<std::common_type_t<decltype(T().begin()), decltype(T().end())>>>
    : std::true_type {};


struct tuple {
  int a{0};
  double b{0};
  tuple& operator++() {
    ++a;
    ++b;
    return *this;
  }
};

inline bool operator==(const tuple& t1, const tuple& t2) {
  return t1.a == t2.a and t1.b == t2.b;
}

inline tuple operator+(const tuple& t1, const tuple& t2) {
  return tuple{t1.a + t2.a, t1.b + t2.b};
}

MPLR_REFLECTION(tuple, a, b)


template<typename T>
struct wrapper {
  T data{};
};

template<typename T>
inline bool operator==(const wrapper<T>& w1, const wrapper<T>& w2) {
  return w1.data == w2.data;
}

template<typename T>
inline wrapper<T> operator+(const wrapper<T>& w1, const wrapper<T>& w2) {
  return wrapper<T>{w1.data + w2.data};
}

template<typename T, typename U>
struct pair_like {
  T first{};
  U second{};
};

template<typename T, typename U>
inline bool operator==(const pair_like<T, U>& p1, const pair_like<T, U>& p2) {
  return p1.first == p2.first && p1.second == p2.second;
}

MPLR_REFLECTION_TEMPLATE((typename T), (wrapper<T>), data)
MPLR_REFLECTION_TEMPLATE((typename T, typename U), (pair_like<T, U>), first, second)


template<typename T>
class add {
public:
  T operator()(const T& a, const T& b) const {
    return a + b;
  }
};


enum class use_non_root_overload { no, yes };

#endif  // MPLR_TEST_HELPER_HPP
