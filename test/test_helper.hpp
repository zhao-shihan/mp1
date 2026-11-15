#if !(defined MP1_TEST_HELPER_HPP)
#define MP1_TEST_HELPER_HPP

#include "mp1/mp1.hpp"

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>


// SFINAE probes shared by the p2p tests.  The free-function API
// has distinct scalar / contiguous-container / sequence-container / iterator
// forms, so the generic test helpers select the right form from these traits.

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


template<typename T>
struct is_std_array : std::false_type {};

template<typename T, std::size_t N>
struct is_std_array<std::array<T, N>> : std::true_type {};

template<typename T>
inline constexpr bool is_std_array_v = is_std_array<T>::value;


// Buffered-send payload size in bytes (the bsend_buffer constructor adds
// MPI_BSEND_OVERHEAD itself).  Scalar send forms have a payload of sizeof(T);
// container/iterator forms have size() * sizeof(value_type).
template<typename T, typename = void>
struct payload_size {
  static int get(const T&) {
    return static_cast<int>(sizeof(T));
  }
};

template<typename T>
struct payload_size<
    T, std::void_t<decltype(std::declval<const T&>().size()), typename T::value_type>> {
  static int get(const T& data) {
    return static_cast<int>(data.size() * sizeof(typename T::value_type));
  }
};

template<typename T>
int payload_size_bytes(const T& data) {
  return payload_size<T>::get(data);
}


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

MP1_REFLECTION(tuple, a, b)


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

MP1_REFLECTION_TEMPLATE((typename T), (wrapper<T>), data)
MP1_REFLECTION_TEMPLATE((typename T, typename U), (pair_like<T, U>), first, second)


template<typename T>
class add {
public:
  T operator()(const T& a, const T& b) const {
    return a + b;
  }
};


enum class use_non_root_overload { no, yes };

#endif  // MP1_TEST_HELPER_HPP
