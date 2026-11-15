#if !(defined MP1_IMPL_DETAIL_CONCEPTS_HPP)

#define MP1_IMPL_DETAIL_CONCEPTS_HPP

// The seven public predicates and the four buffer-position spellings.
//
// The old detail layer classified buffers with eleven `container_traits` specializations, a
// hand-written iterator table and five SFINAE constants.  This header replaces that layer with
// the corrected lattice:
//
//   1. element                 T has an MPI datatype mapping
//   2. element_value           a single value, not a range (std::array/C-array disambiguator)
//   3. element_buffer          contiguous, sized range of elements, not keyed, not char[N]
//   4. element_sequence        forward, sized, non-contiguous sequence with lvalue references
//   5. resizable_element_buffer  element_buffer that can size itself (vector, string)
//   6. borrowable              a value whose lifetime the library may borrow
//   7. buffer_at<Role>         the position predicate (one concept, four spellings)
//
// `detail::associative_like` is the associative-container exclusion: ordered associative
// containers iterate in comparator order, unordered ones in unspecified order, duplicate keys
// collapse and map's element is pair<const K, V> (const key, ABI padding).  They are keyed, not
// flat element sequences.
//
// The predicates are constraints, not diagnostics: the compile-fail probes assert the
// constraint failure, and the diagnostic text is not a stable interface.

#include "mp1/impl/datatype.hpp"
#include "mp1/impl/vocabulary.hpp"

#include <concepts>
#include <cstddef>
#include <ranges>
#include <type_traits>


namespace mp1 {

  namespace detail {

    // Standard associative containers expose key_type.  They are keyed, not flat element
    // sequences: ordered containers have comparator-defined iteration order, unordered ones
    // have unspecified order, duplicate keys collapse, and map's element is pair<const K,V>
    // (whose native image includes ABI padding and a const key).  Exclude all of them.
    template<class R>
    concept associative_like = requires { typename std::remove_cvref_t<R>::key_type; };

    // A built-in character array (a string literal, a char buffer) carries no explicit
    // length and would send its terminating NUL as an element.  std::array<char, N>,
    // std::string, std::string_view and std::span<char> stay accepted.
    template<class T>
    inline constexpr bool is_character_v =
        std::is_same_v<T, char> or std::is_same_v<T, wchar_t> or
#if defined(__cpp_char8_t)
        std::is_same_v<T, char8_t> or
#endif
        std::is_same_v<T, char16_t> or std::is_same_v<T, char32_t>;

    template<class R>
    inline constexpr bool builtin_character_array =
        std::is_array_v<std::remove_reference_t<R>> and
        is_character_v<std::remove_cv_t<std::remove_extent_t<std::remove_reference_t<R>>>>;

  }  // namespace detail

  // The element type of any buffer position: the region element for a region, the range value
  // type otherwise.  Kept here rather than in vocabulary.hpp: the range_value_t mapping needs
  // <ranges>, and vocabulary.hpp is the light base of the include graph (fwd.hpp includes it).
  template<class B, bool = is_region_v<std::remove_cvref_t<B>>>
  struct buffer_element;
  template<class B>
  struct buffer_element<B, true> {
    using type = std::remove_cv_t<region_element_t<std::remove_cvref_t<B>>>;
  };
  template<class B>
  struct buffer_element<B, false> {
    using type = std::ranges::range_value_t<std::remove_cvref_t<B>>;
  };
  template<class B>
  using buffer_element_t = typename buffer_element<B>::type;

  // 1. element: T has an MPI datatype mapping (predefined, reflected, or MP1_DATATYPE_TRAITS).
  template<class T>
  concept element = has_datatype_v<std::remove_cvref_t<T>>;

  // 2. element_value: a single value, not a range.  This is the std::array/C-array
  //    disambiguator.  A built-in C array is a range, so it can never be a value; a
  //    std::pair is a value.
  template<class T>
  concept element_value = element<T> && !std::ranges::range<std::remove_cvref_t<T>>;

  // 3. element_buffer: contiguous, sized range of elements; excludes built-in character
  //    arrays (a string literal would send its NUL), vector<bool> (not contiguous; proxy
  //    reference), and all associative containers (non-flat, keyed semantics).
  template<class R>
  concept element_buffer = std::ranges::contiguous_range<R> && std::ranges::sized_range<R> &&
                           element<std::ranges::range_value_t<R>> &&
                           !detail::builtin_character_array<R> && !detail::associative_like<R>;

  // 4. element_sequence: forward, sized, non-contiguous sequence with real lvalue
  //    references; send-only (staging).  Rejects vector<bool> proxies, forward_list
  //    (unsized), and all associative containers.
  template<class R>
  concept element_sequence =
      std::ranges::forward_range<R> && std::ranges::sized_range<R> &&
      !std::ranges::contiguous_range<R> &&
      std::is_lvalue_reference_v<std::ranges::range_reference_t<R>> &&
      element<std::ranges::range_value_t<R>> && !detail::builtin_character_array<R> &&
      !detail::associative_like<R>;

  // 5. resizable_element_buffer: the auto-size/bounded receive target (vector, string).
  template<class R>
  concept resizable_element_buffer = element_buffer<R> && requires(R& r, std::size_t n) {
    r.resize(n);
    r.max_size();
  };

  // 6. borrowable: an lvalue, a borrowed range (span/string_view), a region, or the stateless
  //    `in_place_t` sentinel.  This is the async/persistent lifetime guard; it rejects an
  //    rvalue container.  `in_place_t{}` is a value with no lifetime, so it is borrowable even
  //    when written as a temporary.
  template<class R>
  concept borrowable =
      std::is_lvalue_reference_v<R> || std::ranges::borrowed_range<R> ||
      is_region_v<std::remove_cvref_t<R>> || std::same_as<std::remove_cvref_t<R>, in_place_t>;

  // 7. buffer_at: the position predicate.  Role names both the direction and whether MPI
  //    allows MPI_IN_PLACE in that position.  `region<const T>` is send-only; `region<T>` can
  //    send or receive (receive requires a mutable region); recv buffers must be writable;
  //    non-region recv buffers must be borrowable (no rvalue container); value sends may be
  //    rvalues (blocking), value receives must be mutable lvalues.
  enum class buffer_role { send, recv, send_in_place, recv_in_place };

  template<buffer_role Role>
  inline constexpr bool in_place_role =
      Role == buffer_role::send_in_place || Role == buffer_role::recv_in_place;
  template<buffer_role Role>
  inline constexpr bool send_role =
      Role == buffer_role::send || Role == buffer_role::send_in_place;

  template<buffer_role Role, class B>
  concept buffer_at =
      (in_place_role<Role> && std::same_as<std::remove_cvref_t<B>, in_place_t>) ||
      (is_region_v<std::remove_cvref_t<B>> &&
       (send_role<Role> || !std::is_const_v<region_element_t<std::remove_cvref_t<B>>>)) ||
      (((element_buffer<B> || (send_role<Role> && element_sequence<B>)) &&
        (send_role<Role> || borrowable<B>) &&
        (send_role<Role> || std::ranges::output_range<B, std::ranges::range_value_t<B>>)) ||
       (element_value<B> &&
        (send_role<Role> ||
         (std::is_lvalue_reference_v<B> && !std::is_const_v<std::remove_reference_t<B>>))));

  // Public position vocabulary (four spellings of predicate 7, not four new predicates).
  template<class B>
  concept send_source = buffer_at<buffer_role::send, B>;
  template<class B>
  concept recv_sink = buffer_at<buffer_role::recv, B>;
  template<class B>
  concept in_place_send_source = buffer_at<buffer_role::send_in_place, B>;
  template<class B>
  concept in_place_recv_sink = buffer_at<buffer_role::recv_in_place, B>;

}  // namespace mp1

#endif
