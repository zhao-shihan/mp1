// Public-API compile check: the C++20 buffer concepts and the vocabulary.
//
// Compile-only.  Every check is a static_assert or a function that is compiled but never
// linked or executed; invalid runtime objects are acceptable here.
//
// Covered:
//   * the exact 7-predicate matrix, including the region positives on the real class;
//   * the associative-container exclusion;
//   * std::span<request> as the completion borrow (request_span is gone; the negative probe
//     is test/compile_fail/request_span_removed.cc);
//   * the checked conversion helpers and the checked pointer+count -> span conversion.

#include "mp1/mp1.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <forward_list>
#include <iterator>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

  // ---- 1. element / 2. element_value -----------------------------------------------------
  static_assert(mp1::element<int>);
  static_assert(mp1::element<std::pair<int, int>>);
  static_assert(mp1::element<std::array<int, 3>>);
  static_assert(!mp1::element<std::string>);
  static_assert(!mp1::element<std::vector<int>>);
  static_assert(!mp1::element<std::span<int>>);

  static_assert(mp1::element_value<int>);
  static_assert(mp1::element_value<std::pair<int, int>>);
  static_assert(!mp1::element_value<std::array<int, 3>>);
  static_assert(!mp1::element_value<int[4]>);

  // ---- 3. element_buffer ----------------------------------------------------------------
  static_assert(mp1::element_buffer<std::vector<int>>);
  static_assert(mp1::element_buffer<std::array<int, 3>>);
  static_assert(mp1::element_buffer<std::string>);
  static_assert(mp1::element_buffer<std::span<int>>);
  static_assert(mp1::element_buffer<std::span<const int>>);
  static_assert(mp1::element_buffer<std::string_view>);
  static_assert(mp1::element_buffer<int[4]>);
  static_assert(!mp1::element_buffer<char[4]>);
  static_assert(!mp1::element_buffer<const char[4]>);
  static_assert(mp1::element_buffer<std::array<char, 4>>);
  static_assert(!mp1::element_buffer<std::vector<bool>>);
  static_assert(!mp1::element_buffer<std::list<int>>);
  static_assert(!mp1::element_buffer<std::map<int, int>>);

  // ---- 4. element_sequence --------------------------------------------------------------
  static_assert(mp1::element_sequence<std::list<int>>);
  static_assert(mp1::element_sequence<std::deque<int>>);
  static_assert(mp1::element_sequence<std::list<std::pair<int, int>>>);
  static_assert(!mp1::element_sequence<std::forward_list<int>>);  // unsized
  static_assert(!mp1::element_sequence<std::vector<bool>>);       // proxy reference
  static_assert(!mp1::element_sequence<std::string>);             // contiguous
  static_assert(!mp1::element_sequence<std::set<int>>);           // associative
  static_assert(!mp1::element_sequence<std::unordered_set<int>>);
  static_assert(!mp1::element_sequence<std::map<int, int>>);
  static_assert(!mp1::element_sequence<std::unordered_map<int, int>>);

  // ---- 5. resizable_element_buffer ------------------------------------------------------
  static_assert(mp1::resizable_element_buffer<std::vector<int>>);
  static_assert(mp1::resizable_element_buffer<std::string>);
  static_assert(!mp1::resizable_element_buffer<std::array<int, 3>>);
  static_assert(!mp1::resizable_element_buffer<std::span<int>>);
  static_assert(!mp1::resizable_element_buffer<std::vector<bool>>);

  // ---- 6. borrowable --------------------------------------------------------------------
  static_assert(mp1::borrowable<std::vector<int>&>);
  static_assert(!mp1::borrowable<std::vector<int>>);
  static_assert(mp1::borrowable<std::span<int>>);
  static_assert(mp1::borrowable<std::string_view>);
  static_assert(!mp1::borrowable<std::string>);
  static_assert(!mp1::borrowable<std::vector<int>&&>);
  static_assert(mp1::borrowable<mp1::in_place_t&&>);

  // ---- 7. buffer_at: send positions -----------------------------------------------------
  static_assert(mp1::send_source<int&>);
  static_assert(mp1::send_source<const int&>);
  static_assert(mp1::send_source<int>);
  static_assert(mp1::send_source<std::pair<int, int>>);
  static_assert(mp1::send_source<std::vector<int>&>);
  static_assert(mp1::send_source<std::vector<int>>);
  static_assert(mp1::send_source<std::span<const int>>);
  static_assert(mp1::send_source<std::string_view>);
  static_assert(mp1::send_source<std::list<int>&>);
  static_assert(mp1::send_source<std::list<std::pair<int, int>>&>);
  static_assert(mp1::send_source<std::vector<std::pair<int, int>>&>);
  static_assert(!mp1::send_source<mp1::in_place_t>);
  static_assert(!mp1::send_source<char[4]>);
  static_assert(!mp1::send_source<std::vector<bool>>);
  static_assert(!mp1::send_source<std::forward_list<int>>);
  static_assert(!mp1::send_source<std::map<int, int>&>);
  static_assert(!mp1::send_source<std::set<int>&>);
  static_assert(!mp1::send_source<std::unordered_map<int, int>&>);

  // ---- 7. buffer_at: receive positions --------------------------------------------------
  static_assert(mp1::recv_sink<int&>);
  static_assert(!mp1::recv_sink<const int&>);
  static_assert(!mp1::recv_sink<int>);
  static_assert(mp1::recv_sink<std::vector<int>&>);
  static_assert(!mp1::recv_sink<std::vector<int>>);
  static_assert(mp1::recv_sink<std::span<int>>);
  static_assert(!mp1::recv_sink<std::span<const int>>);
  static_assert(mp1::recv_sink<std::array<int, 3>&>);
  static_assert(!mp1::recv_sink<mp1::in_place_t>);
  static_assert(!mp1::recv_sink<std::list<int>&>);
  static_assert(!mp1::recv_sink<std::string_view>);
  static_assert(!mp1::recv_sink<int[4]>);
  static_assert(mp1::recv_sink<int (&)[4]>);
  static_assert(!mp1::recv_sink<std::map<int, int>&>);
  static_assert(!mp1::recv_sink<std::set<int>&>);

  // ---- in-place positions ---------------------------------------------------------------
  static_assert(mp1::in_place_send_source<mp1::in_place_t>);
  static_assert(mp1::in_place_send_source<int&>);
  static_assert(mp1::in_place_send_source<std::vector<int>&>);
  static_assert(mp1::in_place_recv_sink<mp1::in_place_t>);
  static_assert(mp1::in_place_recv_sink<std::vector<int>&>);
  static_assert(!mp1::in_place_send_source<std::vector<bool>>);

  // ---- region/blocks/in_place with the real vocabulary -----------------------------------
  // `region<T>` is the real class, so the positive rows are asserted on the actual type.
  static_assert(mp1::is_region_v<mp1::region<int>>);
  static_assert(mp1::is_region_v<mp1::region<const int>>);
  static_assert(!mp1::is_region_v<int>);
  static_assert(!mp1::is_region_v<std::vector<int>>);
  static_assert(!mp1::is_region_v<mp1::blocks>);
  static_assert(std::is_void_v<mp1::region_element_t<int>>);
  static_assert(std::is_same_v<mp1::region_element_t<mp1::region<int>>, int>);
  static_assert(std::is_same_v<mp1::region_element_t<mp1::region<const int>>, const int>);
  static_assert(std::is_same_v<mp1::buffer_element_t<mp1::region<int>>, int>);
  static_assert(std::is_same_v<mp1::buffer_element_t<mp1::region<const int>>, int>);
  static_assert(std::is_same_v<mp1::buffer_element_t<mp1::region<const int>&>, int>);

  static_assert(mp1::send_source<mp1::region<int>>);
  static_assert(mp1::send_source<mp1::region<int>&>);
  static_assert(mp1::recv_sink<mp1::region<int>>);
  static_assert(mp1::recv_sink<mp1::region<int>&>);
  static_assert(mp1::send_source<mp1::region<const int>>);
  static_assert(mp1::send_source<mp1::region<const int>&>);
  static_assert(!mp1::recv_sink<mp1::region<const int>>);
  static_assert(!mp1::recv_sink<mp1::region<const int>&>);
  static_assert(mp1::in_place_send_source<mp1::region<const int>>);
  static_assert(mp1::in_place_recv_sink<mp1::region<int>>);
  static_assert(!mp1::in_place_recv_sink<mp1::region<const int>>);

  // `blocks` is a descriptor pair, not a buffer: it carries no element range and no datatype.
  static_assert(!mp1::element<mp1::blocks>);
  static_assert(!mp1::element_value<mp1::blocks>);
  static_assert(!mp1::element_buffer<mp1::blocks>);
  static_assert(!mp1::send_source<mp1::blocks>);
  static_assert(!mp1::recv_sink<mp1::blocks>);

  // `in_place_t` is the stateless sentinel: accepted only in the two in-place roles, and
  // borrowable both as the named lvalue and as a temporary.
  static_assert(mp1::borrowable<mp1::in_place_t>);
  static_assert(mp1::borrowable<mp1::in_place_t&&>);
  static_assert(mp1::borrowable<const mp1::in_place_t&>);
  static_assert(!mp1::element<mp1::in_place_t>);
  static_assert(!mp1::send_source<mp1::in_place_t>);
  static_assert(!mp1::recv_sink<mp1::in_place_t>);
  static_assert(mp1::in_place_send_source<mp1::in_place_t>);
  static_assert(mp1::in_place_recv_sink<mp1::in_place_t>);
  static_assert(mp1::in_place_send_source<mp1::in_place_t&>);
  static_assert(mp1::in_place_recv_sink<mp1::in_place_t&&>);

  // ---- std::span<request> is the completion borrow --------------------------------------
  static_assert(std::is_same_v<decltype(&mp1::wait_all),
                               std::vector<mp1::status> (*)(std::span<mp1::request>)>);
  static_assert(
      std::is_same_v<decltype(&mp1::test_all),
                     std::optional<std::vector<mp1::status>> (*)(std::span<mp1::request>)>);
  static_assert(
      std::is_same_v<decltype(&mp1::wait_any), mp1::completion (*)(std::span<mp1::request>)>);
  static_assert(std::is_same_v<decltype(&mp1::start_all), void (*)(std::span<mp1::request>)>);

  // ---- checked conversions and the checked pointer+count -> span ------------------------
  using check_site = std::source_location;
  static_assert(std::is_same_v<decltype(&mp1::detail::checked_count),
                               mp1::count_t (*)(MPI_Count, const mp1::context&, check_site)>);
  static_assert(std::is_same_v<decltype(&mp1::detail::checked_int),
                               int (*)(std::size_t, const mp1::context&, check_site)>);
  static_assert(std::is_same_v<decltype(&mp1::detail::checked_mpi_count),
                               MPI_Count (*)(mp1::count_t, const mp1::context&, check_site)>);
  static_assert(std::is_same_v<decltype(&mp1::detail::checked_count_add),
                               mp1::count_t (*)(mp1::count_t, mp1::count_t, const mp1::context&,
                                                check_site)>);
  static_assert(std::is_same_v<decltype(&mp1::detail::checked_count_sub),
                               mp1::count_t (*)(mp1::count_t, mp1::count_t, const mp1::context&,
                                                check_site)>);
  static_assert(std::is_same_v<decltype(&mp1::detail::checked_count_mul),
                               mp1::count_t (*)(mp1::count_t, mp1::count_t, const mp1::context&,
                                                check_site)>);
  static_assert(
      std::is_same_v<decltype(&mp1::span_from<int>), std::span<int> (*)(int*, mp1::count_t)>);
  static_assert(std::is_same_v<decltype(&mp1::span_from<const int>),
                               std::span<const int> (*)(const int*, mp1::count_t)>);

  // The checked helpers are callable with their historical argument types (the calls are
  // compiled, never executed).
  [[maybe_unused]] mp1::count_t checked_calls(mp1::count_t n, MPI_Count c) {
    const mp1::count_t from_mpi = mp1::detail::checked_count(c, "probe");
    const int size = mp1::detail::checked_int(static_cast<std::size_t>(n), "probe");
    const MPI_Count back = mp1::detail::checked_mpi_count(n, "probe");
    return from_mpi + size + mp1::detail::checked_count_add(n, 1, "probe") +
           mp1::detail::checked_count_sub(n, 1, "probe") +
           mp1::detail::checked_count_mul(n, 2, "probe") + static_cast<mp1::count_t>(back);
  }

  // ---- the completion functions accept every contiguous request range -------------------
  void completion_spans(std::vector<mp1::request>& vector_requests,
                        std::array<mp1::request, 2>& array_requests,
                        mp1::request (&c_requests)[2]) {
    std::span<mp1::request> span_requests{vector_requests};
    (void)mp1::wait_all(vector_requests);
    (void)mp1::wait_all(array_requests);
    (void)mp1::wait_all(c_requests);
    (void)mp1::wait_all(std::span<mp1::request>{});
    (void)mp1::test_all(span_requests.subspan(0, 1));
    (void)mp1::wait_any(span_requests.first(1));
    (void)mp1::test_any(span_requests.last(1));
    (void)mp1::wait_some(span_requests);
    (void)mp1::test_some(span_requests);
    (void)mp1::get_status_all(span_requests);
    (void)mp1::get_status_any(span_requests);
    (void)mp1::get_status_some(span_requests);
    mp1::start_all(span_requests);
  }

}  // namespace

void api_concepts(std::vector<mp1::request>& requests,
                  std::array<mp1::request, 2>& array_requests, mp1::request (&c_requests)[2]) {
  completion_spans(requests, array_requests, c_requests);
}
