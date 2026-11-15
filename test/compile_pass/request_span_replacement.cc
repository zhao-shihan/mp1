// Positive control for compile_fail/request_span_removed.cc: std::span<request> is the
// completion borrow, and std::vector/std::array/C arrays convert implicitly.
#include "mp1/mp1.hpp"

#include <array>
#include <span>
#include <vector>

void completion_span(std::vector<mp1::request>& requests,
                     std::array<mp1::request, 2>& array_requests) {
  std::span<mp1::request> slice = std::span{requests}.subspan(0, requests.size());
  (void)mp1::wait_all(slice);
  (void)mp1::wait_all(requests);
  (void)mp1::test_all(array_requests);
  (void)mp1::wait_any(std::span<mp1::request>{});
}
