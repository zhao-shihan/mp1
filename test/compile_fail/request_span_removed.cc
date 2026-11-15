// Negative probe: request_span was the hand-rolled MPI-array borrow of the completion
// functions.  The batch functions take std::span<request>, so the type
// must not exist; the positive control uses std::span and the implicit container
// conversions.
#include "mp1/mp1.hpp"

#include <vector>

void use_request_span(std::vector<mp1::request>& requests) {
  mp1::request_span span{requests};
  (void)span;
}
