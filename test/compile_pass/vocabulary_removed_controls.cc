// Positive control for the spelling-removal probes: the target dense/typed spellings of
// the removed transitional vocabulary compile.
#include "mp1/impl/vocabulary.hpp"

#include <span>
#include <vector>

void control(const std::vector<mp1::count_t>& counts, const std::vector<mp1::count_t>& displs) {
  const mp1::blocks dense{counts, displs};
  const std::span<const mp1::count_t> typed_displs{displs};
  (void)dense;
  (void)typed_displs;
}
