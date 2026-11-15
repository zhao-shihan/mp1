// Positive control for the typed (layout-span) variable collectives: named layout and
// displacement arrays are borrowed for the call and copied at initiation when the request
// outlives them.
#include "mp1/mp1.hpp"

int main() {
  int data[4] = {0, 0, 0, 0};
  const std::span<int> buffer{data, 4};
  std::vector<mp1::layout<int>> layout_owner(2, mp1::layout<int>::contiguous(2));
  const std::span<const mp1::layout<int>> layouts{layout_owner};
  const std::vector<mp1::count_t> displs(2, 0);
  mp1::request r = mp1::iallgatherv(mp1::comm_world(), buffer, buffer, layouts, displs);
  (void)r;
  return 0;
}
