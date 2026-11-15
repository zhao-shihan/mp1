// Compile-fail: `mp1::block_count` is not part of the API.  A non-root scatter receive block
// is the receive span's size.
#include "mp1/mp1.hpp"

int main() {
  mp1::block_count count{1};
  (void)count;
  return 0;
}
