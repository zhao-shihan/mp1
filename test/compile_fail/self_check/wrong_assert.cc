// Self-check fixture: a typo must not be accepted as the intended diagnostic.
#include "mp1/mp1.hpp"

int main() {
  return static_cast<int>(mp1::does_not_exist());
}
