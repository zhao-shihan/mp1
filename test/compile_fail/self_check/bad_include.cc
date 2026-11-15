// Self-check fixture: a missing header is a fatal error and is not evidence
// about any library contract, so the runner must reject it.
#include "mp1/does_not_exist.hpp"

int main() {
  return 0;
}
