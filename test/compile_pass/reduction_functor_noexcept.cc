// Positive control for test/compile_fail/throwing_reduction_functor.cc: the same reduction
// with a noexcept functor is accepted.

#include "mp1/mp1.hpp"

struct nothrow_add {
  int operator()(int left, int right) const noexcept {
    return left + right;
  }
};

int main() {
  int value = 0;
  mp1::allreduce(mp1::comm_world(), nothrow_add{}, 1, value);
  return value;
}
