// An mp1-managed reduction callback must not throw.  A functor whose operator() is
// not noexcept is rejected at compile time; an exception cannot unwind through the MPI C
// callback (MPI 4.1 §6.9.5).
//
// Expected diagnostic: mp1: a reduction functor used on the callback path must be
// nothrow-invocable for the element type

#include "mp1/mp1.hpp"

struct throwing_add {
  int operator()(int left, int right) const {
    return left + right;
  }
};

int main() {
  int value = 0;
  mp1::allreduce(mp1::comm_world(), throwing_add{}, 1, value);
  return value;
}
