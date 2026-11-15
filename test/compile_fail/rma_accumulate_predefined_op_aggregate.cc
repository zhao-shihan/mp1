// Negative probe: predefined_op<T> requires a predefined operand type T.  For a homogeneous
// derived aggregate use predefined_op<basic_component_t<T>> (for std::pair<int,int> that is
// predefined_op<int>); the aggregate spelling is a clear compile-time error.

#include "mp1/mp1.hpp"

#include <utility>

int main() {
  mp1::predefined_op<std::pair<int, int>> op{MPI_SUM};
  (void)op;
  return 0;
}
