// Positive control for the custom homogeneous-derived opt-in: MP1_DATATYPE_BASIC(T, U) makes a
// non-predefined custom trait homogeneous over U, so the accumulating families take
// predefined_op<U>.  The committed datatype is validated once at run time before the first
// accumulating call (this control is compile-only).

#include "mp1/mp1.hpp"

#include <functional>
#include <span>

MPI_Datatype make_custom_basic_pair_datatype();

struct custom_basic_pair {
  int a;
  int b;
};

MP1_DATATYPE_TRAITS_PACKED(custom_basic_pair, make_custom_basic_pair_datatype())
MP1_DATATYPE_BASIC(custom_basic_pair, int)

static_assert(mp1::is_homogeneous_datatype_v<custom_basic_pair>);
static_assert(std::is_same_v<mp1::basic_component_t<custom_basic_pair>, int>);
static_assert(mp1::detail::component_info<custom_basic_pair>::runtime_validate);
static_assert(
    std::is_same_v<mp1::detail::accumulate_op_t<custom_basic_pair>, mp1::predefined_op<int>>);

int main() {
  const mp1::mpi_window w;
  custom_basic_pair origin[1] = {};
  const mp1::layout<custom_basic_pair> layout = mp1::layout<custom_basic_pair>::contiguous(1);
  mp1::accumulate(w, mp1::span_from(origin, 1), 0, 0, 1, MPI_SUM);
  mp1::accumulate(w, mp1::span_from(origin, 1), 0, 0, layout, mp1::predefined_op<int>{MPI_SUM});
  mp1::accumulate(w, mp1::span_from(origin, 1), 0, 0, 1, std::plus<int>{});
  mp1::get_accumulate(w, mp1::span_from(origin, 1), mp1::span_from(origin, 1), 0, 0, 1,
                      mp1::predefined_op<int>{MPI_REPLACE});
  mp1::request racc = mp1::raccumulate(w, mp1::span_from(origin, 1), 0, 0, 1, MPI_SUM);
  mp1::request rga = mp1::rget_accumulate(w, mp1::span_from(origin, 1),
                                          mp1::span_from(origin, 1), 0, 0, 1, MPI_SUM);
  (void)racc;
  (void)rga;
  return 0;
}
