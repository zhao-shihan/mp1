// Positive control for the homogeneous-derived RMA accumulating surface: a predefined T and a
// homogeneous derived T both take predefined_op<basic_component_t<T>>, every accepted operator
// spelling compiles for the four accumulating families, and layout<int> + MPI_REPLACE stays a
// source-compatible anchor (MPI 4.1 §12.3.4; MPI 4.1 REPLACE policy).

#include "mp1/mp1.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <tuple>
#include <utility>

enum class accumulate_enum : int { zero = 0, one = 1 };

struct reflected_pair {
  int a;
  int b;
};

MP1_REFLECTION(reflected_pair, a, b)

struct reflected_nested {
  reflected_pair inner;
  int tail;
};

MP1_REFLECTION(reflected_nested, inner, tail)

struct reflected_array_member {
  int values[3];
};

MP1_REFLECTION(reflected_array_member, values)

static_assert(mp1::is_homogeneous_datatype_v<int>);
static_assert(mp1::is_homogeneous_datatype_v<std::pair<int, int>>);
static_assert(mp1::is_homogeneous_datatype_v<std::tuple<int, int>>);
static_assert(mp1::is_homogeneous_datatype_v<std::array<int, 2>>);
static_assert(mp1::is_homogeneous_datatype_v<std::pair<std::pair<int, int>, int>>);
static_assert(mp1::is_homogeneous_datatype_v<reflected_pair>);
static_assert(mp1::is_homogeneous_datatype_v<reflected_nested>);
static_assert(mp1::is_homogeneous_datatype_v<reflected_array_member>);
static_assert(mp1::is_homogeneous_datatype_v<std::pair<accumulate_enum, int>>);
static_assert(mp1::is_homogeneous_datatype_v<std::pair<accumulate_enum, accumulate_enum>>);
static_assert(mp1::is_homogeneous_datatype_v<std::pair<char16_t, std::uint16_t>>);
static_assert(std::is_same_v<mp1::basic_component_t<std::pair<int, int>>, int>);
static_assert(std::is_same_v<mp1::basic_component_t<std::pair<accumulate_enum, int>>, int>);
static_assert(
    std::is_same_v<mp1::basic_component_t<std::pair<accumulate_enum, accumulate_enum>>,
                   accumulate_enum>);
static_assert(std::is_same_v<mp1::basic_component_t<std::pair<char16_t, std::uint16_t>>,
                             std::uint_least16_t>);
static_assert(
    std::is_same_v<mp1::detail::accumulate_op_t<std::pair<int, int>>, mp1::predefined_op<int>>);
static_assert(std::is_same_v<mp1::detail::accumulate_op_t<int>, mp1::predefined_op<int>>);

int main() {
  const mp1::mpi_window w;
  using pair_type = std::pair<int, int>;
  pair_type pair_origin[2] = {};
  pair_type pair_result[2] = {};
  const mp1::layout<pair_type> pair_layout = mp1::layout<pair_type>::contiguous(2);

  // accumulate: four shapes, four accepted operator spellings.
  mp1::accumulate(w, mp1::span_from(pair_origin, 2), 0, 0, 2, MPI_SUM);
  mp1::accumulate(w, mp1::span_from(pair_origin, 2), 0, 0, pair_layout,
                  mp1::predefined_op<int>{MPI_SUM});
  mp1::accumulate(w, mp1::region{pair_origin, pair_layout}, 0, 0, 2, std::plus<int>{});
  mp1::accumulate(w, mp1::region{pair_origin, pair_layout}, 0, 0, pair_layout, std::plus<>{});

  // get_accumulate: eight origin/result/target shapes; MPI_NO_OP is legal here.
  mp1::get_accumulate(w, mp1::span_from(pair_origin, 2), mp1::span_from(pair_result, 2), 0, 0,
                      2, MPI_SUM);
  mp1::get_accumulate(w, mp1::span_from(pair_origin, 2), mp1::span_from(pair_result, 2), 0, 0,
                      pair_layout, MPI_SUM);
  mp1::get_accumulate(w, mp1::span_from(pair_origin, 2), mp1::region{pair_result, pair_layout},
                      0, 0, 2, MPI_SUM);
  mp1::get_accumulate(w, mp1::span_from(pair_origin, 2), mp1::region{pair_result, pair_layout},
                      0, 0, pair_layout, MPI_SUM);
  mp1::get_accumulate(w, mp1::region{pair_origin, pair_layout}, mp1::span_from(pair_result, 2),
                      0, 0, 2, MPI_SUM);
  mp1::get_accumulate(w, mp1::region{pair_origin, pair_layout}, mp1::span_from(pair_result, 2),
                      0, 0, pair_layout, MPI_SUM);
  mp1::get_accumulate(w, mp1::region{pair_origin, pair_layout},
                      mp1::region{pair_result, pair_layout}, 0, 0, 2,
                      mp1::predefined_op<int>{MPI_NO_OP});
  mp1::get_accumulate(w, mp1::region{pair_origin, pair_layout},
                      mp1::region{pair_result, pair_layout}, 0, 0, pair_layout,
                      mp1::predefined_op<int>{MPI_NO_OP});

  // raccumulate: four shapes.
  mp1::request racc1 = mp1::raccumulate(w, mp1::span_from(pair_origin, 2), 0, 0, 2, MPI_SUM);
  mp1::request racc2 =
      mp1::raccumulate(w, mp1::span_from(pair_origin, 2), 0, 0, pair_layout, MPI_SUM);
  mp1::request racc3 =
      mp1::raccumulate(w, mp1::region{pair_origin, pair_layout}, 0, 0, 2, MPI_SUM);
  mp1::request racc4 =
      mp1::raccumulate(w, mp1::region{pair_origin, pair_layout}, 0, 0, pair_layout, MPI_SUM);

  // rget_accumulate: eight shapes; MPI_NO_OP is legal here too.
  mp1::request rga1 = mp1::rget_accumulate(w, mp1::span_from(pair_origin, 2),
                                           mp1::span_from(pair_result, 2), 0, 0, 2, MPI_SUM);
  mp1::request rga2 =
      mp1::rget_accumulate(w, mp1::span_from(pair_origin, 2), mp1::span_from(pair_result, 2), 0,
                           0, pair_layout, MPI_SUM);
  mp1::request rga3 =
      mp1::rget_accumulate(w, mp1::span_from(pair_origin, 2),
                           mp1::region{pair_result, pair_layout}, 0, 0, 2, MPI_SUM);
  mp1::request rga4 =
      mp1::rget_accumulate(w, mp1::span_from(pair_origin, 2),
                           mp1::region{pair_result, pair_layout}, 0, 0, pair_layout, MPI_SUM);
  mp1::request rga5 = mp1::rget_accumulate(w, mp1::region{pair_origin, pair_layout},
                                           mp1::span_from(pair_result, 2), 0, 0, 2, MPI_SUM);
  mp1::request rga6 =
      mp1::rget_accumulate(w, mp1::region{pair_origin, pair_layout},
                           mp1::span_from(pair_result, 2), 0, 0, pair_layout, MPI_SUM);
  mp1::request rga7 = mp1::rget_accumulate(w, mp1::region{pair_origin, pair_layout},
                                           mp1::region{pair_result, pair_layout}, 0, 0, 2,
                                           mp1::predefined_op<int>{MPI_NO_OP});
  mp1::request rga8 = mp1::rget_accumulate(w, mp1::region{pair_origin, pair_layout},
                                           mp1::region{pair_result, pair_layout}, 0, 0,
                                           pair_layout, mp1::predefined_op<int>{MPI_NO_OP});

  // REPLACE with a predefined and a homogeneous-derived datatype (MPI 4.1 REPLACE policy).
  int int_origin[2] = {};
  const mp1::layout<int> int_layout = mp1::layout<int>::contiguous(2);
  mp1::accumulate(w, mp1::span_from(int_origin, 2), 0, 0, int_layout,
                  mp1::predefined_op<int>{MPI_REPLACE});
  mp1::accumulate(w, mp1::span_from(pair_origin, 2), 0, 0, pair_layout,
                  mp1::predefined_op<int>{MPI_REPLACE});

  // The remaining homogeneous derived shapes.
  std::tuple<int, int> tuple_origin[1] = {};
  std::array<int, 2> array_origin[1] = {};
  reflected_pair reflected_origin[1] = {};
  reflected_nested nested_origin[1] = {};
  reflected_array_member array_member_origin[1] = {};
  std::pair<accumulate_enum, int> enum_int_origin[1] = {};
  std::pair<accumulate_enum, accumulate_enum> enum_origin[1] = {};
  std::pair<char16_t, std::uint16_t> utf16_origin[1] = {};
  mp1::accumulate(w, mp1::span_from(tuple_origin, 1), 0, 0, 1, MPI_SUM);
  mp1::accumulate(w, mp1::span_from(array_origin, 1), 0, 0, 1, MPI_SUM);
  mp1::accumulate(w, mp1::span_from(reflected_origin, 1), 0, 0, 1, MPI_SUM);
  mp1::accumulate(w, mp1::span_from(nested_origin, 1), 0, 0, 1, MPI_SUM);
  mp1::accumulate(w, mp1::span_from(array_member_origin, 1), 0, 0, 1, MPI_SUM);
  mp1::accumulate(w, mp1::span_from(enum_int_origin, 1), 0, 0, 1,
                  mp1::predefined_op<int>{MPI_SUM});
  mp1::accumulate(w, mp1::span_from(enum_origin, 1), 0, 0, 1,
                  mp1::predefined_op<accumulate_enum>{MPI_SUM});
  mp1::accumulate(w, mp1::span_from(utf16_origin, 1), 0, 0, 1,
                  mp1::predefined_op<std::uint_least16_t>{MPI_SUM});

  (void)racc1;
  (void)racc2;
  (void)racc3;
  (void)racc4;
  (void)rga1;
  (void)rga2;
  (void)rga3;
  (void)rga4;
  (void)rga5;
  (void)rga6;
  (void)rga7;
  (void)rga8;
  return 0;
}
