// Public-API compile check: windows and one-sided communication.
//
// Compile-only.  MPI 4.1 Ch. 12 defines the window object, the RMA
// communication calls, and the synchronization calls bound here.  The target
// datatype overloads use `layout<T>`; `layout<void>` (absolute addresses) is
// removed and is intentionally absent.

#include "mp1/mp1.hpp"

#include <cstddef>
#include <functional>
#include <span>
#include <utility>

void api_windows(const mp1::mpi_communicator& c) {
  int base = 0;
  const mp1::info table;

  mp1::window created = mp1::window::create(
      c, {.base = &base, .size = sizeof(int), .disp_unit = sizeof(int), .info = table});
  mp1::window allocated =
      mp1::window::allocate(c, {.size = sizeof(int), .disp_unit = sizeof(int), .info = table});
  mp1::window shared = mp1::window::allocate_shared(
      c, {.size = sizeof(int), .disp_unit = sizeof(int), .info = table});
  mp1::window dynamic = mp1::window::create_dynamic(c, table);
  mp1::window adopted = mp1::window::adopt(MPI_WIN_NULL);
  (void)adopted;

  bool valid = created.is_valid();
  const MPI_Win native = created.native_handle();
  void* window_base = created.base();
  const MPI_Aint bytes = created.byte_size();
  const int disp_unit = created.disp_unit();
  const int flavor = created.create_flavor();
  const int model = created.memory_model();
  const int rank = created.rank();
  const int group_size = created.group_size();
  const mp1::group group = created.get_group();
  const mp1::info info_table = created.get_info();
  created.set_info(table);
  const mp1::shared_segment segment = shared.shared_query(0);
  (void)valid;
  (void)native;
  (void)window_base;
  (void)bytes;
  (void)disp_unit;
  (void)flavor;
  (void)model;
  (void)rank;
  (void)group_size;
  (void)group;
  (void)info_table;
  (void)segment;

  dynamic.attach(&base, sizeof(int));
  dynamic.detach(&base);
  created.free();
  allocated.free();
  shared.free();
  dynamic.free();
}

void api_rma(const mp1::mpi_window& w, int rank) {
  int origin = 0;
  int result = 0;
  int compare = 0;
  int target[4] = {0, 0, 0, 0};
  mp1::layout<int> layout = mp1::layout<int>::contiguous(4);
  std::plus<int> plus;
  const MPI_Aint displacement = 0;

  mp1::put(w, mp1::span_from(&origin, 1), rank, displacement, 1);
  mp1::put(w, mp1::span_from(&origin, layout.size()), rank, displacement, layout);
  mp1::put(w, mp1::region{&origin, layout}, rank, displacement, 1);
  mp1::put(w, mp1::region{&origin, layout}, rank, displacement, layout);
  mp1::get(w, mp1::span_from(&origin, 1), rank, displacement, 1);
  mp1::get(w, mp1::span_from(&origin, layout.size()), rank, displacement, layout);
  mp1::get(w, mp1::region{&origin, layout}, rank, displacement, 1);
  mp1::get(w, mp1::region{&origin, layout}, rank, displacement, layout);
  mp1::accumulate(w, mp1::span_from(&origin, 1), rank, displacement, 1, plus);
  mp1::accumulate(w, mp1::span_from(&origin, layout.size()), rank, displacement, layout, plus);
  mp1::accumulate(w, mp1::region{&origin, layout}, rank, displacement, 1, plus);
  mp1::accumulate(w, mp1::region{&origin, layout}, rank, displacement, layout, plus);
  mp1::get_accumulate(w, mp1::span_from(&origin, 1), mp1::span_from(&result, 1), rank,
                      displacement, 1, plus);
  mp1::get_accumulate(w, mp1::span_from(&origin, layout.size()),
                      mp1::span_from(&result, layout.size()), rank, displacement, layout, plus);
  mp1::compare_and_swap(w, &origin, &compare, &result, rank, displacement);
  mp1::fetch_and_op(w, &origin, &result, rank, displacement, plus);

  // All eight mixed origin/result/target shape combinations of get_accumulate.
  int result_data[4] = {0, 0, 0, 0};
  mp1::layout<int> origin_layout = mp1::layout<int>::contiguous(4);
  mp1::layout<int> result_layout = mp1::layout<int>::contiguous(4);
  mp1::get_accumulate(w, mp1::span_from(&origin, 4), mp1::span_from(result_data, 4), rank,
                      displacement, 4, plus);
  mp1::get_accumulate(w, mp1::span_from(&origin, layout.size()), std::span{result_data}, rank,
                      displacement, layout, plus);
  mp1::get_accumulate(w, mp1::span_from(&origin, 4), mp1::region{result_data, result_layout},
                      rank, displacement, 4, plus);
  mp1::get_accumulate(w, mp1::span_from(&origin, layout.size()),
                      mp1::region{result_data, result_layout}, rank, displacement, layout,
                      plus);
  mp1::get_accumulate(w, mp1::region{&origin, origin_layout}, mp1::span_from(result_data, 4),
                      rank, displacement, 4, plus);
  mp1::get_accumulate(w, mp1::region{&origin, origin_layout}, std::span{result_data}, rank,
                      displacement, layout, plus);
  mp1::get_accumulate(w, mp1::region{&origin, origin_layout},
                      mp1::region{result_data, result_layout}, rank, displacement, 4, plus);
  mp1::get_accumulate(w, mp1::region{&origin, origin_layout},
                      mp1::region{result_data, result_layout}, rank, displacement, layout,
                      plus);

  mp1::request rput = mp1::rput(w, mp1::span_from(&origin, 1), rank, displacement, 1);
  mp1::request rget = mp1::rget(w, mp1::span_from(&origin, 1), rank, displacement, 1);
  mp1::request raccumulate =
      mp1::raccumulate(w, mp1::span_from(&origin, 1), rank, displacement, 1, plus);
  mp1::request rget_accumulate = mp1::rget_accumulate(
      w, mp1::span_from(&origin, 1), mp1::span_from(&result, 1), rank, displacement, 1, plus);
  // The same eight mixed shapes for rget_accumulate.
  mp1::request rg1 =
      mp1::rget_accumulate(w, mp1::span_from(&origin, 4), mp1::span_from(result_data, 4), rank,
                           displacement, 4, plus);
  mp1::request rg2 =
      mp1::rget_accumulate(w, mp1::span_from(&origin, layout.size()), std::span{result_data},
                           rank, displacement, layout, plus);
  mp1::request rg3 = mp1::rget_accumulate(w, mp1::span_from(&origin, 4),
                                          mp1::region{result_data, result_layout}, rank,
                                          displacement, 4, plus);
  mp1::request rg4 = mp1::rget_accumulate(w, mp1::span_from(&origin, layout.size()),
                                          mp1::region{result_data, result_layout}, rank,
                                          displacement, layout, plus);
  mp1::request rg5 =
      mp1::rget_accumulate(w, mp1::region{&origin, origin_layout},
                           mp1::span_from(result_data, 4), rank, displacement, 4, plus);
  mp1::request rg6 =
      mp1::rget_accumulate(w, mp1::region{&origin, origin_layout}, std::span{result_data}, rank,
                           displacement, layout, plus);
  mp1::request rg7 = mp1::rget_accumulate(w, mp1::region{&origin, origin_layout},
                                          mp1::region{result_data, result_layout}, rank,
                                          displacement, 4, plus);
  mp1::request rg8 = mp1::rget_accumulate(w, mp1::region{&origin, origin_layout},
                                          mp1::region{result_data, result_layout}, rank,
                                          displacement, layout, plus);
  (void)rput;
  (void)rget;
  (void)raccumulate;
  (void)rget_accumulate;
  (void)rg1;
  (void)rg2;
  (void)rg3;
  (void)rg4;
  (void)rg5;
  (void)rg6;
  (void)rg7;
  (void)rg8;

  mp1::fence(w);
  mp1::lock(w, mp1::lock_type::exclusive, rank);
  mp1::unlock(w, rank);
  mp1::lock_all(w);
  mp1::unlock_all(w);
  mp1::flush(w, rank);
  mp1::flush_all(w);
  mp1::flush_local(w, rank);
  mp1::flush_local_all(w);
  mp1::sync(w);
  mp1::post(w, mp1::get_group(w));
  mp1::start(w, mp1::get_group(w));
  mp1::complete(w);
  mp1::wait(w);
  const bool done = mp1::test(w);
  (void)done;

  // All 24 accumulating shapes with a homogeneous derived element type: the operator type is
  // predefined_op<basic_component_t<std::pair<int,int>>> == predefined_op<int>
  // (MPI 4.1 §12.3.4).  One line per overload, so the 24/24 surface is compile-checked
  // even where the runtime test covers only representative shapes.
  std::pair<int, int> pair_origin[1] = {};
  std::pair<int, int> pair_result[1] = {};
  mp1::layout<std::pair<int, int>> pair_layout =
      mp1::layout<std::pair<int, int>>::contiguous(1);
  mp1::predefined_op<int> pair_op{MPI_SUM};

  mp1::accumulate(w, mp1::span_from(pair_origin, 1), rank, displacement, 1, pair_op);
  mp1::accumulate(w, std::span{pair_origin}, rank, displacement, pair_layout, pair_op);
  mp1::accumulate(w, mp1::region{pair_origin, pair_layout}, rank, displacement, 1, pair_op);
  mp1::accumulate(w, mp1::region{pair_origin, pair_layout}, rank, displacement, pair_layout,
                  pair_op);

  mp1::get_accumulate(w, mp1::span_from(pair_origin, 1), mp1::span_from(pair_result, 1), rank,
                      displacement, 1, pair_op);
  mp1::get_accumulate(w, std::span{pair_origin}, std::span{pair_result}, rank, displacement,
                      pair_layout, pair_op);
  mp1::get_accumulate(w, mp1::span_from(pair_origin, 1), mp1::region{pair_result, pair_layout},
                      rank, displacement, 1, pair_op);
  mp1::get_accumulate(w, std::span{pair_origin}, mp1::region{pair_result, pair_layout}, rank,
                      displacement, pair_layout, pair_op);
  mp1::get_accumulate(w, mp1::region{pair_origin, pair_layout}, mp1::span_from(pair_result, 1),
                      rank, displacement, 1, pair_op);
  mp1::get_accumulate(w, mp1::region{pair_origin, pair_layout}, std::span{pair_result}, rank,
                      displacement, pair_layout, pair_op);
  mp1::get_accumulate(w, mp1::region{pair_origin, pair_layout},
                      mp1::region{pair_result, pair_layout}, rank, displacement, 1, pair_op);
  mp1::get_accumulate(w, mp1::region{pair_origin, pair_layout},
                      mp1::region{pair_result, pair_layout}, rank, displacement, pair_layout,
                      pair_op);

  mp1::request pair_racc1 =
      mp1::raccumulate(w, mp1::span_from(pair_origin, 1), rank, displacement, 1, pair_op);
  mp1::request pair_racc2 =
      mp1::raccumulate(w, std::span{pair_origin}, rank, displacement, pair_layout, pair_op);
  mp1::request pair_racc3 = mp1::raccumulate(w, mp1::region{pair_origin, pair_layout}, rank,
                                             displacement, 1, pair_op);
  mp1::request pair_racc4 = mp1::raccumulate(w, mp1::region{pair_origin, pair_layout}, rank,
                                             displacement, pair_layout, pair_op);

  mp1::request pair_rga1 =
      mp1::rget_accumulate(w, mp1::span_from(pair_origin, 1), mp1::span_from(pair_result, 1),
                           rank, displacement, 1, pair_op);
  mp1::request pair_rga2 =
      mp1::rget_accumulate(w, std::span{pair_origin}, std::span{pair_result}, rank,
                           displacement, pair_layout, pair_op);
  mp1::request pair_rga3 = mp1::rget_accumulate(w, mp1::span_from(pair_origin, 1),
                                                mp1::region{pair_result, pair_layout}, rank,
                                                displacement, 1, pair_op);
  mp1::request pair_rga4 =
      mp1::rget_accumulate(w, std::span{pair_origin}, mp1::region{pair_result, pair_layout},
                           rank, displacement, pair_layout, pair_op);
  mp1::request pair_rga5 =
      mp1::rget_accumulate(w, mp1::region{pair_origin, pair_layout},
                           mp1::span_from(pair_result, 1), rank, displacement, 1, pair_op);
  mp1::request pair_rga6 =
      mp1::rget_accumulate(w, mp1::region{pair_origin, pair_layout}, std::span{pair_result},
                           rank, displacement, pair_layout, pair_op);
  mp1::request pair_rga7 = mp1::rget_accumulate(w, mp1::region{pair_origin, pair_layout},
                                                mp1::region{pair_result, pair_layout}, rank,
                                                displacement, 1, pair_op);
  mp1::request pair_rga8 = mp1::rget_accumulate(w, mp1::region{pair_origin, pair_layout},
                                                mp1::region{pair_result, pair_layout}, rank,
                                                displacement, pair_layout, pair_op);
  (void)pair_racc1;
  (void)pair_racc2;
  (void)pair_racc3;
  (void)pair_racc4;
  (void)pair_rga1;
  (void)pair_rga2;
  (void)pair_rga3;
  (void)pair_rga4;
  (void)pair_rga5;
  (void)pair_rga6;
  (void)pair_rga7;
  (void)pair_rga8;

  const MPI_Aint address = mp1::address_of(origin);
  (void)address;
  (void)target;
  (void)layout;
}

int main() {
  return 0;
}
