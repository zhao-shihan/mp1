// Positive control for the RMA buffer-form compile-fail probes: every one of the 18
// data-movement declarations and all 15 synchronization declarations compiles with the
// span/region origin-result forms and the count/layout target forms.  The raw pointer+count,
// pointer+layout and container data forms are
// deleted; their removal probes use this file as the control.

#include "mp1/mp1.hpp"

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

int main() {
  const mp1::mpi_window w;
  const mp1::group gr;
  int origin[4] = {1, 2, 3, 4};
  int result[4] = {0, 0, 0, 0};
  int compare = 0;

  const mp1::layout<int> count_layout = mp1::layout<int>::contiguous(4);
  const mp1::layout<int> stride_layout = mp1::layout<int>::vector(2, 1, 2);
  const mp1::layout<int> empty = mp1::layout<int>::contiguous(0);
  const mp1::predefined_op<int> sum{MPI_SUM};

  const std::span<const int> origin_span{origin, 4};
  const std::span<int> result_span{result, 4};
  const mp1::region<int> origin_region{origin, count_layout};
  const mp1::region<int> result_region{result, count_layout};
  const mp1::region<const int> const_origin_region{origin, count_layout};

  // put: span/region origin x count/layout target.
  mp1::put(w, origin_span, 0, 0, 4);
  mp1::put(w, origin_span, 0, 0, count_layout);
  mp1::put(w, origin_region, 0, 0, 4);
  mp1::put(w, origin_region, 0, 0, count_layout);
  mp1::put(w, const_origin_region, 0, 0, count_layout);

  // get: span/region result x count/layout target.
  mp1::get(w, result_span, 0, 0, 4);
  mp1::get(w, result_span, 0, 0, count_layout);
  mp1::get(w, result_region, 0, 0, 4);
  mp1::get(w, result_region, 0, 0, count_layout);

  // accumulate: span/region origin x count/layout target, typed op.
  mp1::accumulate(w, origin_span, 0, 0, 4, sum);
  mp1::accumulate(w, origin_span, 0, 0, count_layout, std::plus<int>{});
  mp1::accumulate(w, origin_region, 0, 0, 4, MPI_SUM);
  mp1::accumulate(w, origin_region, 0, 0, count_layout, mp1::predefined_op<int>{MPI_REPLACE});

  // get_accumulate: the eight origin/result/target shapes.
  mp1::get_accumulate(w, origin_span, result_span, 0, 0, 4, sum);
  mp1::get_accumulate(w, origin_span, result_span, 0, 0, count_layout, sum);
  mp1::get_accumulate(w, origin_span, result_region, 0, 0, 4, sum);
  mp1::get_accumulate(w, origin_span, result_region, 0, 0, count_layout, sum);
  mp1::get_accumulate(w, origin_region, result_span, 0, 0, 4, sum);
  mp1::get_accumulate(w, origin_region, result_span, 0, 0, count_layout, sum);
  mp1::get_accumulate(w, origin_region, result_region, 0, 0, 4, sum);
  mp1::get_accumulate(w, origin_region, result_region, 0, 0, count_layout, sum);

  // Request-based families: the same shapes (four per simple family, eight per fetching).
  mp1::request rput1 = mp1::rput(w, origin_span, 0, 0, 4);
  mp1::request rput2 = mp1::rput(w, origin_span, 0, 0, count_layout);
  mp1::request rput3 = mp1::rput(w, origin_region, 0, 0, 4);
  mp1::request rput4 = mp1::rput(w, origin_region, 0, 0, count_layout);
  mp1::request rget1 = mp1::rget(w, result_span, 0, 0, 4);
  mp1::request rget2 = mp1::rget(w, result_span, 0, 0, count_layout);
  mp1::request rget3 = mp1::rget(w, result_region, 0, 0, 4);
  mp1::request rget4 = mp1::rget(w, result_region, 0, 0, count_layout);
  mp1::request racc1 = mp1::raccumulate(w, origin_span, 0, 0, 4, sum);
  mp1::request racc2 = mp1::raccumulate(w, origin_span, 0, 0, count_layout, sum);
  mp1::request racc3 = mp1::raccumulate(w, origin_region, 0, 0, 4, sum);
  mp1::request racc4 = mp1::raccumulate(w, origin_region, 0, 0, count_layout, sum);
  mp1::request rga1 = mp1::rget_accumulate(w, origin_span, result_span, 0, 0, 4, sum);
  mp1::request rga2 =
      mp1::rget_accumulate(w, origin_span, result_span, 0, 0, count_layout, sum);
  mp1::request rga3 = mp1::rget_accumulate(w, origin_span, result_region, 0, 0, 4, sum);
  mp1::request rga4 =
      mp1::rget_accumulate(w, origin_span, result_region, 0, 0, count_layout, sum);
  mp1::request rga5 = mp1::rget_accumulate(w, origin_region, result_span, 0, 0, 4, sum);
  mp1::request rga6 =
      mp1::rget_accumulate(w, origin_region, result_span, 0, 0, count_layout, sum);
  mp1::request rga7 = mp1::rget_accumulate(w, origin_region, result_region, 0, 0, 4, sum);
  mp1::request rga8 =
      mp1::rget_accumulate(w, origin_region, result_region, 0, 0, count_layout, sum);

  // Single-element atomics keep the pointer spelling.
  mp1::fetch_and_op(w, &origin[0], &result[0], 0, 0, MPI_SUM);
  mp1::fetch_and_op(w, static_cast<const int*>(nullptr), &result[0], 0, 0, MPI_NO_OP);
  mp1::compare_and_swap(w, &origin[0], &compare, &result[0], 0, 0);

  // The empty layout is a valid zero-element target.
  mp1::put(w, std::span<const int>{}, 0, 0, empty);

  // The fifteen synchronization calls and the lock_type spelling.
  mp1::fence(w);
  mp1::lock(w, mp1::lock_type::exclusive, 0);
  mp1::unlock(w, 0);
  mp1::lock_all(w);
  mp1::unlock_all(w);
  mp1::flush(w, 0);
  mp1::flush_all(w);
  mp1::flush_local(w, 0);
  mp1::flush_local_all(w);
  mp1::sync(w);
  mp1::post(w, gr);
  mp1::start(w, gr);
  mp1::complete(w);
  mp1::wait(w);
  const bool done = mp1::test(w);

  // Silence unused-variable warnings for the request handles and the sync result.
  (void)rput1;
  (void)rput2;
  (void)rput3;
  (void)rput4;
  (void)rget1;
  (void)rget2;
  (void)rget3;
  (void)rget4;
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
  (void)done;
  (void)stride_layout;
  (void)empty;
  return 0;
}
