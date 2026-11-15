// Public-API compile check: the fixed-collective surface (barrier, bcast, gather, scatter,
// allgather, alltoall, reduce, allreduce, scan, exscan, reduce_scatter, reduce_scatter_block,
// reduce_local) and the variable-collective surface (gatherv, scatterv, allgatherv,
// alltoallv: dense `blocks` and typed layout-span descriptors x blocking/nonblocking/
// persistent).  Compile-only; arguments are valid types but are never executed.
//
// The fixed families are pinned on their target forms: value, std::span, std::array/C array,
// region, non-contiguous sequence send, the in-place sentinel on the MPI-legal side, and the
// blocking/nonblocking/persistent lifetimes (the persistent forms take fixed buffers only).
// The variable families pin both descriptor kinds, the value/span/region single-block forms,
// the documented non-root dummies and the in-place side; a non-contiguous sequence is not a
// variable-family form.

#include "mp1/mp1.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <list>
#include <span>
#include <utility>
#include <vector>

void api_fixed_collectives(const mp1::mpi_communicator& c, int root) {
  int scalar = 0;
  std::array<int, 4> array{0, 0, 0, 0};
  std::array<int, 4> recv_array{0, 0, 0, 0};
  std::vector<int> container(4, 0);
  std::vector<int> recv_container(8, 0);
  std::list<int> sequence{0, 0, 0, 0};
  mp1::layout<int> layout = mp1::layout<int>::contiguous(4);
  mp1::region<int> region_container{container.data(), layout};
  mp1::region<int> region_recv{recv_container.data(), layout};
  std::span<int> span_array{array};
  std::span<int> span_recv{recv_container};

  mp1::barrier(c);
  mp1::bcast(c, root, scalar);
  mp1::bcast(c, root, array);
  mp1::bcast(c, root, container);
  mp1::bcast(c, root, region_container);

  mp1::gather(c, root, scalar, recv_container);
  mp1::gather(c, root, array, recv_container);
  mp1::gather(c, root, container, recv_container);
  mp1::gather(c, root, sequence, recv_container);
  mp1::gather(c, root, region_container, region_recv);
  mp1::gather(c, root, mp1::in_place, recv_container);
  mp1::gather(c, root, scalar, std::span<int>{});

  mp1::scatter(c, root, recv_container, scalar);
  mp1::scatter(c, root, recv_container, array);
  mp1::scatter(c, root, recv_container, container);
  mp1::scatter(c, root, sequence, container);
  mp1::scatter(c, root, region_recv, region_container);
  mp1::scatter(c, root, recv_container, mp1::in_place);
  mp1::scatter(c, root, std::span<const int>{}, scalar);

  mp1::allgather(c, scalar, recv_container);
  mp1::allgather(c, container, recv_container);
  mp1::allgather(c, sequence, recv_container);
  mp1::allgather(c, region_container, region_recv);
  mp1::allgather(c, mp1::in_place, recv_container);

  mp1::alltoall(c, container, recv_container);
  mp1::alltoall(c, sequence, recv_container);
  mp1::alltoall(c, region_container, region_recv);
  mp1::alltoall(c, mp1::in_place, recv_container);

  mp1::reduce(c, std::plus<int>{}, root, scalar, scalar);
  mp1::reduce(c, std::plus<int>{}, root, array, recv_array);
  mp1::reduce(c, std::plus<int>{}, root, container, recv_container);
  mp1::reduce(c, std::plus<int>{}, root, region_container, region_recv);
  mp1::reduce(c, std::plus<int>{}, root, mp1::in_place, recv_container);
  mp1::reduce(c, std::plus<int>{}, root, scalar, std::span<int>{});

  mp1::allreduce(c, std::plus<int>{}, scalar, scalar);
  mp1::allreduce(c, std::plus<int>{}, array, recv_array);
  mp1::allreduce(c, std::plus<int>{}, container, recv_container);
  mp1::allreduce(c, std::plus<int>{}, region_container, region_recv);
  mp1::allreduce(c, std::plus<int>{}, mp1::in_place, recv_container);

  mp1::scan(c, std::plus<int>{}, container, recv_container);
  mp1::scan(c, std::plus<int>{}, mp1::in_place, recv_container);
  mp1::exscan(c, std::plus<int>{}, container, recv_container);
  mp1::exscan(c, std::plus<int>{}, mp1::in_place, recv_container);

  const std::array<mp1::count_t, 2> counts{2, 2};
  const std::span<const mp1::count_t> recvcounts{counts};
  mp1::reduce_scatter(c, std::plus<int>{}, container, recv_array, recvcounts);
  mp1::reduce_scatter(c, std::plus<int>{}, region_container, region_recv, recvcounts);
  mp1::reduce_scatter(c, std::plus<int>{}, mp1::in_place, recv_container, recvcounts);
  mp1::reduce_scatter(c, std::plus<int>{}, container, recv_array,
                      std::span<const mp1::count_t>{});
  mp1::reduce_scatter_block(c, std::plus<int>{}, container, recv_array, 2);
  mp1::reduce_scatter_block(c, std::plus<int>{}, region_container, region_recv, 2);
  mp1::reduce_scatter_block(c, std::plus<int>{}, mp1::in_place, recv_container, 2);

  mp1::reduce_local(std::plus<int>{}, scalar, scalar);
  mp1::reduce_local(std::plus<int>{}, array, recv_array);
  mp1::reduce_local(std::plus<int>{}, container, recv_container);
  mp1::reduce_local(std::plus<int>{}, sequence, recv_container);
  mp1::reduce_local(std::plus<int>{}, region_container, region_recv);
}

void api_nonblocking_collectives(const mp1::mpi_communicator& c, int root) {
  int scalar = 0;
  std::array<int, 4> array{0, 0, 0, 0};
  std::array<int, 4> recv_array{0, 0, 0, 0};
  std::vector<int> container(4, 0);
  std::vector<int> recv_container(8, 0);
  std::list<int> sequence{0, 0, 0, 0};
  mp1::layout<int> layout = mp1::layout<int>::contiguous(4);
  mp1::region<int> region_container{container.data(), layout};
  mp1::region<int> region_recv{recv_container.data(), layout};
  const std::array<mp1::count_t, 2> counts{2, 2};
  const std::span<const mp1::count_t> recvcounts{counts};
  std::plus<int> plus;

  mp1::request r1 = mp1::ibarrier(c);
  mp1::request r2 = mp1::ibcast(c, root, container);
  mp1::request r2b = mp1::ibcast(c, root, region_container);
  mp1::request r3 = mp1::igather(c, root, container, recv_container);
  mp1::request r3b = mp1::igather(c, root, mp1::in_place, recv_container);
  mp1::request r4 = mp1::iscatter(c, root, recv_container, container);
  mp1::request r4b = mp1::iscatter(c, root, recv_container, mp1::in_place);
  mp1::request r5 = mp1::iallgather(c, container, recv_container);
  mp1::request r5b = mp1::iallgather(c, region_container, region_recv);
  mp1::request r6 = mp1::ialltoall(c, container, recv_container);
  mp1::request r7 = mp1::ireduce(c, plus, root, container, recv_container);
  mp1::request r8 = mp1::iallreduce(c, plus, container, recv_container);
  mp1::request r9 = mp1::iscan(c, plus, container, recv_container);
  mp1::request r10 = mp1::iexscan(c, plus, container, recv_container);
  mp1::request r11 = mp1::ireduce_scatter(c, plus, container, recv_array, recvcounts);
  mp1::request r12 = mp1::ireduce_scatter_block(c, plus, container, recv_array, 2);
  // The fixed nonblocking families also accept the value and std::array forms.
  mp1::request r13 = mp1::ibcast(c, root, scalar);
  mp1::request r14 = mp1::ibcast(c, root, array);
  mp1::request r15 = mp1::ireduce(c, plus, root, array, recv_array);
  (void)r1;
  (void)r2;
  (void)r2b;
  (void)r3;
  (void)r3b;
  (void)r4;
  (void)r4b;
  (void)r5;
  (void)r5b;
  (void)r6;
  (void)r7;
  (void)r8;
  (void)r9;
  (void)r10;
  (void)r11;
  (void)r12;
  (void)r13;
  (void)r14;
  (void)r15;

  // A non-contiguous sequence send is staged; the request owns the staged flat vector.
  mp1::request staged = mp1::iallgather(c, sequence, recv_container);
  (void)staged;

  // The counts array is copied at initiation: a temporary is safe.
  mp1::request temporary_counts =
      mp1::ireduce_scatter(c, plus, container, recv_array, std::array<mp1::count_t, 2>{2, 2});
  (void)temporary_counts;
}

void api_persistent_collectives(const mp1::mpi_communicator& c, int root) {
  int scalar = 0;
  std::array<int, 4> array{0, 0, 0, 0};
  std::array<int, 4> recv_array{0, 0, 0, 0};
  std::vector<int> recv_container(8, 0);
  std::plus<int> plus;
  mp1::layout<int> layout = mp1::layout<int>::contiguous(4);
  mp1::region<int> region_array{array.data(), layout};
  mp1::region<int> region_recv{recv_array.data(), layout};
  const std::array<mp1::count_t, 2> counts{2, 2};
  const std::span<const mp1::count_t> recvcounts{counts};
  const mp1::info info;

  mp1::request r1 = mp1::barrier_init(c, info);
  mp1::request r2 = mp1::bcast_init(c, root, array, info);
  mp1::request r2b = mp1::bcast_init(c, root, region_array, info);
  mp1::request r3 = mp1::gather_init(c, root, array, recv_array, info);
  mp1::request r3b = mp1::gather_init(c, root, mp1::in_place, recv_array, info);
  mp1::request r4 = mp1::scatter_init(c, root, recv_array, array, info);
  mp1::request r4b = mp1::scatter_init(c, root, recv_array, mp1::in_place, info);
  mp1::request r5 = mp1::allgather_init(c, array, recv_array, info);
  mp1::request r5b = mp1::allgather_init(c, region_array, region_recv, info);
  mp1::request r6 = mp1::alltoall_init(c, array, recv_array, info);
  mp1::request r7 = mp1::reduce_init(c, plus, root, array, recv_array, info);
  mp1::request r7b = mp1::reduce_init(c, plus, root, mp1::in_place, recv_array, info);
  mp1::request r8 = mp1::allreduce_init(c, plus, array, recv_array, info);
  mp1::request r9 = mp1::scan_init(c, plus, array, recv_array, info);
  mp1::request r10 = mp1::exscan_init(c, plus, array, recv_array, info);
  mp1::request r11 = mp1::reduce_scatter_init(c, plus, array, recv_array, recvcounts, info);
  mp1::request r12 = mp1::reduce_scatter_block_init(c, plus, array, recv_array, 2, info);
  // The persistent fixed families also accept the value form.
  mp1::request r13 = mp1::bcast_init(c, root, scalar, info);
  (void)r1;
  (void)r2;
  (void)r2b;
  (void)r3;
  (void)r3b;
  (void)r4;
  (void)r4b;
  (void)r5;
  (void)r5b;
  (void)r6;
  (void)r7;
  (void)r7b;
  (void)r8;
  (void)r9;
  (void)r10;
  (void)r11;
  (void)r12;
  (void)r13;

  // The counts array is copied at initiation.
  mp1::request temporary_counts = mp1::reduce_scatter_init(
      c, plus, array, recv_array, std::array<mp1::count_t, 2>{2, 2}, info);
  (void)temporary_counts;
}

void api_variable_collectives(const mp1::mpi_communicator& c, int root) {
  int send[4] = {0, 0, 0, 0};
  int recv[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int scalar = 0;
  const std::span<const int> send_span{send, 4};
  const std::span<int> recv_span{recv, 8};
  const std::span<int> local_span{recv, 2};
  mp1::layout<int> send_layout = mp1::layout<int>::contiguous(2);
  const mp1::region<const int> send_region{send, send_layout};
  mp1::region<int> local_region{recv, send_layout};
  std::vector<mp1::layout<int>> layout_owner(2, mp1::layout<int>::contiguous(2));
  const std::span<const mp1::layout<int>> layouts{layout_owner};
  std::vector<mp1::count_t> count_owner{2, 2};
  std::vector<mp1::count_t> displ_owner{0, 2};
  const mp1::blocks dense{count_owner, displ_owner};
  const std::span<const mp1::count_t> displs{displ_owner};
  const mp1::blocks empty{};

  // Blocking dense: gatherv / scatterv / allgatherv / alltoallv, including the value and
  // region single-block forms, in-place on the legal side, and the non-root dummies.
  mp1::gatherv(c, root, send_span, recv_span, dense);
  mp1::gatherv(c, root, scalar, recv_span, dense);
  mp1::gatherv(c, root, send_region, recv_span, dense);
  mp1::gatherv(c, root, mp1::in_place, recv_span, dense);
  mp1::gatherv(c, root, send_span, std::span<int>{}, empty);
  mp1::scatterv(c, root, send_span, dense, local_span);
  mp1::scatterv(c, root, send_span, dense, local_region);
  mp1::scatterv(c, root, send_span, dense, mp1::in_place);
  mp1::scatterv(c, root, send_span, dense, scalar);
  mp1::scatterv(c, root, std::span<const int>{}, empty, local_span);
  mp1::allgatherv(c, send_span, recv_span, dense);
  mp1::allgatherv(c, scalar, recv_span, dense);
  mp1::allgatherv(c, send_region, recv_span, dense);
  mp1::allgatherv(c, mp1::in_place, recv_span, dense);
  mp1::alltoallv(c, send_span, dense, recv_span, dense);
  mp1::alltoallv(c, mp1::in_place, empty, recv_span, dense);

  // Blocking typed: one layout per peer plus element displacements.
  mp1::gatherv(c, root, send_span, recv_span, layouts, displs);
  mp1::gatherv(c, root, scalar, recv_span, layouts, displs);
  mp1::gatherv(c, root, send_region, recv_span, layouts, displs);
  mp1::gatherv(c, root, mp1::in_place, recv_span, layouts, displs);
  mp1::gatherv(c, root, send_span, std::span<int>{}, std::span<const mp1::layout<int>>{},
               std::span<const mp1::count_t>{});
  mp1::scatterv(c, root, send_span, layouts, displs, local_span);
  mp1::scatterv(c, root, send_span, layouts, displs, local_region);
  mp1::scatterv(c, root, send_span, layouts, displs, mp1::in_place);
  mp1::scatterv(c, root, std::span<const int>{}, std::span<const mp1::layout<int>>{},
                std::span<const mp1::count_t>{}, local_span);
  mp1::allgatherv(c, send_span, recv_span, layouts, displs);
  mp1::allgatherv(c, scalar, recv_span, layouts, displs);
  mp1::allgatherv(c, send_region, recv_span, layouts, displs);
  mp1::allgatherv(c, mp1::in_place, recv_span, layouts, displs);
  mp1::alltoallv(c, send_span, layouts, displs, recv_span, layouts, displs);
  mp1::alltoallv(c, mp1::in_place, std::span<const mp1::layout<int>>{},
                 std::span<const mp1::count_t>{}, recv_span, layouts, displs);

  // Nonblocking: both descriptors, both single-block forms; the request copies every
  // descriptor/layout it reads after initiation.
  mp1::request v1 = mp1::igatherv(c, root, send_span, recv_span, dense);
  mp1::request v1t = mp1::igatherv(c, root, send_span, recv_span, layouts, displs);
  mp1::request v2 = mp1::igatherv(c, root, scalar, recv_span, dense);
  mp1::request v2t = mp1::igatherv(c, root, send_region, recv_span, layouts, displs);
  mp1::request v2b = mp1::igatherv(c, root, send_span, std::span<int>{}, empty);
  mp1::request v3 = mp1::iscatterv(c, root, send_span, dense, local_span);
  mp1::request v3t = mp1::iscatterv(c, root, send_span, layouts, displs, local_region);
  mp1::request v3b = mp1::iscatterv(c, root, std::span<const int>{}, empty, local_span);
  mp1::request v4 = mp1::iallgatherv(c, send_span, recv_span, dense);
  mp1::request v4t = mp1::iallgatherv(c, scalar, recv_span, layouts, displs);
  mp1::request v5 = mp1::ialltoallv(c, send_span, dense, recv_span, dense);
  mp1::request v5t = mp1::ialltoallv(c, send_span, layouts, displs, recv_span, layouts, displs);
  mp1::request v5i = mp1::ialltoallv(c, mp1::in_place, empty, recv_span, dense);
  (void)v1;
  (void)v1t;
  (void)v2;
  (void)v2t;
  (void)v2b;
  (void)v3;
  (void)v3t;
  (void)v3b;
  (void)v4;
  (void)v4t;
  (void)v5;
  (void)v5t;
  (void)v5i;

  // Persistent: fixed buffers only (span/array/C array/value/region), descriptors copied at
  // *_init; no resizable container and no sequence.
  const mp1::info info;
  mp1::request p1 = mp1::gatherv_init(c, root, send_span, recv_span, dense, info);
  mp1::request p1t = mp1::gatherv_init(c, root, send_span, recv_span, layouts, displs, info);
  mp1::request p1v = mp1::gatherv_init(c, root, scalar, recv_span, dense, info);
  mp1::request p1i = mp1::gatherv_init(c, root, mp1::in_place, recv_span, dense, info);
  mp1::request p2 = mp1::scatterv_init(c, root, send_span, dense, local_region, info);
  mp1::request p2t = mp1::scatterv_init(c, root, send_span, layouts, displs, local_span, info);
  mp1::request p2i = mp1::scatterv_init(c, root, send_span, dense, mp1::in_place, info);
  mp1::request p3 = mp1::allgatherv_init(c, send_span, recv_span, dense, info);
  mp1::request p3t = mp1::allgatherv_init(c, scalar, recv_span, layouts, displs, info);
  mp1::request p3i = mp1::allgatherv_init(c, mp1::in_place, recv_span, dense, info);
  mp1::request p4 = mp1::alltoallv_init(c, send_span, dense, recv_span, dense, info);
  mp1::request p4t =
      mp1::alltoallv_init(c, send_span, layouts, displs, recv_span, layouts, displs, info);
  mp1::request p4i = mp1::alltoallv_init(c, mp1::in_place, empty, recv_span, dense, info);
  (void)p1;
  (void)p1t;
  (void)p1v;
  (void)p1i;
  (void)p2;
  (void)p2t;
  (void)p2i;
  (void)p3;
  (void)p3t;
  (void)p3i;
  (void)p4;
  (void)p4t;
  (void)p4i;
}

int main() {
  return 0;
}
