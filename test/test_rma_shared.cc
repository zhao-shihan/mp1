// Shared-memory RMA scenario (KIND scenario, 2 ranks).
//
// MPI_Win_allocate_shared requires every process of the communicator to be in the same
// shared-memory domain.  Some providers and launch configurations do not expose a
// multi-process shared-memory domain: MPI_Comm_split_type(SHARED) then returns a smaller
// (for example singleton) domain, and a shared window cannot cover MPI_COMM_WORLD.  In that
// case every rank returns 77 and CTest reports the entry as skipped.  A job that requires the
// capability exports MP1_REQUIRE_SHARED=1 (for example from
// -DMP1_TEST_REQUIRE_FEATURES=shared); the test then fails instead of skipping.  The
// capability decision is reduced across MPI_COMM_WORLD first, so all ranks skip or run
// together.
//
// The scenario body needs at least two ranks (peer direct load and non-root put).  A one-rank
// run cannot exercise it, so it returns 77 with a distinct "scenario-requires-2-ranks" message
// before capability probing; that is scenario eligibility, not a missing shared-memory domain.
//
// Covered here (not in the unit contract test, which needs no shared-memory domain):
//   allocate_shared, shared_query(rank/PROC_NULL/self, size=0 segment),
//   MPI_Win_lock_all + MPI_Win_sync + barrier direct load/store, and put/get on a shared
//   window.  The skip path is capability-only; it never weakens the API.

#include "mp1/mp1.hpp"

#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>


namespace {

  int failures = 0;
  int rank_id = 0;

  void expect(bool condition, const char* what) {
    if (not condition) {
      std::fprintf(stderr, "[%d] shared RMA check failed: %s\n", rank_id, what);
      ++failures;
    }
  }

  // Runtime-requirement switches are set by the CTest harness when the job
  // opts into the corresponding platform feature.  An empty value is not a
  // requirement.
  bool feature_required(const char* variable) {
    const char* value = std::getenv(variable);
    return value != nullptr and value[0] != '\0';
  }

}  // namespace


int main(int argc, char** argv) {
  mp1::init(argc, argv);
  const mp1::mpi_communicator world = mp1::comm_world();
  rank_id = world.rank();

  int local_ok = 0;
  int domain_size = 0;
  mp1::communicator shm;
  try {
    shm = mp1::communicator::split_type(world, mp1::comm_type_shared);
    local_ok = shm.is_valid() ? 1 : 0;
    domain_size = local_ok != 0 ? shm.size() : 0;
  } catch (...) {
    local_ok = 0;
  }

  const int local_shared =
      (local_ok != 0 && shm.is_valid() && shm.size() == world.size() && world.size() >= 2) ? 1
                                                                                           : 0;
  int shared = 0;
  MPI_Allreduce(&local_shared, &shared, 1, MPI_INT, MPI_MIN, world.native_handle());
  if (shared == 0) {
    if (local_ok != 0 && shm.is_valid()) {
      try {
        shm.free();
        // NOLINTNEXTLINE(bugprone-empty-catch): finalize cleanup must not throw
      } catch (...) {
      }
    }
    std::fprintf(stderr,
                 "MP1_SKIP: no-shared-domain -- no multi-process shared-memory domain "
                 "available (world=%d, domain=%d)\n",
                 world.size(), domain_size);
    if (feature_required("MP1_REQUIRE_SHARED")) {
      std::fprintf(stderr,
                   "test_rma_shared: shared-memory RMA is required but no multi-process "
                   "shared-memory domain is available\n");
      try {
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): finalize cleanup must not throw
      } catch (...) {
      }
      return 1;
    }
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): finalize cleanup must not throw
    } catch (...) {
    }
    return 77;
  }

  const int local_rank = shm.rank();
  const int local_bytes = local_rank == 0 ? static_cast<int>(4 * sizeof(int)) : 0;

  mp1::window w =
      mp1::window::allocate_shared(shm, {.size = local_bytes, .disp_unit = sizeof(int)});
  expect(w.is_valid(), "allocate_shared returned an invalid window");
  expect(w.create_flavor() == MPI_WIN_FLAVOR_SHARED, "create_flavor");
  expect(w.byte_size() == local_bytes, "local byte_size");
  expect(w.disp_unit() == static_cast<int>(sizeof(int)), "local disp_unit");
  expect(w.rank() == local_rank, "local rank in window group");
  expect(w.group_size() == shm.size(), "window group size");
  const int model = w.memory_model();
  expect(model == MPI_WIN_UNIFIED or model == MPI_WIN_SEPARATE, "MPI_WIN_MODEL value");

  // Every rank can query its own segment; a zero-size segment is legal.
  const mp1::shared_segment self = w.shared_query(local_rank);
  expect(self.size == local_bytes, "shared_query(self).size");
  if (local_bytes > 0) {
    expect(self.base != nullptr, "shared_query(self).base");
    expect(self.disp_unit == static_cast<int>(sizeof(int)), "shared_query(self).disp_unit");
  }

  // MPI_PROC_NULL returns the lowest rank with a non-zero segment (rank 0 here).
  const mp1::shared_segment any = w.shared_query(MPI_PROC_NULL);
  expect(any.size == static_cast<MPI_Aint>(4 * sizeof(int)), "shared_query(PROC_NULL).size");
  expect(any.base != nullptr, "shared_query(PROC_NULL).base");
  expect(any.disp_unit == static_cast<int>(sizeof(int)), "shared_query(PROC_NULL).disp_unit");

  // The returned disp_unit belongs to the queried rank, not to the caller.
  const mp1::shared_segment rank0 = w.shared_query(0);
  expect(rank0.size == static_cast<MPI_Aint>(4 * sizeof(int)), "shared_query(0).size");
  expect(rank0.base != nullptr, "shared_query(0).base");

  // Direct load/store through the shared segment.  Writer syncs before the barrier, readers
  // sync after it (MPI 4.1 12.5.4 shared-memory example).
  mp1::lock_all(w);
  if (local_rank == 0) {
    // volatile: the compiler cannot see the cross-process synchronization performed by MPI
    // (MPI 4.1 12.7.4); use it for window memory accessed by both load/store and peers.
    volatile int* local = static_cast<volatile int*>(w.base());
    for (int i = 0; i < 4; ++i)
      local[i] = 42 + i;
    mp1::sync(w);
  }
  mp1::barrier(shm);
  if (local_rank != 0) {
    mp1::sync(w);
    if (model == MPI_WIN_UNIFIED) {
      volatile const int* remote = static_cast<volatile const int*>(rank0.base);
      for (int i = 0; i < 4; ++i)
        expect(remote[i] == 42 + i, "direct load from a peer shared segment");
    }
  }
  mp1::unlock_all(w);
  mp1::barrier(shm);  // all readers are done before rank 0 reuses its segment below

  // RMA on a shared window: all non-root ranks put the same value into rank 0's segment.
  // The local store goes to rank 0's window memory before the fence that opens the epoch,
  // so it is visible to the remote put.
  const int payload = 55;
  if (local_rank == 0)
    *static_cast<volatile int*>(w.base()) = 0;
  mp1::fence(w);
  if (local_rank != 0)
    mp1::put(w, mp1::span_from(&payload, 1), 0, 0, 1);
  mp1::fence(w);
  if (local_rank == 0)
    expect(*static_cast<volatile int*>(w.base()) == payload, "put into a shared window");

  w.free();

  int total_failures = 0;
  MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_SUM, world.native_handle());
  if (shm.is_valid())
    shm.free();
  try {
    mp1::finalize();
    // NOLINTNEXTLINE(bugprone-empty-catch): finalize cleanup must not throw
  } catch (...) {
  }
  return total_failures == 0 ? 0 : 1;
}
