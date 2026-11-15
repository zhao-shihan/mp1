// Spawn release protocol: parent/child two-executable scenario test.
//
// The parent spawns ./mp1_test_spawn_contract_child (its first program argument) in one of the
// modes implemented there and checks the parent-side behavior:
//
//   0. the option payloads: spawn_options defaults and designated initialization, and
//      spawn_multiple_options::add() keeping commands/maxprocs/infos aligned;
//   1. the natural RAII pair (parent drops its spawn result, child drops parent()) terminates;
//   2. free() is rejected on a spawn result, disconnect() is the collective release;
//   3. release() + raw MPI is the non-cooperating-peer escape and makes no MPI call;
//   4. the root is positional: the payload is read only at the named root, here a non-zero one;
//   5. the errcodes variants fill the OUT array on every rank, sized maxprocs / sum(maxprocs)
//      (or the caller-supplied explicit length, including a non-root default payload), and
//      soft success is not an error;
//   6. spawn_multiple works and returns a disconnect-policy owner;
//   7. spawn_multiple's child ranks follow MPI's command order (§11.8.3);
//   8. two child threads racing on parent() produce exactly one owner (atomic one-shot).
//
// This scenario covers the single `spawn_options` form and the `spawn_multiple_options`
// builder form.  The soft-spawn clause ("a soft spawn is a success with a smaller
// remote_size()") is unobservable on both pinned providers: an independent probe with info
// soft="0:1", maxprocs=2 hard-spawns 2 processes on Open MPI 6.0.0rc2 and MPICH 4.3.2
// (remote_size == 2).  The success path is still exercised; only the size reduction is
// provider-contingent, so the clause is documented here rather than asserted falsely.
//
// MPI 4.1 basis: §11.8.2 (Starting Processes; root-only arguments, pp. 521-524), §11.8.3
// (spawn_multiple, pp. 525-527), §11.10.4 (Releasing Connections: both ends must disconnect,
// pp. 543-544).  The wrap counters assert the release operation and that no agreement
// broadcast is hidden in spawn (raw passthrough).
//
// The test is deliberately independent of the Boost.Test harness because it needs its own
// child executable; the harness passes that executable as argv[1].

#include "mp1/mp1.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>


#if MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#endif

namespace {

  int failures = 0;

  void check_failure(const char* file, int line, const char* expression) {
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expression);
    ++failures;
  }

#define MP1_CHECK(condition)                         \
  do {                                               \
    if (not(condition))                              \
      check_failure(__FILE__, __LINE__, #condition); \
  } while (false)

  bool wrap_supported() {
#if MP1_TEST_WRAP
    return mpi_wrap::supported();
#else
    return false;
#endif
  }

  std::size_t call_count(const char* symbol) {
#if MP1_TEST_WRAP
    return wrap_supported() ? mpi_wrap::count(symbol) : 0;
#else
    (void)symbol;
    return 0;
#endif
  }

  std::size_t total_calls() {
#if MP1_TEST_WRAP
    return wrap_supported() ? mpi_wrap::total() : 0;
#else
    return 0;
#endif
  }

  void reset_calls() {
#if MP1_TEST_WRAP
    if (wrap_supported())
      mpi_wrap::reset_counts();
#endif
  }

  template<class F>
  bool throws_logic_error(F&& function) {
    try {
      function();
    } catch (const std::logic_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  mp1::command_line child_command(const std::string& child, const char* mode) {
    return mp1::command_line{child, mode};
  }

}  // namespace


int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <spawn-child-executable>\n", argv[0]);
    return 2;
  }
  const std::string child = argv[1];

  try {
    mp1::init();
    const mp1::mpi_communicator world = mp1::comm_world();
    const int rank = world.rank();

    // The root-only arguments are deliberately garbage on the non-root ranks in several cases:
    // MPI ignores them there, and mp1 must not inspect them (raw passthrough).
    const mp1::command_line root_command = child_command(child, "raii");

    // ---- 0. option payload shape --------------------------------------------------------
    {
      const mp1::spawn_options defaults;
      MP1_CHECK(defaults.maxprocs == 1);
      MP1_CHECK(defaults.command.empty());
      MP1_CHECK(defaults.info.native_handle() == MPI_INFO_NULL);

      mp1::spawn_multiple_options multiple;
      MP1_CHECK(multiple.size() == 0);
      MP1_CHECK(multiple.commands().empty());
      MP1_CHECK(multiple.maxprocs().empty());
      MP1_CHECK(multiple.infos().empty());

      mp1::info hint;
      hint.set("host", "localhost");
      multiple.add(child_command(child, "raii"), 2, hint)
          .add(child_command(child, "disconnect"), 1);
      MP1_CHECK(multiple.size() == 2);
      MP1_CHECK(multiple.commands().size() == 2);
      MP1_CHECK(multiple.maxprocs().size() == 2);
      MP1_CHECK(multiple.infos().size() == 2);
      MP1_CHECK(multiple.maxprocs()[0] == 2);
      MP1_CHECK(multiple.maxprocs()[1] == 1);
      MP1_CHECK(multiple.commands()[0].size() == 2);
      MP1_CHECK(multiple.infos()[0].native_handle() != MPI_INFO_NULL);

      mp1::spawn_multiple_options copy = multiple;  // a copyable value payload
      MP1_CHECK(copy.size() == 2);
      MP1_CHECK(copy.maxprocs()[1] == 1);
    }

    // ---- 1. RAII pair: both ends disconnect from their destructors ----------------------
    reset_calls();
    {
      const mp1::spawn_options options{
          .maxprocs = rank == 0 ? 1 : -7,
          .command = rank == 0 ? root_command : mp1::command_line{}};
      mp1::communicator children = mp1::spawn(world, 0, options);
      MP1_CHECK(children.is_valid());
      MP1_CHECK(children.is_inter());
      MP1_CHECK(children.remote_size() == 1);
      MP1_CHECK(children.size() == world.size());
      MP1_CHECK(children.policy() == mp1::release_policy::disconnect);
      MP1_CHECK(call_count("MPI_Comm_spawn") == 1);
      MP1_CHECK(call_count("MPI_Bcast") == 0);  // no hidden spawn agreement collective
      reset_calls();
    }
    MP1_CHECK(call_count("MPI_Comm_disconnect") == 1);
    MP1_CHECK(call_count("MPI_Comm_free") == 0);

    // ---- 2. free() rejected; explicit collective disconnect -----------------------------
    {
      const mp1::spawn_options options{
          .maxprocs = 1,
          .command = rank == 0 ? child_command(child, "disconnect") : mp1::command_line{}};
      mp1::communicator children = mp1::spawn(world, 0, options);
      MP1_CHECK(children.is_valid());
      MP1_CHECK(throws_logic_error([&] { children.free(); }));
      MP1_CHECK(children.is_valid());
      reset_calls();
      children.disconnect();
      MP1_CHECK(not children.is_valid());
      MP1_CHECK(call_count("MPI_Comm_disconnect") == 1);
      MP1_CHECK(call_count("MPI_Comm_free") == 0);
    }

    // ---- 3. release() escape for a non-cooperating peer ---------------------------------
    {
      const mp1::spawn_options options{
          .maxprocs = 1,
          .command = rank == 0 ? child_command(child, "release") : mp1::command_line{}};
      mp1::communicator children = mp1::spawn(world, 0, options);
      MP1_CHECK(children.is_valid());
      reset_calls();
      MPI_Comm raw = children.release();
      MP1_CHECK(not children.is_valid());
      MP1_CHECK(total_calls() == 0);
      MP1_CHECK(MPI_Comm_free(&raw) == MPI_SUCCESS);
    }

    // ---- 4. parent one-shot adoption (child side) ---------------------------------------
    {
      const mp1::spawn_options options{
          .maxprocs = 1,
          .command = rank == 0 ? child_command(child, "one_shot") : mp1::command_line{}};
      mp1::communicator children = mp1::spawn(world, 0, options);
      MP1_CHECK(children.is_valid());
      children.disconnect();
      MP1_CHECK(not children.is_valid());
    }

    // ---- 5. root is positional: a non-zero root reads its own payload -------------------
    {
      const int root = world.size() - 1;
      const mp1::spawn_options options{
          .maxprocs = 1,
          .command = rank == root ? child_command(child, "disconnect") : mp1::command_line{}};
      mp1::communicator children = mp1::spawn(world, root, options);
      MP1_CHECK(children.is_valid());
      MP1_CHECK(children.is_inter());
      MP1_CHECK(children.remote_size() == 1);
      children.disconnect();
    }

    // ---- 6. errcodes variants -----------------------------------------------------------
    {
      // Default sizing: the OUT array is sized from each rank's local maxprocs, so every rank
      // must pass the root's maxprocs (caller obligation, documented).  maxprocs = 2 is not
      // the default 1, so a wrong local size would be visible.
      const mp1::spawn_options options{
          .maxprocs = 2, .command = rank == 0 ? root_command : mp1::command_line{}};
      mp1::spawn_result result = mp1::spawn_with_errcodes(world, 0, options);
      MP1_CHECK(result.intercomm.is_valid());
      MP1_CHECK(result.intercomm.policy() == mp1::release_policy::disconnect);
      MP1_CHECK(result.intercomm.remote_size() == 2);
      MP1_CHECK(result.errcodes.size() == 2);
      if (result.errcodes.size() == 2) {
        MP1_CHECK(result.errcodes[0] == MPI_SUCCESS);
        MP1_CHECK(result.errcodes[1] == MPI_SUCCESS);
      }
      result.intercomm.disconnect();
      MP1_CHECK(not result.intercomm.is_valid());
    }

    // ---- 6b. errcodes explicit length: safe non-root default payload ---------------------
    {
      // The non-root ranks keep the default payload (maxprocs = 1) while the root asks for 2.
      // Passing the root's OUT length explicitly (2) sizes every rank's array correctly, so
      // the ignored non-root maxprocs cannot truncate it.
      const mp1::spawn_options options{
          .maxprocs = 2, .command = rank == 0 ? root_command : mp1::command_line{}};
      mp1::spawn_result result = mp1::spawn_with_errcodes(world, 0, options, 2);
      MP1_CHECK(result.intercomm.is_valid());
      MP1_CHECK(result.intercomm.remote_size() == 2);
      MP1_CHECK(result.errcodes.size() == 2);
      if (result.errcodes.size() == 2) {
        MP1_CHECK(result.errcodes[0] == MPI_SUCCESS);
        MP1_CHECK(result.errcodes[1] == MPI_SUCCESS);
      }
      result.intercomm.disconnect();
    }

    // ---- 7. spawn_multiple --------------------------------------------------------------
    {
      mp1::spawn_multiple_options spawns;
      if (rank == 0)
        spawns.add(child_command(child, "raii"), 1).add(child_command(child, "disconnect"), 1);
      MP1_CHECK(spawns.size() == (rank == 0 ? 2u : 0u));
      MP1_CHECK(spawns.maxprocs().size() == spawns.size());
      MP1_CHECK(spawns.infos().size() == spawns.size());
      mp1::communicator children = mp1::spawn_multiple(world, 0, spawns);
      MP1_CHECK(children.is_valid());
      MP1_CHECK(children.is_inter());
      MP1_CHECK(children.remote_size() == 2);
      MP1_CHECK(children.policy() == mp1::release_policy::disconnect);
      reset_calls();
      children.disconnect();
      MP1_CHECK(call_count("MPI_Comm_disconnect") == 1);
    }

    // ---- 8. spawn_multiple_with_errcodes: sum(maxprocs), not count ----------------------
    {
      mp1::spawn_multiple_options spawns;
      // maxprocs has two entries {1,2}: the OUT array has 1+2 = 3 entries, not 2 (the count).
      // The non-root ranks repeat the same maxprocs (the default-sizing contract).
      spawns.add(rank == 0 ? child_command(child, "raii") : mp1::command_line{}, 1)
          .add(rank == 0 ? child_command(child, "raii") : mp1::command_line{}, 2);
      mp1::spawn_result result = mp1::spawn_multiple_with_errcodes(world, 0, spawns);
      MP1_CHECK(result.intercomm.is_valid());
      MP1_CHECK(result.intercomm.remote_size() == 3);
      MP1_CHECK(result.errcodes.size() == 3);
      if (result.errcodes.size() == 3) {
        MP1_CHECK(result.errcodes[0] == MPI_SUCCESS);
        MP1_CHECK(result.errcodes[1] == MPI_SUCCESS);
        MP1_CHECK(result.errcodes[2] == MPI_SUCCESS);
      }
      result.intercomm.disconnect();
    }

    // ---- 8b. spawn_multiple errcodes explicit length: empty non-root payload -------------
    {
      mp1::spawn_multiple_options spawns;
      if (rank == 0) {
        spawns.add(child_command(child, "raii"), 1).add(child_command(child, "raii"), 2);
      }
      // The non-root payload is empty (sum(maxprocs) = 0 locally); the explicit length 3 is the
      // root's sum and sizes every rank's OUT array correctly.
      mp1::spawn_result result = mp1::spawn_multiple_with_errcodes(world, 0, spawns, 3);
      MP1_CHECK(result.intercomm.is_valid());
      MP1_CHECK(result.intercomm.remote_size() == 3);
      MP1_CHECK(result.errcodes.size() == 3);
      if (result.errcodes.size() == 3) {
        MP1_CHECK(result.errcodes[0] == MPI_SUCCESS);
        MP1_CHECK(result.errcodes[1] == MPI_SUCCESS);
        MP1_CHECK(result.errcodes[2] == MPI_SUCCESS);
      }
      result.intercomm.disconnect();
    }

    // ---- 9. spawn_multiple command order ------------------------------------------------
    // MPI 4.1 §11.8.3 p. 525: the first maxprocs[0] processes spawned by the first command get
    // ranks 0..maxprocs[0]-1 in the children group.  The child reports its local rank and which
    // command line launched it; parent rank 0 checks the result against the remote rank.
    {
      mp1::spawn_multiple_options spawns;
      if (rank == 0)
        spawns.add(mp1::command_line{child, "report_rank", "first"}, 1)
            .add(mp1::command_line{child, "report_rank", "second"}, 1);
      mp1::communicator children = mp1::spawn_multiple(world, 0, spawns);
      MP1_CHECK(children.is_valid());
      MP1_CHECK(children.remote_size() == 2);
      if (rank == 0) {
        for (int remote = 0; remote < 2; ++remote) {
          int reported_rank = -1;
          int order = -1;
          (void)mp1::recv(children, reported_rank, remote, mp1::tag{8801});
          (void)mp1::recv(children, order, remote, mp1::tag{8802});
          MP1_CHECK(reported_rank == remote);
          MP1_CHECK(order == remote);
        }
      }
      children.disconnect();
    }

    // ---- 10. concurrent parent() claim in the child (atomic one-shot adoption) ----------
    // Two child threads call parent() at MPI_THREAD_MULTIPLE; exactly one may own the single
    // MPI_Comm_get_parent handle (MPI 4.1 §11.8.2 advice to users) and the loser must get
    // std::logic_error.  The child reports the counts over the parent intercommunicator
    // because spawn does not expose the child's exit code.
    {
      const mp1::spawn_options options{
          .maxprocs = 1,
          .command = rank == 0 ? child_command(child, "parent_threads") : mp1::command_line{}};
      mp1::communicator children = mp1::spawn(world, 0, options);
      MP1_CHECK(children.is_valid());
      MP1_CHECK(children.is_inter());
      if (rank == 0) {
        int report = -1;
        (void)mp1::recv(children, report, 0, mp1::tag{99});
        // owners * 100 + logic_errors * 10 + other_failures
        MP1_CHECK((report / 100) == 1);
        MP1_CHECK((((report / 10) % 10) == 1));
        MP1_CHECK((report % 10) == 0);
      }
      children.disconnect();
      MP1_CHECK(not children.is_valid());
    }

    mp1::finalize();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "spawn contract test exception: %s\n", error.what());
    return 1;
  } catch (...) {
    return 1;
  }
  return failures == 0 ? 0 : 1;
}
