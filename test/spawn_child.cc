// Child executable for test_spawn_contract (not matched by the test_*.cc glob).
//
// The parent spawns this program with one mode argument.  Each mode exercises one release
// protocol of the parent side of an MPI_Comm_spawn intercommunicator:
//
//   raii            obtain parent() and drop it (destructor disconnects)
//   disconnect      reject free(), then call disconnect() explicitly
//   one_shot        a second parent() throws, then disconnect; after that parent() is invalid
//   release         release() the handle without an MPI call and free it with raw MPI
//   report_rank     report this child's rank and command-order marker to parent rank 0
//   parent_threads  two threads call parent() at MPI_THREAD_MULTIPLE; exactly one must own
//                   the handle and the other must get std::logic_error (atomic one-shot)
//
// MPI 4.1 §11.8.2 (p. 524): MPI_Comm_get_parent returns the same intercommunicator the parents
// obtained from MPI_Comm_spawn; §11.10.4 (p. 544): both ends must disconnect.

#include "mp1/mp1.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>


#if MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#endif

namespace {

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

  void reset_counts() {
#if MP1_TEST_WRAP
    if (wrap_supported())
      mpi_wrap::reset_counts();
#endif
  }

  int fail(int code, const char* what) {
    std::fprintf(stderr, "spawn child failure %d: %s\n", code, what);
    return code;
  }

}  // namespace


int main(int argc, char* argv[]) {
  if (argc < 2)
    return fail(2, "missing mode argument");
  const std::string mode = argv[1];

  try {
    // The concurrent-parent mode is the only one that needs several threads inside MPI.
    if (mode == "parent_threads")
      mp1::init(mp1::threading_mode::multiple);
    else
      mp1::init();
    if (mode == "raii") {
      {
        mp1::communicator parent_comm = mp1::parent();
        if (not parent_comm.is_valid())
          return fail(3, "parent() is invalid in a spawned process");
        if (parent_comm.policy() != mp1::release_policy::disconnect)
          return fail(4, "parent() does not use the disconnect policy");
        // Destructor disconnects; the parent does the same.
      }
    } else if (mode == "disconnect") {
      mp1::communicator parent_comm = mp1::parent();
      if (not parent_comm.is_valid())
        return fail(3, "parent() is invalid in a spawned process");

      // free() must be rejected before any MPI release call; both ends must disconnect.
      bool free_threw = false;
      try {
        parent_comm.free();
      } catch (const std::logic_error&) {
        free_threw = true;
      }
      if (not free_threw)
        return fail(5, "free() did not throw std::logic_error");
      if (wrap_supported() and call_count("MPI_Comm_disconnect") != 0)
        return fail(6, "free() made an MPI_Comm_disconnect call");
      if (wrap_supported() and call_count("MPI_Comm_free") != 0)
        return fail(7, "free() made an MPI_Comm_free call");

      parent_comm.disconnect();
      if (parent_comm.is_valid())
        return fail(8, "disconnect() left the owner valid");
      if (wrap_supported() and call_count("MPI_Comm_disconnect") != 1)
        return fail(9, "disconnect() was not exactly one MPI_Comm_disconnect call");
    } else if (mode == "one_shot") {
      mp1::communicator first = mp1::parent();
      if (not first.is_valid())
        return fail(3, "parent() is invalid in a spawned process");
      bool second_threw = false;
      try {
        (void)mp1::parent();
      } catch (const std::logic_error&) {
        second_threw = true;
      }
      if (not second_threw)
        return fail(10, "a second parent() adoption did not throw std::logic_error");
      first.disconnect();
      if (mp1::parent().is_valid())
        return fail(11,
                    "parent() is valid after the parent intercommunicator was disconnected");
    } else if (mode == "release") {
      mp1::communicator parent_comm = mp1::parent();
      if (not parent_comm.is_valid())
        return fail(3, "parent() is invalid in a spawned process");
      reset_counts();
      MPI_Comm raw = parent_comm.release();
      if (parent_comm.is_valid())
        return fail(12, "release() left the owner valid");
      if (wrap_supported() and call_count("MPI_Comm_disconnect") != 0)
        return fail(13, "release() made an MPI_Comm_disconnect call");
      // Non-cooperating pair: this end frees, the parent frees too.
      if (MPI_Comm_free(&raw) != MPI_SUCCESS)
        return fail(14, "MPI_Comm_free failed for the released parent handle");
    } else if (mode == "report_rank") {
      // Used by the spawn_multiple command-order case: the child's local rank in the parent
      // intercommunicator is MPI's order of the commands/processes (MPI 4.1 §11.8.3).
      mp1::communicator parent_comm = mp1::parent();
      if (not parent_comm.is_valid())
        return fail(3, "parent() is invalid in a spawned process");
      const int order = (argc > 2 and std::string{argv[2]} == "second") ? 1 : 0;
      const int child_rank = parent_comm.rank();
      mp1::send(parent_comm, child_rank, 0, mp1::tag{8801});
      mp1::send(parent_comm, order, 0, mp1::tag{8802});
      parent_comm.disconnect();
    } else if (mode == "parent_threads") {
      if (mp1::provided() != mp1::threading_mode::multiple)
        return fail(18, "parent_threads requires MPI_THREAD_MULTIPLE");

      // MPI 4.1 §11.8.2 advice to users (p. 524): a second MPI_Comm_get_parent returns the
      // same intercommunicator, so two adopters are two owners of one handle and one
      // disconnect invalidates the other.  The one-shot claim must be an atomic
      // compare-exchange: exactly one owner and one std::logic_error for the loser.
      std::atomic<int> owners{0};
      std::atomic<int> rejected{0};
      std::atomic<int> unexpected{0};
      mp1::communicator owned[2];
      std::mutex gate_mutex;
      std::condition_variable gate;
      int at_start = 0;
      int at_end = 0;
      auto worker = [&](int index) {
        {
          std::unique_lock<std::mutex> lock{gate_mutex};
          ++at_start;
          gate.notify_all();
          gate.wait(lock, [&] { return at_start == 2; });
        }
        try {
          mp1::communicator parent_comm = mp1::parent();
          if (parent_comm.is_valid()) {
            ++owners;
            owned[index] = std::move(parent_comm);
          } else {
            ++unexpected;
          }
        } catch (const std::logic_error&) {
          ++rejected;
        } catch (...) {
          ++unexpected;
        }
        // Both attempts must observe the handle before any disconnect; otherwise the loser
        // could see MPI_COMM_NULL instead of the one-shot rejection.
        {
          std::unique_lock<std::mutex> lock{gate_mutex};
          ++at_end;
          gate.notify_all();
          gate.wait(lock, [&] { return at_end == 2; });
        }
      };
      std::thread first{worker, 0};
      std::thread second{worker, 1};
      first.join();
      second.join();

      // Report over the raw parent intercommunicator: the parent cannot observe this
      // process's exit code, so a broken invariant has to be visible in the message.
      const int report = owners.load() * 100 + rejected.load() * 10 + unexpected.load();
      MPI_Comm raw_parent{MPI_COMM_NULL};
      if (MPI_Comm_get_parent(&raw_parent) != MPI_SUCCESS or raw_parent == MPI_COMM_NULL)
        return fail(19, "MPI_Comm_get_parent failed in parent_threads");
      if (MPI_Send(&report, 1, MPI_INT, 0, 99, raw_parent) != MPI_SUCCESS)
        return fail(20, "reporting the concurrent parent() outcome failed");

      mp1::communicator* winner = nullptr;
      for (auto& slot : owned) {
        if (not slot.is_valid())
          continue;
        if (winner == nullptr) {
          winner = &slot;
        } else {
          // Only reachable if the one-shot claim regressed: keep one owner for the matched
          // disconnect and drop the extra reference without a second MPI call.
          (void)slot.release();
        }
      }
      if (winner != nullptr) {
        winner->disconnect();
        if (winner->is_valid())
          return fail(21, "disconnect() left the concurrent parent owner valid");
      } else {
        MPI_Comm_disconnect(&raw_parent);
      }
    } else {
      return fail(15, "unknown mode");
    }
    mp1::finalize();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "spawn child exception: %s\n", error.what());
    return 16;
  } catch (...) {
    return 17;
  }
  return 0;
}
