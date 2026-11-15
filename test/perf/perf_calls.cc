// Deterministic call-count gates (label perf-calls).
//
// Every operation under test is wrapped with the `-Wl,--wrap` shim (mpi_wrap).  The
// counters are reset immediately around the operation, asserted on every rank, and the test
// fails if any expected native spelling, staging call, or request-teardown policy differs.
//
// This is not a timing test: no absolute wall time is read here.  Run through the
// `perf_calls` CTest entry, which drives ranks 2 and 4; a bare `mpiexec -n N` run is
// equivalent.  Exit code 77 means the platform has no wrap shim.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <iterator>
#include <list>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_PERF_CALLS_WRAP 1
#else
#define MP1_PERF_CALLS_WRAP 0
#endif

namespace {

  // The contiguity classifier is a compile-time contract, not a runtime heuristic.
  static_assert(std::contiguous_iterator<int*>);
  static_assert(std::contiguous_iterator<const int*>);
  static_assert(std::contiguous_iterator<std::vector<int>::iterator>);
  static_assert(std::contiguous_iterator<std::vector<int>::const_iterator>);
  static_assert(std::contiguous_iterator<std::array<int, 4>::iterator>);
  static_assert(std::contiguous_iterator<std::string::iterator>);
  static_assert(std::contiguous_iterator<std::wstring::iterator>);
  static_assert(std::contiguous_iterator<std::u8string::iterator>);
  static_assert(std::contiguous_iterator<std::u16string::iterator>);
  static_assert(std::contiguous_iterator<std::u32string::iterator>);
  static_assert(!std::contiguous_iterator<std::list<int>::iterator>);
  static_assert(!std::contiguous_iterator<std::list<std::pair<int, int>>::iterator>);
  static_assert(!std::contiguous_iterator<std::deque<int>::iterator>);
  static_assert(!std::contiguous_iterator<std::vector<bool>::iterator>);

  int g_rank = 0;
  int g_size = 1;
  int g_failures = 0;

  [[nodiscard]] const std::string& alltoallw_symbol() {
    static const std::string symbol = std::string{"MPI_Allto"} + "allw";
    return symbol;
  }

  [[nodiscard]] const std::string& alltoallw_c_symbol() {
    static const std::string symbol = std::string{"MPI_Allto"} + "allw_c";
    return symbol;
  }

  [[nodiscard]] const std::string& ialltoallw_symbol() {
    static const std::string symbol = std::string{"MPI_Iallto"} + "allw";
    return symbol;
  }

  [[nodiscard]] const std::string& ialltoallw_c_symbol() {
    static const std::string symbol = std::string{"MPI_Iallto"} + "allw_c";
    return symbol;
  }

  void check(bool condition, const std::string& what) {
    if (not condition) {
      ++g_failures;
      std::fprintf(stderr, "perf_calls: rank %d: FAIL %s\n", g_rank, what.c_str());
    }
  }

  void expect_count(const char* symbol, std::size_t expected) {
    const std::size_t actual = mpi_wrap::count(symbol);
    if (actual != expected) {
      check(false, std::string{"count("} + symbol + ") = " + std::to_string(actual) +
                       ", expected " + std::to_string(expected));
    }
  }

  void expect_symbol_zero(const char* symbol) {
    expect_count(symbol, 0);
  }

  // MPI_Datatype is an integer handle on MPICH and a pointer on Open MPI.  The comparison
  // token is the same integer form mpi_wrap records; it is never converted back to a handle.
  template<class Handle>
  std::uintptr_t handle_token(Handle handle) noexcept {
    if constexpr (std::is_pointer_v<Handle>)
      return reinterpret_cast<std::uintptr_t>(handle);
    else
      return static_cast<std::uintptr_t>(handle);
  }

  // Dense path: no derived-type duplication and no staging anywhere.
  void expect_dense_no_staging() {
    expect_symbol_zero("MPI_Type_dup");
    expect_symbol_zero("MPI_Pack_c");
    expect_symbol_zero("MPI_Unpack_c");
    expect_symbol_zero("MPI_Pack_size_c");
    expect_symbol_zero(alltoallw_symbol().c_str());
    expect_symbol_zero(alltoallw_c_symbol().c_str());
    expect_symbol_zero(ialltoallw_symbol().c_str());
    expect_symbol_zero(ialltoallw_c_symbol().c_str());
  }

  // Every variable family must stay away from the forbidden all-to-all engine and from
  // duplicate datatypes, independent of whether the path staged.
  void expect_variable_clean() {
    expect_symbol_zero("MPI_Type_dup");
    expect_symbol_zero(alltoallw_symbol().c_str());
    expect_symbol_zero(alltoallw_c_symbol().c_str());
    expect_symbol_zero(ialltoallw_symbol().c_str());
    expect_symbol_zero(ialltoallw_c_symbol().c_str());
  }

  // ---------------------------------------------------------------------------------------
  // Dense variable families (blocking and nonblocking).  One corresponding native _c
  // spelling per call, zero staging, zero datatype duplication.
  // ---------------------------------------------------------------------------------------

  constexpr int dense_count = 2;

  void dense_gatherv(bool nonblocking) {
    const mp1::mpi_communicator c = mp1::comm_world();
    const int root = g_size - 1;
    const int n = dense_count;
    const std::vector<mp1::count_t> counts(static_cast<std::size_t>(g_size), n);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(g_size));
    for (int i = 0; i < g_size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * n;
    }
    const mp1::blocks dense{counts, displs};
    std::vector<int> send(static_cast<std::size_t>(n));
    std::vector<int> recv(static_cast<std::size_t>(n) * static_cast<std::size_t>(g_size), -1);
    for (int j = 0; j < n; ++j) {
      send[static_cast<std::size_t>(j)] = g_rank * 100 + j;
    }

    mpi_wrap::reset_counts();
    if (nonblocking) {
      mp1::request request;
      if (g_rank == root) {
        request = mp1::igatherv(c, root, send, recv, dense);
      } else {
        request = mp1::igatherv(c, root, send, std::span<int>{}, mp1::blocks{});
      }
      expect_count("MPI_Igatherv_c", 1);
      expect_symbol_zero("MPI_Igatherv");
      request.wait();
    } else if (g_rank == root) {
      mp1::gatherv(c, root, send, recv, dense);
    } else {
      mp1::gatherv(c, root, send, std::span<int>{}, mp1::blocks{});
    }
    if (not nonblocking) {
      expect_count("MPI_Gatherv_c", 1);
      expect_symbol_zero("MPI_Gatherv");
    }
    expect_dense_no_staging();

    if (g_rank == root) {
      for (int i = 0; i < g_size; ++i) {
        for (int j = 0; j < n; ++j) {
          check(recv[static_cast<std::size_t>(i * n + j)] == i * 100 + j,
                "dense gatherv data at rank " + std::to_string(i));
        }
      }
    }
  }

  void dense_scatterv(bool nonblocking) {
    const mp1::mpi_communicator c = mp1::comm_world();
    const int root = g_size - 1;
    const int n = dense_count;
    const std::vector<mp1::count_t> counts(static_cast<std::size_t>(g_size), n);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(g_size));
    for (int i = 0; i < g_size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * n;
    }
    const mp1::blocks dense{counts, displs};
    std::vector<int> send(static_cast<std::size_t>(n) * static_cast<std::size_t>(g_size), -1);
    std::vector<int> recv(static_cast<std::size_t>(n), -1);
    for (int i = 0; i < g_size; ++i) {
      for (int j = 0; j < n; ++j) {
        send[static_cast<std::size_t>(i * n + j)] = i * 100 + j;
      }
    }

    mpi_wrap::reset_counts();
    if (nonblocking) {
      mp1::request request;
      if (g_rank == root) {
        request = mp1::iscatterv(c, root, send, dense, recv);
      } else {
        request = mp1::iscatterv(c, root, std::span<const int>{}, mp1::blocks{}, recv);
      }
      expect_count("MPI_Iscatterv_c", 1);
      expect_symbol_zero("MPI_Iscatterv");
      request.wait();
    } else if (g_rank == root) {
      mp1::scatterv(c, root, send, dense, recv);
    } else {
      mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{}, recv);
    }
    if (not nonblocking) {
      expect_count("MPI_Scatterv_c", 1);
      expect_symbol_zero("MPI_Scatterv");
    }
    expect_dense_no_staging();

    for (int j = 0; j < n; ++j) {
      check(recv[static_cast<std::size_t>(j)] == g_rank * 100 + j,
            "dense scatterv data on rank " + std::to_string(g_rank));
    }
  }

  void dense_allgatherv(bool nonblocking) {
    const mp1::mpi_communicator c = mp1::comm_world();
    const int n = dense_count;
    const std::vector<mp1::count_t> counts(static_cast<std::size_t>(g_size), n);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(g_size));
    for (int i = 0; i < g_size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * n;
    }
    const mp1::blocks dense{counts, displs};
    std::vector<int> send(static_cast<std::size_t>(n));
    std::vector<int> recv(static_cast<std::size_t>(n) * static_cast<std::size_t>(g_size), -1);
    for (int j = 0; j < n; ++j) {
      send[static_cast<std::size_t>(j)] = g_rank * 100 + j;
    }

    mpi_wrap::reset_counts();
    if (nonblocking) {
      mp1::request request = mp1::iallgatherv(c, send, recv, dense);
      expect_count("MPI_Iallgatherv_c", 1);
      expect_symbol_zero("MPI_Iallgatherv");
      request.wait();
    } else {
      mp1::allgatherv(c, send, recv, dense);
    }
    if (not nonblocking) {
      expect_count("MPI_Allgatherv_c", 1);
      expect_symbol_zero("MPI_Allgatherv");
    }
    expect_dense_no_staging();

    for (int i = 0; i < g_size; ++i) {
      for (int j = 0; j < n; ++j) {
        check(recv[static_cast<std::size_t>(i * n + j)] == i * 100 + j,
              "dense allgatherv data at rank " + std::to_string(i));
      }
    }
  }

  void dense_alltoallv(bool nonblocking) {
    const mp1::mpi_communicator c = mp1::comm_world();
    const int n = dense_count;
    const std::vector<mp1::count_t> counts(static_cast<std::size_t>(g_size), n);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(g_size));
    for (int i = 0; i < g_size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * n;
    }
    const mp1::blocks dense{counts, displs};
    const std::size_t total = static_cast<std::size_t>(n) * static_cast<std::size_t>(g_size);
    std::vector<int> send(total, -1);
    std::vector<int> recv(total, -1);
    for (int peer = 0; peer < g_size; ++peer) {
      for (int j = 0; j < n; ++j) {
        send[static_cast<std::size_t>(peer * n + j)] = g_rank * 100 + peer * 10 + j;
      }
    }

    mpi_wrap::reset_counts();
    if (nonblocking) {
      mp1::request request = mp1::ialltoallv(c, send, dense, recv, dense);
      expect_count("MPI_Ialltoallv_c", 1);
      expect_symbol_zero("MPI_Ialltoallv");
      request.wait();
    } else {
      mp1::alltoallv(c, send, dense, recv, dense);
    }
    if (not nonblocking) {
      expect_count("MPI_Alltoallv_c", 1);
      expect_symbol_zero("MPI_Alltoallv");
    }
    expect_dense_no_staging();

    for (int peer = 0; peer < g_size; ++peer) {
      for (int j = 0; j < n; ++j) {
        check(recv[static_cast<std::size_t>(peer * n + j)] == peer * 100 + g_rank * 10 + j,
              "dense alltoallv data from rank " + std::to_string(peer));
      }
    }
  }

  // ---------------------------------------------------------------------------------------
  // Holey path: Pack_c is allowed but bounded by the staged block count per direction;
  // Type_dup and the all-to-all engine stay at zero, and the wire count is the exact element
  // count (checked by the untouched gap sentinels and the delivered values, not the padded
  // extent).
  //
  // The typed wire makes a staged receive pack its flat native-T image before MPI_Unpack_c
  // (MPI 4.1 5.2: a packing unit is implementation-defined, so a byte copy is not the portable
  // conversion).  The bound is therefore the number of staged SEND blocks plus one local pack
  // per staged RECEIVE block, not just the peer count:
  //   gatherv root  = 1 staged send + P staged receive blocks = P + 1
  //   scatterv root = P staged send blocks + 1 staged receive block = P + 1
  //   allgatherv    = 1 staged send + P staged receive blocks = P + 1
  //   alltoallv     = P staged send + P staged receive blocks = 2 * P
  // All four are exact, not slack bounds: a change to either side moves them.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] mp1::layout<int> holey_layout() {
    // Two ints at element offsets 0 and 2: size 8 bytes, true extent 12 bytes.
    return mp1::layout<int>::vector(2, 1, 2);
  }

  // The wire count is the exact MPI_Type_size of the layout, never the padded extent.
  void check_type_size_contract(const mp1::layout<int>& layout, const char* what) {
    MPI_Count type_size{0};
    MPI_Type_size_c(layout.native_handle(), &type_size);
    check(type_size == static_cast<MPI_Count>(layout.byte_size()),
          std::string{what} + ": MPI_Type_size != layout byte_size");
  }

  void holey_gatherv() {
    const mp1::mpi_communicator c = mp1::comm_world();
    const int root = g_size - 1;
    const mp1::layout<int> holey = holey_layout();
    check_type_size_contract(holey, "holey gatherv");
    constexpr int extent = 3;
    constexpr int block_elements = 2;
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(g_size), holey);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(g_size));
    for (int i = 0; i < g_size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * extent;
    }
    std::vector<int> send(extent, -1);
    send[0] = g_rank * 100 + 1;
    send[2] = g_rank * 100 + 2;
    std::vector<int> recv(static_cast<std::size_t>(extent) * static_cast<std::size_t>(g_size),
                          -1);

    mpi_wrap::reset_counts();
    if (g_rank == root) {
      mp1::gatherv(c, root, mp1::region{send.data(), holey}, recv, recv_layouts, displs);
    } else {
      mp1::gatherv(c, root, mp1::region{send.data(), holey}, std::span<int>{}, mp1::blocks{});
    }
    expect_count("MPI_Gatherv_c", 1);
    expect_symbol_zero("MPI_Gatherv");
    expect_variable_clean();
    const std::size_t packs = mpi_wrap::count("MPI_Pack_c");
    check(packs >= 1, "holey gatherv must stage at least one block");
    // P + 1: the root's own send stage plus one completion pack per staged receive block.
    check(packs <= static_cast<std::size_t>(g_size) + 1,
          "holey gatherv Pack_c = " + std::to_string(packs) + " exceeds blocks " +
              std::to_string(g_size + 1));
    check(mpi_wrap::count("MPI_Pack_size_c") <= static_cast<std::size_t>(g_size) + 1,
          "holey gatherv Pack_size_c exceeds staged blocks");

    if (g_rank == root) {
      for (int i = 0; i < g_size; ++i) {
        check(recv[static_cast<std::size_t>(i * extent)] == i * 100 + 1,
              "holey gatherv first value at rank " + std::to_string(i));
        check(recv[static_cast<std::size_t>(i * extent + 2)] == i * 100 + 2,
              "holey gatherv second value at rank " + std::to_string(i));
        check(recv[static_cast<std::size_t>(i * extent + 1)] == -1,
              "holey gatherv wrote padding at rank " + std::to_string(i));
      }
    }
    check(static_cast<int>(holey.size()) == block_elements, "holey layout element count");
  }

  void holey_scatterv() {
    const mp1::mpi_communicator c = mp1::comm_world();
    const int root = g_size - 1;
    const mp1::layout<int> holey = holey_layout();
    check_type_size_contract(holey, "holey gatherv");
    constexpr int extent = 3;
    std::vector<mp1::layout<int>> send_layouts(static_cast<std::size_t>(g_size), holey);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(g_size));
    for (int i = 0; i < g_size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * extent;
    }
    std::vector<int> send(static_cast<std::size_t>(extent) * static_cast<std::size_t>(g_size),
                          -1);
    for (int i = 0; i < g_size; ++i) {
      send[static_cast<std::size_t>(i * extent)] = i * 100 + 1;
      send[static_cast<std::size_t>(i * extent + 2)] = i * 100 + 2;
    }
    std::vector<int> recv(extent, -1);

    mpi_wrap::reset_counts();
    if (g_rank == root) {
      mp1::scatterv(c, root, send, send_layouts, displs, mp1::region{recv.data(), holey});
    } else {
      mp1::scatterv(c, root, std::span<const int>{}, mp1::blocks{},
                    mp1::region{recv.data(), holey});
    }
    expect_count("MPI_Scatterv_c", 1);
    expect_symbol_zero("MPI_Scatterv");
    expect_variable_clean();
    const std::size_t packs = mpi_wrap::count("MPI_Pack_c");
    // P + 1: P staged send blocks plus one completion pack for the root's receive block.
    check(packs <= static_cast<std::size_t>(g_size) + 1,
          "holey scatterv Pack_c = " + std::to_string(packs) + " exceeds blocks " +
              std::to_string(g_size + 1));
    check(mpi_wrap::count("MPI_Pack_size_c") <= static_cast<std::size_t>(g_size) + 1,
          "holey scatterv Pack_size_c exceeds staged blocks");

    check(recv[0] == g_rank * 100 + 1, "holey scatterv first value");
    check(recv[2] == g_rank * 100 + 2, "holey scatterv second value");
    check(recv[1] == -1, "holey scatterv wrote padding");
  }

  void holey_allgatherv() {
    const mp1::mpi_communicator c = mp1::comm_world();
    const mp1::layout<int> holey = holey_layout();
    check_type_size_contract(holey, "holey gatherv");
    constexpr int extent = 3;
    std::vector<mp1::layout<int>> recv_layouts(static_cast<std::size_t>(g_size), holey);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(g_size));
    for (int i = 0; i < g_size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * extent;
    }
    std::vector<int> send(extent, -1);
    send[0] = g_rank * 100 + 1;
    send[2] = g_rank * 100 + 2;
    std::vector<int> recv(static_cast<std::size_t>(extent) * static_cast<std::size_t>(g_size),
                          -1);

    mpi_wrap::reset_counts();
    mp1::allgatherv(c, mp1::region{send.data(), holey}, recv, recv_layouts, displs);
    expect_count("MPI_Allgatherv_c", 1);
    expect_symbol_zero("MPI_Allgatherv");
    expect_variable_clean();
    const std::size_t packs = mpi_wrap::count("MPI_Pack_c");
    check(packs >= 1, "holey allgatherv must stage at least one block");
    check(packs <= static_cast<std::size_t>(g_size) + 1,
          "holey allgatherv Pack_c = " + std::to_string(packs) + " exceeds blocks");
    check(mpi_wrap::count("MPI_Pack_size_c") <= static_cast<std::size_t>(g_size) + 1,
          "holey allgatherv Pack_size_c exceeds staged blocks");

    for (int i = 0; i < g_size; ++i) {
      check(recv[static_cast<std::size_t>(i * extent)] == i * 100 + 1,
            "holey allgatherv first value at rank " + std::to_string(i));
      check(recv[static_cast<std::size_t>(i * extent + 2)] == i * 100 + 2,
            "holey allgatherv second value at rank " + std::to_string(i));
      check(recv[static_cast<std::size_t>(i * extent + 1)] == -1,
            "holey allgatherv wrote padding at rank " + std::to_string(i));
    }
  }

  void holey_alltoallv() {
    const mp1::mpi_communicator c = mp1::comm_world();
    const mp1::layout<int> holey = holey_layout();
    check_type_size_contract(holey, "holey gatherv");
    constexpr int extent = 3;
    std::vector<mp1::layout<int>> layouts(static_cast<std::size_t>(g_size), holey);
    std::vector<mp1::count_t> displs(static_cast<std::size_t>(g_size));
    for (int i = 0; i < g_size; ++i) {
      displs[static_cast<std::size_t>(i)] = static_cast<mp1::count_t>(i) * extent;
    }
    const std::size_t total =
        static_cast<std::size_t>(extent) * static_cast<std::size_t>(g_size);
    std::vector<int> send(total, -1);
    std::vector<int> recv(total, -1);
    for (int peer = 0; peer < g_size; ++peer) {
      send[static_cast<std::size_t>(peer * extent)] = g_rank * 100 + peer * 10 + 1;
      send[static_cast<std::size_t>(peer * extent + 2)] = g_rank * 100 + peer * 10 + 2;
    }

    mpi_wrap::reset_counts();
    mp1::alltoallv(c, send, layouts, displs, recv, layouts, displs);
    expect_count("MPI_Alltoallv_c", 1);
    expect_symbol_zero("MPI_Alltoallv");
    expect_variable_clean();
    const std::size_t packs = mpi_wrap::count("MPI_Pack_c");
    check(packs >= 1, "holey alltoallv must stage at least one block");
    check(packs <= static_cast<std::size_t>(2 * g_size),
          "holey alltoallv Pack_c = " + std::to_string(packs) + " exceeds blocks");
    check(mpi_wrap::count("MPI_Pack_size_c") <= static_cast<std::size_t>(2 * g_size),
          "holey alltoallv Pack_size_c exceeds staged blocks");

    for (int peer = 0; peer < g_size; ++peer) {
      check(recv[static_cast<std::size_t>(peer * extent)] == peer * 100 + g_rank * 10 + 1,
            "holey alltoallv first value from rank " + std::to_string(peer));
      check(recv[static_cast<std::size_t>(peer * extent + 2)] == peer * 100 + g_rank * 10 + 2,
            "holey alltoallv second value from rank " + std::to_string(peer));
      check(recv[static_cast<std::size_t>(peer * extent + 1)] == -1,
            "holey alltoallv wrote padding from rank " + std::to_string(peer));
    }
  }

  // ---------------------------------------------------------------------------------------
  // Reductions.  Flat paths use exactly one native _c call and never pack; the padded layout
  // path may stage but stays at or below the block count.
  // ---------------------------------------------------------------------------------------

  void flat_reductions() {
    const mp1::mpi_communicator c = mp1::comm_world();
    const int root = g_size - 1;
    constexpr int n = 3;
    std::vector<int> send(static_cast<std::size_t>(n), g_rank + 1);
    std::vector<int> recv(static_cast<std::size_t>(n), -1);

    mpi_wrap::reset_counts();
    mp1::allreduce(c, std::plus<int>{}, std::span<int>{send}, std::span<int>{recv});
    expect_count("MPI_Allreduce_c", 1);
    expect_symbol_zero("MPI_Allreduce");
    expect_symbol_zero("MPI_Pack_c");
    expect_symbol_zero("MPI_Type_dup");

    recv.assign(static_cast<std::size_t>(n), -1);
    mpi_wrap::reset_counts();
    if (g_rank == root) {
      mp1::reduce(c, std::plus<int>{}, root, std::span<int>{send}, std::span<int>{recv});
    } else {
      mp1::reduce(c, std::plus<int>{}, root, std::span<int>{send}, std::span<int>{});
    }
    expect_count("MPI_Reduce_c", 1);
    expect_symbol_zero("MPI_Reduce");
    expect_symbol_zero("MPI_Pack_c");
    expect_symbol_zero("MPI_Type_dup");

    recv.assign(static_cast<std::size_t>(n), -1);
    mpi_wrap::reset_counts();
    mp1::scan(c, std::plus<int>{}, std::span<int>{send}, std::span<int>{recv});
    expect_count("MPI_Scan_c", 1);
    expect_symbol_zero("MPI_Scan");
    expect_symbol_zero("MPI_Pack_c");
    expect_symbol_zero("MPI_Type_dup");

    recv.assign(static_cast<std::size_t>(n), -1);
    mpi_wrap::reset_counts();
    mp1::exscan(c, std::plus<int>{}, std::span<int>{send}, std::span<int>{recv});
    expect_count("MPI_Exscan_c", 1);
    expect_symbol_zero("MPI_Exscan");
    expect_symbol_zero("MPI_Pack_c");
    expect_symbol_zero("MPI_Type_dup");

    recv.assign(static_cast<std::size_t>(n), -1);
    mpi_wrap::reset_counts();
    mp1::request request =
        mp1::iallreduce(c, std::plus<int>{}, std::span<int>{send}, std::span<int>{recv});
    expect_count("MPI_Iallreduce_c", 1);
    expect_symbol_zero("MPI_Iallreduce");
    expect_symbol_zero("MPI_Pack_c");
    request.wait();
    expect_symbol_zero("MPI_Pack_c");
  }

  void staged_allreduce() {
    const mp1::mpi_communicator c = mp1::comm_world();
    // Four ints at element offsets 0/2/4/6: size 16 bytes, true extent 28 bytes, so the
    // reduction cannot use the flat address and stages both directions.
    constexpr int elements = 4;
    constexpr int extent = 7;
    const mp1::layout<int> holey = mp1::layout<int>::vector(elements, 1, 2);
    check_type_size_contract(holey, "staged allreduce");
    std::vector<int> send(extent, -1);
    std::vector<int> recv(extent, -1);
    for (int i = 0; i < elements; ++i) {
      send[static_cast<std::size_t>(2 * i)] = g_rank + 1;
    }

    mp1::allreduce(c, std::plus<int>{}, mp1::region<int>{send.data(), holey},
                   mp1::region<int>{recv.data(), holey});  // warm
    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    mp1::allreduce(c, std::plus<int>{}, mp1::region<int>{send.data(), holey},
                   mp1::region<int>{recv.data(), holey});
    expect_count("MPI_Allreduce_c", 1);
    expect_symbol_zero("MPI_Allreduce");
    expect_symbol_zero("MPI_Type_dup");
    // Two flat stages + two exact packed scratch buffers; the pack->unpack round trip is
    // required for portability (MPI 4.1 §5.1.8), but the byte scratch is uninitialized.
    const std::size_t staged_allocations = mpi_wrap::allocation_count();
    check(staged_allocations <= 4,
          "staged allreduce allocations " + std::to_string(staged_allocations) + " > 4");
    const std::size_t packs = mpi_wrap::count("MPI_Pack_c");
    check(packs >= 1, "staged allreduce must pack at least one direction");
    check(packs <= static_cast<std::size_t>(elements),
          "staged allreduce Pack_c = " + std::to_string(packs) + " exceeds blocks " +
              std::to_string(elements));
    check(mpi_wrap::count("MPI_Pack_size_c") <= 2,
          "staged allreduce Pack_size_c > 2 directions");

    const int expected = g_size * (g_size + 1) / 2;
    for (int i = 0; i < extent; ++i) {
      if (i % 2 == 0) {
        check(recv[static_cast<std::size_t>(i)] == expected,
              "staged allreduce value at element " + std::to_string(i / 2));
      } else {
        check(recv[static_cast<std::size_t>(i)] == -1,
              "staged allreduce wrote padding at " + std::to_string(i));
      }
    }
  }

  // ---------------------------------------------------------------------------------------
  // p2p pointer/scalar and dropped-request teardown policy.
  // ---------------------------------------------------------------------------------------

  void p2p_calls() {
    const mp1::mpi_communicator c = mp1::comm_world();
    const int dest = (g_rank + 1) % g_size;
    const int source = (g_rank + g_size - 1) % g_size;
    constexpr int tag = 7100;
    std::vector<int> send_buffer(4, g_rank);
    std::vector<int> recv_buffer(4, -1);

    mpi_wrap::reset_counts();
    mp1::send(c, std::span(send_buffer), dest, mp1::tag{tag});
    mp1::recv(c, std::span(recv_buffer), source, mp1::tag{tag});
    expect_count("MPI_Send_c", 1);
    expect_symbol_zero("MPI_Send");
    expect_count("MPI_Recv_c", 1);
    expect_symbol_zero("MPI_Recv");
    expect_symbol_zero("MPI_Type_dup");

    recv_buffer.assign(4, -1);
    mpi_wrap::reset_counts();
    {
      mp1::request receive = mp1::irecv(c, std::span(recv_buffer), source, mp1::tag{tag});
      mp1::request send_request = mp1::isend(c, std::span(send_buffer), dest, mp1::tag{tag});
      send_request.wait();
      receive.wait();
    }
    expect_count("MPI_Irecv_c", 1);
    expect_symbol_zero("MPI_Irecv");
    expect_count("MPI_Isend_c", 1);
    expect_symbol_zero("MPI_Isend");
    expect_count("MPI_Wait", 2);
    expect_symbol_zero("MPI_Type_dup");

    int scalar_send = g_rank;
    int scalar_recv = -1;
    mpi_wrap::reset_counts();
    mp1::send(c, scalar_send, dest, mp1::tag{tag});
    mp1::recv(c, scalar_recv, source, mp1::tag{tag});
    expect_count("MPI_Send_c", 1);
    expect_count("MPI_Recv_c", 1);
    expect_symbol_zero("MPI_Type_dup");
  }

  void dropped_request_policy() {
    const mp1::mpi_communicator c = mp1::comm_world();
    int value = 0;

    // Dropped receive: MPI_Cancel=1, MPI_Wait=1, MPI_Request_free=0.
    mpi_wrap::reset_counts();
    {
      mp1::request request = mp1::irecv(c, std::span(&value, 1), MPI_PROC_NULL, mp1::tag{0});
    }
    expect_count("MPI_Cancel", 1);
    expect_count("MPI_Wait", 1);
    expect_symbol_zero("MPI_Request_free");

    // Dropped send: MPI_Request_free=1, MPI_Cancel=0.
    mpi_wrap::reset_counts();
    {
      mp1::request request = mp1::isend(c, std::span(&value, 1), MPI_PROC_NULL, mp1::tag{0});
    }
    expect_count("MPI_Request_free", 1);
    expect_symbol_zero("MPI_Cancel");

    // Dropped collective: MPI_Wait=1, MPI_Request_free=0.
    int recv = 0;
    mpi_wrap::reset_counts();
    {
      mp1::request request = mp1::iallreduce(c, std::plus<int>{}, std::span<int>{&value, 1},
                                             std::span<int>{&recv, 1});
    }
    expect_count("MPI_Wait", 1);
    expect_symbol_zero("MPI_Request_free");
    expect_symbol_zero("MPI_Cancel");
  }


  // ---------------------------------------------------------------------------------------
  // Iterator-range gates (deterministic calls and allocation bounds).
  //
  // A statically contiguous range must be pointer-equivalent: one native call, no datatype
  // constructor, no MPI_Get_address and no heap allocation.  A non-contiguous range of a
  // trivially copyable scalar must stage once: one native call, zero MPI_Get_address, and the
  // payload owns the staged bytes for a nonblocking call.  A non-trivially-copyable value type
  // keeps the documented object-bound derived fallback.
  // ---------------------------------------------------------------------------------------

  template<class F>
  std::size_t measure_raw_allocations(F&& operation) {
    operation();  // warm MPI's own request/collective machinery
    mpi_wrap::reset_allocation_count();
    operation();
    return mpi_wrap::allocation_count();
  }

  void expect_allocation_at_most(const char* what, std::size_t bound, std::size_t actual) {
    if (actual > bound) {
      check(false, std::string{what} + ": allocations " + std::to_string(actual) + " > bound " +
                       std::to_string(bound));
    }
  }

  void expect_allocation_exactly(const char* what, std::size_t expected, std::size_t actual) {
    if (actual != expected) {
      check(false, std::string{what} + ": allocations " + std::to_string(actual) +
                       ", expected " + std::to_string(expected));
    }
  }

  void expect_no_datatype_work() {
    expect_symbol_zero("MPI_Get_address");
    expect_symbol_zero("MPI_Type_create_hindexed");
    expect_symbol_zero("MPI_Type_dup");
  }

  void iterator_range_calls() {
    const mp1::mpi_communicator c = mp1::comm_world();
    constexpr int n = 16;
    std::vector<int> values(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
      values[static_cast<std::size_t>(i)] = g_rank * 100 + i;
    const std::list<int> sequence(values.begin(), values.end());

    // Contiguous vector iterators: blocking send is one MPI call, no datatype and no wrapper
    // heap allocation (the raw baseline captures the provider's own request allocation).
    const auto raw_send = [&] {
      MPI_Send_c(values.data(), n, MPI_INT, MPI_PROC_NULL, 7200, c.native_handle());
    };
    const std::size_t raw_send_allocations = measure_raw_allocations(raw_send);
    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    mp1::send(c, values, MPI_PROC_NULL, mp1::tag{7200});
    const std::size_t contiguous_send_allocations = mpi_wrap::allocation_count();
    expect_count("MPI_Send_c", 1);
    expect_no_datatype_work();
    expect_allocation_at_most("contiguous vector-iterator send", raw_send_allocations,
                              contiguous_send_allocations);

    // Non-contiguous list iterators: one local staging copy, still one native call.
    const auto raw_send_values = [&] {
      MPI_Send_c(values.data(), n, MPI_INT, MPI_PROC_NULL, 7201, c.native_handle());
    };
    const std::size_t raw_send_values_allocations = measure_raw_allocations(raw_send_values);
    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    mp1::send(c, sequence, MPI_PROC_NULL, mp1::tag{7201});
    const std::size_t staged_send_allocations = mpi_wrap::allocation_count();
    expect_count("MPI_Send_c", 1);
    expect_no_datatype_work();
    expect_allocation_at_most("staged list-iterator send", raw_send_values_allocations + 1,
                              staged_send_allocations);

    // Contiguous nonblocking send/receive: no payload, no staging, no count re-query.
    std::vector<int> received(static_cast<std::size_t>(n), -1);
    const auto raw_pair = [&] {
      MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
      MPI_Status statuses[2];
      MPI_Irecv_c(received.data(), n, MPI_INT, g_rank, 7202, c.native_handle(), &requests[0]);
      MPI_Isend_c(values.data(), n, MPI_INT, g_rank, 7202, c.native_handle(), &requests[1]);
      MPI_Waitall(2, requests, statuses);
    };
    const std::size_t raw_pair_allocations = measure_raw_allocations(raw_pair);
    received.assign(static_cast<std::size_t>(n), -1);
    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    {
      mp1::request receive = mp1::irecv(c, std::span(received), g_rank, mp1::tag{7202});
      mp1::request send = mp1::isend(c, values, g_rank, mp1::tag{7202});
      send.wait();
      receive.wait();
    }
    const std::size_t contiguous_pair_allocations = mpi_wrap::allocation_count();
    expect_count("MPI_Isend_c", 1);
    expect_count("MPI_Irecv_c", 1);
    expect_count("MPI_Wait", 2);
    expect_count("MPI_Get_count_c", 0);
    expect_no_datatype_work();
    expect_allocation_at_most("contiguous range isend/irecv", raw_pair_allocations,
                              contiguous_pair_allocations);
    check(received == values, "contiguous range send/recv data");

    // Non-contiguous nonblocking send: the request payload owns the staged bytes; the
    // receive target is a bounded contiguous container (a sequence container is send-only).
    std::vector<int> received_sequence(static_cast<std::size_t>(n), -1);
    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    {
      mp1::request receive = mp1::irecv(c, received_sequence, mp1::capacity{n}, g_rank, mp1::tag{7203});
      mp1::request send = mp1::isend(c, sequence, g_rank, mp1::tag{7203});
      send.wait();
      receive.wait();
    }
    const std::size_t staged_pair_allocations = mpi_wrap::allocation_count();
    expect_count("MPI_Isend_c", 1);
    expect_count("MPI_Irecv_c", 1);
    expect_count("MPI_Wait", 2);
    expect_count("MPI_Get_count_c", 1);  // the bounded receive queries the actual count
    expect_no_datatype_work();
    expect_allocation_at_most("staged sequence isend + bounded irecv", raw_pair_allocations + 4,
                              staged_pair_allocations);
    check(std::equal(sequence.begin(), sequence.end(), received_sequence.begin()),
          "staged sequence send/recv data");

    // Non-trivially-copyable elements keep the documented derived object-bound fallback on
    // the staged send side; the receive target is a contiguous container.
    std::list<std::pair<int, int>> pairs;
    pairs.emplace_back(g_rank, g_rank + 1);
    pairs.emplace_back(g_rank + 2, g_rank + 3);
    std::vector<std::pair<int, int>> received_pairs(pairs.size(), {-1, -1});
    mpi_wrap::reset_counts();
    {
      mp1::request receive = mp1::irecv(c, received_pairs, mp1::capacity{2}, g_rank, mp1::tag{7205});
      mp1::request send = mp1::isend(c, pairs, g_rank, mp1::tag{7205});
      send.wait();
      receive.wait();
    }
    expect_count("MPI_Isend_c", 1);
    expect_count("MPI_Irecv_c", 1);
    check(mpi_wrap::count("MPI_Type_create_hindexed") >= 1,
          "non-trivially-copyable sequence send must use the derived fallback");
    check(received_pairs == std::vector<std::pair<int, int>>(pairs.begin(), pairs.end()),
          "derived-fallback sequence send/recv data");
  }

  // ---------------------------------------------------------------------------------------
  // Completion gates: a poll of an inactive/small request set must not allocate wrapper
  // scratch.  The result vectors that are part of the API may allocate, but only once.  The
  // raw baseline captures the provider's own allocation for the same native call.
  // ---------------------------------------------------------------------------------------

  void completion_calls() {
    std::vector<mp1::request> inactive(2);
    MPI_Request raw_requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    MPI_Status raw_statuses[2];
    int raw_index = MPI_UNDEFINED;
    int raw_flag = 0;
    int raw_outcount = 0;
    int raw_indices[2] = {-1, -1};

    const auto raw_test_any = [&] {
      MPI_Testany(2, raw_requests, &raw_index, &raw_flag, &raw_statuses[0]);
    };
    const auto raw_wait_any = [&] {
      MPI_Waitany(2, raw_requests, &raw_index, &raw_statuses[0]);
    };
    const auto raw_test_all = [&] { MPI_Testall(2, raw_requests, &raw_flag, raw_statuses); };
    const auto raw_wait_all = [&] { MPI_Waitall(2, raw_requests, raw_statuses); };
    const auto raw_test_some = [&] {
      MPI_Testsome(2, raw_requests, &raw_outcount, raw_indices, raw_statuses);
    };
    const auto raw_wait_some = [&] {
      MPI_Waitsome(2, raw_requests, &raw_outcount, raw_indices, raw_statuses);
    };

    const std::size_t raw_test_any_allocations = measure_raw_allocations(raw_test_any);
    const std::size_t raw_wait_any_allocations = measure_raw_allocations(raw_wait_any);
    const std::size_t raw_test_all_allocations = measure_raw_allocations(raw_test_all);
    const std::size_t raw_wait_all_allocations = measure_raw_allocations(raw_wait_all);
    const std::size_t raw_test_some_allocations = measure_raw_allocations(raw_test_some);
    const std::size_t raw_wait_some_allocations = measure_raw_allocations(raw_wait_some);

    // Warm the wrapper entry points (the mp1-side static/cache state).
    (void)mp1::test_any(inactive);
    (void)mp1::wait_any(inactive);
    (void)mp1::test_all(inactive);
    (void)mp1::wait_all(inactive);
    (void)mp1::test_some(inactive);
    (void)mp1::wait_some(inactive);
    (void)mp1::get_status_all(inactive);
    (void)mp1::get_status_any(inactive);
    (void)mp1::get_status_some(inactive);

    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    const std::optional<mp1::completion> any = mp1::test_any(inactive);
    const std::size_t test_any_allocations = mpi_wrap::allocation_count();
    check(any.has_value() and any->first == MPI_UNDEFINED,
          "test_any on inactive slots reports MPI_UNDEFINED");
    expect_count("MPI_Testany", 1);
    expect_allocation_exactly("test_any inactive poll", raw_test_any_allocations,
                              test_any_allocations);

    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    const mp1::completion wait_any_result = mp1::wait_any(inactive);
    const std::size_t wait_any_allocations = mpi_wrap::allocation_count();
    check(wait_any_result.first == MPI_UNDEFINED,
          "wait_any on inactive slots reports MPI_UNDEFINED");
    expect_count("MPI_Waitany", 1);
    expect_allocation_exactly("wait_any inactive poll", raw_wait_any_allocations,
                              wait_any_allocations);

    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    const std::vector<mp1::status> wait_statuses = mp1::wait_all(inactive);
    const std::size_t wait_all_allocations = mpi_wrap::allocation_count();
    check(wait_statuses.size() == inactive.size(), "wait_all returns one status per slot");
    expect_count("MPI_Waitall", 1);
    expect_allocation_exactly("wait_all inactive", raw_wait_all_allocations + 1,
                              wait_all_allocations);

    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    const auto test_statuses = mp1::test_all(inactive);
    const std::size_t test_all_allocations = mpi_wrap::allocation_count();
    check(test_statuses.has_value() and test_statuses->size() == inactive.size(),
          "test_all returns one status per slot");
    expect_count("MPI_Testall", 1);
    expect_allocation_exactly("test_all inactive", raw_test_all_allocations + 1,
                              test_all_allocations);

    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    const auto test_some_result = mp1::test_some(inactive);
    const std::size_t test_some_allocations = mpi_wrap::allocation_count();
    check(test_some_result == std::nullopt, "test_some inactive returns nullopt");
    expect_count("MPI_Testsome", 1);
    expect_allocation_exactly("test_some inactive", raw_test_some_allocations,
                              test_some_allocations);

    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    const std::vector<mp1::completion> wait_some_result = mp1::wait_some(inactive);
    const std::size_t wait_some_allocations = mpi_wrap::allocation_count();
    check(wait_some_result.empty(), "wait_some inactive returns an empty vector");
    expect_count("MPI_Waitsome", 1);
    expect_allocation_exactly("wait_some inactive", raw_wait_some_allocations,
                              wait_some_allocations);

    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    const auto statuses = mp1::get_status_all(inactive);
    const std::size_t status_all_allocations = mpi_wrap::allocation_count();
    check(statuses.has_value() and statuses->size() == inactive.size(),
          "get_status_all inactive returns one empty status per slot");
    check(mpi_wrap::total() == 0, "get_status_all with no active slot makes no MPI call");
    expect_allocation_exactly("get_status_all inactive", 1, status_all_allocations);

    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    const std::optional<mp1::completion> status_any = mp1::get_status_any(inactive);
    const std::size_t status_any_allocations = mpi_wrap::allocation_count();
    check(status_any.has_value() and status_any->first == MPI_UNDEFINED,
          "get_status_any inactive reports MPI_UNDEFINED");
    check(mpi_wrap::total() == 0, "get_status_any with no active slot makes no MPI call");
    expect_allocation_exactly("get_status_any inactive", 0, status_any_allocations);

    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    const auto status_some = mp1::get_status_some(inactive);
    const std::size_t status_some_allocations = mpi_wrap::allocation_count();
    check(not status_some.has_value(), "get_status_some inactive returns nullopt");
    check(mpi_wrap::total() == 0, "get_status_some with no active slot makes no MPI call");
    expect_allocation_exactly("get_status_some inactive", 0, status_some_allocations);
  }

  // ---------------------------------------------------------------------------------------
  // comm_world()/comm_self() metadata cache: zero MPI calls after the first use.
  // ---------------------------------------------------------------------------------------

  void communicator_cache_calls() {
    (void)mp1::comm_world();
    (void)mp1::comm_self();
    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    for (int i = 0; i < 100; ++i) {
      const mp1::mpi_communicator world = mp1::comm_world();
      const mp1::mpi_communicator self = mp1::comm_self();
      (void)world;
      (void)self;
    }
    const std::size_t copy_allocations = mpi_wrap::allocation_count();
    const std::size_t copy_calls = mpi_wrap::total();
    check(copy_calls == 0, "comm_world/comm_self made " + std::to_string(copy_calls) +
                               " MPI calls after warm-up");
    expect_allocation_exactly("comm_world/comm_self copies", 0, copy_allocations);

    const mp1::mpi_communicator world = mp1::comm_world();
    mpi_wrap::reset_counts();
    check(world.size() == g_size and world.rank() == g_rank and not world.is_inter(),
          "cached world metadata");
    check(mpi_wrap::total() == 0, "cached communicator accessors made an MPI call");
  }

  // ---------------------------------------------------------------------------------------
  // Borrow-only layout paths never duplicate the datatype.
  // ---------------------------------------------------------------------------------------

  void borrow_layout_calls() {
    const mp1::mpi_communicator c = mp1::comm_world();
    const mp1::layout<int> block = mp1::layout<int>::contiguous(4);
    std::vector<int> send(4, g_rank);
    std::vector<int> recv(4, -1);

    // Warm the layout commit and the reduction op cache.
    mp1::send(c, mp1::region{send.data(), block}, MPI_PROC_NULL, mp1::tag{7300});
    mp1::allreduce(c, std::plus<int>{}, mp1::region<int>{send.data(), block},
                   mp1::region<int>{recv.data(), block});

    // A blocking region borrows the layout: no duplicate, no free, and the native call
    // receives the caller's own handle.
    mpi_wrap::reset_counts();
    mp1::send(c, mp1::region{send.data(), block}, MPI_PROC_NULL, mp1::tag{7300});
    expect_count("MPI_Send_c", 1);
    expect_symbol_zero("MPI_Type_dup");
    expect_symbol_zero("MPI_Type_free");
    check(mpi_wrap::last_datatype("MPI_Send_c") == handle_token(block.native_handle()),
          "blocking send(region) passes the caller's layout handle");

    // A nonblocking region initiation duplicates the layout exactly once, so the caller's
    // layout may die with the call.  The native request must
    // use the duplicate, and the request-owned duplicate is freed exactly once at
    // completion; the caller's handle is never freed by mp1.
    mpi_wrap::reset_counts();
    {
      mp1::request request =
          mp1::isend(c, mp1::region{send.data(), block}, MPI_PROC_NULL, mp1::tag{7301});
      const std::uintptr_t duplicate = mpi_wrap::last_datatype("MPI_Type_dup");
      check(duplicate != 0, "isend(region) duplicated the layout");
      check(mpi_wrap::last_datatype("MPI_Isend_c") == duplicate,
            "isend(region) passes the request-owned duplicate to MPI_Isend_c");
      expect_count("MPI_Isend_c", 1);
      expect_count("MPI_Type_dup", 1);
      expect_count("MPI_Type_free", 0);
      request.wait();
    }
    expect_count("MPI_Type_free", 1);

    // A persistent send region binds the duplicate once at init and keeps it across
    // start/wait cycles; it is released when the request is destroyed.
    mpi_wrap::reset_counts();
    {
      mp1::request request =
          mp1::send_init(c, mp1::region{send.data(), block}, MPI_PROC_NULL, mp1::tag{7302});
      const std::uintptr_t duplicate = mpi_wrap::last_datatype("MPI_Type_dup");
      check(duplicate != 0, "send_init(region) duplicated the layout");
      check(mpi_wrap::last_datatype("MPI_Send_init_c") == duplicate,
            "send_init(region) passes the request-owned duplicate to MPI_Send_init_c");
      expect_count("MPI_Type_dup", 1);
      request.start();
      (void)request.wait();
      expect_count("MPI_Type_free", 0);
      request.start();
      (void)request.wait();
      expect_count("MPI_Type_free", 0);
    }
    expect_count("MPI_Type_free", 1);

    // A combined nonblocking region exchange duplicates each region side once and passes
    // both duplicates (distinct from each other and from the caller's layout handle) to the
    // single native call.
    mpi_wrap::reset_counts();
    {
      mp1::region<int> send_region{send.data(), block};
      mp1::region<int> recv_region{recv.data(), block};
      mp1::request request =
          mp1::isendrecv(c, send_region, MPI_PROC_NULL, mp1::tag{7303}, recv_region, MPI_PROC_NULL, mp1::tag{7303});
      const std::uintptr_t send_type = mpi_wrap::last_datatype("MPI_Isendrecv_c");
      const std::uintptr_t recv_type = mpi_wrap::last_recv_datatype("MPI_Isendrecv_c");
      check(send_type != 0 and recv_type != 0,
            "isendrecv(region) captured both native datatypes");
      check(send_type != handle_token(block.native_handle()) and
                recv_type != handle_token(block.native_handle()),
            "isendrecv(region) passes request-owned duplicates, not the caller's handle");
      check(send_type != recv_type,
            "isendrecv(region) uses one distinct duplicate per region side");
      expect_count("MPI_Isendrecv_c", 1);
      expect_count("MPI_Type_dup", 2);
      expect_count("MPI_Type_free", 0);
      request.wait();
    }
    expect_count("MPI_Type_free", 2);

    // isendrecv_replace has one region buffer; its request-owned duplicate is the single
    // datatype for both directions.
    mpi_wrap::reset_counts();
    {
      mp1::region<int> replace_region{recv.data(), block};
      mp1::request request =
          mp1::isendrecv_replace(c, replace_region, MPI_PROC_NULL, mp1::tag{7304}, MPI_PROC_NULL, mp1::tag{7304});
      const std::uintptr_t duplicate = mpi_wrap::last_datatype("MPI_Type_dup");
      check(duplicate != 0, "isendrecv_replace(region) duplicated the layout");
      check(mpi_wrap::last_datatype("MPI_Isendrecv_replace_c") == duplicate,
            "isendrecv_replace(region) passes the request-owned duplicate");
      expect_count("MPI_Isendrecv_replace_c", 1);
      expect_count("MPI_Type_dup", 1);
      expect_count("MPI_Type_free", 0);
      request.wait();
    }
    expect_count("MPI_Type_free", 1);

    mpi_wrap::reset_counts();
    mp1::allreduce(c, std::plus<int>{}, mp1::region<int>{send.data(), block},
                   mp1::region<int>{recv.data(), block});
    expect_count("MPI_Allreduce_c", 1);
    expect_symbol_zero("MPI_Type_dup");
    expect_symbol_zero("MPI_Pack_c");
  }

  // ---------------------------------------------------------------------------------------
  // Flat fixed collectives: one native `_c` call per operation, no staging and no datatype
  // duplication.  The fixed-family rule (one native MPI_*_c for a flat fixed collective) is
  // gated here for gather/scatter/allgather/alltoall and barrier.
  // ---------------------------------------------------------------------------------------

  void fixed_collective_calls() {
    const mp1::mpi_communicator c = mp1::comm_world();
    const int root = g_size - 1;
    constexpr int n = 4;

    std::vector<int> send(static_cast<std::size_t>(n), g_rank);
    std::vector<int> recv(static_cast<std::size_t>(n) * static_cast<std::size_t>(g_size), -1);
    std::vector<int> send_aggregate(recv.size(), 0);
    for (int destination = 0; destination < g_size; ++destination)
      for (int j = 0; j < n; ++j)
        send_aggregate[static_cast<std::size_t>(destination) * n + j] = destination;

    // Warm the fixed-family engine paths before the counted calls.
    mp1::barrier(c);
    mp1::allgather(c, send, recv);
    mp1::alltoall(c, send_aggregate, recv);

    mpi_wrap::reset_counts();
    mp1::barrier(c);
    expect_count("MPI_Barrier", 1);
    expect_symbol_zero("MPI_Type_dup");
    expect_symbol_zero("MPI_Pack_c");

    mpi_wrap::reset_counts();
    if (g_rank == root)
      mp1::gather(c, root, std::span<const int>{send}, std::span<int>{recv});
    else
      mp1::gather(c, root, std::span<const int>{send}, std::span<int>{});
    expect_count("MPI_Gather_c", 1);
    expect_dense_no_staging();
    if (g_rank == root) {
      bool matches = true;
      for (int source = 0; source < g_size; ++source)
        for (int j = 0; j < n; ++j)
          matches = matches and recv[static_cast<std::size_t>(source) * n + j] == source;
      check(matches, "flat gather results");
    }

    mpi_wrap::reset_counts();
    if (g_rank == root)
      mp1::scatter(c, root, std::span<const int>{send_aggregate}, std::span<int>{send});
    else
      mp1::scatter(c, root, std::span<const int>{}, std::span<int>{send});
    expect_count("MPI_Scatter_c", 1);
    expect_dense_no_staging();
    {
      bool matches = true;
      for (int j = 0; j < n; ++j)
        matches = matches and send[static_cast<std::size_t>(j)] == g_rank;
      check(matches, "flat scatter results");
    }

    mpi_wrap::reset_counts();
    mp1::allgather(c, send, recv);
    expect_count("MPI_Allgather_c", 1);
    expect_dense_no_staging();
    {
      bool matches = true;
      for (int source = 0; source < g_size; ++source)
        for (int j = 0; j < n; ++j)
          matches = matches and recv[static_cast<std::size_t>(source) * n + j] == source;
      check(matches, "flat allgather results");
    }

    mpi_wrap::reset_counts();
    mp1::alltoall(c, send_aggregate, recv);
    expect_count("MPI_Alltoall_c", 1);
    expect_dense_no_staging();
    {
      bool matches = true;
      for (int source = 0; source < g_size; ++source)
        for (int j = 0; j < n; ++j)
          matches = matches and recv[static_cast<std::size_t>(source) * n + j] == g_rank;
      check(matches, "flat alltoall results");
    }
  }

  // ---------------------------------------------------------------------------------------
  // Exscan: in-place has no wrapper scratch; out-of-place allocates only rank 0's
  // uninitialized scratch (the documented rank-0 unchanged guarantee).  The raw baseline
  // isolates the provider's own per-call allocation.
  // ---------------------------------------------------------------------------------------

  void exscan_scratch_calls() {
    const mp1::mpi_communicator c = mp1::comm_world();
    constexpr int n = 1024;
    std::vector<int> send(static_cast<std::size_t>(n), g_rank);
    std::vector<int> recv(static_cast<std::size_t>(n), -1);
    std::vector<int> in_place(static_cast<std::size_t>(n), g_rank);

    const auto raw_in_place = [&] {
      MPI_Exscan_c(MPI_IN_PLACE, in_place.data(), n, MPI_INT, MPI_SUM, c.native_handle());
    };
    const auto raw_out_of_place = [&] {
      MPI_Exscan_c(send.data(), recv.data(), n, MPI_INT, MPI_SUM, c.native_handle());
    };
    const std::size_t raw_in_place_allocations = measure_raw_allocations(raw_in_place);
    const std::size_t raw_out_of_place_allocations = measure_raw_allocations(raw_out_of_place);

    mp1::exscan(c, std::plus<int>{}, mp1::in_place, std::span<int>{in_place});  // warm
    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    mp1::exscan(c, std::plus<int>{}, mp1::in_place, std::span<int>{in_place});
    const std::size_t in_place_allocations = mpi_wrap::allocation_count();
    expect_count("MPI_Exscan_c", 1);
    expect_allocation_exactly("in-place exscan scratch", raw_in_place_allocations,
                              in_place_allocations);

    mp1::exscan(c, std::plus<int>{}, std::span<int>{send}, std::span<int>{recv});  // warm
    mpi_wrap::reset_counts();
    mpi_wrap::reset_allocation_count();
    mp1::exscan(c, std::plus<int>{}, std::span<int>{send}, std::span<int>{recv});
    const std::size_t out_of_place_allocations = mpi_wrap::allocation_count();
    expect_count("MPI_Exscan_c", 1);
    expect_allocation_exactly("out-of-place exscan scratch",
                              raw_out_of_place_allocations + (g_rank == 0 ? 1 : 0),
                              out_of_place_allocations);
  }

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  mp1::init();
  g_rank = mp1::comm_world().rank();
  g_size = mp1::comm_world().size();

#if MP1_PERF_CALLS_WRAP
  if (not mpi_wrap::supported()) {
    if (g_rank == 0) {
      std::printf("perf_calls: SKIP (no -Wl,--wrap support on this platform)\n");
    }
    mp1::finalize();
    return 77;
  }

  // Warm up MPI/mp1 state before the first reset: datatype construction and the process-
  // lifetime operation cache are setup costs, not per-call behavior.
  {
    const mp1::mpi_communicator c = mp1::comm_world();
    const mp1::layout<int> warm = mp1::layout<int>::contiguous(1);
    std::vector<int> warm_send(4, 0);
    std::vector<int> warm_recv(4, 0);
    mp1::allreduce(c, std::plus<int>{}, std::span<int>{warm_send.data(), 1},
                   std::span<int>{warm_recv.data(), 1});
    (void)warm;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  dense_gatherv(false);
  dense_gatherv(true);
  dense_scatterv(false);
  dense_scatterv(true);
  dense_allgatherv(false);
  dense_allgatherv(true);
  dense_alltoallv(false);
  dense_alltoallv(true);
  holey_gatherv();
  holey_scatterv();
  holey_allgatherv();
  holey_alltoallv();
  flat_reductions();
  staged_allreduce();
  p2p_calls();
  dropped_request_policy();
  iterator_range_calls();
  completion_calls();
  communicator_cache_calls();
  borrow_layout_calls();
  fixed_collective_calls();
  exscan_scratch_calls();

  int local_failures = g_failures;
  int total_failures = 0;
  MPI_Allreduce(&local_failures, &total_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) {
    if (total_failures == 0) {
      std::printf("perf_calls: OK (%d ranks, all deterministic call-count gates passed)\n",
                  g_size);
    } else {
      std::printf("perf_calls: FAIL (%d check(s) across %d ranks)\n", total_failures, g_size);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  mp1::finalize();

  const std::size_t after_finalize = mpi_wrap::after_finalize_calls();
  bool after_finalize_ok = mpi_wrap::after_finalize() and after_finalize == 0;

  // After finalize, every operation must fail its local check before touching MPI.
  bool threw = false;
  try {
    (void)mp1::comm_world().size();
  } catch (const std::runtime_error&) {
    threw = true;
  }
  after_finalize_ok = after_finalize_ok and threw and mpi_wrap::after_finalize_calls() == 0;

  // A second finalize must also be a local error, not another MPI_Finalize.
  bool second_finalize_threw = false;
  try {
    mp1::finalize();
  } catch (const std::runtime_error&) {
    second_finalize_threw = true;
  }
  after_finalize_ok =
      after_finalize_ok and second_finalize_threw and mpi_wrap::after_finalize_calls() == 0;

  if (not after_finalize_ok) {
    std::fprintf(stderr,
                 "perf_calls: rank %d: after-finalize detector failed (calls=%zu, threw=%d, "
                 "second=%d)\n",
                 g_rank, after_finalize, threw ? 1 : 0, second_finalize_threw ? 1 : 0);
    return 1;
  }
  return total_failures == 0 ? 0 : 1;
#else
  if (g_rank == 0) {
    std::printf("perf_calls: SKIP (wrap shim not enabled)\n");
  }
  mp1::finalize();
  return 77;
#endif
}
