#if !(defined MP1_TEST_SUPPORT_MPI_WRAP_HPP)
#define MP1_TEST_SUPPORT_MPI_WRAP_HPP

#include <cstddef>
#include <cstdint>

// Process-global MPI call counters used by the deterministic performance gates.
//
// The implementation in mpi_wrap.cc interposes the exact native entry points
// listed in mpi_wrap_symbols.hpp with GNU/Clang `-Wl,--wrap=`.  Counters are
// rank-local (each MPI process has its own copy).  Tests must reset the
// counters around the operation under test and assert on every rank.
//
// `count()` takes an exact symbol name, for example "MPI_Gatherv_c".  An
// unknown symbol is a test bug and throws std::invalid_argument; it is never
// silently reported as zero.  The datatype capture below is a test-only token:
// an integer copy of the handle for equality comparison, never passed to MPI
// or dereferenced.
namespace mpi_wrap {

  // True when the build used -Wl,--wrap= and the wrapper table is active.
  // Elsewhere callers should skip (return 77) rather than assert on counters.
  bool supported() noexcept;

  // Exact number of wrapped calls to `symbol` since the last reset.
  std::size_t count(const char* symbol);

  // Sum of all wrapped-call counters.
  std::size_t total() noexcept;

  // Zero every wrapped-call counter and the datatype, buffer and commute
  // captures.  Leaves the after-finalize detector and the allocation counter
  // untouched.
  void reset_counts() noexcept;

  // Last datatype handle token observed in the named wrapped call since the
  // last reset_counts().  For the p2p initiation symbols this is the datatype
  // argument; for MPI_Type_dup it is the duplicate written to the output
  // parameter.  The second accessor reports MPI_Isendrecv_c's recvtype; all
  // other symbols return 0 there.  A zero token means "not observed"; a known
  // symbol without a wired capture also returns 0.  An unknown symbol throws
  // std::invalid_argument exactly like count().
  std::uintptr_t last_datatype(const char* symbol);
  std::uintptr_t last_recv_datatype(const char* symbol);

  // Last buffer address token observed in the named wrapped p2p initiation
  // call (MPI_Send_c, MPI_Isend_c) since the last reset_counts().  The token is
  // std::uintptr_t(reinterpret_cast<void*>(buf)), so MPI_BOTTOM reads as 0 on
  // every provider that defines it as a null address; it is never passed back
  // to MPI.  A zero token means "not observed"; a known symbol without a wired
  // capture also returns 0.  An unknown symbol throws std::invalid_argument.
  std::uintptr_t last_buffer(const char* symbol);

  // Commute argument observed in the last wrapped MPI_Op_create call since the
  // last reset_counts(); -1 when the symbol has no wired capture or the call
  // has not been observed (so 0 and 1 are unambiguous).  A known symbol other
  // than MPI_Op_create returns -1; an unknown symbol throws
  // std::invalid_argument.
  int last_commute(const char* symbol);

  // True after a successful MPI_Finalize.  A wrapped call made after that
  // point is counted in after_finalize_calls() and is not forwarded to MPI
  // (it returns MPI_ERR_OTHER), so destructors can be checked safely.
  // MPI 4.1 Table 11.1 entry points (MPI_Info_*, MPI_Error_*, lifecycle
  // queries) are forwarded and not counted: they are always available.
  bool after_finalize() noexcept;
  std::size_t after_finalize_calls() noexcept;

  // Heap allocation counter (global operator new replacements in
  // mpi_wrap.cc).  Used by the allocation-bound performance gates.
  std::size_t allocation_count() noexcept;
  void reset_allocation_count() noexcept;

}  // namespace mpi_wrap

#endif
