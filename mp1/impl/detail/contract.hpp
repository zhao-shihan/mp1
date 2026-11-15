#if !(defined MP1_DETAIL_CONTRACT_HPP)

#define MP1_DETAIL_CONTRACT_HPP

// The one local-check and MPI-return-code contract of the library.
//
// Closed taxonomy: local argument misuse throws std::invalid_argument before any MPI call;
// object-state or API-policy misuse throws std::logic_error; environment/lifecycle misuse
// throws std::runtime_error (detail::require_mpi in detail/mpi_call.hpp); an MPI value that
// cannot be represented internally throws std::overflow_error (checked_count/checked_mpi_*
// in detail/mpi_call.hpp); an MPI return code other than MPI_SUCCESS throws mp1::mpi_error.
// There is no mp1 exception hierarchy beyond mpi_error.
//
// Every diagnostic is `mp1: <operation> (<file>:<line>): <message>`; the file:line is the
// check site inside the library (see mp1::fail / mp1::require / mp1::throw_invalid in
// error.hpp).  error.hpp is the MPI-version boundary for error formatting; including it here
// keeps this header self-contained without a second <mpi.h> include.
#include "mp1/impl/error.hpp"

#include <source_location>


namespace mp1::detail {

  /// Throw mp1::mpi_error iff the MPI procedure returned a non-success code.
  ///
  /// slot is -1 except for multi-completion calls (wait_any/test_any), where it names the
  /// completed slot.  The code/class/operation/slot accessors of mpi_error are MPI-free;
  /// class_name()/what() may call MPI_Error_string, which is always available (Table 11.1).
  /// The defaulted `where` is evaluated at the check's call site, so the exception reports the
  /// line that tested the MPI return code.
  inline void check_mpi(int rc, const context& ctx, int slot = -1,
                        std::source_location where = std::source_location::current()) {
    if (rc != MPI_SUCCESS)
      throw ::mp1::mpi_error{rc, ctx.operation, slot, where};
  }

}  // namespace mp1::detail

#endif
