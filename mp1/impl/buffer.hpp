#if !(defined MP1_IMPL_BUFFER_HPP)

#define MP1_IMPL_BUFFER_HPP

#include "mp1/impl/communicator.hpp"
#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/environment.hpp"
#include "mp1/impl/request.hpp"

#include <utility>


namespace mp1 {

  /// MPI_Buffer_flush: block until all messages in the process-specific buffered-send buffer
  /// have been transmitted. Requires active MPI and an attached buffer (explicit or
  /// automatic).
  inline void buffer_flush() {
    detail::require_mpi("mp1::buffer_flush");
    detail::check_mpi(MPI_Buffer_flush(), {"MPI_Buffer_flush"});
  }

  /// MPI_Buffer_iflush: start a nonblocking flush of the process-specific buffer. The returned
  /// request is inactive when the implementation completes the flush immediately (Open MPI
  /// returns MPI_REQUEST_NULL); otherwise it is a wait-only request. Requires active MPI and
  /// an attached buffer.
  [[nodiscard]] inline request buffer_iflush() {
    detail::require_mpi("mp1::buffer_iflush");
    MPI_Request handle{MPI_REQUEST_NULL};
    detail::check_mpi(MPI_Buffer_iflush(&handle), {"MPI_Buffer_iflush"});
    return request::from_flush(handle);
  }

  /// MPI_Comm_attach_buffer_c: attach a communicator-specific buffered-send buffer.
  ///
  /// The communicator keeps the buffer until detach_buffer; MPI_Comm_dup does not copy it.
  /// Pass buffer_automatic (from environment.hpp) to enable automatic buffering; MPI then
  /// ignores the size, so any value (including a negative one) is accepted. A negative
  /// automatic size is normalized to 0 before the MPI call because some providers validate
  /// the otherwise-ignored argument and would reject a legal automatic attach. For an
  /// explicit buffer, `size` is in bytes and a negative size throws std::invalid_argument
  /// before any MPI call. Throws std::invalid_argument for an invalid communicator.
  inline void attach_buffer(const mpi_communicator& comm, void* buffer, count_t size) {
    detail::require_valid(comm, "mp1::attach_buffer");
    detail::require_mpi("mp1::attach_buffer");
    if (buffer == buffer_automatic) {
      // Same normalization as buffer_attach: MPI ignores the size for automatic buffering
      // (MPI 4.1 §3.6), but some providers validate it and reject a negative value.
      const MPI_Count mpi_size = size < 0 ? MPI_Count{0} : static_cast<MPI_Count>(size);
      detail::check_mpi(MPI_Comm_attach_buffer_c(comm.native_handle(), buffer, mpi_size),
                        {"MPI_Comm_attach_buffer_c"});
      return;
    }
    detail::check_mpi(
        MPI_Comm_attach_buffer_c(comm.native_handle(), buffer,
                                 detail::checked_mpi_count(size, "mp1::attach_buffer")),
        {"MPI_Comm_attach_buffer_c"});
  }

  /// MPI_Comm_detach_buffer_c: detach the communicator-specific buffer and return its address
  /// and size in bytes. May block until pending buffered sends complete. The size is reported
  /// as MPI_UNDEFINED when it cannot be represented in count_t (or, for buffer_automatic,
  /// when MPI reports an undefined size); the conversion never throws after MPI has
  /// successfully detached. If automatic buffering was attached, buffer_automatic is
  /// returned: do not use the size as an allocation size.
  [[nodiscard]] inline std::pair<void*, count_t> detach_buffer(const mpi_communicator& comm) {
    detail::require_valid(comm, "mp1::detach_buffer");
    detail::require_mpi("mp1::detach_buffer");
    void* buffer{nullptr};
    MPI_Count size{0};
    detail::check_mpi(MPI_Comm_detach_buffer_c(comm.native_handle(), &buffer, &size),
                      {"MPI_Comm_detach_buffer_c"});
    return {buffer, detail::checked_count_or_undefined(size)};
  }

  /// MPI_Comm_flush_buffer: block until all messages in the communicator-specific buffer have
  /// been transmitted without detaching it. Requires active MPI and an attached buffer.
  inline void flush_buffer(const mpi_communicator& comm) {
    detail::require_valid(comm, "mp1::flush_buffer");
    detail::require_mpi("mp1::flush_buffer");
    detail::check_mpi(MPI_Comm_flush_buffer(comm.native_handle()), {"MPI_Comm_flush_buffer"});
  }

  /// MPI_Comm_iflush_buffer: start a nonblocking flush of the communicator-specific buffer.
  /// The request is inactive if the implementation completes immediately; otherwise it is a
  /// wait-only request. Requires active MPI and an attached buffer.
  [[nodiscard]] inline request iflush_buffer(const mpi_communicator& comm) {
    detail::require_valid(comm, "mp1::iflush_buffer");
    detail::require_mpi("mp1::iflush_buffer");
    MPI_Request handle{MPI_REQUEST_NULL};
    detail::check_mpi(MPI_Comm_iflush_buffer(comm.native_handle(), &handle),
                      {"MPI_Comm_iflush_buffer"});
    return request::from_flush(handle);
  }

}  // namespace mp1

#endif
