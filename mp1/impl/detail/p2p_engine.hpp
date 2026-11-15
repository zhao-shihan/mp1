#if !(defined MP1_DETAIL_P2P_ENGINE_HPP)

#define MP1_DETAIL_P2P_ENGINE_HPP

#include "mp1/impl/datatype.hpp"
#include "mp1/impl/detail/buffer.hpp"
#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/detail/request_payload.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>


namespace mp1::detail {

  /// Normalize the completion status of a receive or probe (MPI 4.1 §3.10).
  ///
  /// A receive or probe (blocking or nonblocking) with source = MPI_PROC_NULL succeeds
  /// without modifying the receive buffer, and MPI defines the whole status object: source =
  /// MPI_PROC_NULL, tag = MPI_ANY_TAG, count = 0. The helper also selects the receive side of
  /// a combined send-receive. Synthesize that status with make_proc_null() instead of
  /// forwarding the provider's envelope, whose count field is not guaranteed to be configured
  /// for an MPI count query. A receive from a real source keeps the native, count-defining
  /// status. The source argument is the value passed to the MPI call, never a status field.
  [[nodiscard]] inline status p2p_recv_status(int source, const MPI_Status& native) noexcept {
    if (source != MPI_PROC_NULL)
      return status_access::make(native);
    return status_access::make_proc_null();
  }

  // The four blocking/nonblocking/persistent send families. The explicit values index the
  // consteval tables below, so the enum and the tables cannot drift silently.
  enum class send_mode : unsigned char {
    standard = 0,
    buffered = 1,
    synchronous = 2,
    ready = 3
  };

  // An out-of-range send_mode is rejected by the static_assert in every dispatch below (the
  // fixed underlying type keeps the value well-defined), so the static_assert is the only
  // rejection point. The enumerator values index the consteval tables below.

  namespace send_mode_detail {

    // Two consteval tables replace the three per-mode name chains and every per-mode native
    // call chain. One row per mode; the row index is the enumerator value.
    // The native entries are the MPI 4.1 large-count C bindings, whose signatures are fixed by
    // the standard, so the table is the single spelling of mode -> MPI entry point.
    struct name_entry {
      const char* send;
      const char* isend;
      const char* send_init;
    };

    using blocking_call = int (*)(const void*, MPI_Count, MPI_Datatype, int, int, MPI_Comm);
    using request_call = int (*)(const void*, MPI_Count, MPI_Datatype, int, int, MPI_Comm,
                                 MPI_Request*);

    struct call_entry {
      blocking_call send;
      request_call isend;
      request_call send_init;
    };

    [[nodiscard]] consteval std::array<name_entry, 4> make_name_table() noexcept {
      return {{
          {"MPI_Send_c", "MPI_Isend_c", "MPI_Send_init_c"},
          {"MPI_Bsend_c", "MPI_Ibsend_c", "MPI_Bsend_init_c"},
          {"MPI_Ssend_c", "MPI_Issend_c", "MPI_Ssend_init_c"},
          {"MPI_Rsend_c", "MPI_Irsend_c", "MPI_Rsend_init_c"},
      }};
    }

    [[nodiscard]] consteval std::array<call_entry, 4> make_call_table() noexcept {
      return {{
          {MPI_Send_c, MPI_Isend_c, MPI_Send_init_c},
          {MPI_Bsend_c, MPI_Ibsend_c, MPI_Bsend_init_c},
          {MPI_Ssend_c, MPI_Issend_c, MPI_Ssend_init_c},
          {MPI_Rsend_c, MPI_Irsend_c, MPI_Rsend_init_c},
      }};
    }

    inline constexpr auto names = make_name_table();
    inline constexpr auto calls = make_call_table();

    template<send_mode M>
    [[nodiscard]] constexpr std::size_t position() noexcept {
      return static_cast<std::size_t>(M);
    }

  }  // namespace send_mode_detail

  template<send_mode M>
  [[nodiscard]] constexpr const char* send_mode_name() noexcept {
    static_assert(M == send_mode::standard or M == send_mode::buffered or
                      M == send_mode::synchronous or M == send_mode::ready,
                  "mp1: send_mode_name: send_mode must be standard, buffered, synchronous, or "
                  "ready");
    return send_mode_detail::names[send_mode_detail::position<M>()].send;
  }

  template<send_mode M>
  [[nodiscard]] constexpr const char* isend_mode_name() noexcept {
    static_assert(M == send_mode::standard or M == send_mode::buffered or
                      M == send_mode::synchronous or M == send_mode::ready,
                  "mp1: isend_mode_name: send_mode must be standard, buffered, synchronous, "
                  "or ready");
    return send_mode_detail::names[send_mode_detail::position<M>()].isend;
  }

  template<send_mode M>
  [[nodiscard]] constexpr const char* send_init_mode_name() noexcept {
    static_assert(M == send_mode::standard or M == send_mode::buffered or
                      M == send_mode::synchronous or M == send_mode::ready,
                  "mp1: send_init_mode_name: send_mode must be standard, buffered, "
                  "synchronous, or ready");
    return send_mode_detail::names[send_mode_detail::position<M>()].send_init;
  }

  template<send_mode M>
  inline void native_send(const buffer_view& buffer, int dest, int tag, MPI_Comm comm) {
    static_assert(M == send_mode::standard or M == send_mode::buffered or
                      M == send_mode::synchronous or M == send_mode::ready,
                  "mp1: native_send: send_mode must be standard, buffered, synchronous, or "
                  "ready");
    const auto& entry = send_mode_detail::calls[send_mode_detail::position<M>()];
    check_mpi(entry.send(buffer.data, buffer.count, buffer.type, dest, tag, comm),
              send_mode_detail::names[send_mode_detail::position<M>()].send);
  }

  template<send_mode M>
  [[nodiscard]] inline request native_isend(const buffer_view& buffer, int dest, int tag,
                                            MPI_Comm comm,
                                            std::unique_ptr<request_payload> payload) {
    static_assert(M == send_mode::standard or M == send_mode::buffered or
                      M == send_mode::synchronous or M == send_mode::ready,
                  "mp1: native_isend: send_mode must be standard, buffered, synchronous, or "
                  "ready");
    MPI_Request handle{MPI_REQUEST_NULL};
    const auto& entry = send_mode_detail::calls[send_mode_detail::position<M>()];
    check_mpi(entry.isend(buffer.data, buffer.count, buffer.type, dest, tag, comm, &handle),
              send_mode_detail::names[send_mode_detail::position<M>()].isend);
    request result = request::from_send(handle);
    attach_payload(result, std::move(payload));
    return result;
  }

  [[nodiscard]] inline status native_sendrecv(const buffer_view& send_buffer, int dest,
                                              int send_tag,
                                              const mutable_buffer_view& recv_buffer,
                                              int source, int recv_tag, MPI_Comm comm) {
    MPI_Status native{};
    check_mpi(MPI_Sendrecv_c(send_buffer.data, send_buffer.count, send_buffer.type, dest,
                             send_tag, recv_buffer.data, recv_buffer.count, recv_buffer.type,
                             source, recv_tag, comm, &native),
              "MPI_Sendrecv_c");
    return p2p_recv_status(source, native);
  }

  [[nodiscard]] inline status native_sendrecv_replace(const mutable_buffer_view& buffer,
                                                      int dest, int send_tag, int source,
                                                      int recv_tag, MPI_Comm comm) {
    MPI_Status native{};
    check_mpi(MPI_Sendrecv_replace_c(buffer.data, buffer.count, buffer.type, dest, send_tag,
                                     source, recv_tag, comm, &native),
              "MPI_Sendrecv_replace_c");
    return p2p_recv_status(source, native);
  }

  // MPI 4.1 §3.7.2: one native call produces one request that completes when both directions
  // have completed. The wrapper never decomposes it into MPI_Isend + MPI_Irecv: the combined
  // request is the documented one-call path, and MPI_Cancel is not allowed on it (§3.8).
  [[nodiscard]] inline request native_isendrecv(
      const buffer_view& send_buffer, int dest, int send_tag,
      const mutable_buffer_view& recv_buffer, int source, int recv_tag, MPI_Comm comm,
      std::unique_ptr<request_payload> payload = nullptr) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Isendrecv_c(send_buffer.data, send_buffer.count, send_buffer.type, dest,
                              send_tag, recv_buffer.data, recv_buffer.count, recv_buffer.type,
                              source, recv_tag, comm, &handle),
              "MPI_Isendrecv_c");
    // The receive side of a combined request follows the same MPI 4.1 3.10 status rule as a
    // plain receive: record the requested source so completion normalizes MPI_PROC_NULL.
    request result = request::from_sendrecv(handle, source == MPI_PROC_NULL);
    attach_payload(result, std::move(payload));
    return result;
  }

  [[nodiscard]] inline request native_isendrecv_replace(
      const mutable_buffer_view& buffer, int dest, int send_tag, int source, int recv_tag,
      MPI_Comm comm, std::unique_ptr<request_payload> payload = nullptr) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Isendrecv_replace_c(buffer.data, buffer.count, buffer.type, dest, send_tag,
                                      source, recv_tag, comm, &handle),
              "MPI_Isendrecv_replace_c");
    // The single in-place buffer is the receive side of the combined request, so a requested
    // source of MPI_PROC_NULL selects the synthetic 3.10 status like the two-buffer form.
    request result = request::from_sendrecv(handle, source == MPI_PROC_NULL);
    attach_payload(result, std::move(payload));
    return result;
  }

  template<send_mode M>
  [[nodiscard]] inline request native_send_init(
      const buffer_view& buffer, int dest, int tag, MPI_Comm comm,
      std::unique_ptr<request_payload> payload = nullptr) {
    static_assert(M == send_mode::standard or M == send_mode::buffered or
                      M == send_mode::synchronous or M == send_mode::ready,
                  "mp1: native_send_init: send_mode must be standard, buffered, synchronous, "
                  "or ready");
    MPI_Request handle{MPI_REQUEST_NULL};
    const auto& entry = send_mode_detail::calls[send_mode_detail::position<M>()];
    check_mpi(entry.send_init(buffer.data, buffer.count, buffer.type, dest, tag, comm, &handle),
              send_mode_detail::names[send_mode_detail::position<M>()].send_init);
    request result = request::from_persistent_send(handle);
    attach_payload(result, std::move(payload));
    return result;
  }

  [[nodiscard]] inline status native_recv(const mutable_buffer_view& buffer, int source,
                                          int tag, MPI_Comm comm) {
    MPI_Status native{};
    check_mpi(MPI_Recv_c(buffer.data, buffer.count, buffer.type, source, tag, comm, &native),
              "MPI_Recv_c");
    // MPI 4.1 3.10 defines the status of a source == MPI_PROC_NULL receive; a real receive
    // keeps its native, count-defining status.
    return p2p_recv_status(source, native);
  }

  [[nodiscard]] inline request native_irecv(const mutable_buffer_view& buffer, int source,
                                            int tag, MPI_Comm comm,
                                            std::unique_ptr<request_payload> payload) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Irecv_c(buffer.data, buffer.count, buffer.type, source, tag, comm, &handle),
              "MPI_Irecv_c");
    // MPI 4.1 3.10 defines the status of a source == MPI_PROC_NULL receive; record it so the
    // wrapper synthesizes that status instead of forwarding a provider-specific envelope.
    request result = request::from_recv(handle, source == MPI_PROC_NULL);
    attach_payload(result, std::move(payload));
    return result;
  }

  [[nodiscard]] inline request native_recv_init(
      const mutable_buffer_view& buffer, int source, int tag, MPI_Comm comm,
      std::unique_ptr<request_payload> payload = nullptr) {
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(
        MPI_Recv_init_c(buffer.data, buffer.count, buffer.type, source, tag, comm, &handle),
        "MPI_Recv_init_c");
    request result = request::from_persistent_recv(handle, source == MPI_PROC_NULL);
    attach_payload(result, std::move(payload));
    return result;
  }

  [[nodiscard]] inline status native_mrecv(const mutable_buffer_view& buffer,
                                           MPI_Message& message) {
    MPI_Status native{};
    check_mpi(MPI_Mrecv_c(buffer.data, buffer.count, buffer.type, &message, &native),
              "MPI_Mrecv_c");
    return status_access::make(native);
  }

  [[nodiscard]] inline request native_imrecv(const mutable_buffer_view& buffer,
                                             MPI_Message& message,
                                             std::unique_ptr<request_payload> payload) {
    // MPI_Imrecv consumes the handle (it is set to MPI_MESSAGE_NULL), so capture the
    // MPI_MESSAGE_NO_PROC fact before the call: MPI 4.1 3.10 defines that completion status,
    // and a provider may not fill its SOURCE field with MPI_PROC_NULL.
    const bool no_proc = message == MPI_MESSAGE_NO_PROC;
    MPI_Request handle{MPI_REQUEST_NULL};
    check_mpi(MPI_Imrecv_c(buffer.data, buffer.count, buffer.type, &message, &handle),
              "MPI_Imrecv_c");
    request result = request::from_matched_recv(handle, no_proc);
    attach_payload(result, std::move(payload));
    return result;
  }

  // ---------------------------------------------------------------------------------------
  // Loss-free consumption of a matched message.
  //
  // A message handle returned by MPI_MPROBE/MPI_IMPROBE has no free operation: it must be
  // received with MPI_MRECV or MPI_IMRECV, and no later probe or receive can match it again
  // (MPI 4.1 §3.8.2).  If mp1 must fail after the match (a container type mismatch, an
  // unrepresentable element count, a failed resize, an allocation failure), dropping the
  // handle strands the matched message; a synchronous-mode peer then never completes.
  //
  // There is no receive that ignores the message's type signature without a datatype, but MPI
  // 4.1 §5.2 explicitly relaxes type matching for MPI_PACKED: "A message sent with any type
  // (including MPI_PACKED) can be received using the type MPI_PACKED."  The byte length comes
  // from the probe status with MPI_Get_count_c(status, MPI_PACKED): MPI 4.1 §3.2.5 requires the
  // datatype argument of MPI_GET_COUNT to match the receive, and its rationale allows, for a
  // probe status, the same datatypes as a receive of that message; §5.2 makes MPI_PACKED such a
  // datatype for every message.  A same-length MPI_PACKED receive therefore consumes the
  // message without a type-signature error.  (MPI_BYTE would *not* be portable here: §3.3.1
  // Example 3.3 is explicit that an MPI_INT message received as MPI_BYTE is erroneous.)
  //
  // The exact-length path needs a scratch buffer of the message's packed size.  If that
  // allocation itself fails (out of memory), the fallback is a deliberately truncated
  // MPI_PACKED receive: MPI 4.1 §3.2.4 makes an overflow an error for an erroneous program and
  // leaves the handle state implementation-defined, so it is a best-effort last resort that
  // keeps a synchronous sender from hanging rather than stranding the message.  This is the
  // documented OOM-only unrecoverable path: no error handler is installed or manipulated here.
  // MPI handles the overflow with the error handler of the communicator used for the matching
  // probe (MPI 4.1 §3.8.3), not necessarily MPI_COMM_WORLD; install_recoverable_errors() sets
  // only WORLD and SELF.  A communicator duplicated from WORLD after that install inherits its
  // returning handler at duplication time (§7.4.2); one created or duplicated before it, or
  // duplicated from a still-fatal parent, needs its own set_error_handler.  Under
  // a fatal handler the truncation aborts the job; with a returning handler this function
  // throws mpi_error.  message::discard() surfaces that mpi_error, whereas the auto-size recv
  // guard (and the message destructor) suppress it and let the original
  // invalid_argument/bad_alloc (or nothing, in the destructor) continue, counting the handle.
  // Applications that must survive OOM should use a returning handler or the
  // mprobe + container mrecv protocol, which validates before consuming.  The handle is
  // invalidated after every attempt and never received twice.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] inline MPI_Count matched_message_bytes(const MPI_Status& probed) noexcept {
    MPI_Count bytes{static_cast<MPI_Count>(-1)};
    if (MPI_Get_count_c(&probed, MPI_PACKED, &bytes) != MPI_SUCCESS)
      return static_cast<MPI_Count>(-1);
    return bytes;
  }

  /// Consume a matched message without interpreting its type or data.
  ///
  /// On return `message` is MPI_MESSAGE_NULL.  Throws mpi_error if the MPI receive itself
  /// fails (the message is still invalidated: MPI leaves the handle state implementation-
  /// defined after an erroneous receive, so a retry is never attempted).  If the exact-size
  /// scratch allocation fails, the best-effort truncated fallback either consumes a message
  /// that fits the small stack buffer or reports its own truncation error; it is the
  /// documented OOM-only path and installs no error handler (MPI 4.1 §3.2.4).  The overflow is
  /// handled by the matching probe communicator's handler (§3.8.3): with a returning handler
  /// this function throws mpi_error; message::discard() lets that reach the caller, while the
  /// auto-size recv guard and the message destructor call the noexcept wrapper below, which
  /// suppresses it and counts the handle.
  inline void consume_matched_message(MPI_Message& message, const MPI_Status& probed,
                                      const context& ctx) {
    if (message == MPI_MESSAGE_NULL)
      return;
    if (message == MPI_MESSAGE_NO_PROC) {
      // MPI 4.1 §3.8.2: it is not necessary to receive MPI_MESSAGE_NO_PROC.
      message = MPI_MESSAGE_NULL;
      return;
    }
    if (not mpi_active())
      detail::fail_with<std::runtime_error>(ctx.operation, std::source_location::current(),
                                            "MPI is not active");

    const MPI_Count raw_bytes = matched_message_bytes(probed);
    const bool exact = raw_bytes >= 0 and
                       static_cast<std::uintmax_t>(raw_bytes) <=
                           static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max());
    MPI_Status discarded{};
    int rc = MPI_SUCCESS;
    try {
      std::unique_ptr<unsigned char[]> scratch;
      if (exact and raw_bytes > 0)
        scratch = std::make_unique<unsigned char[]>(static_cast<std::size_t>(raw_bytes));
      const MPI_Count count = exact ? raw_bytes : static_cast<MPI_Count>(0);
      rc = MPI_Mrecv_c(scratch.get(), count, MPI_PACKED, &message, &discarded);
    } catch (const std::bad_alloc&) {
      // OOM-only fallback (documented, not recoverable in general): the typed container could
      // not be resized and even the exact-size byte scratch could not be allocated.  Consume
      // the message with a deliberately too-small MPI_PACKED buffer so a synchronous sender can
      // finish instead of hanging.  MPI 4.1 §5.2 makes MPI_PACKED a legal receive for every
      // message; if the message does not fit this buffer the receive is a §3.2.4 overflow,
      // i.e. an erroneous program.  MPI applies the error handler of the communicator used for
      // the matching probe (§3.8.3): a fatal handler aborts here, while a returning handler
      // makes the call return an error and this function throws mpi_error.  No error handler is
      // installed or changed by mp1.  The noexcept guard/destructor path below catches that
      // mpi_error and abandons the handle instead of letting it escape.
      unsigned char dummy[1] = {};
      rc = MPI_Mrecv_c(dummy, 1, MPI_PACKED, &message, &discarded);
    } catch (...) {
      message = MPI_MESSAGE_NULL;  // never retry: the matched handle is single-use
      throw;
    }
    if (rc != MPI_SUCCESS) {
      message = MPI_MESSAGE_NULL;
      throw mpi_error{rc, std::format("{}: MPI_Mrecv_c (MPI_PACKED discard)", ctx.operation)};
    }
    message = MPI_MESSAGE_NULL;
  }

  /// Destructor-safe wrapper: no MPI when the environment is gone, and never throws.
  inline void consume_matched_message_noexcept(MPI_Message& message,
                                               const MPI_Status& probed) noexcept {
    if (message == MPI_MESSAGE_NULL)
      return;
    if (message == MPI_MESSAGE_NO_PROC) {
      message = MPI_MESSAGE_NULL;
      return;
    }
    if (not mpi_active()) {
      note_abandoned_handle();
      message = MPI_MESSAGE_NULL;
      return;
    }
    try {
      consume_matched_message(message, probed, "mp1::message");
    } catch (...) {  // NOLINT(bugprone-empty-catch): noexcept destructor path; counted below
      note_abandoned_handle();
      message = MPI_MESSAGE_NULL;
    }
  }

}  // namespace mp1::detail

#endif
