#if !(defined MP1_IMPL_REQUEST_HPP)

#define MP1_IMPL_REQUEST_HPP

#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/detail/request_payload.hpp"
#include "mp1/impl/detail/vector.hpp"
#include "mp1/impl/status.hpp"

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace mp1 {

  class request;

  /// Request-completion result: index into the span, or MPI_UNDEFINED.
  using completion = std::pair<int, status>;
  using completions = std::vector<completion>;

  /// Operation family of a request. It is fixed by the factory that produced the handle; it is
  /// never inferred from the MPI_Request value.
  enum class request_kind : unsigned char {
    none,        ///< default, moved-from, released or broken-handle request
    send,        ///< isend/ibsend/issend/irsend, persistent or not
    receive,     ///< irecv, imrecv, recv_init
    collective,  ///< i<collective>, persistent collectives
    file,        ///< MPI_File_i*
    rma,         ///< Rput/Rget/Raccumulate/Rget_accumulate
    flush,       ///< buffer_iflush / iflush_buffer
    sendrecv     ///< isendrecv / isendrecv_replace (wait-only, not cancelable)
  };

  /// Teardown policy fixed by the factory. An inactive persistent request is always freed with
  /// MPI_Request_free regardless of this value; the policy describes the active-state action.
  enum class request_teardown : unsigned char {
    none,                  ///< no wrapped request (default/moved-from/released)
    request_free,          ///< MPI_Request_free; non-blocking, payload abandoned
    cancel_wait,           ///< MPI_Cancel + MPI_Wait; payload completed unless cancelled
    cancel_wait_and_free,  ///< MPI_Cancel + MPI_Wait + MPI_Request_free (persistent receive)
    wait,                  ///< MPI_Wait; may block
    wait_and_free          ///< MPI_Wait + MPI_Request_free (persistent collective)
  };

  /// Explicit request state. The state machine is:
  ///
  ///   freed (empty) --factory(valid, non-persistent)--> active
  ///   freed --factory(valid, persistent)--> inactive <--start--+
  ///                                            |             |
  ///                                            +--start()----+
  ///                                            |             |
  ///                                            v             |
  ///                                         active --wait/test--> completed
  ///                                            |                   |
  ///                                   payload finish       persistent: inactive
  ///                                            |            non-persistent: freed
  ///                                            v
  ///                       completing (private interval between the MPI return and the commit)
  ///
  ///   any --release()/move-from--> freed/moved_from
  ///   completion error with unknown handle state --abandon--> broken (null handle)
  ///   persistent payload completion threw --broken (valid handle, payload lost)
  ///
  /// "completing" and "completed" are transition states: no public function observes a slot in
  /// them after the completion transaction returns. "freed" means the request owns no MPI
  /// request handle (default, released, completed non-persistent). "broken" is terminal:
  /// start()/cancel() throw, wait()/test()/get_status() return the empty status without MPI,
  /// only release()/destruction are usable.
  enum class request_state : unsigned char {
    inactive,    ///< valid persistent handle, no operation in flight
    active,      ///< operation in flight
    completing,  ///< private: MPI has returned, commit is in progress
    completed,   ///< private: commit done, payload finish pending
    freed,       ///< no wrapped MPI_Request handle
    broken,      ///< terminal unusable state (payload lost or handle state unknown)
    moved_from   ///< source of a move: empty, with no metadata retained
  };

  namespace detail {

    // Internal payload attach point; public factories do not take payloads.
    void attach_payload(request& r, std::unique_ptr<request_payload> payload);

    // Grants the free completion functions access to request state.
    struct request_access;

    void note_dropped_request() noexcept;
    std::size_t dropped_request_count() noexcept;

  }  // namespace detail

  /// One request type for nonblocking and persistent operations.
  ///
  /// # State and kernel
  ///
  /// A request is exactly one of: freed (null/default/released/completed non-persistent),
  /// inactive persistent, active, completing/completed (private transition states), broken or
  /// moved-from. kind(), teardown_policy(), state(), drop_would_block(), cancelable(),
  /// is_broken() and native_handle() expose the state machine without an MPI call.
  ///
  /// # Drop table (destructor, move-assignment target, vector::erase)
  ///
  /// | kind/state                          | MPI actions                       | payload        | blocks |
  /// |-------------------------------------|-----------------------------------|----------------|--------|
  /// | invalid/released/moved-from/broken-handle | none                        | destroy        | no     |
  /// | inactive persistent (any kind)      | MPI_Request_free                  | destroy        | no     |
  /// | active send (persistent or not)     | MPI_Request_free                  | abandon/leak   | no     |
  /// | active receive                      | MPI_Cancel + MPI_Wait (+ free if persistent) | complete unless cancelled | usually bounded |
  /// | active collective (non-persistent)  | MPI_Wait                          | complete on success | can be unbounded |
  /// | active persistent collective        | MPI_Wait; MPI_Request_free only if the wait succeeded | complete on success | can be unbounded |
  /// | active file/RMA/flush               | MPI_Wait                          | destroy (none today) | yes |
  /// | active send-recv (`isendrecv*`)     | MPI_Wait                          | none           | can be unbounded |
  /// | stale active==false, non-null handle| none; abandon (leak)              | abandon/leak   | no     |
  /// | MPI inactive                        | none; abandon and count           | destroy        | no     |
  ///
  /// Data-carrying requests complete ctx they wait for: a staged receive or reduction is
  /// delivered by the destructor, not discarded. A send is freed, never waited, and its owned
  /// staging is intentionally leaked because MPI may still read it; the caller must keep a
  /// borrowed data buffer alive for the (unobservable) operation. release() is the only
  /// no-MPI-call escape: the operation continues, the completion becomes unobservable, and the
  /// handle/payload are leaked. Destructors are noexcept; a destructor can block (see
  /// drop_would_block()).
  ///
  /// # Native handle
  ///
  /// native_handle() is non-owning. Passing it to MPI_Wait/MPI_Test/MPI_Request_free detaches
  /// the operation from the wrapper; afterwards call release() (no MPI call) instead of letting
  /// the destructor run its policy on a stale handle.
  ///
  /// # MPI 4.1 references
  ///
  /// 3.7.3 (wait/test completion, empty status for null/inactive handles; MPI_Request_free of
  /// an active request is permitted only for point-to-point requests, and is erroneous for a
  /// collective/I/O/request-based-RMA request), 3.7.5 (multi-completion semantics and
  /// MPI_Waitany/MPI_Testany error/index rules), 3.7.6 (non-destructive status queries),
  /// 3.8.4 (MPI_Cancel; cancellation status via MPI_Test_cancelled), 3.9 (persistent
  /// point-to-point start/inactive/free), 6.12 (nonblocking collective completion; SOURCE/TAG
  /// undefined; REQUEST_FREE/CANCEL erroneous), 6.13 (persistent collective inactive/free/
  /// cancel rules), 12.3.5 (request-based RMA status undefined except MPI_ERROR), 14.4.1 (file
  /// status: count/elements meaningful, other fields undefined, MPI_Test_cancelled allowed).
  class [[nodiscard]] request {
  public:
    request() noexcept = default;

    request(request&& other) noexcept;
    request& operator=(request&& other) noexcept;
    request(const request&) = delete;
    request& operator=(const request&) = delete;
    ~request() noexcept;

    // Named factories. Choose the factory matching the MPI call that produced the handle.
    // `proc_null` records that MPI ignores the receive because the peer is MPI_PROC_NULL
    // (MPI 4.1 3.10); the completion status is then synthesized on every provider instead of
    // forwarding the provider's possibly non-standard field values. It is metadata, never a
    // request-state value: the handle, state machine and teardown are unchanged.
    static request from_send(MPI_Request handle) noexcept;
    static request from_recv(MPI_Request handle, bool proc_null = false) noexcept;
    static request from_persistent_send(MPI_Request handle) noexcept;
    static request from_persistent_recv(MPI_Request handle, bool proc_null = false) noexcept;
    static request from_collective(MPI_Request handle) noexcept;
    static request from_persistent_collective(MPI_Request handle) noexcept;
    static request from_file(MPI_Request handle) noexcept;
    static request from_matched_recv(MPI_Request handle, bool proc_null = false) noexcept;

    /// Adopt an MPI_Request produced by MPI_Isendrecv/MPI_Isendrecv_replace.
    ///
    /// `proc_null` records that the receive direction was posted with source =
    /// MPI_PROC_NULL, so completion returns the synthesized MPI 4.1 §3.10 status. For a real
    /// receive direction wait()/test() return the provider's status (MPI 4.1 §3.7.2 intends
    /// the receive status; some providers leave tag/count, and for an ignored direction even
    /// source, unspecified, so treat those fields as provider-defined).
    ///
    /// A send-recv request is wait-only: MPI 4.1 §3.8 forbids MPI_Cancel for it, so
    /// cancel() throws std::logic_error and cancelable() is false. The destructor performs
    /// one MPI_Wait and can block until the matching receive arrives; release() is the
    /// no-MPI-call escape.
    static request from_sendrecv(MPI_Request handle, bool proc_null = false) noexcept;

    static request from_flush(MPI_Request handle) noexcept;

    /// Adopt an MPI_Request produced by a request-based RMA call (MPI_Rput/Rget/Raccumulate/
    /// Rget_accumulate). RMA completions are wait-only: cancel() throws std::logic_error and
    /// the destructor waits. MPI_Request_free and MPI_Cancel are never called for it. The
    /// status returned by wait()/test() is normalized (source/tag/count undefined, MPI 4.1
    /// section 12.3.5); error() is preserved for array completions.
    static request from_rma(MPI_Request handle) noexcept;

    /// \return operation family; request_kind::none for default/moved-from/released/broken
    /// requests
    [[nodiscard]] request_kind kind() const noexcept;

    /// \return fixed teardown policy; request_teardown::none for requests that own no handle
    [[nodiscard]] request_teardown teardown_policy() const noexcept;

    /// \return current state (see the class comment)
    [[nodiscard]] request_state state() const noexcept;

    /// \return true if the wrapped MPI request handle is not MPI_REQUEST_NULL
    [[nodiscard]] bool is_valid() const noexcept;

    /// \return true if started/issued and not yet reaped (persistent inactive handles are
    /// valid but not active)
    [[nodiscard]] bool is_active() const noexcept;

    /// \return true if this request was created by a persistent factory
    [[nodiscard]] bool is_persistent() const noexcept;

    /// \return true if this request is terminal because a payload completion threw or MPI left
    /// the handle state unknown; is_broken() implies the request must be released/destroyed
    [[nodiscard]] bool is_broken() const noexcept;

    /// \return true if MPI_Cancel is legal for this request in its current state; false for an
    /// invalid, inactive, collective/file/RMA/flush/send-recv request
    [[nodiscard]] bool cancelable() const noexcept;

    /// \return true if destroying this request may block waiting for an operation to complete
    /// (active receive/collective/file/RMA/flush/send-recv); false for sends, inactive
    /// persistent requests and invalid requests
    [[nodiscard]] bool drop_would_block() const noexcept;

    /// \return non-owning MPI_Request handle; do not free or wait on it through raw MPI without
    /// releasing the wrapper afterwards
    [[nodiscard]] MPI_Request native_handle() const noexcept;

    /// Wait for completion; returns the empty status without an MPI call for an inactive or
    /// broken request. May block. On a per-operation error a persistent request stays valid and
    /// inactive (the payload is kept for the next start) and a non-persistent request becomes
    /// broken (the handle is abandoned).
    status wait();

    /// Test for completion; nullopt only for an active, incomplete request. May complete the
    /// request and run the payload completion. An error with flag==0 is treated as an unknown
    /// handle state: the request becomes broken.
    [[nodiscard]] std::optional<status> test();

    /// Non-reaping completion query; never changes active state and never runs the payload.
    /// Returns the empty status for an inactive/broken request.
    [[nodiscard]] std::optional<status> get_status();

    /// MPI_Cancel for an active send/receive request; no-op for an invalid
    /// (default/moved-from/released) request; throws std::logic_error for a broken request
    /// (even when its MPI_Request handle was abandoned) and for a valid non-cancelable request
    /// (inactive persistent, collective, file/RMA/flush/send-recv). Cancelling a send is
    /// deprecated by MPI 4.1 section 3.8.4 but is kept for compatibility; its effect is
    /// implementation-dependent.
    void cancel();

    /// MPI_Start for an inactive persistent request; runs the payload's prepare_start() hook
    /// first (local only). Throws std::logic_error for a broken or already-active request and
    /// std::invalid_argument for a non-persistent/invalid/released request.
    void start();

    /// No MPI call: drop the handle and payload intentionally (both leaked).
    void release() noexcept;

    explicit operator bool() const noexcept {
      return is_valid();
    }

  private:
    static request make(MPI_Request handle, request_kind kind, request_teardown policy,
                        bool persistent, request_state state, bool proc_null = false) noexcept;

    // Drop this request with its own teardown policy; used by the destructor and move
    // assignment. It may block: an active receive runs MPI_Cancel + MPI_Wait and an active
    // collective/file/RMA/flush runs MPI_Wait (drop_would_block() reports it); release() is
    // the only no-MPI-call escape.
    void drop() noexcept;

    MPI_Request h_{MPI_REQUEST_NULL};
    std::unique_ptr<detail::request_payload> payload_;
    request_kind kind_{request_kind::none};
    request_teardown td_{request_teardown::none};
    request_state state_{request_state::freed};
    bool persistent_{false};
    // True when the receive was posted with source = MPI_PROC_NULL. Completion status is then
    // the synthetic standard status (MPI 4.1 3.10), not the provider's field values.
    bool proc_null_{false};

    friend void detail::attach_payload(request& r,
                                       std::unique_ptr<detail::request_payload> payload);
    friend struct detail::request_access;
  };

  // -----------------------------------------------------------------------------------------
  // Free completion functions. std::span<request> is the MPI-array-shaped borrow; a
  // std::vector, std::array, C array or std::span converts implicitly, and std::span supplies
  // the slice spelling (subspan/first/last). All scratch allocations and validation happen
  // before the MPI call, and the API result storage is allocated before the call whenever a
  // completion can be reported; the post-MPI completion only assigns into preallocated
  // storage. Completion is a transaction:
  // preflight -> MPI -> classify return code -> noexcept commit of handle/active state ->
  // payload finish. A payload exception is caught per slot, the remaining slots are still
  // finished, and the first exception is rethrown after state is consistent.
  // Invariant: a request is committed only after MPI has confirmed that it completed (success
  // or a specific completion error); MPI_ERR_PENDING never reaches commit_completed or
  // finish_committed.
  // -----------------------------------------------------------------------------------------

  /// MPI_Waitall; returns one status per slot. MPI_ERR_IN_STATUS is not thrown: per-slot errors
  /// are returned in status.error() (MPI 4.1 3.7.5). A slot whose per-slot error is
  /// MPI_ERR_PENDING "has neither failed nor completed" (3.7.5): it is returned as a PENDING
  /// status (error() == MPI_ERR_PENDING, source/tag ANY, count/elements 0) and remains active
  /// with its handle and payload intact; call the same completion function again to finish it.
  /// Only a status whose operation completed may be committed. Empty input -> empty vector, no
  /// MPI.
  std::vector<status> wait_all(std::span<request> requests);

  /// MPI_Testall; nullopt if not all complete (state unchanged; MPI 4.1 3.7.5 says flag=false
  /// modifies no request). With flag=true every operation completed, so the PENDING rule of
  /// wait_all is defensive here; it is applied identically through the shared transaction.
  /// Empty input -> engaged empty vector, no MPI.
  [[nodiscard]] std::optional<std::vector<status>> test_all(std::span<request> requests);

  /// MPI_Waitany; {MPI_UNDEFINED, status{}} when no slot is active (empty input included). A
  /// per-operation error commits only the completed index and throws mpi_error whose context
  /// names the slot; every other slot is left untouched (MPI 4.1 3.7.5: Waitany behaves as
  /// Wait on the returned index only). The thrown code is the function return code: for a
  /// single-status completion MPI_Status.MPI_ERROR is undefined (3.7.3).
  [[nodiscard]] completion wait_any(std::span<request> requests);

  /// MPI_Testany; nullopt = nothing completed, {MPI_UNDEFINED, status{}} = no active slots. A
  /// per-operation error commits only the completed index and throws with the slot in context;
  /// the thrown code is the function return code (MPI 4.1 3.7.3).
  [[nodiscard]] std::optional<completion> test_any(std::span<request> requests);

  /// MPI_Waitsome; empty vector = no active slots (or empty input). A per-slot error is
  /// reported in its status; the completed set is committed and payloads are finished
  /// (MPI 4.1 3.7.5: IN_STATUS lists every succeeded or failed completion). A provider-reported
  /// MPI_ERR_PENDING slot is defensively left active and omitted from the result.
  [[nodiscard]] std::vector<completion> wait_some(std::span<request> requests);

  /// MPI_Testsome; nullopt = no active slots, empty vector = active but none completed. A
  /// provider-reported MPI_ERR_PENDING slot is defensively left active and omitted.
  [[nodiscard]] std::optional<std::vector<completion>> test_some(std::span<request> requests);

  // -----------------------------------------------------------------------------------------
  // Non-destructive multi-request status queries (MPI 4.1 section 3.7.6). They never
  // deallocate, inactivate or otherwise modify a request; a later test/wait/free still owns
  // the request state. The queries are implemented as one MPI_Request_get_status per active
  // slot: some providers abort inside the bulk MPI_Request_get_status_all for an active
  // persistent collective, and the per-slot procedure is portable. An inactive or
  // null slot never causes an MPI call and reports the empty status.
  // -----------------------------------------------------------------------------------------

  /// MPI_Request_get_status_all; nullopt while any active request is incomplete. Empty input
  /// -> engaged empty vector, no MPI. MPI_ERR_IN_STATUS is not thrown: per-slot errors are
  /// returned in status.error(). Implemented as one MPI_Request_get_status per active slot
  /// (MPI 4.1 3.7.6) because the per-slot call is portable.
  [[nodiscard]] std::optional<std::vector<status>> get_status_all(std::span<request> requests);

  /// MPI_Request_get_status_any; nullopt = active requests exist but none completed,
  /// {MPI_UNDEFINED, status{}} = no active slots (empty input included). The returned index
  /// addresses the original span.
  [[nodiscard]] std::optional<completion> get_status_any(std::span<request> requests);

  /// MPI_Request_get_status_some; nullopt = no active slots, empty vector = active but none
  /// completed, otherwise the completed subset. The returned indices address the original
  /// span.
  [[nodiscard]] std::optional<std::vector<completion>> get_status_some(
      std::span<request> requests);

  /// MPI_Startall; every slot must be a valid inactive persistent request, else the call throws
  /// before any prepare/MPI work (std::logic_error for a broken/active slot,
  /// std::invalid_argument otherwise) and leaves every slot unchanged. Empty input -> no-op.
  /// A failed MPI_Startall abandons every slot (counted leak) and throws.
  void start_all(std::span<request> requests);

  // -----------------------------------------------------------------------------------------
  // Implementation
  // -----------------------------------------------------------------------------------------

  namespace detail {

    inline std::atomic<std::size_t>& dropped_request_counter() noexcept {
      static std::atomic<std::size_t> counter{0};
      return counter;
    }

    inline void note_dropped_request() noexcept {
      dropped_request_counter().fetch_add(1, std::memory_order_relaxed);
    }

    inline std::size_t dropped_request_count() noexcept {
      return dropped_request_counter().load(std::memory_order_relaxed);
    }

    enum class completion_verdict : unsigned char { success, cancelled, error, pending };

    struct request_access;

    class request_batch;

  }  // namespace detail

  inline request request::make(MPI_Request handle, request_kind kind, request_teardown policy,
                               bool persistent, request_state state, bool proc_null) noexcept {
    request result;
    result.h_ = handle;
    result.kind_ = kind;
    result.td_ = policy;
    result.state_ = state;
    result.persistent_ = persistent;
    result.proc_null_ = proc_null;
    return result;
  }

  inline request request::from_send(MPI_Request handle) noexcept {
    return make(handle, request_kind::send, request_teardown::request_free, false,
                handle == MPI_REQUEST_NULL ? request_state::freed : request_state::active);
  }

  inline request request::from_recv(MPI_Request handle, bool proc_null) noexcept {
    return make(handle, request_kind::receive, request_teardown::cancel_wait, false,
                handle == MPI_REQUEST_NULL ? request_state::freed : request_state::active,
                proc_null);
  }

  inline request request::from_persistent_send(MPI_Request handle) noexcept {
    return make(handle, request_kind::send, request_teardown::request_free, true,
                request_state::inactive);
  }

  inline request request::from_persistent_recv(MPI_Request handle, bool proc_null) noexcept {
    return make(handle, request_kind::receive, request_teardown::cancel_wait_and_free, true,
                request_state::inactive, proc_null);
  }

  inline request request::from_collective(MPI_Request handle) noexcept {
    return make(handle, request_kind::collective, request_teardown::wait, false,
                handle == MPI_REQUEST_NULL ? request_state::freed : request_state::active);
  }

  inline request request::from_persistent_collective(MPI_Request handle) noexcept {
    // Persistent collectives start inactive, keep their handle across completions, and must
    // never be freed while active (MPI 4.1 section 6.13); the destructor waits then frees.
    return make(handle, request_kind::collective, request_teardown::wait_and_free, true,
                request_state::inactive);
  }

  inline request request::from_file(MPI_Request handle) noexcept {
    return make(handle, request_kind::file, request_teardown::wait, false,
                handle == MPI_REQUEST_NULL ? request_state::freed : request_state::active);
  }

  inline request request::from_matched_recv(MPI_Request handle, bool proc_null) noexcept {
    // MPI 4.1 section 3.8.3 permits cancelling a request returned by MPI_Imrecv, so
    // the matched receive uses the same cancel+wait policy as a normal receive. `proc_null`
    // records that the consumed message was MPI_MESSAGE_NO_PROC.
    return make(handle, request_kind::receive, request_teardown::cancel_wait, false,
                handle == MPI_REQUEST_NULL ? request_state::freed : request_state::active,
                proc_null);
  }

  inline request request::from_rma(MPI_Request handle) noexcept {
    // RMA requests are reaped by MPI_Wait/MPI_Test only; the wait-only policy makes cancel()
    // reject them and drop() perform exactly one MPI_Wait. Their completion status is
    // normalized (source/tag/count undefined for RMA, MPI 4.1 section 12.3.5).
    return make(handle, request_kind::rma, request_teardown::wait, false,
                handle == MPI_REQUEST_NULL ? request_state::freed : request_state::active);
  }

  inline request request::from_sendrecv(MPI_Request handle, bool proc_null) noexcept {
    // A combined send-receive is one wait-only request (MPI 4.1 §3.7.2). Cancelling it is
    // explicitly not allowed (§3.8), and MPI_Request_free is not the portable teardown, so the
    // policy is one MPI_Wait. A provider may return MPI_REQUEST_NULL when both sides are local
    // (for example both are MPI_PROC_NULL); that becomes a freed request with an empty status.
    // `proc_null` is status metadata only, so it never changes the handle, state or teardown.
    return make(handle, request_kind::sendrecv, request_teardown::wait, false,
                handle == MPI_REQUEST_NULL ? request_state::freed : request_state::active,
                proc_null);
  }

  inline request request::from_flush(MPI_Request handle) noexcept {
    // A buffer-flush request is not cancelable and MPI_Request_free is not portable for it in
    // every implementation. Wait is the portable teardown, so cancel() rejects it. A null
    // MPI_REQUEST_NULL handle (Open MPI can return it from MPI_Buffer_iflush) becomes an
    // invalid, inactive request.
    return make(handle, request_kind::flush, request_teardown::wait, false,
                handle == MPI_REQUEST_NULL ? request_state::freed : request_state::active);
  }

  inline request::request(request&& other) noexcept
      : h_{other.h_},
        payload_{std::move(other.payload_)},
        kind_{other.kind_},
        td_{other.td_},
        state_{other.state_},
        persistent_{other.persistent_},
        proc_null_{other.proc_null_} {
    other.h_ = MPI_REQUEST_NULL;
    other.payload_.reset();
    other.kind_ = request_kind::none;
    other.td_ = request_teardown::none;
    other.state_ = request_state::moved_from;
    other.persistent_ = false;
    other.proc_null_ = false;
  }

  inline request& request::operator=(request&& other) noexcept {
    if (this != &other) {
      drop();
      h_ = other.h_;
      payload_ = std::move(other.payload_);
      kind_ = other.kind_;
      td_ = other.td_;
      state_ = other.state_;
      persistent_ = other.persistent_;
      proc_null_ = other.proc_null_;
      other.h_ = MPI_REQUEST_NULL;
      other.payload_.reset();
      other.kind_ = request_kind::none;
      other.td_ = request_teardown::none;
      other.state_ = request_state::moved_from;
      other.persistent_ = false;
      other.proc_null_ = false;
    }
    return *this;
  }

  inline request::~request() noexcept {
    drop();
  }

  inline request_kind request::kind() const noexcept {
    return kind_;
  }

  inline request_teardown request::teardown_policy() const noexcept {
    return td_;
  }

  inline request_state request::state() const noexcept {
    return state_;
  }

  inline bool request::is_valid() const noexcept {
    return h_ != MPI_REQUEST_NULL;
  }

  inline bool request::is_active() const noexcept {
    return state_ == request_state::active;
  }

  inline bool request::is_persistent() const noexcept {
    return persistent_;
  }

  inline bool request::is_broken() const noexcept {
    return state_ == request_state::broken;
  }

  inline bool request::cancelable() const noexcept {
    return is_valid() and state_ == request_state::active and
           (kind_ == request_kind::send or kind_ == request_kind::receive);
  }

  inline bool request::drop_would_block() const noexcept {
    if (state_ != request_state::active)
      return false;
    switch (kind_) {
      case request_kind::receive:
      case request_kind::collective:
      case request_kind::file:
      case request_kind::rma:
      case request_kind::flush:
      case request_kind::sendrecv:
        return true;
      case request_kind::send:
      case request_kind::none:
        return false;
    }
    return false;
  }

  inline MPI_Request request::native_handle() const noexcept {
    return h_;
  }

  namespace detail {

    struct request_access {
      static MPI_Request handle(const request& r) noexcept {
        return r.h_;
      }

      static request_kind kind(const request& r) noexcept {
        return r.kind_;
      }

      static request_state state(const request& r) noexcept {
        return r.state_;
      }

      static bool active(const request& r) noexcept {
        return r.state_ == request_state::active;
      }

      static bool persistent(const request& r) noexcept {
        return r.persistent_;
      }

      static bool valid(const request& r) noexcept {
        return r.h_ != MPI_REQUEST_NULL;
      }

      static bool has_payload(const request& r) noexcept {
        return r.payload_ != nullptr;
      }

      static bool proc_null(const request& r) noexcept {
        return r.proc_null_;
      }

      static request_payload* payload(request& r) noexcept {
        return r.payload_.get();
      }

      static void clear_payload(request& r) noexcept {
        r.payload_.reset();
      }

      static void set_handle(request& r, MPI_Request handle) noexcept {
        r.h_ = handle;
      }

      static void set_state(request& r, request_state state) noexcept {
        r.state_ = state;
      }

      // Local pre-start hook (re-pack staged sends, re-prime in-place staging). May throw; the
      // caller must leave every request inactive on failure.
      static void prepare_start(request& r) {
        if (r.payload_)
          r.payload_->prepare_start();
      }

      // Noexcept commit of one completed slot: a persistent handle stays valid for the next
      // start; a non-persistent completed request owns no handle. No payload code runs here.
      static void commit_completed(request& r) noexcept {
        r.state_ = request_state::completing;
        if (not r.persistent_)
          r.h_ = MPI_REQUEST_NULL;
        r.state_ = request_state::completed;
      }

      // Abandon a slot whose MPI state is unknown: leak the payload (MPI may still reference
      // it), null the handle, and reset the provenance. The handle is counted as abandoned.
      static void abandon_slot(request& r) noexcept {
        if (r.h_ != MPI_REQUEST_NULL)
          note_abandoned_handle();
        abandon_forever(r.payload_.release());
        r.h_ = MPI_REQUEST_NULL;
        r.kind_ = request_kind::none;
        r.td_ = request_teardown::none;
        r.persistent_ = false;
        r.proc_null_ = false;
        r.state_ = request_state::broken;
      }
    };

    inline void attach_payload(request& r, std::unique_ptr<request_payload> payload) {
      r.payload_ = std::move(payload);
    }

    // Status provenance is selected from the request_kind fixed by the factory, never from
    // MPI_Status field values:
    //   collective/RMA/flush -> normalized (only MPI_ERROR, for IN_STATUS, is exposed)
    //   send                 -> count/cancellation-undefined source/tag-normalized status
    //   file                 -> count/cancellation-defined, source/tag-normalized status
    //   receive/sendrecv     -> native receive status (count/cancellation defined), except
    //                           a combined receive posted with MPI_PROC_NULL
    // A receive whose source was MPI_PROC_NULL is the exception: MPI 4.1 3.10 defines the
    // whole status, and some providers (pinned MPICH 4.3.2 for a persistent receive) leave
    // SOURCE at MPI_ANY_SOURCE, so the wrapper synthesizes the standard status instead of
    // forwarding the provider's. The mailbox metadata is set by the factory, never inferred.
    [[nodiscard]] inline status status_for(const request& r, const MPI_Status& native,
                                           bool error_defined = false) noexcept {
      if (request_access::proc_null(r))
        return status_access::make_proc_null();
      const request_kind kind = request_access::kind(r);
      if (kind == request_kind::collective or kind == request_kind::rma or
          kind == request_kind::flush)
        return status_access::make_normalized(native, error_defined);
      if (kind == request_kind::send)
        return status_access::make_send(native, error_defined);
      if (kind == request_kind::file)
        return status_access::make_file(native, error_defined);
      return status_access::make(native);
    }

    [[nodiscard]] inline bool needs_cancelled_check(const request& r) noexcept {
      return request_access::kind(r) == request_kind::receive and
             request_access::has_payload(r);
    }

    // Finish a committed slot. May throw only from MPI_Test_cancelled / payload completion;
    // the request state has already been committed, so the slot is never left active. On a
    // payload failure a persistent request becomes broken (payload destroyed, handle kept) and
    // a non-persistent one is freed. Returns the status and the first exception through
    // failure (empty when the slot finished normally).
    [[nodiscard]] inline status finish_committed(request& r, completion_verdict verdict,
                                                 const MPI_Status& native, bool error_defined,
                                                 std::exception_ptr& failure) noexcept {
      status result = status_for(r, native, error_defined);
      failure = nullptr;
      try {
        if (verdict == completion_verdict::success and request_access::has_payload(r)) {
          // MPI 4.1 3.8.4 defines only MPI_Test_cancelled for a cancelled operation, not the
          // SOURCE/TAG fields: a provider may return a non-empty-looking status (concrete
          // source/tag) with MPI_Test_cancelled true, while another may return the empty
          // status. The query is therefore required for every payload-carrying receive;
          // skipping it when the status looks non-empty would run the payload of a cancelled
          // receive and modify the caller's buffer.
          if (needs_cancelled_check(r)) {
            MPI_Status cancel_status = native;
            int cancelled{0};
            check_mpi(MPI_Test_cancelled(&cancel_status, &cancelled), "MPI_Test_cancelled");
            if (cancelled != 0)
              verdict = completion_verdict::cancelled;
          }
          if (verdict == completion_verdict::success)
            request_access::payload(r)->complete(result);
        }
      } catch (...) {
        failure = std::current_exception();
      }

      if (failure) {
        // MPI completion has already run, so destroying the payload is safe; a persistent
        // request cannot be restarted because its payload is gone.
        request_access::clear_payload(r);
        request_access::set_state(
            r, request_access::persistent(r) ? request_state::broken : request_state::freed);
        if (not request_access::persistent(r))
          request_access::set_handle(r, MPI_REQUEST_NULL);
      } else if (request_access::persistent(r)) {
        request_access::set_state(r, request_state::inactive);
      } else {
        request_access::clear_payload(r);
        request_access::set_handle(r, MPI_REQUEST_NULL);
        request_access::set_state(r, request_state::freed);
      }
      return result;
    }

    // Throw a completion error after the slot's state is made consistent. `completed` is true
    // when MPI reported that this persistent request completed (or may have completed) its
    // operation with an error; then the request stays valid/inactive with its payload for the
    // next start. Otherwise the handle state is unknown and the slot is abandoned. `code` is
    // the function return code: for a single-status completion MPI_Status.MPI_ERROR is
    // undefined (MPI 4.1 3.7.3) and must not be consulted.
    [[noreturn]] inline void throw_single_completion_error(request& r, int code,
                                                           const context& ctx, bool completed) {
      if (completed and request_access::persistent(r)) {
        request_access::set_state(r, request_state::inactive);
      } else {
        request_access::abandon_slot(r);
      }
      throw mpi_error{code, ctx.operation};
    }

    class request_batch {
    public:
      // `status_count` is the number of MPI_Status slots the native call may write:
      // Waitall/Testall/Waitsome/Testsome write one status per request slot, while
      // Waitany/Testany write exactly one status.  Bounding it keeps a poll of a large
      // request set from allocating the full status array.
      request_batch(std::span<request> requests, const context& ctx, std::size_t status_count)
          : requests_{requests},
            handles_(requests.size()),
            active_(requests.size()),
            native_(status_count) {
        count_ = checked_int(requests.size(), ctx);
        for (std::size_t i = 0; i < requests_.size(); ++i) {
          request& r = requests_[i];
          active_[i] = request_access::active(r) ? 1 : 0;
          if (active_[i] != 0) {
            any_active_ = true;
            const MPI_Request handle = request_access::handle(r);
            if (handle == MPI_REQUEST_NULL) {
              detail::fail_as<std::logic_error>(
                  ctx, std::source_location::current(),
                  "active request slot {} has a null MPI_Request handle", i);
            }
            handles_[i] = handle;
          } else {
            // An inactive persistent collective slot is not progressed by the bulk call, so
            // every inactive slot is passed as MPI_REQUEST_NULL and never touched.
            handles_[i] = MPI_REQUEST_NULL;
          }
        }
      }

      [[nodiscard]] int count() const noexcept {
        return count_;
      }

      [[nodiscard]] std::size_t size() const noexcept {
        return requests_.size();
      }

      /// True when the native call was given at least one active handle, so it can report a
      /// completion.  With no active handle MPI_Testsome/MPI_Waitsome return MPI_UNDEFINED
      /// (MPI 4.1 3.7.5), so no result storage is needed.
      [[nodiscard]] bool any_active() const noexcept {
        return any_active_;
      }

      [[nodiscard]] request& slot(std::size_t index) noexcept {
        return requests_[index];
      }

      [[nodiscard]] bool was_active(std::size_t index) const noexcept {
        return active_[index] != 0;
      }

      [[nodiscard]] MPI_Request* handles() noexcept {
        return handles_.data();
      }

      [[nodiscard]] MPI_Status* native() noexcept {
        return native_.data();
      }

      [[nodiscard]] MPI_Status& native(std::size_t index) noexcept {
        return native_[index];
      }

      /// Status slot for a Waitsome/Testsome output position (status j belongs to indices[j],
      /// not to the request-array slot).
      [[nodiscard]] MPI_Status& native_at(std::size_t position) noexcept {
        return native_[position];
      }

      void commit(std::size_t index) noexcept {
        if (active_[index] != 0)
          request_access::commit_completed(requests_[index]);
      }

      [[nodiscard]] completion_verdict verdict(std::size_t index, int rc) const noexcept {
        return verdict_for(native_[index], rc);
      }

      [[nodiscard]] completion_verdict verdict_at(std::size_t position, int rc) const noexcept {
        return verdict_for(native_[position], rc);
      }

      // Fill the caller's preflight result vector (one element per request, default
      // constructed before the MPI call) and finish every completed active payload. A payload
      // exception is recorded per slot; all remaining slots are still finished. Assigning into
      // the preallocated vector performs no allocation, so nothing between the MPI return and
      // the state/payload commit can throw for lack of memory.
      //
      // A slot MPI reports as MPI_ERR_PENDING "has neither failed nor completed" (MPI 4.1
      // 3.7.5): it is never committed or finished. The wrapper state stays active, the handle
      // stays live and the payload (which MPI may still write through) stays owned; only the
      // preallocated status entry is assigned.
      void finish_all(int rc, std::vector<status>& result, std::exception_ptr& failure) {
        failure = nullptr;
        for (std::size_t i = 0; i < requests_.size(); ++i) {
          if (not was_active(i)) {
            result[i] = status{};
            continue;
          }
          const completion_verdict slot_verdict = verdict(i, rc);
          if (slot_verdict == completion_verdict::pending) {
            result[i] = status_access::make_pending(native_[i]);
            continue;
          }
          std::exception_ptr slot_failure;
          result[i] = finish_committed(requests_[i], slot_verdict, native_[i],
                                       rc == MPI_ERR_IN_STATUS, slot_failure);
          if (slot_failure and not failure)
            failure = slot_failure;
        }
      }

      void abandon_active() noexcept {
        for (std::size_t i = 0; i < requests_.size(); ++i) {
          if (active_[i] != 0)
            request_access::abandon_slot(requests_[i]);
        }
      }

    private:
      [[nodiscard]] static completion_verdict verdict_for(const MPI_Status& native,
                                                          int rc) noexcept {
        if (rc == MPI_SUCCESS)
          return completion_verdict::success;
        if (rc == MPI_ERR_IN_STATUS) {
          // Only the per-slot MPI_ERROR field is meaningful for an array completion. A
          // PENDING slot has neither failed nor completed and must not be committed.
          if (native.MPI_ERROR == MPI_SUCCESS)
            return completion_verdict::success;
          if (native.MPI_ERROR == MPI_ERR_PENDING)
            return completion_verdict::pending;
          return completion_verdict::error;
        }
        // A function-level error: the native error fields are undefined and are not consulted.
        // The callers abandon/throwing paths never reach finish_all with such an rc; this is
        // the defensive classification only.
        return completion_verdict::error;
      }

      std::span<request> requests_;
      small_vector<MPI_Request, 32> handles_;
      small_vector<unsigned char, 32> active_;
      small_vector<MPI_Status, 8> native_;
      bool any_active_{false};
      int count_{0};
    };

  }  // namespace detail

  inline status request::wait() {
    if (state_ != request_state::active)
      return status{};
    detail::require_mpi("mp1::request::wait");

    MPI_Status native{};
    const int rc = MPI_Wait(&h_, &native);
    if (rc != MPI_SUCCESS) {
      // MPI 4.1 3.7.3: MPI_Wait takes a single status, so the error code is the function
      // result and native.MPI_ERROR is undefined.
      detail::throw_single_completion_error(*this, rc, "MPI_Wait", true);
    }

    detail::request_access::commit_completed(*this);
    std::exception_ptr failure;
    const status result = detail::finish_committed(*this, detail::completion_verdict::success,
                                                   native, false, failure);
    if (failure)
      std::rethrow_exception(failure);
    return result;
  }

  inline std::optional<status> request::test() {
    if (state_ != request_state::active)
      return status{};
    detail::require_mpi("mp1::request::test");

    MPI_Status native{};
    int flag{0};
    const int rc = MPI_Test(&h_, &flag, &native);
    if (rc != MPI_SUCCESS) {
      // MPI_Test with flag == 0 leaves the handle state unknown even for a persistent request;
      // with flag != 0 the completed persistent request can stay valid/inactive. The thrown
      // code is the function result: native.MPI_ERROR is undefined for a single-status
      // completion (MPI 4.1 3.7.3).
      const bool completed = flag != 0;
      detail::throw_single_completion_error(*this, rc, "MPI_Test", completed);
    }
    if (flag == 0)
      return std::nullopt;

    detail::request_access::commit_completed(*this);
    std::exception_ptr failure;
    const status result = detail::finish_committed(*this, detail::completion_verdict::success,
                                                   native, false, failure);
    if (failure)
      std::rethrow_exception(failure);
    return result;
  }

  inline std::optional<status> request::get_status() {
    if (state_ != request_state::active)
      return status{};
    detail::require_mpi("mp1::request::get_status");

    MPI_Status native{};
    int flag{0};
    detail::check_mpi(MPI_Request_get_status(h_, &flag, &native), "MPI_Request_get_status");
    if (flag == 0)
      return std::nullopt;
    return detail::status_for(*this, native);
  }

  inline void request::cancel() {
    // "broken" is terminal and observable even when the broken state was reached by abandoning
    // the MPI_Request handle: this must throw before the null-handle no-op check.
    if (state_ == request_state::broken) {
      detail::fail_with<std::logic_error>("mp1::request::cancel",
                                          std::source_location::current(),
                                          "the request is broken and cannot be cancelled");
    }
    // An invalid (default/moved-from/released) request is a plain null request: no-op.
    if (not is_valid())
      return;
    if (not cancelable()) {
      // Valid but not cancelable in this state: inactive persistent, collective, file/RMA/
      // flush, send-recv. State/policy misuse is std::logic_error.
      detail::fail_with<std::logic_error>("mp1::request::cancel",
                                          std::source_location::current(),
                                          "this request kind/state is not cancelable");
    }
    detail::require_mpi("mp1::request::cancel");
    detail::check_mpi(MPI_Cancel(&h_), "MPI_Cancel");
  }

  inline void request::start() {
    if (state_ == request_state::broken)
      detail::fail_with<std::logic_error>(
          "mp1::request::start", std::source_location::current(), "the request is broken");
    if (not persistent_)
      throw_invalid("mp1::request::start", "not a persistent request");
    if (not is_valid())
      throw_invalid("mp1::request::start", "invalid request");
    if (state_ == request_state::active)
      detail::fail_with<std::logic_error>(
          "mp1::request::start", std::source_location::current(), "request is already active");
    if (state_ != request_state::inactive) {
      throw_invalid("mp1::request::start", "request is not in the inactive persistent state");
    }
    detail::require_mpi("mp1::request::start");
    detail::request_access::prepare_start(*this);
    const int rc = MPI_Start(&h_);
    if (rc != MPI_SUCCESS) {
      detail::request_access::abandon_slot(*this);
      throw mpi_error{rc, "MPI_Start"};
    }
    state_ = request_state::active;
  }

  inline void request::release() noexcept {
    if (is_valid()) {
      detail::note_dropped_request();
      detail::note_abandoned_handle();
    }
    detail::abandon_forever(payload_.release());
    h_ = MPI_REQUEST_NULL;
    kind_ = request_kind::none;
    td_ = request_teardown::none;
    state_ = request_state::freed;
    persistent_ = false;
    proc_null_ = false;
  }

  inline void request::drop() noexcept {
    if (not is_valid()) {
      payload_.reset();
      kind_ = request_kind::none;
      td_ = request_teardown::none;
      state_ = request_state::freed;
      persistent_ = false;
      proc_null_ = false;
      return;
    }

    detail::note_dropped_request();

    if (not detail::mpi_active()) {
      detail::note_abandoned_handle();
      payload_.reset();
      h_ = MPI_REQUEST_NULL;
      kind_ = request_kind::none;
      td_ = request_teardown::none;
      state_ = request_state::freed;
      persistent_ = false;
      proc_null_ = false;
      return;
    }

    // Broken-payload: the handle is a valid inactive persistent request and no operation is in
    // flight, so it is safe (and required) to free it. A broken-handle has h_ == null and took
    // the early return above.
    if (state_ == request_state::broken) {
      (void)MPI_Request_free(&h_);
      payload_.reset();
      h_ = MPI_REQUEST_NULL;
      kind_ = request_kind::none;
      td_ = request_teardown::none;
      state_ = request_state::freed;
      persistent_ = false;
      proc_null_ = false;
      return;
    }

    // Inactive persistent (or completed persistent) request: MPI_Request_free is local and
    // legal; there is no operation to wait for.
    if (persistent_ and state_ != request_state::active) {
      (void)MPI_Request_free(&h_);
      payload_.reset();
      h_ = MPI_REQUEST_NULL;
      kind_ = request_kind::none;
      td_ = request_teardown::none;
      state_ = request_state::freed;
      persistent_ = false;
      proc_null_ = false;
      return;
    }

    // A non-persistent request with a valid handle is normally active. A failed completion
    // path leaves the handle state unknown; never wait/free it a second time.
    if (state_ != request_state::active) {
      detail::request_access::abandon_slot(*this);
      return;
    }

    switch (td_) {
      case request_teardown::request_free:
        // The operation may still read the borrowed buffer and owned staging; free the handle
        // and leak the payload. The caller must keep the data buffer alive.
        (void)MPI_Request_free(&h_);
        detail::abandon_forever(payload_.release());
        break;

      case request_teardown::cancel_wait:
      case request_teardown::cancel_wait_and_free: {
        MPI_Status native{};
        (void)MPI_Cancel(&h_);
        const bool wait_ok = MPI_Wait(&h_, &native) == MPI_SUCCESS;
        if (wait_ok) {
          bool cancelled{false};
          if (payload_) {
            MPI_Status cancel_status = native;
            int flag{0};
            if (MPI_Test_cancelled(&cancel_status, &flag) != MPI_SUCCESS)
              cancelled = true;  // unknown: never unpack on an unverifiable completion
            else
              cancelled = flag != 0;
          }
          if (not cancelled and payload_) {
            try {
              payload_->complete(detail::status_for(*this, native));
            } catch (...) {  // NOLINT(bugprone-empty-catch): noexcept destructor; see comment
              // Destructors are noexcept; the payload is destroyed below (MPI already
              // completed the operation, so destruction is safe).
            }
          }
          payload_.reset();
          if (td_ == request_teardown::cancel_wait_and_free)
            (void)MPI_Request_free(&h_);
        } else {
          detail::note_abandoned_handle();
          detail::abandon_forever(payload_.release());
        }
        break;
      }

      case request_teardown::wait:
      case request_teardown::wait_and_free: {
        MPI_Status native{};
        const bool wait_ok = MPI_Wait(&h_, &native) == MPI_SUCCESS;
        if (wait_ok) {
          if (payload_) {
            try {
              payload_->complete(detail::status_for(*this, native));
            } catch (...) {  // NOLINT(bugprone-empty-catch): noexcept destructor; see comment
              // Swallowed by the noexcept destructor; the operation completed, so destroying
              // the payload is safe.
            }
          }
          payload_.reset();
          if (td_ == request_teardown::wait_and_free)
            (void)MPI_Request_free(&h_);
        } else {
          // The local MPI state is unknown: null the handle (counted) and leak the payload
          // because MPI may still reference its staging.
          detail::note_abandoned_handle();
          detail::abandon_forever(payload_.release());
        }
        break;
      }

      case request_teardown::none:
      default:
        // A valid active request with no teardown policy means the request state is corrupt.
        // drop() is noexcept (destructor / noexcept move assignment); abort with a diagnostic
        // instead of silently leaking an un-reaped handle.
        std::fputs("mp1: request::drop: invalid teardown policy\n", stderr);
        std::abort();
    }

    h_ = MPI_REQUEST_NULL;
    kind_ = request_kind::none;
    td_ = request_teardown::none;
    state_ = request_state::freed;
    persistent_ = false;
    proc_null_ = false;
  }

  // -----------------------------------------------------------------------------------------
  // Free completion functions
  // -----------------------------------------------------------------------------------------

  inline std::vector<status> wait_all(std::span<request> requests) {
    const std::size_t size = requests.size();
    if (size == 0)
      return {};

    detail::require_mpi("mp1::wait_all");
    detail::request_batch batch{requests, "mp1::wait_all", size};
    std::vector<status> result(size);  // preflight: no allocation after MPI returns
    const int rc = MPI_Waitall(batch.count(), batch.handles(), batch.native());
    if (rc != MPI_SUCCESS and rc != MPI_ERR_IN_STATUS) {
      // A function-level error is not MPI_ERR_IN_STATUS: MPI does not report which slots the
      // call touched.  Inactive slots are passed to MPI as null request entries (MPI 4.1
      // §3.7.5 allows null/inactive entries), so MPI cannot have modified their wrappers;
      // abandoning only the active slots leaves an inactive persistent handle valid for a later
      // start.
      batch.abandon_active();
      throw mpi_error{rc, "mp1::wait_all: MPI_Waitall"};
    }

    // The shared transaction commits each slot that completed (success or a specific
    // completion error) and leaves an MPI_ERR_PENDING slot active. Do not commit before
    // classifying: a PENDING slot is not a completed operation.
    std::exception_ptr failure;
    batch.finish_all(rc, result, failure);
    if (failure)
      std::rethrow_exception(failure);
    return result;
  }

  inline std::optional<std::vector<status>> test_all(std::span<request> requests) {
    const std::size_t size = requests.size();
    if (size == 0)
      return std::vector<status>{};

    detail::require_mpi("mp1::test_all");
    detail::request_batch batch{requests, "mp1::test_all", size};
    std::vector<status> result(size);  // preflight: no allocation after MPI returns
    int flag{0};
    const int rc = MPI_Testall(batch.count(), batch.handles(), &flag, batch.native());
    if (flag == 0) {
      // No request is modified. MPI_ERR_IN_STATUS with flag == false is contradictory; treat
      // it as a function-level error and leave every slot untouched.
      if (rc == MPI_SUCCESS)
        return std::nullopt;
      throw mpi_error{rc, "mp1::test_all: MPI_Testall"};
    }
    if (rc != MPI_SUCCESS and rc != MPI_ERR_IN_STATUS) {
      // Same policy as wait_all: only active slots can have been touched by the call.
      batch.abandon_active();
      throw mpi_error{rc, "mp1::test_all: MPI_Testall"};
    }

    // flag == true means every operation completed (MPI 4.1 3.7.5), so the pending rule in
    // finish_all is defensive here; it is applied identically through the shared transaction.
    std::exception_ptr failure;
    batch.finish_all(rc, result, failure);
    if (failure)
      std::rethrow_exception(failure);
    return result;
  }

  inline completion wait_any(std::span<request> requests) {
    const std::size_t size = requests.size();
    if (size == 0)
      return completion{MPI_UNDEFINED, status{}};

    detail::require_mpi("mp1::wait_any");
    detail::request_batch batch{requests, "mp1::wait_any", 1};
    int index{MPI_UNDEFINED};
    MPI_Status native{};
    const int rc = MPI_Waitany(batch.count(), batch.handles(), &index, &native);

    if (rc == MPI_SUCCESS) {
      if (index == MPI_UNDEFINED)
        return completion{MPI_UNDEFINED, status{}};
      if (index < 0 or static_cast<std::size_t>(index) >= size or
          not batch.was_active(static_cast<std::size_t>(index))) {
        batch.abandon_active();
        detail::fail_with<std::logic_error>("mp1::wait_any", std::source_location::current(),
                                            "MPI returned an invalid completion index");
      }
      const std::size_t slot = static_cast<std::size_t>(index);
      batch.commit(slot);
      std::exception_ptr failure;
      status result{};
      result = detail::finish_committed(batch.slot(slot), detail::completion_verdict::success,
                                        native, false, failure);
      if (failure)
        std::rethrow_exception(failure);
      return completion{index, result};
    }

    // A single-status completion error: only the returned index was completed/deallocated.
    // Every other slot is untouched (MPI 4.1 section 3.7.5, pp. 84-85).
    if (index == MPI_UNDEFINED) {
      // Function-level error; no slot is known completed.
      throw mpi_error{rc, "mp1::wait_any: MPI_Waitany"};
    }
    if (index < 0 or static_cast<std::size_t>(index) >= size or
        not batch.was_active(static_cast<std::size_t>(index))) {
      batch.abandon_active();
      detail::fail_with<std::logic_error>("mp1::wait_any", std::source_location::current(),
                                          "MPI returned an invalid completion index");
    }

    const std::size_t slot = static_cast<std::size_t>(index);
    // MPI 4.1 3.7.3: MPI_Waitany takes a single MPI_Status, so the operation error is the
    // function return code; the status MPI_ERROR field is undefined and is not consulted.
    const int code = rc;
    batch.commit(slot);
    std::exception_ptr failure;
    status ignored{};
    (void)detail::finish_committed(batch.slot(slot), detail::completion_verdict::error, native,
                                   false, failure);
    if (failure)
      std::rethrow_exception(failure);
    throw mpi_error{code, "mp1::wait_any: MPI_Waitany", static_cast<int>(slot)};
  }

  inline std::optional<completion> test_any(std::span<request> requests) {
    const std::size_t size = requests.size();
    if (size == 0)
      return completion{MPI_UNDEFINED, status{}};

    detail::require_mpi("mp1::test_any");
    detail::request_batch batch{requests, "mp1::test_any", 1};
    int index{MPI_UNDEFINED};
    int flag{0};
    MPI_Status native{};
    const int rc = MPI_Testany(batch.count(), batch.handles(), &index, &flag, &native);

    if (rc == MPI_SUCCESS) {
      // The outputs are defined only for a successful call (MPI 4.1 9.4).
      if (flag == 0)
        return std::nullopt;  // no operation completed; no request is modified
      if (index == MPI_UNDEFINED)
        return completion{MPI_UNDEFINED, status{}};  // the list has no active handles
      if (index < 0 or static_cast<std::size_t>(index) >= size or
          not batch.was_active(static_cast<std::size_t>(index))) {
        batch.abandon_active();
        detail::fail_with<std::logic_error>("mp1::test_any", std::source_location::current(),
                                            "MPI returned an invalid completion index");
      }

      const std::size_t slot = static_cast<std::size_t>(index);
      batch.commit(slot);
      std::exception_ptr failure;
      status result{};
      result = detail::finish_committed(batch.slot(slot), detail::completion_verdict::success,
                                        native, false, failure);
      if (failure)
        std::rethrow_exception(failure);
      return completion{index, result};
    }

    // Non-success: MPI 4.1 3.7.3 and 9.4. For a single-status completion the operation error
    // is the function return code; MPI_Status.MPI_ERROR is undefined and is not consulted.
    // MPI still deallocates the request that completed, so `index` (initialized to
    // MPI_UNDEFINED by this wrapper and validated below) identifies the slot that must be
    // committed. When no index is reported the call was function-level: no slot is known
    // completed and every wrapper state is left untouched.
    if (index == MPI_UNDEFINED)
      throw mpi_error{rc, "mp1::test_any: MPI_Testany"};
    if (index < 0 or static_cast<std::size_t>(index) >= size or
        not batch.was_active(static_cast<std::size_t>(index))) {
      batch.abandon_active();
      detail::fail_with<std::logic_error>("mp1::test_any", std::source_location::current(),
                                          "MPI returned an invalid completion index");
    }

    const std::size_t slot = static_cast<std::size_t>(index);
    batch.commit(slot);
    std::exception_ptr failure;
    status ignored{};
    (void)detail::finish_committed(batch.slot(slot), detail::completion_verdict::error, native,
                                   false, failure);
    if (failure)
      std::rethrow_exception(failure);
    throw mpi_error{rc, "mp1::test_any: MPI_Testany", static_cast<int>(slot)};
  }

  inline std::vector<completion> wait_some(std::span<request> requests) {
    const std::size_t size = requests.size();
    if (size == 0)
      return {};

    detail::require_mpi("mp1::wait_some");
    detail::request_batch batch{requests, "mp1::wait_some", size};
    detail::small_vector<int, 32> indices{size};
    std::vector<completion> result;
    // Preflight the result storage before the MPI call so a bad_alloc cannot discard a
    // completion MPI already committed.  With no active handle the call must report
    // MPI_UNDEFINED and returns an empty vector, so an all-inactive poll allocates nothing.
    if (batch.any_active())
      result.reserve(size);
    int outcount{0};
    const int rc =
        MPI_Waitsome(batch.count(), batch.handles(), &outcount, indices.data(), batch.native());
    if (rc != MPI_SUCCESS and rc != MPI_ERR_IN_STATUS) {
      // Function-level error: no completion index is reported; inactive slots were passed as
      // null request entries and stay untouched, active slots are abandoned (leaked) because
      // their state is unknown.
      batch.abandon_active();
      throw mpi_error{rc, "mp1::wait_some: MPI_Waitsome"};
    }
    if (outcount == MPI_UNDEFINED) {
      // MPI_UNDEFINED means "no active handles" and is defined only for MPI_SUCCESS. An
      // MPI_ERR_IN_STATUS with an undefined outcount is contradictory: no OUT value is valid,
      // so the touched slots are unknown and the active slots are abandoned (leaked) before
      // the function-level error is reported.
      if (rc == MPI_SUCCESS)
        return {};
      batch.abandon_active();
      throw mpi_error{rc, "mp1::wait_some: MPI_Waitsome"};
    }
    if (outcount < 0 or static_cast<std::size_t>(outcount) > size) {
      batch.abandon_active();
      detail::fail_with<std::logic_error>("mp1::wait_some", std::source_location::current(),
                                          "MPI returned an invalid completion count");
    }

    for (int j = 0; j < outcount; ++j) {
      const int index = indices[static_cast<std::size_t>(j)];
      if (index < 0 or static_cast<std::size_t>(index) >= size or
          not batch.was_active(static_cast<std::size_t>(index))) {
        batch.abandon_active();
        detail::fail_with<std::logic_error>("mp1::wait_some", std::source_location::current(),
                                            "MPI returned an invalid completion index");
      }
    }

    std::exception_ptr failure;
    for (int j = 0; j < outcount; ++j) {
      const std::size_t slot = static_cast<std::size_t>(indices[static_cast<std::size_t>(j)]);
      const detail::completion_verdict slot_verdict =
          batch.verdict_at(static_cast<std::size_t>(j), rc);
      if (slot_verdict == detail::completion_verdict::pending) {
        // Defensive: a PENDING slot has not completed. Leave it active with its handle and
        // payload and omit it from the completed set.
        continue;
      }
      batch.commit(slot);
      std::exception_ptr slot_failure;
      status slot_status = detail::finish_committed(
          batch.slot(slot), slot_verdict, batch.native_at(static_cast<std::size_t>(j)),
          rc == MPI_ERR_IN_STATUS, slot_failure);
      result.emplace_back(static_cast<int>(slot), slot_status);
      if (slot_failure and not failure)
        failure = slot_failure;
    }
    if (failure)
      std::rethrow_exception(failure);
    return result;
  }

  inline std::optional<std::vector<completion>> test_some(std::span<request> requests) {
    const std::size_t size = requests.size();
    if (size == 0)
      return std::nullopt;

    detail::require_mpi("mp1::test_some");
    detail::request_batch batch{requests, "mp1::test_some", size};
    detail::small_vector<int, 32> indices{size};
    std::vector<completion> result;
    // Preflight the result storage before the MPI call so a bad_alloc cannot discard a
    // completion MPI already committed.  With no active handle the call must report
    // MPI_UNDEFINED and returns nullopt, so an all-inactive poll allocates nothing.
    if (batch.any_active())
      result.reserve(size);
    int outcount{0};
    const int rc =
        MPI_Testsome(batch.count(), batch.handles(), &outcount, indices.data(), batch.native());
    if (rc != MPI_SUCCESS and rc != MPI_ERR_IN_STATUS) {
      batch.abandon_active();
      throw mpi_error{rc, "mp1::test_some: MPI_Testsome"};
    }
    if (outcount == MPI_UNDEFINED) {
      // Only MPI_SUCCESS defines MPI_UNDEFINED, which means "no active handles";
      // MPI_ERR_IN_STATUS with an undefined outcount is contradictory: no OUT value is valid,
      // so the touched slots are unknown and the active slots are abandoned (leaked) before the
      // function-level error is reported.
      if (rc == MPI_SUCCESS)
        return std::nullopt;
      batch.abandon_active();
      throw mpi_error{rc, "mp1::test_some: MPI_Testsome"};
    }
    if (outcount < 0 or static_cast<std::size_t>(outcount) > size) {
      batch.abandon_active();
      detail::fail_with<std::logic_error>("mp1::test_some", std::source_location::current(),
                                          "MPI returned an invalid completion count");
    }
    if (outcount == 0) {
      if (rc == MPI_ERR_IN_STATUS) {
        // Contradictory: no index is reported; leave every slot untouched.
        throw mpi_error{rc, "mp1::test_some: MPI_Testsome"};
      }
      return std::vector<completion>{};
    }

    for (int j = 0; j < outcount; ++j) {
      const int index = indices[static_cast<std::size_t>(j)];
      if (index < 0 or static_cast<std::size_t>(index) >= size or
          not batch.was_active(static_cast<std::size_t>(index))) {
        batch.abandon_active();
        detail::fail_with<std::logic_error>("mp1::test_some", std::source_location::current(),
                                            "MPI returned an invalid completion index");
      }
    }

    std::exception_ptr failure;
    for (int j = 0; j < outcount; ++j) {
      const std::size_t slot = static_cast<std::size_t>(indices[static_cast<std::size_t>(j)]);
      const detail::completion_verdict slot_verdict =
          batch.verdict_at(static_cast<std::size_t>(j), rc);
      if (slot_verdict == detail::completion_verdict::pending) {
        // Defensive: a PENDING slot has not completed. Leave it active with its handle and
        // payload and omit it from the completed set.
        continue;
      }
      batch.commit(slot);
      std::exception_ptr slot_failure;
      status slot_status = detail::finish_committed(
          batch.slot(slot), slot_verdict, batch.native_at(static_cast<std::size_t>(j)),
          rc == MPI_ERR_IN_STATUS, slot_failure);
      result.emplace_back(static_cast<int>(slot), slot_status);
      if (slot_failure and not failure)
        failure = slot_failure;
    }
    if (failure)
      std::rethrow_exception(failure);
    return result;
  }

  inline std::optional<std::vector<status>> get_status_all(std::span<request> requests) {
    const std::size_t size = requests.size();
    if (size == 0)
      return std::vector<status>{};

    bool any_active{false};
    for (std::size_t i = 0; i < size; ++i) {
      if (detail::request_access::active(requests[i])) {
        any_active = true;
        break;
      }
    }
    if (not any_active)
      return std::vector<status>(size);  // every slot reports the empty status; no MPI call

    detail::require_mpi("mp1::get_status_all");
    std::vector<status> result(size);
    for (std::size_t i = 0; i < size; ++i) {
      if (not detail::request_access::active(requests[i]))
        continue;
      MPI_Status native{};
      int flag{0};
      detail::check_mpi(
          MPI_Request_get_status(detail::request_access::handle(requests[i]), &flag, &native),
          "MPI_Request_get_status");
      if (flag == 0)
        return std::nullopt;
      result[i] = detail::status_for(requests[i], native);
    }
    return result;
  }

  inline std::optional<completion> get_status_any(std::span<request> requests) {
    const std::size_t size = requests.size();
    if (size == 0)
      return completion{MPI_UNDEFINED, status{}};

    bool any_active{false};
    for (std::size_t i = 0; i < size; ++i) {
      if (detail::request_access::active(requests[i])) {
        any_active = true;
        break;
      }
    }
    if (not any_active)
      return completion{MPI_UNDEFINED, status{}};

    detail::require_mpi("mp1::get_status_any");
    for (std::size_t i = 0; i < size; ++i) {
      if (not detail::request_access::active(requests[i]))
        continue;
      MPI_Status native{};
      int flag{0};
      detail::check_mpi(
          MPI_Request_get_status(detail::request_access::handle(requests[i]), &flag, &native),
          "MPI_Request_get_status");
      if (flag != 0)
        return completion{static_cast<int>(i), detail::status_for(requests[i], native)};
    }
    return std::nullopt;
  }

  inline std::optional<std::vector<completion>> get_status_some(std::span<request> requests) {
    const std::size_t size = requests.size();
    if (size == 0)
      return std::nullopt;

    bool any_active{false};
    for (std::size_t i = 0; i < size; ++i) {
      if (detail::request_access::active(requests[i])) {
        any_active = true;
        break;
      }
    }
    if (not any_active)
      return std::nullopt;

    detail::require_mpi("mp1::get_status_some");
    std::vector<completion> result;
    for (std::size_t i = 0; i < size; ++i) {
      if (not detail::request_access::active(requests[i]))
        continue;
      MPI_Status native{};
      int flag{0};
      detail::check_mpi(
          MPI_Request_get_status(detail::request_access::handle(requests[i]), &flag, &native),
          "MPI_Request_get_status");
      if (flag != 0)
        result.emplace_back(static_cast<int>(i), detail::status_for(requests[i], native));
    }
    return result;
  }

  inline void start_all(std::span<request> requests) {
    if (requests.empty())
      return;

    // All-or-nothing local validation before prepare_start or any MPI call. A broken or
    // already-active slot is state/policy misuse (std::logic_error); a non-persistent or
    // invalid/released slot is std::invalid_argument.
    for (std::size_t i = 0; i < requests.size(); ++i) {
      request& r = requests[i];
      if (detail::request_access::state(r) == request_state::broken)
        detail::fail_as<std::logic_error>({"mp1::start_all"}, std::source_location::current(),
                                          "request slot {} is broken", i);
      if (detail::request_access::state(r) == request_state::active)
        detail::fail_as<std::logic_error>({"mp1::start_all"}, std::source_location::current(),
                                          "request slot {} is already active", i);
      if (not detail::request_access::persistent(r))
        detail::fail_as<std::invalid_argument>(
            {"mp1::start_all"}, std::source_location::current(),
            "every request must be a persistent request (slot {})", i);
      if (not detail::request_access::valid(r))
        detail::fail_as<std::invalid_argument>({"mp1::start_all"},
                                               std::source_location::current(),
                                               "request slot {} is invalid", i);
      if (detail::request_access::state(r) != request_state::inactive)
        detail::fail_as<std::invalid_argument>({"mp1::start_all"},
                                               std::source_location::current(),
                                               "request slot {} is not inactive", i);
    }

    detail::require_mpi("mp1::start_all");

    // Every payload prepares locally before MPI sees a single start; if one throws, no request
    // has been started and the whole set stays inactive.
    for (auto& request : requests)
      detail::request_access::prepare_start(request);

    const int count = detail::checked_int(requests.size(), "mp1::start_all");
    detail::small_vector<MPI_Request, 32> handles{requests.size()};
    for (std::size_t i = 0; i < requests.size(); ++i)
      handles[i] = detail::request_access::handle(requests[i]);

    const int rc = MPI_Startall(count, handles.data());
    if (rc != MPI_SUCCESS) {
      for (auto& request : requests)
        detail::request_access::abandon_slot(request);
      throw mpi_error{rc, "mp1::start_all: MPI_Startall"};
    }

    // A persistent handle is valid by definition across MPI_Startall; never copy the output
    // array back. Inactive slots were already substituted with their original handle above
    // (all slots of a start_all are persistent/inactive by construction).
    for (auto& request : requests)
      detail::request_access::set_state(request, request_state::active);
  }

}  // namespace mp1

#endif
