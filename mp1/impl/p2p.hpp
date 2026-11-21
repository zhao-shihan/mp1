#if !(defined MP1_IMPL_P2P_HPP)

#define MP1_IMPL_P2P_HPP

#include "mp1/impl/communicator.hpp"
#include "mp1/impl/detail/buffer.hpp"
#include "mp1/impl/detail/layout_lowering.hpp"
#include "mp1/impl/detail/p2p_engine.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/status.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace mp1::detail {

  // ---------------------------------------------------------------------------------------
  // Always-on p2p envelope validation.
  // ---------------------------------------------------------------------------------------

  inline void p2p_require_comm(const mpi_communicator& c, const context& ctx) {
    require_valid(c, ctx);
    require_mpi(ctx);
  }

  [[nodiscard]] inline int p2p_peer_count(const mpi_communicator& c) {
    return c.is_inter() ? c.remote_size() : c.size();
  }

  inline void p2p_require_dest(const mpi_communicator& c, int dest, const context& ctx) {
    if (dest == MPI_PROC_NULL)
      return;
    const int peers = p2p_peer_count(c);
    if (dest < 0 or dest >= peers) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "dest {} is outside [0,{})", dest, peers);
    }
  }

  inline void p2p_require_source(const mpi_communicator& c, int source, const context& ctx) {
    if (source == MPI_PROC_NULL or source == MPI_ANY_SOURCE)
      return;
    const int peers = p2p_peer_count(c);
    if (source < 0 or source >= peers) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "source {} is outside [0,{})", source, peers);
    }
  }

  // MPI 4.1 §3.2.3: a send tag must be in [0, MPI_TAG_UB] and is never a wildcard; a receive
  // or probe tag may be MPI_ANY_TAG in addition to that range.  Both checks read only the
  // communicator's cached tag_ub(), so validating a tag never makes an MPI call.
  inline void p2p_require_send_tag(const mpi_communicator& c, tag t, const context& ctx) {
    if (t == any_tag) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "MPI_ANY_TAG is not a send tag");
    }
    if (t.value < 0) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "tag {} is negative", t.value);
    }
    if (t.value > c.tag_ub()) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "tag {} exceeds MPI_TAG_UB {}", t.value,
                                             c.tag_ub());
    }
  }

  inline void p2p_require_recv_tag(const mpi_communicator& c, tag t, const context& ctx) {
    if (t == any_tag)
      return;
    if (t.value < 0) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "tag {} is negative and not any_tag", t.value);
    }
    if (t.value > c.tag_ub()) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "tag {} exceeds MPI_TAG_UB {}", t.value,
                                             c.tag_ub());
    }
  }

  // MPI 4.1 §3.2.7 requires the send and receive buffers of a combined send-receive to be
  // disjoint, and allows different lengths and datatypes. Only the trivially identifiable
  // overlaps are rejected here, before any MPI call; proving general disjointness is never
  // attempted. When a direction is MPI_PROC_NULL its buffer is not accessed (MPI 4.1 §3.10), so
  // the same object may be passed for both directions in that case.
  [[noreturn]] inline void throw_sendrecv_alias(const context& ctx, const char* replacement) {
    detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                           "send and receive buffers must be disjoint (MPI 4.1 "
                                           "3.2.7); use {} for an in-place exchange",
                                           replacement);
  }

  [[nodiscard]] constexpr bool sendrecv_both_directions_active(int dest, int source) noexcept {
    return dest != MPI_PROC_NULL and source != MPI_PROC_NULL;
  }

  // MPI 4.1 §5.1.11 makes a datatype erroneous only "in association with a buffer updated by
  // the operation": a receive that MPI ignores (source = MPI_PROC_NULL, §3.10) never applies
  // the datatype, so a proven overlap in its layout is harmless and is not rejected.  Every
  // receive that can write still takes the full check.  `used` describes the receive
  // direction: false only when it is an explicit no-op, never to relax an active receive.
  template<class T>
  inline void require_used_receive_layout(const layout<T>& l, bool used, const context& ctx) {
    if (used)
      require_receive_layout(l, ctx);
  }

  // ---------------------------------------------------------------------------------------
  // Request payloads owned by nonblocking p2p requests.
  // ---------------------------------------------------------------------------------------

  template<class It>
  class range_send_payload final : public request_payload {
  public:
    range_send_payload(It first, It last) : buffer_{first, last} {
    }

    [[nodiscard]] buffer_view view() const noexcept {
      return buffer_.view();
    }

    void complete(const status&) override {
    }

  private:
    iterator_send_buffer<It> buffer_;
  };

  template<class C>
  class capacity_recv_payload final : public request_payload {
  public:
    using value_type = container_value_t<C>;

    capacity_recv_payload(C* data, capacity cap) : data_{data}, capacity_{cap.value()} {
    }

    void complete(const status& s) override {
      const count_t actual = s.get_count<value_type>();
      if (is_undefined_count(actual) or actual < 0) [[unlikely]] {
        throw_invalid("mp1::irecv container",
                      "message type does not match container value_type");
      }
      if (actual > capacity_) [[unlikely]] {
        throw_invalid("mp1::irecv container", "receive capacity exceeded");
      }
      data_->resize(static_cast<std::size_t>(actual));
    }

  private:
    C* data_;
    count_t capacity_;
  };

  // ---------------------------------------------------------------------------------------
  // Persistent-request payloads.
  //
  // A persistent request binds one buffer address for its whole lifetime (MPI 4.1 §3.9: the
  // same argument list "with the exception of the buffer contents"), so the payload validates
  // the borrowed storage before every start and never reallocates after *_init.
  // ---------------------------------------------------------------------------------------

  /// Contiguous lvalue send: a zero-copy view plus a pointer/size validator. A const lvalue
  /// borrows exactly like a non-const one; the payload stores a const pointer.
  template<class C>
  class persistent_contiguous_send_payload final : public request_payload {
  public:
    using value_type = container_value_t<C>;

    persistent_contiguous_send_payload(const C* data, const context& ctx)
        : data_{data},
          count_{buffer_to_mpi_count(data->size(), ctx)},
          size_{data->size()},
          expected_{data->data()},
          ctx_{ctx} {
    }

    [[nodiscard]] buffer_view view() const noexcept {
      return {expected_, count_, native_datatype<value_type>()};
    }

    void prepare_start() override {
      if (data_->data() != expected_ or data_->size() != size_) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            ctx_, std::source_location::current(),
            "the borrowed container was resized or reallocated after the persistent send was "
            "initialized");
      }
    }

    void complete(const status&) override {
    }

  private:
    const C* data_;
    MPI_Count count_;
    std::size_t size_;
    const value_type* expected_;
    context ctx_;
  };

  /// Non-contiguous lvalue send: one staging array allocated at init, re-filled with the
  /// container's current contents before every start. The MPI request keeps pointing at the
  /// same array, so no per-start allocation and no reallocation is possible.
  template<class C>
  class persistent_sequence_send_payload final : public request_payload {
  public:
    using value_type = container_value_t<C>;

    persistent_sequence_send_payload(const C* data, const context& ctx)
        : data_{data},
          count_{buffer_iterator_count(std::distance(data->begin(), data->end()), ctx)},
          ctx_{ctx} {
      if (count_ > 0) {
        stage_.reset(new value_type[static_cast<std::size_t>(count_)]);
        std::copy(data->begin(), data->end(), stage_.get());
      }
    }

    [[nodiscard]] buffer_view view() const noexcept {
      return {stage_.get(), count_, native_datatype<value_type>()};
    }

    void prepare_start() override {
      const std::ptrdiff_t distance = std::distance(data_->begin(), data_->end());
      if (distance != static_cast<std::ptrdiff_t>(count_)) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            ctx_, std::source_location::current(),
            "the borrowed container size changed after the persistent send was initialized");
      }
      if (count_ > 0)
        std::copy(data_->begin(), data_->end(), stage_.get());
    }

    void complete(const status&) override {
    }

  private:
    const C* data_;
    MPI_Count count_;
    std::unique_ptr<value_type[]> stage_;
    context ctx_;
  };

  /// Fixed-capacity persistent receive: the container is resized to `capacity` once at init;
  /// before every start it is grown back to the capacity (never past the initial allocation),
  /// and after every completed, non-cancelled cycle it shrinks to the received count. A
  /// reallocation by the caller is a contract violation detected as std::invalid_argument.
  template<class C>
  class persistent_capacity_recv_payload final : public request_payload {
  public:
    using value_type = container_value_t<C>;

    persistent_capacity_recv_payload(C* data, count_t capacity, const context& ctx)
        : data_{data}, capacity_{capacity}, expected_{data->data()}, ctx_{ctx} {
    }

    void prepare_start() override {
      if (data_->data() != expected_ or data_->capacity() < static_cast<std::size_t>(capacity_))
          [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            ctx_, std::source_location::current(),
            "the receive container was reallocated after recv_init; the fixed-capacity buffer "
            "must not be resized past its capacity or reallocated");
      }
      data_->resize(static_cast<std::size_t>(capacity_));
    }

    void complete(const status& s) override {
      const count_t actual = s.get_count<value_type>();
      if (is_undefined_count(actual) or actual < 0) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            ctx_, std::source_location::current(),
            "message type does not match container value_type");
      }
      if (actual > capacity_) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(ctx_, std::source_location::current(),
                                               "received count exceeds the fixed capacity");
      }
      data_->resize(static_cast<std::size_t>(actual));
    }

  private:
    C* data_;
    count_t capacity_;
    const value_type* expected_;
    context ctx_;
  };

  // ---------------------------------------------------------------------------------------
  // Blocking receive helpers.
  // ---------------------------------------------------------------------------------------

  template<class T>
  [[nodiscard]] inline status recv_scalar_impl(const mpi_communicator& c, T& data, int source,
                                               tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    return native_recv(make_recv_buffer(data), source, tag.value, c.native_handle());
  }

  /// Normalize the status of a matched probe (MPI 4.1 §3.8.2, §3.10).  A matching probe with
  /// source = MPI_PROC_NULL returns MPI_MESSAGE_NO_PROC and the status source = MPI_PROC_NULL,
  /// tag = MPI_ANY_TAG, count = 0.  The wrapper synthesizes that standard status instead of
  /// forwarding the provider's envelope (the count field of the provider's status is not
  /// guaranteed to be internally configured for a count query).
  [[nodiscard]] inline status p2p_probed_status(MPI_Message handle, const MPI_Status& native) {
    if (handle != MPI_MESSAGE_NO_PROC)
      return status_access::make(native);
    return status_access::make_proc_null();
  }

  /// Same normalization for a completed MPI_Mrecv/MPI_Imrecv of MPI_MESSAGE_NO_PROC.
  [[nodiscard]] inline status p2p_matched_recv_status(status result, bool was_no_proc) {
    if (not was_no_proc)
      return result;
    return status_access::make_proc_null();
  }

  /// Scope guard for the gap between MPI_Mprobe and MPI_Mrecv in the auto-sizing container
  /// receive.  Every throw in that window (type mismatch, size overflow, resize/allocation
  /// failure) would otherwise strand a matched message forever (MPI 4.1 §3.8.2) and block a
  /// synchronous peer.  release() is called immediately before the matched receive, which owns
  /// consumption from then on; after an MPI receive error the handle state is implementation-
  /// defined, so the guard never retries.
  class matched_message_guard {
  public:
    matched_message_guard(MPI_Message& message, const MPI_Status& probed) noexcept
        : message_{message}, probed_{probed} {
    }

    matched_message_guard(const matched_message_guard&) = delete;
    matched_message_guard& operator=(const matched_message_guard&) = delete;

    ~matched_message_guard() {
      if (active_)
        consume_matched_message_noexcept(message_, probed_);
    }

    void release() noexcept {
      active_ = false;
    }

  private:
    MPI_Message& message_;
    const MPI_Status& probed_;
    bool active_{true};
  };

  template<class C>
  [[nodiscard]] inline status recv_container_impl(const mpi_communicator& c, C& data,
                                                  int source, tag tag, const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<C>>;
    using value_type = container_value_t<plain>;
    static_assert(is_contiguous_resizable_v<plain> or is_sequence_container_v<plain>,
                  "mp1: container receive requires a resizable or sequence container");
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    if (source == MPI_PROC_NULL) {
      return native_recv(
          mutable_buffer_view{static_cast<value_type*>(dummy_buffer(buffer_slot::recv)), 0,
                              native_datatype<value_type>()},
          source, tag.value, c.native_handle());
    }

    // The auto-sizing container form has to learn the count before it can size the container,
    // so it is a two-call protocol: MPI_Mprobe then MPI_Mrecv.  A Probe+Recv pair would also be
    // two calls and would additionally lose the matched-message guarantee: MPI 4.1 §3.8.1 only
    // promises that a receive using the probed source/tag gets that message "if no other
    // intervening receive occurs"; under MPI_THREAD_MULTIPLE another thread may receive it.
    // MPI_Mprobe matches the message so only MPI_Mrecv can consume it (MPI 4.1 §3.8.2).  Keep
    // the matched receive; callers that need one native call use the capacity form.
    MPI_Message message{MPI_MESSAGE_NULL};
    MPI_Status native{};
    check_mpi(MPI_Mprobe(source, tag.value, c.native_handle(), &message, &native),
              {"MPI_Mprobe"});
    if (message == MPI_MESSAGE_NULL) [[unlikely]] {
      // A successful matching probe returns a matched message or MPI_MESSAGE_NO_PROC
      // (MPI 4.1 §3.8.2); a null handle is a provider contract violation, not a message.
      detail::fail_with<std::runtime_error>(ctx.operation, std::source_location::current(),
                                            "MPI_Mprobe returned a null message handle");
    }
    // From here to the matched receive, any throw must consume the matched message.
    matched_message_guard guard{message, native};
    const status probed = p2p_probed_status(message, native);
    // MPI_Get_count returns MPI_UNDEFINED when the message's type signature cannot be
    // expressed with value_type.  A wrong type of the *same* size is not detectable here:
    // MPI 4.1 §3.3.1 makes that program erroneous and the matched receive below behaves
    // exactly like the underlying MPI receive.
    const count_t count = probed.get_count<value_type>();
    if (is_undefined_count(count) or count < 0) [[unlikely]] {
      throw_invalid(ctx.operation, "message type does not match container value_type");
    }
    if constexpr (is_contiguous_resizable_v<plain>) {
      if (static_cast<std::size_t>(count) > data.max_size()) [[unlikely]] {
        throw_invalid(ctx.operation, "message is larger than container max_size");
      }
      data.resize(static_cast<std::size_t>(count));
      guard.release();  // MPI_Mrecv owns the handle from here; no retry after an MPI error
      return native_mrecv(make_recv_buffer(data), message);
    } else {
      std::vector<value_type> staging(static_cast<std::size_t>(count));
      guard.release();
      const status result = native_mrecv(make_recv_buffer(staging), message);
      assign_sequence(data, staging);
      return result;
    }
  }

  template<class C>
  inline void finish_capacity(C& data, capacity cap, const status& s, const context& ctx) {
    using value_type = container_value_t<C>;
    const count_t actual = s.get_count<value_type>();
    if (is_undefined_count(actual) or actual < 0) [[unlikely]] {
      throw_invalid(ctx.operation, "message type does not match container value_type");
    }
    if (actual > cap.value()) [[unlikely]] {
      throw_invalid(ctx.operation, "receive capacity exceeded");
    }
    data.resize(static_cast<std::size_t>(actual));
  }

  template<class C>
  [[nodiscard]] inline status recv_capacity_impl(const mpi_communicator& c, C& data,
                                                 capacity cap, int source, tag tag,
                                                 const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<C>>;
    using value_type = container_value_t<plain>;
    static_assert(is_contiguous_resizable_v<plain>,
                  "mp1: capacity receive requires a contiguous resizable container");
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    if (cap.value() < 0) [[unlikely]]
      throw_invalid(ctx.operation, "negative capacity");
    if (source == MPI_PROC_NULL) {
      return native_recv(
          mutable_buffer_view{static_cast<value_type*>(dummy_buffer(buffer_slot::recv)), 0,
                              native_datatype<value_type>()},
          source, tag.value, c.native_handle());
    }
    const mutable_buffer_view buffer = make_recv_buffer(data, cap.value());
    const status result = native_recv(buffer, source, tag.value, c.native_handle());
    finish_capacity(data, cap, result, ctx);
    return result;
  }

  // ---------------------------------------------------------------------------------------
  // Nonblocking receive helpers.
  // ---------------------------------------------------------------------------------------

  template<class T>
  [[nodiscard]] inline request irecv_scalar_impl(const mpi_communicator& c, T& data, int source,
                                                 tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    return native_irecv(make_recv_buffer(data), source, tag.value, c.native_handle(), nullptr);
  }

  template<class C, std::enable_if_t<is_contiguous_resizable_v<C>, int> = 0>
  [[nodiscard]] inline request irecv_capacity_impl(const mpi_communicator& c, C& data,
                                                   capacity cap, int source, tag tag,
                                                   const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<C>>;
    using value_type = container_value_t<plain>;
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    if (cap.value() < 0) [[unlikely]]
      throw_invalid(ctx.operation, "negative capacity");
    if (source == MPI_PROC_NULL) {
      return native_irecv(
          mutable_buffer_view{static_cast<value_type*>(dummy_buffer(buffer_slot::recv)), 0,
                              native_datatype<value_type>()},
          source, tag.value, c.native_handle(), nullptr);
    }
    const mutable_buffer_view buffer = make_recv_buffer(data, cap.value());
    auto payload = std::make_unique<capacity_recv_payload<plain>>(&data, cap);
    return native_irecv(buffer, source, tag.value, c.native_handle(), std::move(payload));
  }

  // ---------------------------------------------------------------------------------------
  // Persistent receive helpers.
  // ---------------------------------------------------------------------------------------

  template<class T>
  [[nodiscard]] inline request recv_init_scalar_impl(const mpi_communicator& c, T& data,
                                                     int source, tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    return native_recv_init(make_recv_buffer(data), source, tag.value, c.native_handle(),
                            nullptr);
  }

  // Fixed-capacity persistent receive. One allocation at init; before every start the container
  // is grown back to the capacity (never past the initial allocation), after every completed
  // non-cancelled cycle it shrinks to the received count. No auto-size form exists: an
  // auto-size receive would resize (and possibly reallocate) under a restartable request.
  template<class C, std::enable_if_t<is_contiguous_resizable_v<C>, int> = 0>
  [[nodiscard]] inline request recv_init_capacity_impl(const mpi_communicator& c, C& data,
                                                       capacity cap, int source, tag tag,
                                                       const context& ctx) {
    using value_type = container_value_t<C>;
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    if (cap.value() < 0) [[unlikely]]
      throw_invalid(ctx.operation, "negative capacity");
    if (source == MPI_PROC_NULL) {
      // MPI 4.1 §3.10: a receive from MPI_PROC_NULL succeeds without touching the buffer. Do
      // not resize or allocate; every cycle returns the PROC_NULL status and leaves the
      // container exactly as it was.
      return native_recv_init(
          mutable_buffer_view{static_cast<value_type*>(dummy_buffer(buffer_slot::recv)), 0,
                              native_datatype<value_type>()},
          source, tag.value, c.native_handle(), nullptr);
    }
    if (static_cast<std::size_t>(cap.value()) > data.max_size()) [[unlikely]]
      throw_invalid(ctx.operation, "capacity exceeds container max_size");
    data.resize(static_cast<std::size_t>(cap.value()));
    auto payload = std::make_unique<persistent_capacity_recv_payload<C>>(std::addressof(data),
                                                                         cap.value(), ctx);
    const mutable_buffer_view buffer = make_recv_buffer(data);
    return native_recv_init(buffer, source, tag.value, c.native_handle(), std::move(payload));
  }

  // Grants the public message type and the matched receive/matched probe functions access to
  // the owned MPI_Message; defined after class message below.
  struct message_access;


  // ---------------------------------------------------------------------------------------
  // Buffer forms: value / range / region.
  //
  // The value form is one object of its MPI datatype; the range form is a contiguous
  // (zero-copy) or non-contiguous (staged) element range; the region form is one occurrence
  // of a committed layout over the caller's data (count 1, datatype = layout).  A blocking
  // call borrows the region's layout; a nonblocking/persistent initiation duplicates it
  // (MPI_Type_dup, MPI 4.1 §5.1.10) into the request payload so the caller's layout may die
  // when the call returns.  The data pointer is captured and
  // never owned.
  // ---------------------------------------------------------------------------------------

  template<class T>
  class region_send_payload final : public request_payload {
  public:
    /// `ignored` is `dest == MPI_PROC_NULL`.  The payload validates and normalizes the data
    /// pointer exactly like the blocking region path (`make_send_buffer`): an active send
    /// rejects a null pointer for a non-empty layout, while an ignored send substitutes the
    /// per-slot dummy address.  `view()` is what the native request must use, so the
    /// request-owned duplicate is the datatype MPI retains, not the caller's handle.
    region_send_payload(const T* data, const layout<std::remove_const_t<T>>& map, bool ignored)
        : data_{static_cast<const T*>(make_send_buffer(data, map, ignored).data)}, map_{map} {
    }

    [[nodiscard]] buffer_view view() const noexcept {
      return {data_, 1, map_.native_handle()};
    }

    void complete(const status&) override {
    }

  private:
    const T* data_;
    layout<std::remove_const_t<T>> map_;  // owning duplicate of the caller's layout
  };

  template<class T>
  class region_recv_payload final : public request_payload {
  public:
    /// `ignored` is `source == MPI_PROC_NULL` (or the matched receive's MPI_MESSAGE_NO_PROC).
    /// Same null-pointer rule as the blocking region path; `view()` is what the native
    /// request must use.
    region_recv_payload(T* data, const layout<T>& map, bool ignored)
        : data_{static_cast<T*>(make_recv_buffer(data, map, ignored).data)}, map_{map} {
    }

    [[nodiscard]] mutable_buffer_view view() const noexcept {
      return {data_, 1, map_.native_handle()};
    }

    void complete(const status&) override {
    }

  private:
    T* data_;
    layout<T> map_;  // owning duplicate of the caller's layout
  };

  /// A combined nonblocking request can own a staged/duplicated send side and a duplicated
  /// receive region at the same time; the composite keeps both alive until completion.
  class combined_payload final : public request_payload {
  public:
    combined_payload(std::unique_ptr<request_payload> send,
                     std::unique_ptr<request_payload> recv)
        : send_{std::move(send)}, recv_{std::move(recv)} {
    }

    void complete(const status&) override {
    }

  private:
    std::unique_ptr<request_payload> send_;
    std::unique_ptr<request_payload> recv_;
  };

  // ---------------------------------------------------------------------------------------
  // Flat position descriptors for the combined families (sendrecv/isendrecv and replace).
  // The address/bytes pair is the trivially identifiable byte extent used by the MPI 4.1
  // §3.2.7 disjointness check; it is null/zero for a staged (copied) send, which cannot
  // alias the caller's receive buffer.
  // ---------------------------------------------------------------------------------------

  struct send_position {
    buffer_view buffer;
    const void* address{nullptr};
    count_t bytes{0};
    MPI_Datatype map{MPI_DATATYPE_NULL};
  };

  struct recv_position {
    mutable_buffer_view buffer;
    void* address{nullptr};
    count_t bytes{0};
    MPI_Datatype map{MPI_DATATYPE_NULL};
  };

  template<class T>
  [[nodiscard]] inline count_t buffer_bytes(count_t elements) noexcept {
    if (elements < 0)
      return -1;
    if (elements != 0 and
        static_cast<std::size_t>(elements) >
            static_cast<std::size_t>(std::numeric_limits<count_t>::max() / sizeof(T)))
      return -1;
    return elements * static_cast<count_t>(sizeof(T));
  }

  /// The region form: count 1 with the layout's datatype.  `ignored` (MPI_PROC_NULL) keeps a
  /// null pointer legal for a non-empty layout, exactly like the pointer+layout builds.
  template<class T>
  [[nodiscard]] inline send_position make_region_send_position(region<T> buffer, bool ignored) {
    using value_type = std::remove_const_t<T>;
    const layout<value_type>& map = buffer.map();
    const value_type* data = buffer.data();
    return {make_send_buffer(data, map, ignored), data, map.byte_size(), map.native_handle()};
  }

  template<class T>
  [[nodiscard]] inline recv_position make_region_recv_position(region<T> buffer, bool used,
                                                               const context& ctx) {
    require_used_receive_layout(buffer.map(), used, ctx);
    return {make_recv_buffer(buffer.data(), buffer.map(), not used), buffer.data(),
            buffer.map().byte_size(), buffer.map().native_handle()};
  }

  template<class T>
  requires element_value<T>
  [[nodiscard]] inline send_position make_value_send_position(const T& value) {
    return {make_send_buffer(value), std::addressof(value), static_cast<count_t>(sizeof(T)),
            MPI_DATATYPE_NULL};
  }

  template<class T>
  requires element_value<T>
  [[nodiscard]] inline recv_position make_value_recv_position(T& value) {
    return {make_recv_buffer(value), std::addressof(value), static_cast<count_t>(sizeof(T)),
            MPI_DATATYPE_NULL};
  }

  template<class R>
  requires element_buffer<R>
  [[nodiscard]] inline send_position make_range_send_position(const R& range, bool ignored,
                                                              const context& ctx) {
    using value_type = std::ranges::range_value_t<R>;
    const count_t elements = static_cast<count_t>(std::ssize(range));
    const value_type* data = std::ranges::data(range);
    if (data == nullptr) {
      if (elements > 0 and not ignored) [[unlikely]] {
        throw_invalid(ctx.operation, "null data pointer for a non-empty buffer");
      }
      // MPI_PROC_NULL ignores the buffer and a zero-count buffer has no elements; some
      // providers still reject a null pointer, so pass the stable dummy address (never
      // dereferenced, zero elements counted).
      data = static_cast<const value_type*>(dummy_buffer(buffer_slot::send));
    }
    return {buffer_view{data, buffer_to_mpi_count(static_cast<std::size_t>(elements), ctx),
                        container_element_datatype<value_type>()},
            data, buffer_bytes<value_type>(elements), MPI_DATATYPE_NULL};
  }

  template<class R>
  requires element_buffer<R>
  [[nodiscard]] inline recv_position make_range_recv_position(R& range, bool used,
                                                              const context& ctx) {
    using value_type = std::ranges::range_value_t<R>;
    const count_t elements = static_cast<count_t>(std::ssize(range));
    value_type* data = std::ranges::data(range);
    if (data == nullptr) {
      if (used and elements > 0) [[unlikely]] {
        throw_invalid(ctx.operation, "null data pointer for a non-empty buffer");
      }
      data = static_cast<value_type*>(dummy_buffer(buffer_slot::recv));
    }
    return {
        mutable_buffer_view{data, buffer_to_mpi_count(static_cast<std::size_t>(elements), ctx),
                            container_element_datatype<value_type>()},
        data, buffer_bytes<value_type>(elements), MPI_DATATYPE_NULL};
  }

  /// One dispatch point for the three send forms of a combined call.
  template<class S>
  [[nodiscard]] inline send_position make_send_position(S&& send, bool ignored,
                                                        const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<S>>;
    if constexpr (is_region_v<plain>) {
      return make_region_send_position(send, ignored);
    } else if constexpr (element_value<plain>) {
      return make_value_send_position(send);
    } else {
      static_assert(element_buffer<plain>);
      return make_range_send_position(send, ignored, ctx);
    }
  }

  /// One dispatch point for the three receive forms of a combined call.  A receive into a
  /// non-contiguous container is not a recv_sink, so it never reaches this function.
  template<class R>
  [[nodiscard]] inline recv_position make_recv_position(R&& recv, bool used,
                                                        const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<R>>;
    if constexpr (is_region_v<plain>) {
      return make_region_recv_position(recv, used, ctx);
    } else if constexpr (element_value<plain>) {
      return make_value_recv_position(recv);
    } else {
      static_assert(element_buffer<plain>);
      return make_range_recv_position(recv, used, ctx);
    }
  }

  /// MPI 4.1 §3.2.7: the send and receive buffers of a combined operation must be disjoint.
  /// Only the trivially identifiable overlaps are rejected: the same starting address, or two
  /// positive byte extents that intersect.  A staged send (no address) and an ignored direction
  /// (MPI_PROC_NULL does not access its buffer, §3.10) are never compared.
  inline void require_disjoint_sendrecv_positions(const send_position& send,
                                                  const recv_position& recv, bool both_active,
                                                  const context& ctx, const char* replacement) {
    if (not both_active or send.address == nullptr or recv.address == nullptr)
      return;
    if (send.address == recv.address) [[unlikely]]
      throw_sendrecv_alias(ctx, replacement);
    if (send.bytes > 0 and recv.bytes > 0) {
      const auto* send_begin = static_cast<const char*>(send.address);
      const auto* recv_begin = static_cast<const char*>(recv.address);
      const auto* send_end = send_begin + send.bytes;
      const auto* recv_end = recv_begin + recv.bytes;
      const std::less<const char*> less;
      if (less(send_begin, recv_end) and less(recv_begin, send_end)) [[unlikely]]
        throw_sendrecv_alias(ctx, replacement);
    }
  }

  // ---------------------------------------------------------------------------------------
  // Blocking send: three forms.
  // ---------------------------------------------------------------------------------------

  template<send_mode M, class T>
  requires element_value<T>
  inline void send_value_impl(const mpi_communicator& c, const T& value, int dest, tag tag,
                              const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_send_tag(c, tag, ctx);
    native_send<M>(make_send_buffer(value), dest, tag.value, c.native_handle());
  }

  template<send_mode M, class R>
  inline void send_range_impl(const mpi_communicator& c, R&& range, int dest, tag tag,
                              const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<R>>;
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_send_tag(c, tag, ctx);
    if constexpr (element_buffer<plain>) {
      const send_position position =
          make_range_send_position(range, dest == MPI_PROC_NULL, ctx);
      native_send<M>(position.buffer, dest, tag.value, c.native_handle());
    } else {
      static_assert(element_sequence<plain>);
      iterator_send_buffer<decltype(std::ranges::begin(range))> buffer{
          std::ranges::begin(range), std::ranges::end(range)};
      native_send<M>(buffer.view(), dest, tag.value, c.native_handle());
    }
  }

  template<send_mode M, class T>
  inline void send_region_impl(const mpi_communicator& c, region<T> buffer, int dest, tag tag,
                               const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_send_tag(c, tag, ctx);
    const send_position position = make_region_send_position(buffer, dest == MPI_PROC_NULL);
    native_send<M>(position.buffer, dest, tag.value, c.native_handle());
  }

  // ---------------------------------------------------------------------------------------
  // Nonblocking send: three forms.  The value form is lvalue-only (the caller's object is
  // borrowed); the range form needs borrowable<R&&> (a container lvalue or a borrowed view);
  // the region form duplicates the layout into the request.
  // ---------------------------------------------------------------------------------------

  template<send_mode M, class T>
  requires element_value<T>
  [[nodiscard]] inline request isend_value_impl(const mpi_communicator& c, T& value, int dest,
                                                tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_send_tag(c, tag, ctx);
    return native_isend<M>(make_send_buffer(value), dest, tag.value, c.native_handle(),
                           nullptr);
  }

  template<send_mode M, class R>
  [[nodiscard]] inline request isend_range_impl(const mpi_communicator& c, R&& range, int dest,
                                                tag tag, const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<R>>;
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_send_tag(c, tag, ctx);
    if constexpr (element_buffer<plain>) {
      const send_position position =
          make_range_send_position(range, dest == MPI_PROC_NULL, ctx);
      return native_isend<M>(position.buffer, dest, tag.value, c.native_handle(), nullptr);
    } else {
      static_assert(element_sequence<plain>);
      auto payload = std::make_unique<range_send_payload<decltype(std::ranges::begin(range))>>(
          std::ranges::begin(range), std::ranges::end(range));
      const buffer_view view = payload->view();
      return native_isend<M>(view, dest, tag.value, c.native_handle(), std::move(payload));
    }
  }

  template<send_mode M, class T>
  [[nodiscard]] inline request isend_region_impl(const mpi_communicator& c, region<T> buffer,
                                                 int dest, tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_send_tag(c, tag, ctx);
    auto payload = std::make_unique<region_send_payload<T>>(buffer.data(), buffer.map(),
                                                            dest == MPI_PROC_NULL);
    const buffer_view view = payload->view();
    return native_isend<M>(view, dest, tag.value, c.native_handle(), std::move(payload));
  }

  // ---------------------------------------------------------------------------------------
  // Persistent send: three forms.  A contiguous container lvalue is revalidated before every
  // start (data() and size() must be unchanged); a borrowed view is captured at init; a
  // region duplicates its layout once at init.
  // ---------------------------------------------------------------------------------------

  template<send_mode M, class T>
  requires element_value<T>
  [[nodiscard]] inline request send_init_value_impl(const mpi_communicator& c, T& value,
                                                    int dest, tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_send_tag(c, tag, ctx);
    return native_send_init<M>(make_send_buffer(value), dest, tag.value, c.native_handle());
  }

  template<send_mode M, class R>
  [[nodiscard]] inline request send_init_range_impl(const mpi_communicator& c, R&& range,
                                                    int dest, tag tag, const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<R>>;
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_send_tag(c, tag, ctx);
    if constexpr (is_contiguous_container_v<plain>) {
      auto payload = std::make_unique<persistent_contiguous_send_payload<plain>>(
          std::addressof(range), ctx);
      const buffer_view view = payload->view();
      return native_send_init<M>(view, dest, tag.value, c.native_handle(), std::move(payload));
    } else if constexpr (element_buffer<plain>) {
      // A borrowed view (std::span/string_view, or a C array): capture pointer + count now;
      // there is no container object to revalidate.
      const send_position position =
          make_range_send_position(range, dest == MPI_PROC_NULL, ctx);
      return native_send_init<M>(position.buffer, dest, tag.value, c.native_handle());
    } else {
      static_assert(element_sequence<plain>);
      auto payload =
          std::make_unique<persistent_sequence_send_payload<plain>>(std::addressof(range), ctx);
      const buffer_view view = payload->view();
      return native_send_init<M>(view, dest, tag.value, c.native_handle(), std::move(payload));
    }
  }

  template<send_mode M, class T>
  [[nodiscard]] inline request send_init_region_impl(const mpi_communicator& c,
                                                     region<T> buffer, int dest, tag tag,
                                                     const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_send_tag(c, tag, ctx);
    auto payload = std::make_unique<region_send_payload<T>>(buffer.data(), buffer.map(),
                                                            dest == MPI_PROC_NULL);
    const buffer_view view = payload->view();
    return native_send_init<M>(view, dest, tag.value, c.native_handle(), std::move(payload));
  }

  // ---------------------------------------------------------------------------------------
  // Receive policies 2 and 3: fixed buffer (span/array/C array, count = extent) and typed
  // region (count 1, datatype = layout).  A receive layout is a written type map, so a
  // proven overlap is rejected locally (MPI 4.1 §5.1.11) unless the receive is ignored.
  // ---------------------------------------------------------------------------------------

  template<class R>
  [[nodiscard]] inline mutable_buffer_view make_recv_fixed_buffer(R&& range, bool ignored,
                                                                  const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<R>>;
    using value_type = std::ranges::range_value_t<plain>;
    static_assert(element_buffer<plain>);
    const count_t elements = static_cast<count_t>(std::ssize(range));
    value_type* data = std::ranges::data(range);
    if (data == nullptr) {
      if (not ignored and elements > 0) [[unlikely]] {
        throw_invalid(ctx.operation, "null data pointer for a non-empty buffer");
      }
      // MPI_PROC_NULL ignores the buffer and a zero-count buffer has no elements; use the
      // stable dummy address so a provider that validates the pointer still accepts the call.
      data = static_cast<value_type*>(dummy_buffer(buffer_slot::recv));
    }
    return {data, buffer_to_mpi_count(static_cast<std::size_t>(elements), ctx),
            container_element_datatype<value_type>()};
  }

  template<class R>
  [[nodiscard]] inline status recv_fixed_impl(const mpi_communicator& c, R&& range, int source,
                                              tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    const mutable_buffer_view view =
        make_recv_fixed_buffer(range, source == MPI_PROC_NULL, ctx);
    return native_recv(view, source, tag.value, c.native_handle());
  }

  template<class R>
  [[nodiscard]] inline request irecv_fixed_impl(const mpi_communicator& c, R&& range,
                                                int source, tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    const mutable_buffer_view view =
        make_recv_fixed_buffer(range, source == MPI_PROC_NULL, ctx);
    return native_irecv(view, source, tag.value, c.native_handle(), nullptr);
  }

  template<class R>
  [[nodiscard]] inline request recv_init_fixed_impl(const mpi_communicator& c, R&& range,
                                                    int source, tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    const mutable_buffer_view view =
        make_recv_fixed_buffer(range, source == MPI_PROC_NULL, ctx);
    return native_recv_init(view, source, tag.value, c.native_handle(), nullptr);
  }

  template<class T>
  [[nodiscard]] inline status recv_region_impl(const mpi_communicator& c, region<T> buffer,
                                               int source, tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    require_used_receive_layout(buffer.map(), source != MPI_PROC_NULL, ctx);
    return native_recv(make_recv_buffer(buffer.data(), buffer.map(), source == MPI_PROC_NULL),
                       source, tag.value, c.native_handle());
  }

  template<class T>
  [[nodiscard]] inline request irecv_region_impl(const mpi_communicator& c, region<T> buffer,
                                                 int source, tag tag, const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    require_used_receive_layout(buffer.map(), source != MPI_PROC_NULL, ctx);
    auto payload = std::make_unique<region_recv_payload<T>>(buffer.data(), buffer.map(),
                                                            source == MPI_PROC_NULL);
    const mutable_buffer_view view = payload->view();
    return native_irecv(view, source, tag.value, c.native_handle(), std::move(payload));
  }

  template<class T>
  [[nodiscard]] inline request recv_init_region_impl(const mpi_communicator& c,
                                                     region<T> buffer, int source, tag tag,
                                                     const context& ctx) {
    p2p_require_comm(c, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_recv_tag(c, tag, ctx);
    require_used_receive_layout(buffer.map(), source != MPI_PROC_NULL, ctx);
    auto payload = std::make_unique<region_recv_payload<T>>(buffer.data(), buffer.map(),
                                                            source == MPI_PROC_NULL);
    const mutable_buffer_view view = payload->view();
    return native_recv_init(view, source, tag.value, c.native_handle(), std::move(payload));
  }

  // ---------------------------------------------------------------------------------------
  // Combined families: one declaration each, dispatching on the three forms.  The send side
  // may be a non-contiguous sequence (staged locally for sendrecv, request-owned for
  // isendrecv); the receive side is value, fixed buffer or region (a receive into a
  // non-contiguous container is excluded by recv_sink).
  // ---------------------------------------------------------------------------------------

  template<class S, class R>
  [[nodiscard]] inline status sendrecv_impl(const mpi_communicator& c, S&& send, int dest,
                                            tag send_tag, R&& recv, int source, tag recv_tag,
                                            const context& ctx) {
    using send_plain = std::remove_cv_t<std::remove_reference_t<S>>;
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_send_tag(c, send_tag, ctx);
    p2p_require_recv_tag(c, recv_tag, ctx);
    const bool both_active = sendrecv_both_directions_active(dest, source);
    if constexpr (element_sequence<send_plain>) {
      iterator_send_buffer<decltype(std::ranges::begin(send))> stage{std::ranges::begin(send),
                                                                     std::ranges::end(send)};
      const recv_position receive = make_recv_position(recv, source != MPI_PROC_NULL, ctx);
      require_disjoint_sendrecv_positions(
          send_position{stage.view(), nullptr, 0, MPI_DATATYPE_NULL}, receive, both_active, ctx,
          "mp1::sendrecv_replace");
      return native_sendrecv(stage.view(), dest, send_tag.value, receive.buffer, source,
                             recv_tag.value, c.native_handle());
    } else {
      const send_position transmit = make_send_position(send, dest == MPI_PROC_NULL, ctx);
      const recv_position receive = make_recv_position(recv, source != MPI_PROC_NULL, ctx);
      require_disjoint_sendrecv_positions(transmit, receive, both_active, ctx,
                                          "mp1::sendrecv_replace");
      return native_sendrecv(transmit.buffer, dest, send_tag.value, receive.buffer, source,
                             recv_tag.value, c.native_handle());
    }
  }

  template<class B>
  [[nodiscard]] inline status sendrecv_replace_impl(const mpi_communicator& c, B&& buffer,
                                                    int dest, tag send_tag, int source,
                                                    tag recv_tag, const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<B>>;
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_send_tag(c, send_tag, ctx);
    p2p_require_recv_tag(c, recv_tag, ctx);
    // Only a both-directions MPI_PROC_NULL call ignores the buffer entirely (MPI 4.1 §3.10);
    // with one direction active the buffer is read or written and must be valid.
    const bool ignored = dest == MPI_PROC_NULL and source == MPI_PROC_NULL;
    if constexpr (is_region_v<plain>) {
      require_used_receive_layout(buffer.map(), source != MPI_PROC_NULL, ctx);
      return native_sendrecv_replace(make_recv_buffer(buffer.data(), buffer.map(), ignored),
                                     dest, send_tag.value, source, recv_tag.value,
                                     c.native_handle());
    } else if constexpr (element_value<plain>) {
      return native_sendrecv_replace(make_recv_buffer(buffer), dest, send_tag.value, source,
                                     recv_tag.value, c.native_handle());
    } else {
      static_assert(element_buffer<plain>);
      const mutable_buffer_view view = make_recv_fixed_buffer(buffer, ignored, ctx);
      return native_sendrecv_replace(view, dest, send_tag.value, source, recv_tag.value,
                                     c.native_handle());
    }
  }

  template<class S, class R>
  [[nodiscard]] inline request isendrecv_impl(const mpi_communicator& c, S&& send, int dest,
                                              tag send_tag, R&& recv, int source, tag recv_tag,
                                              const context& ctx) {
    using send_plain = std::remove_cv_t<std::remove_reference_t<S>>;
    using recv_plain = std::remove_cv_t<std::remove_reference_t<R>>;
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_send_tag(c, send_tag, ctx);
    p2p_require_recv_tag(c, recv_tag, ctx);
    const bool both_active = sendrecv_both_directions_active(dest, source);

    std::unique_ptr<request_payload> send_payload;
    buffer_view send_buffer{nullptr, 0, MPI_DATATYPE_NULL};
    send_position transmit{send_buffer, nullptr, 0, MPI_DATATYPE_NULL};
    if constexpr (is_region_v<send_plain>) {
      transmit = make_region_send_position(send, dest == MPI_PROC_NULL);
      auto region_payload = std::make_unique<region_send_payload<region_element_t<send_plain>>>(
          send.data(), send.map(), dest == MPI_PROC_NULL);
      send_buffer = region_payload->view();
      send_payload = std::move(region_payload);
    } else if constexpr (element_value<send_plain>) {
      transmit = make_value_send_position(send);
      send_buffer = transmit.buffer;
    } else if constexpr (element_buffer<send_plain>) {
      transmit = make_range_send_position(send, dest == MPI_PROC_NULL, ctx);
      send_buffer = transmit.buffer;
    } else {
      static_assert(element_sequence<send_plain>);
      auto staged = std::make_unique<range_send_payload<decltype(std::ranges::begin(send))>>(
          std::ranges::begin(send), std::ranges::end(send));
      send_buffer = staged->view();
      send_payload = std::move(staged);
    }

    std::unique_ptr<request_payload> recv_payload;
    mutable_buffer_view recv_buffer{nullptr, 0, MPI_DATATYPE_NULL};
    recv_position receive{recv_buffer, nullptr, 0, MPI_DATATYPE_NULL};
    if constexpr (is_region_v<recv_plain>) {
      receive = make_region_recv_position(recv, source != MPI_PROC_NULL, ctx);
      auto region_payload = std::make_unique<region_recv_payload<region_element_t<recv_plain>>>(
          recv.data(), recv.map(), source == MPI_PROC_NULL);
      recv_buffer = region_payload->view();
      recv_payload = std::move(region_payload);
    } else if constexpr (element_value<recv_plain>) {
      receive = make_value_recv_position(recv);
      recv_buffer = receive.buffer;
    } else {
      static_assert(element_buffer<recv_plain>);
      receive = make_range_recv_position(recv, source != MPI_PROC_NULL, ctx);
      recv_buffer = receive.buffer;
    }

    require_disjoint_sendrecv_positions(transmit, receive, both_active, ctx,
                                        "mp1::isendrecv_replace");
    if (not send_payload and not recv_payload) {
      return native_isendrecv(send_buffer, dest, send_tag.value, recv_buffer, source,
                              recv_tag.value, c.native_handle());
    }
    auto payload =
        std::make_unique<combined_payload>(std::move(send_payload), std::move(recv_payload));
    return native_isendrecv(send_buffer, dest, send_tag.value, recv_buffer, source,
                            recv_tag.value, c.native_handle(), std::move(payload));
  }

  template<class B>
  [[nodiscard]] inline request isendrecv_replace_impl(const mpi_communicator& c, B&& buffer,
                                                      int dest, tag send_tag, int source,
                                                      tag recv_tag, const context& ctx) {
    using plain = std::remove_cv_t<std::remove_reference_t<B>>;
    p2p_require_comm(c, ctx);
    p2p_require_dest(c, dest, ctx);
    p2p_require_source(c, source, ctx);
    p2p_require_send_tag(c, send_tag, ctx);
    p2p_require_recv_tag(c, recv_tag, ctx);
    // Only a both-directions MPI_PROC_NULL call ignores the buffer entirely (MPI 4.1 §3.10).
    const bool ignored = dest == MPI_PROC_NULL and source == MPI_PROC_NULL;
    if constexpr (is_region_v<plain>) {
      require_used_receive_layout(buffer.map(), source != MPI_PROC_NULL, ctx);
      auto payload = std::make_unique<region_recv_payload<region_element_t<plain>>>(
          buffer.data(), buffer.map(), ignored);
      // Materialize the view before the payload is moved into the native call: function
      // argument evaluation is only indeterminately sequenced, so `payload->view()` and
      // `std::move(payload)` in one call would be unsequenced.
      const mutable_buffer_view view = payload->view();
      return native_isendrecv_replace(view, dest, send_tag.value, source, recv_tag.value,
                                      c.native_handle(), std::move(payload));
    } else if constexpr (element_value<plain>) {
      return native_isendrecv_replace(make_recv_buffer(buffer), dest, send_tag.value, source,
                                      recv_tag.value, c.native_handle());
    } else {
      static_assert(element_buffer<plain>);
      const mutable_buffer_view view = make_recv_fixed_buffer(buffer, ignored, ctx);
      return native_isendrecv_replace(view, dest, send_tag.value, source, recv_tag.value,
                                      c.native_handle());
    }
  }

}  // namespace mp1::detail


namespace mp1 {

  /// Owning, move-only handle of a message matched by mp1::mprobe.
  ///
  /// MPI gives a matched message no free operation: it must be consumed exactly once with
  /// MPI_Mrecv/MPI_Imrecv (MPI 4.1 §3.8.2), and copying a raw MPI_Message would leave a stale
  /// handle that a second receive could use (undefined behaviour, §3.8.3).  This type owns one
  /// handle; the copy operations are deleted, moves transfer ownership and reset the source.
  ///
  /// # Lifecycle
  ///
  /// * A default-constructed, moved-from, consumed or released message is invalid
  ///   (`is_valid() == false`); passing it to mrecv/imrecv throws std::invalid_argument before
  ///   any MPI call.  Use mprobe/improbe to obtain a valid one.
  /// * `is_no_proc()` is true for MPI_MESSAGE_NO_PROC, the predefined "message" returned by a
  ///   matching probe with source = MPI_PROC_NULL (MPI 4.1 §3.8.2, §3.10).  It is a successful
  ///   no-op, not an invalid handle; it is *not* compared against MPI_MESSAGE_NULL.
  /// * mrecv/imrecv consume the message: after the call the object is invalid whether MPI
  ///   succeeded or failed.  The MPI handle state after an error is implementation-defined
  ///   (§3.8.3), so a retry is forbidden by contract.  MPI_MESSAGE_NO_PROC may legally be
  ///   passed to mrecv/imrecv; the call returns the PROC_NULL status and the container forms
  ///   leave the container untouched (MPI 4.1 §3.10: the no-op receive does not modify the
  ///   receive buffer).
  /// * `discard()` consumes a real matched message without interpreting it.  This is portable:
  ///   MPI 4.1 §5.2 allows receiving a message sent with *any* type with MPI_PACKED, and the
  ///   packed byte count comes from the probe status (§3.2.5, §5.2).  It is the safe way to
  ///   free a message whose type the caller cannot or does not want to receive.
  /// * The destructor calls `discard()` and is noexcept: a still-matched message is consumed
  ///   (so a synchronous sender completes) unless MPI is no longer active or consumption
  ///   fails, in which case the handle is abandoned and counted.  `release()` is the explicit
  ///   no-MPI-call escape (the message is abandoned and counted).
  ///
  /// Matched receives take the message by lvalue reference: `mrecv(x, mprobe(c))` is not a
  /// supported spelling (the temporary would be consumed before its destructor, but the
  /// asymmetry with the failure contract is not worth the second declaration); name the
  /// message object.
  class message {
  public:
    message() noexcept = default;

    message(const message&) = delete;
    message& operator=(const message&) = delete;

    message(message&& other) noexcept : handle_(other.handle_), probed_(other.probed_) {
      other.handle_ = MPI_MESSAGE_NULL;
      other.probed_ = status{};
    }

    message& operator=(message&& other) noexcept {
      if (this != &other) {
        discard_noexcept();
        handle_ = other.handle_;
        probed_ = other.probed_;
        other.handle_ = MPI_MESSAGE_NULL;
        other.probed_ = status{};
      }
      return *this;
    }

    ~message() {
      discard_noexcept();
    }

    /// \return true if this object owns a matched message (including MPI_MESSAGE_NO_PROC);
    /// false for the default, moved-from, consumed and released states
    [[nodiscard]] bool is_valid() const noexcept {
      return handle_ != MPI_MESSAGE_NULL;
    }

    /// \return true iff the handle is MPI_MESSAGE_NO_PROC, a successful PROC_NULL no-op
    [[nodiscard]] bool is_no_proc() const noexcept {
      return handle_ == MPI_MESSAGE_NO_PROC;
    }

    /// \return status returned by the matching probe; no MPI call
    ///
    /// For MPI_MESSAGE_NO_PROC the envelope is normalized to the symbolic standard values
    /// source = MPI_PROC_NULL, tag = MPI_ANY_TAG, count 0 (MPI 4.1 §3.8.2, §3.10).  The
    /// integer values of MPI_PROC_NULL/MPI_ANY_SOURCE differ between providers (Open MPI and
    /// MPICH assign them in opposite order), so callers must compare against the constants.
    [[nodiscard]] const status& probed_status() const noexcept {
      return probed_;
    }

    /// \return copy of the raw MPI_Message for raw-MPI interop.  The wrapper still owns it:
    /// a raw consumer must take it over and call release() so the destructor does not receive
    /// it a second time.
    [[nodiscard]] MPI_Message native_handle() const noexcept {
      return handle_;
    }

    /// Consume the matched message without interpreting its type or data (MPI 4.1 §5.2).
    /// No-op for an invalid or MPI_MESSAGE_NO_PROC handle.  Throws mpi_error if the discard
    /// receive itself fails; the object is invalid afterwards in every case.
    void discard() {
      if (handle_ == MPI_MESSAGE_NULL)
        return;
      if (handle_ == MPI_MESSAGE_NO_PROC) {
        // §3.8.2: receiving MPI_MESSAGE_NO_PROC is not necessary.
        handle_ = MPI_MESSAGE_NULL;
        probed_ = status{};
        return;
      }
      detail::require_mpi("mp1::message::discard");
      detail::consume_matched_message(handle_, probed_.native_handle(),
                                      "mp1::message::discard");
      probed_ = status{};
    }

    /// Abandon the matched message without an MPI call; counted as an abandoned handle.
    /// The message stays unreceived (a synchronous sender may block), so this is the raw-MPI
    /// escape, not a discard.
    void release() noexcept {
      // MPI_MESSAGE_NO_PROC is a predefined no-op handle, not a matched resource.
      if (handle_ != MPI_MESSAGE_NULL and handle_ != MPI_MESSAGE_NO_PROC)
        detail::note_abandoned_handle();
      handle_ = MPI_MESSAGE_NULL;
      probed_ = status{};
    }

  private:
    message(MPI_Message handle, status probed) noexcept : handle_(handle), probed_(probed) {
    }

    void discard_noexcept() noexcept {
      if (handle_ == MPI_MESSAGE_NULL)
        return;
      detail::consume_matched_message_noexcept(handle_, probed_.native_handle());
      probed_ = status{};
    }

    MPI_Message handle_{MPI_MESSAGE_NULL};
    status probed_;

    friend struct detail::message_access;
  };

  namespace detail {

    struct message_access {
      [[nodiscard]] static MPI_Message& handle(message& msg) noexcept {
        return msg.handle_;
      }

      [[nodiscard]] static message make(MPI_Message handle, status probed) noexcept {
        return message{handle, probed};
      }

      /// Mark the object invalid after an MPI receive: the handle is single-use and its state
      /// after an error is implementation-defined (MPI 4.1 §3.8.3), so it is never retried.
      static void invalidate(message& msg) noexcept {
        msg.handle_ = MPI_MESSAGE_NULL;
        msg.probed_ = status{};
      }
    };

    /// Common mrecv protocol: validate, call MPI_Mrecv_c with the message handle, invalidate
    /// the owner, and normalize the MPI_MESSAGE_NO_PROC status.
    [[nodiscard]] inline status mrecv_message(const mutable_buffer_view& buffer, message& msg,
                                              const context& ctx) {
      if (not msg.is_valid()) [[unlikely]] {
        throw_invalid(ctx.operation, "invalid message handle (null, consumed, or moved-from)");
      }
      const bool no_proc = msg.is_no_proc();
      status result{};
      try {
        result = native_mrecv(buffer, message_access::handle(msg));
      } catch (...) {
        message_access::invalidate(msg);
        throw;
      }
      message_access::invalidate(msg);
      return p2p_matched_recv_status(result, no_proc);
    }

    /// Common imrecv protocol: same contract as mrecv_message; the returned request owns the
    /// completion (MPI 4.1 §3.8.3, cancelable per §3.8.4).
    [[nodiscard]] inline request imrecv_message(const mutable_buffer_view& buffer, message& msg,
                                                std::unique_ptr<request_payload> payload,
                                                const context& ctx) {
      if (not msg.is_valid()) [[unlikely]] {
        throw_invalid(ctx.operation, "invalid message handle (null, consumed, or moved-from)");
      }
      request result{};
      try {
        result = native_imrecv(buffer, message_access::handle(msg), std::move(payload));
      } catch (...) {
        message_access::invalidate(msg);
        throw;
      }
      message_access::invalidate(msg);
      return result;
    }

    template<class T>
    [[nodiscard]] inline status mrecv_value_impl(T& value, message& msg, const context& ctx) {
      require_mpi(ctx);
      return mrecv_message(make_recv_buffer(value), msg, ctx);
    }

    template<class R>
    [[nodiscard]] inline status mrecv_fixed_impl(R&& range, message& msg, const context& ctx) {
      require_mpi(ctx);
      const mutable_buffer_view view = make_recv_fixed_buffer(range, msg.is_no_proc(), ctx);
      return mrecv_message(view, msg, ctx);
    }

    template<class T>
    [[nodiscard]] inline status mrecv_region_impl(region<T> buffer, message& msg,
                                                  const context& ctx) {
      require_mpi(ctx);
      require_used_receive_layout(buffer.map(), not msg.is_no_proc(), ctx);
      return mrecv_message(make_recv_buffer(buffer.data(), buffer.map(), msg.is_no_proc()), msg,
                           ctx);
    }

    /// Auto-size / bounded matched receive: the probe status gives the element count before
    /// the message is consumed, so the container is sized once; a count above `bound` throws
    /// while the message is still valid.  MPI_MESSAGE_NO_PROC leaves the container untouched
    /// (MPI 4.1 §3.10).
    template<class C>
    class matched_resizable_recv_plan {
    public:
      using value_type = std::ranges::range_value_t<C>;

      matched_resizable_recv_plan(C& data, const message& msg, count_t bound,
                                  const context& ctx)
          : data_{std::addressof(data)}, bound_{bound} {
        if (msg.is_no_proc()) {
          no_proc_ = true;
          return;
        }
        const count_t count = msg.probed_status().get_count<value_type>();
        if (is_undefined_count(count) or count < 0) [[unlikely]] {
          throw_invalid(ctx.operation, "message type does not match container value_type");
        }
        if (bound_ >= 0 and count > bound_) [[unlikely]] {
          throw_invalid(ctx.operation, "receive capacity exceeded");
        }
        if (static_cast<std::size_t>(count) > data.max_size()) [[unlikely]] {
          throw_invalid(ctx.operation, "message is larger than container max_size");
        }
        data.resize(static_cast<std::size_t>(count));
      }

      [[nodiscard]] mutable_buffer_view view() {
        if (no_proc_)
          return {static_cast<value_type*>(dummy_buffer(buffer_slot::recv)), 0,
                  native_datatype<value_type>()};
        return make_recv_buffer(*data_);
      }

      void commit(const status&) {
      }

    private:
      C* data_;
      count_t bound_{-1};
      bool no_proc_{false};
    };

    template<class C>
    [[nodiscard]] inline status mrecv_auto_impl(C& data, message& msg, const context& ctx) {
      require_mpi(ctx);
      if (not msg.is_valid()) [[unlikely]] {
        throw_invalid(ctx.operation, "invalid message handle (null, consumed, or moved-from)");
      }
      matched_resizable_recv_plan<C> plan{data, msg, -1, ctx};
      const status result = mrecv_message(plan.view(), msg, ctx);
      plan.commit(result);
      return result;
    }

    template<class C>
    [[nodiscard]] inline status mrecv_bounded_impl(C& data, capacity bound, message& msg,
                                                   const context& ctx) {
      require_mpi(ctx);
      if (bound.value() < 0) [[unlikely]]
        throw_invalid(ctx.operation, "negative capacity");
      if (not msg.is_valid()) [[unlikely]] {
        throw_invalid(ctx.operation, "invalid message handle (null, consumed, or moved-from)");
      }
      matched_resizable_recv_plan<C> plan{data, msg, bound.value(), ctx};
      const status result = mrecv_message(plan.view(), msg, ctx);
      plan.commit(result);
      return result;
    }

    template<class T>
    [[nodiscard]] inline request imrecv_value_impl(T& value, message& msg, const context& ctx) {
      require_mpi(ctx);
      return imrecv_message(make_recv_buffer(value), msg, nullptr, ctx);
    }

    template<class R>
    [[nodiscard]] inline request imrecv_fixed_impl(R&& range, message& msg,
                                                   const context& ctx) {
      require_mpi(ctx);
      const mutable_buffer_view view = make_recv_fixed_buffer(range, msg.is_no_proc(), ctx);
      return imrecv_message(view, msg, nullptr, ctx);
    }

    template<class T>
    [[nodiscard]] inline request imrecv_region_impl(region<T> buffer, message& msg,
                                                    const context& ctx) {
      require_mpi(ctx);
      require_used_receive_layout(buffer.map(), not msg.is_no_proc(), ctx);
      auto payload = std::make_unique<region_recv_payload<T>>(buffer.data(), buffer.map(),
                                                              msg.is_no_proc());
      const mutable_buffer_view view = payload->view();
      return imrecv_message(view, msg, std::move(payload), ctx);
    }

    template<class C>
    [[nodiscard]] inline request imrecv_bounded_impl(C& data, capacity bound, message& msg,
                                                     const context& ctx) {
      require_mpi(ctx);
      if (bound.value() < 0) [[unlikely]]
        throw_invalid(ctx.operation, "negative capacity");
      if (not msg.is_valid()) [[unlikely]] {
        throw_invalid(ctx.operation, "invalid message handle (null, consumed, or moved-from)");
      }
      matched_resizable_recv_plan<C> plan{data, msg, bound.value(), ctx};
      return imrecv_message(plan.view(), msg, nullptr, ctx);
    }

  }  // namespace detail

  // =========================================================================================
  // Point-to-point: twelve send modes (standard / buffered / synchronous / ready, each
  // blocking / nonblocking / persistent) with three forms each (value / range / region).
  //
  //   value   one object of its MPI datatype: count 1.
  //   range   std::span, std::array, C array, std::vector, std::string, std::string_view
  //           (contiguous, zero-copy), or std::list/std::deque on send only (staged).
  //   region  one occurrence of a committed layout: (data, 1, layout).  The layout is
  //           borrowed for a blocking call and duplicated at initiation for a request.
  //
  // The empty spelling is a zero-count span: `mp1::empty_data` for the explicit byte message,
  // or any `std::span<T>{}`.  It sends/receives zero elements with T's datatype, which matches
  // any zero-count message (the datatype of a zero-count message contributes no elements to
  // the type signature).  There are no `*_empty` function names.
  // =========================================================================================

  // ---- blocking send: 4 modes × 3 forms ---------------------------------------------------

  template<element_value T>
  void send(const mpi_communicator& c, const T& value, int dest, tag tag = {}) {
    detail::send_value_impl<detail::send_mode::standard>(c, value, dest, tag, "mp1::send");
  }

  template<class R>
  requires(element_buffer<R> || element_sequence<R>)
  void send(const mpi_communicator& c, R&& range, int dest, tag tag = {}) {
    detail::send_range_impl<detail::send_mode::standard>(c, range, dest, tag, "mp1::send");
  }

  template<class T>
  void send(const mpi_communicator& c, region<T> buffer, int dest, tag tag = {}) {
    detail::send_region_impl<detail::send_mode::standard>(c, buffer, dest, tag, "mp1::send");
  }

  template<element_value T>
  void bsend(const mpi_communicator& c, const T& value, int dest, tag tag = {}) {
    detail::send_value_impl<detail::send_mode::buffered>(c, value, dest, tag, "mp1::bsend");
  }

  template<class R>
  requires(element_buffer<R> || element_sequence<R>)
  void bsend(const mpi_communicator& c, R&& range, int dest, tag tag = {}) {
    detail::send_range_impl<detail::send_mode::buffered>(c, range, dest, tag, "mp1::bsend");
  }

  template<class T>
  void bsend(const mpi_communicator& c, region<T> buffer, int dest, tag tag = {}) {
    detail::send_region_impl<detail::send_mode::buffered>(c, buffer, dest, tag, "mp1::bsend");
  }

  template<element_value T>
  void ssend(const mpi_communicator& c, const T& value, int dest, tag tag = {}) {
    detail::send_value_impl<detail::send_mode::synchronous>(c, value, dest, tag, "mp1::ssend");
  }

  template<class R>
  requires(element_buffer<R> || element_sequence<R>)
  void ssend(const mpi_communicator& c, R&& range, int dest, tag tag = {}) {
    detail::send_range_impl<detail::send_mode::synchronous>(c, range, dest, tag, "mp1::ssend");
  }

  template<class T>
  void ssend(const mpi_communicator& c, region<T> buffer, int dest, tag tag = {}) {
    detail::send_region_impl<detail::send_mode::synchronous>(c, buffer, dest, tag,
                                                             "mp1::ssend");
  }

  template<element_value T>
  void rsend(const mpi_communicator& c, const T& value, int dest, tag tag = {}) {
    detail::send_value_impl<detail::send_mode::ready>(c, value, dest, tag, "mp1::rsend");
  }

  template<class R>
  requires(element_buffer<R> || element_sequence<R>)
  void rsend(const mpi_communicator& c, R&& range, int dest, tag tag = {}) {
    detail::send_range_impl<detail::send_mode::ready>(c, range, dest, tag, "mp1::rsend");
  }

  template<class T>
  void rsend(const mpi_communicator& c, region<T> buffer, int dest, tag tag = {}) {
    detail::send_region_impl<detail::send_mode::ready>(c, buffer, dest, tag, "mp1::rsend");
  }

  // ---- nonblocking send: 4 modes × 3 forms; an rvalue value/range is rejected -------------

  template<element_value T>
  requires(std::is_lvalue_reference_v<T &&>)
  [[nodiscard]] request isend(const mpi_communicator& c, T&& value, int dest, tag tag = {}) {
    return detail::isend_value_impl<detail::send_mode::standard>(c, value, dest, tag,
                                                                 "mp1::isend");
  }

  template<class R>
  requires((element_buffer<R> || element_sequence<R>) && borrowable<R &&>)
  [[nodiscard]] request isend(const mpi_communicator& c, R&& range, int dest, tag tag = {}) {
    return detail::isend_range_impl<detail::send_mode::standard>(c, range, dest, tag,
                                                                 "mp1::isend");
  }

  template<class T>
  [[nodiscard]] request isend(const mpi_communicator& c, region<T> buffer, int dest,
                              tag tag = {}) {
    return detail::isend_region_impl<detail::send_mode::standard>(c, buffer, dest, tag,
                                                                  "mp1::isend");
  }

  template<element_value T>
  requires(std::is_lvalue_reference_v<T &&>)
  [[nodiscard]] request ibsend(const mpi_communicator& c, T&& value, int dest, tag tag = {}) {
    return detail::isend_value_impl<detail::send_mode::buffered>(c, value, dest, tag,
                                                                 "mp1::ibsend");
  }

  template<class R>
  requires((element_buffer<R> || element_sequence<R>) && borrowable<R &&>)
  [[nodiscard]] request ibsend(const mpi_communicator& c, R&& range, int dest, tag tag = {}) {
    return detail::isend_range_impl<detail::send_mode::buffered>(c, range, dest, tag,
                                                                 "mp1::ibsend");
  }

  template<class T>
  [[nodiscard]] request ibsend(const mpi_communicator& c, region<T> buffer, int dest,
                               tag tag = {}) {
    return detail::isend_region_impl<detail::send_mode::buffered>(c, buffer, dest, tag,
                                                                  "mp1::ibsend");
  }

  template<element_value T>
  requires(std::is_lvalue_reference_v<T &&>)
  [[nodiscard]] request issend(const mpi_communicator& c, T&& value, int dest, tag tag = {}) {
    return detail::isend_value_impl<detail::send_mode::synchronous>(c, value, dest, tag,
                                                                    "mp1::issend");
  }

  template<class R>
  requires((element_buffer<R> || element_sequence<R>) && borrowable<R &&>)
  [[nodiscard]] request issend(const mpi_communicator& c, R&& range, int dest, tag tag = {}) {
    return detail::isend_range_impl<detail::send_mode::synchronous>(c, range, dest, tag,
                                                                    "mp1::issend");
  }

  template<class T>
  [[nodiscard]] request issend(const mpi_communicator& c, region<T> buffer, int dest,
                               tag tag = {}) {
    return detail::isend_region_impl<detail::send_mode::synchronous>(c, buffer, dest, tag,
                                                                     "mp1::issend");
  }

  template<element_value T>
  requires(std::is_lvalue_reference_v<T &&>)
  [[nodiscard]] request irsend(const mpi_communicator& c, T&& value, int dest, tag tag = {}) {
    return detail::isend_value_impl<detail::send_mode::ready>(c, value, dest, tag,
                                                              "mp1::irsend");
  }

  template<class R>
  requires((element_buffer<R> || element_sequence<R>) && borrowable<R &&>)
  [[nodiscard]] request irsend(const mpi_communicator& c, R&& range, int dest, tag tag = {}) {
    return detail::isend_range_impl<detail::send_mode::ready>(c, range, dest, tag,
                                                              "mp1::irsend");
  }

  template<class T>
  [[nodiscard]] request irsend(const mpi_communicator& c, region<T> buffer, int dest,
                               tag tag = {}) {
    return detail::isend_region_impl<detail::send_mode::ready>(c, buffer, dest, tag,
                                                               "mp1::irsend");
  }

  // ---- persistent send: 4 modes × 3 forms; a container lvalue is revalidated every start --

  template<element_value T>
  requires(std::is_lvalue_reference_v<T &&>)
  [[nodiscard]] request send_init(const mpi_communicator& c, T&& value, int dest,
                                  tag tag = {}) {
    return detail::send_init_value_impl<detail::send_mode::standard>(c, value, dest, tag,
                                                                     "mp1::send_init");
  }

  template<class R>
  requires((element_buffer<R> || element_sequence<R>) && borrowable<R &&>)
  [[nodiscard]] request send_init(const mpi_communicator& c, R&& range, int dest,
                                  tag tag = {}) {
    return detail::send_init_range_impl<detail::send_mode::standard>(c, range, dest, tag,
                                                                     "mp1::send_init");
  }

  template<class T>
  [[nodiscard]] request send_init(const mpi_communicator& c, region<T> buffer, int dest,
                                  tag tag = {}) {
    return detail::send_init_region_impl<detail::send_mode::standard>(c, buffer, dest, tag,
                                                                      "mp1::send_init");
  }

  template<element_value T>
  requires(std::is_lvalue_reference_v<T &&>)
  [[nodiscard]] request bsend_init(const mpi_communicator& c, T&& value, int dest,
                                   tag tag = {}) {
    return detail::send_init_value_impl<detail::send_mode::buffered>(c, value, dest, tag,
                                                                     "mp1::bsend_init");
  }

  template<class R>
  requires((element_buffer<R> || element_sequence<R>) && borrowable<R &&>)
  [[nodiscard]] request bsend_init(const mpi_communicator& c, R&& range, int dest,
                                   tag tag = {}) {
    return detail::send_init_range_impl<detail::send_mode::buffered>(c, range, dest, tag,
                                                                     "mp1::bsend_init");
  }

  template<class T>
  [[nodiscard]] request bsend_init(const mpi_communicator& c, region<T> buffer, int dest,
                                   tag tag = {}) {
    return detail::send_init_region_impl<detail::send_mode::buffered>(c, buffer, dest, tag,
                                                                      "mp1::bsend_init");
  }

  template<element_value T>
  requires(std::is_lvalue_reference_v<T &&>)
  [[nodiscard]] request ssend_init(const mpi_communicator& c, T&& value, int dest,
                                   tag tag = {}) {
    return detail::send_init_value_impl<detail::send_mode::synchronous>(c, value, dest, tag,
                                                                        "mp1::ssend_init");
  }

  template<class R>
  requires((element_buffer<R> || element_sequence<R>) && borrowable<R &&>)
  [[nodiscard]] request ssend_init(const mpi_communicator& c, R&& range, int dest,
                                   tag tag = {}) {
    return detail::send_init_range_impl<detail::send_mode::synchronous>(c, range, dest, tag,
                                                                        "mp1::ssend_init");
  }

  template<class T>
  [[nodiscard]] request ssend_init(const mpi_communicator& c, region<T> buffer, int dest,
                                   tag tag = {}) {
    return detail::send_init_region_impl<detail::send_mode::synchronous>(c, buffer, dest, tag,
                                                                         "mp1::ssend_init");
  }

  template<element_value T>
  requires(std::is_lvalue_reference_v<T &&>)
  [[nodiscard]] request rsend_init(const mpi_communicator& c, T&& value, int dest,
                                   tag tag = {}) {
    return detail::send_init_value_impl<detail::send_mode::ready>(c, value, dest, tag,
                                                                  "mp1::rsend_init");
  }

  template<class R>
  requires((element_buffer<R> || element_sequence<R>) && borrowable<R &&>)
  [[nodiscard]] request rsend_init(const mpi_communicator& c, R&& range, int dest,
                                   tag tag = {}) {
    return detail::send_init_range_impl<detail::send_mode::ready>(c, range, dest, tag,
                                                                  "mp1::rsend_init");
  }

  template<class T>
  [[nodiscard]] request rsend_init(const mpi_communicator& c, region<T> buffer, int dest,
                                   tag tag = {}) {
    return detail::send_init_region_impl<detail::send_mode::ready>(c, buffer, dest, tag,
                                                                   "mp1::rsend_init");
  }

  // =========================================================================================
  // Receive: five blocking policies, four nonblocking and persistent policies.
  //
  //   1 value       T&                  one element; more than one is a truncation error
  //   2 fixed       span/array/C array  extent is the capacity; no resize
  //   3 region      region<T>           count 1 with the layout; overlap map rejected
  //   4 auto-size   vector/string       MPI_Mprobe + MPI_Mrecv; resized to the count
  //   5 bounded     vector/string+capacity  MPI_Recv_c with an upper bound; resized down
  //
  // Nonblocking/persistent: 1, 2, 3, 5 (no auto-size: MPI_Irecv needs a count, and resizing
  // after initiation would invalidate the buffer).  A raw count_t in the third argument of a
  // resizable receive is rejected: `recv(c, v, n, tag{0})` with count_t n would otherwise read
  // n as the source rank, so the capacity must be spelled `capacity{n}`.
  // =========================================================================================

  template<element_value T>
  requires(!std::is_const_v<T>)
  status recv(const mpi_communicator& c, T& value, int source, tag tag = {}) {
    return detail::recv_scalar_impl(c, value, source, tag, "mp1::recv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
           !is_region_v<std::remove_cvref_t<R>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>>)
  status recv(const mpi_communicator& c, R&& buffer, int source, tag tag = {}) {
    return detail::recv_fixed_impl(c, buffer, source, tag, "mp1::recv");
  }

  template<class T>
  requires(!std::is_const_v<T>)
  status recv(const mpi_communicator& c, region<T> buffer, int source, tag tag = {}) {
    return detail::recv_region_impl(c, buffer, source, tag, "mp1::recv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R&> && resizable_element_buffer<R>)
  status recv(const mpi_communicator& c, R& range, int source, tag tag = {}) {
    return detail::recv_container_impl(c, range, source, tag, "mp1::recv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R&> && resizable_element_buffer<R>)
  status recv(const mpi_communicator& c, R& range, capacity bound, int source, tag tag = {}) {
    return detail::recv_capacity_impl(c, range, bound, source, tag, "mp1::recv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R &&> &&
           resizable_element_buffer<std::remove_cvref_t<R>>)
  status recv(const mpi_communicator& c, R&&, count_t, tag) = delete;

  template<element_value T>
  requires(!std::is_const_v<T>)
  [[nodiscard]] request irecv(const mpi_communicator& c, T& value, int source, tag tag = {}) {
    return detail::irecv_scalar_impl(c, value, source, tag, "mp1::irecv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
           !is_region_v<std::remove_cvref_t<R>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>>)
  [[nodiscard]] request irecv(const mpi_communicator& c, R&& buffer, int source, tag tag = {}) {
    return detail::irecv_fixed_impl(c, buffer, source, tag, "mp1::irecv");
  }

  template<class T>
  requires(!std::is_const_v<T>)
  [[nodiscard]] request irecv(const mpi_communicator& c, region<T> buffer, int source,
                              tag tag = {}) {
    return detail::irecv_region_impl(c, buffer, source, tag, "mp1::irecv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R&> && resizable_element_buffer<R>)
  [[nodiscard]] request irecv(const mpi_communicator& c, R& range, capacity bound, int source,
                              tag tag = {}) {
    return detail::irecv_capacity_impl(c, range, bound, source, tag, "mp1::irecv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R &&> &&
           resizable_element_buffer<std::remove_cvref_t<R>>)
  [[nodiscard]] request irecv(const mpi_communicator& c, R&&, count_t, tag) = delete;

  template<element_value T>
  requires(!std::is_const_v<T>)
  [[nodiscard]] request recv_init(const mpi_communicator& c, T& value, int source,
                                  tag tag = {}) {
    return detail::recv_init_scalar_impl(c, value, source, tag, "mp1::recv_init");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
           !is_region_v<std::remove_cvref_t<R>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>>)
  [[nodiscard]] request recv_init(const mpi_communicator& c, R&& buffer, int source,
                                  tag tag = {}) {
    return detail::recv_init_fixed_impl(c, buffer, source, tag, "mp1::recv_init");
  }

  template<class T>
  requires(!std::is_const_v<T>)
  [[nodiscard]] request recv_init(const mpi_communicator& c, region<T> buffer, int source,
                                  tag tag = {}) {
    return detail::recv_init_region_impl(c, buffer, source, tag, "mp1::recv_init");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R&> && resizable_element_buffer<R>)
  [[nodiscard]] request recv_init(const mpi_communicator& c, R& range, capacity bound,
                                  int source, tag tag = {}) {
    return detail::recv_init_capacity_impl(c, range, bound, source, tag, "mp1::recv_init");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R &&> &&
           resizable_element_buffer<std::remove_cvref_t<R>>)
  [[nodiscard]] request recv_init(const mpi_communicator& c, R&&, count_t, tag) = delete;

  // =========================================================================================
  // Probe and matched probe.  The matched handles feed mrecv/imrecv, which take `message&`
  // only: no rvalue duplication.
  // =========================================================================================

  /// Block until a matching message is available and return its status.
  ///
  /// A probe with source = MPI_PROC_NULL succeeds immediately and returns the standard
  /// MPI 4.1 §3.10 status (source MPI_PROC_NULL, tag `any_tag`, count 0), which the wrapper
  /// synthesizes as a normalized status: its count/elements are 0 and is_cancelled() is false
  /// without an MPI query. A probe of a real message returns the provider's native status.
  [[nodiscard]] inline status probe(const mpi_communicator& c, int source = MPI_ANY_SOURCE,
                                    tag tag = any_tag) {
    detail::p2p_require_comm(c, "mp1::probe");
    detail::p2p_require_source(c, source, "mp1::probe");
    detail::p2p_require_recv_tag(c, tag, "mp1::probe");
    MPI_Status native{};
    detail::check_mpi(MPI_Probe(source, tag.value, c.native_handle(), &native), {"MPI_Probe"});
    return detail::p2p_recv_status(source, native);
  }

  /// Nonblocking probe; nullopt means no matching message is available.
  ///
  /// A successful probe with source = MPI_PROC_NULL returns the standard MPI 4.1 §3.10
  /// status (source MPI_PROC_NULL, tag `any_tag`, count 0), synthesized with normalized
  /// provenance exactly like probe: its count/elements are 0 and is_cancelled() is false
  /// without an MPI query. A probe of a real message returns the provider's native status.
  [[nodiscard]] inline std::optional<status> iprobe(const mpi_communicator& c,
                                                    int source = MPI_ANY_SOURCE,
                                                    tag tag = any_tag) {
    detail::p2p_require_comm(c, "mp1::iprobe");
    detail::p2p_require_source(c, source, "mp1::iprobe");
    detail::p2p_require_recv_tag(c, tag, "mp1::iprobe");
    int flag{0};
    MPI_Status native{};
    detail::check_mpi(MPI_Iprobe(source, tag.value, c.native_handle(), &flag, &native),
                      {"MPI_Iprobe"});
    if (flag == 0)
      return std::nullopt;
    return detail::p2p_recv_status(source, native);
  }

  /// Block a matching probe and return the owning message handle plus its status.
  ///
  /// Unlike probe/iprobe, the matched message can only be received with MPI_Mrecv/MPI_Imrecv
  /// (MPI 4.1 §3.8.2): intervening receives cannot steal it, which is what makes the
  /// auto-sizing container receive safe under MPI_THREAD_MULTIPLE.  The returned message must
  /// be consumed, discarded or released; its destructor discards it (see mp1::message).
  [[nodiscard]] inline message mprobe(const mpi_communicator& c, int source = MPI_ANY_SOURCE,
                                      tag tag = any_tag) {
    detail::p2p_require_comm(c, "mp1::mprobe");
    detail::p2p_require_source(c, source, "mp1::mprobe");
    detail::p2p_require_recv_tag(c, tag, "mp1::mprobe");
    MPI_Message handle{MPI_MESSAGE_NULL};
    MPI_Status native{};
    detail::check_mpi(MPI_Mprobe(source, tag.value, c.native_handle(), &handle, &native),
                      {"MPI_Mprobe"});
    if (handle == MPI_MESSAGE_NULL) [[unlikely]] {
      detail::fail_with<std::runtime_error>("mp1::mprobe", std::source_location::current(),
                                            "MPI_Mprobe returned a null message handle");
    }
    return detail::message_access::make(handle, detail::p2p_probed_status(handle, native));
  }

  /// Nonblocking matching probe: nullopt means no matching message (state unchanged); an
  /// engaged message (including MPI_MESSAGE_NO_PROC for source = MPI_PROC_NULL) is a
  /// successful match that owns one message.
  [[nodiscard]] inline std::optional<message> improbe(const mpi_communicator& c,
                                                      int source = MPI_ANY_SOURCE,
                                                      tag tag = any_tag) {
    detail::p2p_require_comm(c, "mp1::improbe");
    detail::p2p_require_source(c, source, "mp1::improbe");
    detail::p2p_require_recv_tag(c, tag, "mp1::improbe");
    int flag{0};
    MPI_Message handle{MPI_MESSAGE_NULL};
    MPI_Status native{};
    detail::check_mpi(
        MPI_Improbe(source, tag.value, c.native_handle(), &flag, &handle, &native),
        {"MPI_Improbe"});
    if (flag == 0)
      return std::nullopt;
    if (handle == MPI_MESSAGE_NULL) [[unlikely]] {
      // A successful matching probe must return a message handle or MPI_MESSAGE_NO_PROC
      // (MPI 4.1 §3.8.2).  A providers' null handle here is a contract violation, not a
      // message; refusing it is safer than handing out an invalid owner.
      detail::fail_with<std::runtime_error>(
          "mp1::improbe", std::source_location::current(),
          "MPI_Improbe returned flag=true with a null MPI_Message");
    }
    return detail::message_access::make(handle, detail::p2p_probed_status(handle, native));
  }

  // =========================================================================================
  // Matched receive: five policies for mrecv, four for imrecv, `message&` only.
  // =========================================================================================

  template<element_value T>
  requires(!std::is_const_v<T>)
  status mrecv(T& value, message& msg) {
    return detail::mrecv_value_impl(value, msg, "mp1::mrecv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
           !is_region_v<std::remove_cvref_t<R>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>>)
  status mrecv(R&& buffer, message& msg) {
    return detail::mrecv_fixed_impl(buffer, msg, "mp1::mrecv");
  }

  template<class T>
  requires(!std::is_const_v<T>)
  status mrecv(region<T> buffer, message& msg) {
    return detail::mrecv_region_impl(buffer, msg, "mp1::mrecv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R&> && resizable_element_buffer<R>)
  status mrecv(R& range, message& msg) {
    return detail::mrecv_auto_impl(range, msg, "mp1::mrecv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R&> && resizable_element_buffer<R>)
  status mrecv(R& range, capacity bound, message& msg) {
    return detail::mrecv_bounded_impl(range, bound, msg, "mp1::mrecv");
  }

  template<element_value T>
  requires(!std::is_const_v<T>)
  [[nodiscard]] request imrecv(T& value, message& msg) {
    return detail::imrecv_value_impl(value, msg, "mp1::imrecv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R &&> && !element_value<std::remove_cvref_t<R>> &&
           !is_region_v<std::remove_cvref_t<R>> &&
           !resizable_element_buffer<std::remove_cvref_t<R>>)
  [[nodiscard]] request imrecv(R&& buffer, message& msg) {
    return detail::imrecv_fixed_impl(buffer, msg, "mp1::imrecv");
  }

  template<class T>
  requires(!std::is_const_v<T>)
  [[nodiscard]] request imrecv(region<T> buffer, message& msg) {
    return detail::imrecv_region_impl(buffer, msg, "mp1::imrecv");
  }

  template<class R>
  requires(buffer_at<buffer_role::recv, R&> && resizable_element_buffer<R>)
  [[nodiscard]] request imrecv(R& range, capacity bound, message& msg) {
    return detail::imrecv_bounded_impl(range, bound, msg, "mp1::imrecv");
  }

  // =========================================================================================
  // Combined send-receive: one declaration each.
  //
  // MPI 4.1 §3.2.7 requires the send and receive buffers to be disjoint; the wrapper rejects
  // only trivially identical/overlapping byte extents and points at sendrecv_replace for the
  // in-place form.  Both directions accept the three forms; the send and receive extents are
  // independent.  A temporary send buffer is accepted by the blocking form and rejected by
  // the nonblocking form (the request would capture a dead address).
  // =========================================================================================

  template<class S, class R>
  requires(send_source<S> && recv_sink<R>)
  status sendrecv(const mpi_communicator& c, S&& send, int dest, tag send_tag, R&& recv,
                  int source, tag recv_tag) {
    return detail::sendrecv_impl(c, send, dest, send_tag, recv, source, recv_tag,
                                 "mp1::sendrecv");
  }

  template<class B>
  requires(send_source<B> && recv_sink<B>)
  status sendrecv_replace(const mpi_communicator& c, B&& buffer, int dest, tag send_tag,
                          int source, tag recv_tag) {
    return detail::sendrecv_replace_impl(c, buffer, dest, send_tag, source, recv_tag,
                                         "mp1::sendrecv_replace");
  }

  template<class S, class R>
  requires(send_source<S> && recv_sink<R> && borrowable<S &&> && borrowable<R &&>)
  [[nodiscard]] request isendrecv(const mpi_communicator& c, S&& send, int dest, tag send_tag,
                                  R&& recv, int source, tag recv_tag) {
    return detail::isendrecv_impl(c, send, dest, send_tag, recv, source, recv_tag,
                                  "mp1::isendrecv");
  }

  template<class B>
  requires(send_source<B> && recv_sink<B> && borrowable<B &&>)
  [[nodiscard]] request isendrecv_replace(const mpi_communicator& c, B&& buffer, int dest,
                                          tag send_tag, int source, tag recv_tag) {
    return detail::isendrecv_replace_impl(c, buffer, dest, send_tag, source, recv_tag,
                                          "mp1::isendrecv_replace");
  }


}  // namespace mp1

#endif
