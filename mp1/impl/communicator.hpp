#if !(defined MP1_IMPL_COMMUNICATOR_HPP)

#define MP1_IMPL_COMMUNICATOR_HPP

#include "mp1/impl/detail/concepts.hpp"
#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/group.hpp"
#include "mp1/impl/info.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/status.hpp"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>


namespace mp1 {

  // MPI_Comm_split_type constants (MPI 4.1 adds resource-guided to the three MPI 4.0
  // constants); integer constants, no enum. split_type() also accepts implementation-specific
  // positive values.
  inline constexpr int comm_type_shared = MPI_COMM_TYPE_SHARED;
  inline constexpr int comm_type_hw_guided = MPI_COMM_TYPE_HW_GUIDED;
  inline constexpr int comm_type_hw_unguided = MPI_COMM_TYPE_HW_UNGUIDED;
  inline constexpr int comm_type_resource_guided = MPI_COMM_TYPE_RESOURCE_GUIDED;

  // MPI_BSEND_OVERHEAD as an int: a buffered-send buffer needs this many bytes beyond the
  // payload size. bsend_buffer(size) allocates size + bsend_overhead bytes.
  inline constexpr int bsend_overhead = MPI_BSEND_OVERHEAD;

  class mpi_communicator;

  namespace detail {
    void require_valid(const mpi_communicator& c, const context& ctx);
    std::atomic<bool>& parent_adopted() noexcept;

    struct comm_factory;

    /// Validate an MPI stringtag operation tag (MPI 4.1 §7.4, MPI_MAX_STRINGTAG_LEN includes
    /// the terminating NUL) and return a stable C string for the blocking call.  An empty view
    /// is legal and is passed as "".  An embedded NUL is rejected because passing data() would
    /// silently truncate the tag.
    [[nodiscard]] inline const char* stringtag_text(
        std::string_view stringtag, const context& ctx,
        std::source_location where = std::source_location::current()) {
      if (stringtag.find('\0') != std::string_view::npos) [[unlikely]] {
        throw_invalid(ctx.operation, "stringtag contains an embedded NUL byte", where);
      }
      if (stringtag.size() >= static_cast<std::size_t>(MPI_MAX_STRINGTAG_LEN)) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            ctx, where,
            "stringtag length {} is not less than the MPI_MAX_STRINGTAG_LEN "
            "limit ({})",
            stringtag.size(), MPI_MAX_STRINGTAG_LEN);
      }
      return stringtag.empty() ? "" : stringtag.data();
    }

    /// Payload of an MPI_Comm_idup request.  The output MPI_Comm lives in a heap cell whose
    /// address is stable across request moves and release(); MPI writes it asynchronously up to
    /// completion.  complete() only records the completion; the owning communicator view is
    /// built later by take_communicator(), outside the noexcept completion transaction.
    class idup_payload final : public request_payload {
    public:
      idup_payload() : slot_{std::make_unique<MPI_Comm>(MPI_COMM_NULL)} {
      }

      [[nodiscard]] MPI_Comm* slot() noexcept {
        return slot_.get();
      }

      void complete(const status&) noexcept override {
        completed_ = true;
      }

      [[nodiscard]] bool retains_result() const noexcept override {
        return true;
      }

      /// Transfer the completed duplicate.  Must not be called before complete(); returns
      /// MPI_COMM_NULL when the output was null or the payload was never completed.
      [[nodiscard]] MPI_Comm release_comm() noexcept override {
        if (not completed_)
          return MPI_COMM_NULL;
        return std::exchange(*slot_, MPI_COMM_NULL);
      }

      ~idup_payload() noexcept override {
        if (slot_ == nullptr or *slot_ == MPI_COMM_NULL)
          return;
        // A result_pending request's destructor disposes of the duplicate: MPI_Comm_free is
        // collective and can block.  After finalize, or when the slot may not have been
        // written by a completed operation, abandon and count it instead.
        if (not completed_ or not mpi_active()) [[unlikely]] {
          note_abandoned_handle();
          return;
        }
        MPI_Comm handle = std::exchange(*slot_, MPI_COMM_NULL);
        if (MPI_Comm_free(&handle) != MPI_SUCCESS) [[unlikely]]
          note_abandoned_handle();
      }

    private:
      std::unique_ptr<MPI_Comm> slot_;
      bool completed_{false};
    };
  }  // namespace detail

  /// The owning parent intercommunicator factory.
  [[nodiscard]] communicator parent();

  /// Non-owning, copyable, borrowed view of an MPI communicator.
  ///
  /// A view holds a copied MPI handle, not a reference: it is valid only while the MPI object
  /// it borrows is live, so the owner must outlive every view.  free(), disconnect(), release(),
  /// the owner's destruction and MPI_Finalize (including an external one) invalidate every view
  /// of the object.  mp1 cannot detect use of an invalidated view (the handle is a copied value
  /// and native_handle() is a documented raw escape), so such use is a precondition violation
  /// that normally results in an MPI error or a crash inside MPI.  is_valid() only reports
  /// `handle != MPI_COMM_NULL`; it does not promise liveness.
  ///
  /// Constructing a view of a valid raw handle requires active MPI: it eagerly caches the local
  /// size, local rank, inter flag and (for an intercommunicator) the remote size, so every later
  /// validation is MPI-free.  That cache is a snapshot: it never refreshes and is not an
  /// aliveness check.  A valid handle constructed while MPI is inactive (before MPI_Init or
  /// after MPI_Finalize) throws std::runtime_error; there is no deferred view.  Default
  /// construction and mpi_communicator{MPI_COMM_NULL} are MPI-free and remain valid in every
  /// lifecycle state.  The destructor never frees and never calls MPI.
  ///
  /// A const view is a borrowed handle, not a const native object: operations that mutate the
  /// shared communicator through the handle (set_info(), the set_error_handler() free function
  /// and abort()) are const members or take a const view, because constness protects the view's
  /// own cached state and not the MPI object that every copy of the handle refers to.
  class mpi_communicator {
  public:
    /// Invalid view (MPI_COMM_NULL); no MPI call.
    mpi_communicator() noexcept = default;

    /// Borrow h and cache its metadata.  MPI_COMM_NULL is MPI-free; a non-null handle requires
    /// active MPI and throws std::runtime_error when MPI is inactive.
    explicit mpi_communicator(MPI_Comm h) : handle_{h} {
      if (handle_ != MPI_COMM_NULL)
        cache_metadata();
    }

    /// Named form of the borrowing constructor; see mpi_communicator(MPI_Comm).
    [[nodiscard]] static mpi_communicator borrow(MPI_Comm h) {
      return mpi_communicator{h};
    }

    mpi_communicator(const mpi_communicator&) noexcept = default;
    mpi_communicator& operator=(const mpi_communicator&) noexcept = default;
    mpi_communicator(mpi_communicator&&) noexcept = default;
    mpi_communicator& operator=(mpi_communicator&&) noexcept = default;
    ~mpi_communicator() noexcept = default;

    /// \return true if the handle is not MPI_COMM_NULL; no MPI call
    [[nodiscard]] bool is_valid() const noexcept {
      return handle_ != MPI_COMM_NULL;
    }

    /// \return non-owning native handle; safe after finalize
    [[nodiscard]] MPI_Comm native_handle() const noexcept {
      return handle_;
    }

    /// \return cached local size; requires a valid view (and therefore active MPI at construction)
    [[nodiscard]] int size() const {
      require_valid_view("mp1::mpi_communicator::size");
      return size_;
    }

    /// \return cached local rank; requires a valid view (and therefore active MPI at construction)
    [[nodiscard]] int rank() const {
      require_valid_view("mp1::mpi_communicator::rank");
      return rank_;
    }

    /// \return cached MPI_Comm_test_inter result; requires a valid view
    [[nodiscard]] bool is_inter() const {
      require_valid_view("mp1::mpi_communicator::is_inter");
      return inter_;
    }

    /// \return cached remote group size; requires a valid intercommunicator view
    [[nodiscard]] int remote_size() const {
      require_valid_view("mp1::mpi_communicator::remote_size");
      if (not inter_) [[unlikely]]
        throw_invalid("mp1::mpi_communicator::remote_size",
                      "communicator is not an intercommunicator");
      return remote_size_;
    }

    /// \return cached largest legal tag value (MPI_TAG_UB, inclusive); no MPI call.  Queried
    /// once when this view was constructed; falls back to tag_ub_min when the implementation
    /// does not expose the attribute.
    [[nodiscard]] int tag_ub() const {
      require_valid_view("mp1::mpi_communicator::tag_ub");
      return tag_ub_;
    }

    /// MPI_Comm_remote_group; the owning wrapper of the remote group.
    [[nodiscard]] group remote_group() const {
      require_active_view("mp1::mpi_communicator::remote_group");
      if (not inter_) [[unlikely]]
        throw_invalid("mp1::mpi_communicator::remote_group",
                      "communicator is not an intercommunicator");
      MPI_Group result{MPI_GROUP_NULL};
      detail::check_mpi(MPI_Comm_remote_group(handle_, &result), "MPI_Comm_remote_group");
      return detail::group_factory::make(result);
    }

    /// MPI_Comm_compare; both operands must be valid and MPI must be active.  This always issues
    /// the MPI call; operator== short-circuits identical handles before reaching it.
    [[nodiscard]] equality_type compare(const mpi_communicator& other) const {
      require_active_view("mp1::mpi_communicator::compare");
      detail::require_valid(other, "mp1::mpi_communicator::compare");
      int result{MPI_UNEQUAL};
      detail::check_mpi(MPI_Comm_compare(handle_, other.handle_, &result), "MPI_Comm_compare");
      return static_cast<equality_type>(result);
    }

    /// True iff both views are invalid or their handles compare MPI_IDENT.  Identical handles are
    /// equal without an MPI call, so two copies of the same view compare equal even after
    /// finalize; distinct valid handles call compare() and require active MPI.
    [[nodiscard]] bool operator==(const mpi_communicator& other) const {
      if (handle_ == other.handle_)
        return true;
      if (not is_valid() or not other.is_valid())
        return false;
      return compare(other) == equality_type::identical;
    }

    [[nodiscard]] bool operator!=(const mpi_communicator& other) const {
      return not(*this == other);
    }

    /// MPI_Comm_get_info; the returned info owns a newly allocated native table.
    [[nodiscard]] info get_info() const {
      require_active_view("mp1::mpi_communicator::get_info");
      MPI_Info native{MPI_INFO_NULL};
      detail::check_mpi(MPI_Comm_get_info(handle_, &native), "MPI_Comm_get_info");
      return detail::info_access::adopt(native);
    }

    /// MPI_Comm_set_info; collective and non-const on the native communicator.  The method is
    /// const on the view because a view is a borrowed handle: constness protects the view's own
    /// cached state, not the shared MPI object (the same rule as set_error_handler and abort).
    /// A null info throws locally (MPI implementations reject MPI_INFO_NULL here).
    void set_info(const info& i) const {
      require_active_view("mp1::mpi_communicator::set_info");
      if (i.native_handle() == MPI_INFO_NULL) [[unlikely]]
        throw_invalid("mp1::mpi_communicator::set_info", "null info");
      detail::check_mpi(MPI_Comm_set_info(handle_, i.native_handle()), "MPI_Comm_set_info");
    }

    /// MPI_Abort; if it returns, throws mpi_error.
    void abort(int error_code) const {
      require_active_view("mp1::mpi_communicator::abort");
      detail::check_mpi(MPI_Abort(handle_, error_code), "MPI_Abort");
    }

  private:
    MPI_Comm handle_{MPI_COMM_NULL};
    int size_{0};
    int rank_{0};
    int remote_size_{0};
    int tag_ub_{tag_ub_min.value};
    bool inter_{false};

    /// Fill the metadata snapshot.  Only reached for a non-null handle, and only while MPI is
    /// active: a deferred view is not supported (see the class comment).
    void cache_metadata() {
      detail::require_mpi_checked("mp1::mpi_communicator");
      int size{0};
      int rank{0};
      int inter{0};
      detail::check_mpi(MPI_Comm_size(handle_, &size), "MPI_Comm_size");
      detail::check_mpi(MPI_Comm_rank(handle_, &rank), "MPI_Comm_rank");
      detail::check_mpi(MPI_Comm_test_inter(handle_, &inter), "MPI_Comm_test_inter");
      int remote{0};
      if (inter != 0) {
        detail::check_mpi(MPI_Comm_remote_size(handle_, &remote), "MPI_Comm_remote_size");
      }
      // MPI_TAG_UB is a predefined communicator attribute (MPI 4.1 §3.2.3); the value is an
      // int and the bound is inclusive.  It is queried exactly once here, so no later tag
      // validation makes an MPI call.  The attribute must be present for a communicator; the
      // tag_ub_min fallback keeps a nonconforming provider usable without weakening the
      // guaranteed minimum.
      int tag_attribute{tag_ub_min.value};
      int tag_flag{0};
      void* tag_value{nullptr};
      detail::check_mpi(MPI_Comm_get_attr(handle_, MPI_TAG_UB, &tag_value, &tag_flag),
                        "MPI_Comm_get_attr(MPI_TAG_UB)");
      if (tag_flag != 0 and tag_value != nullptr)
        tag_attribute = *static_cast<int*>(tag_value);
      size_ = size;
      rank_ = rank;
      inter_ = inter != 0;
      remote_size_ = inter_ != 0 ? remote : 0;
      tag_ub_ = tag_attribute;
    }

    void require_valid_view(const context& ctx) const {
      if (not is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid communicator");
    }

    void require_active_view(const context& ctx) const {
      require_valid_view(ctx);
      detail::require_mpi(ctx);
    }
  };

  /// Destructor policy of an owning communicator.  Fixed at construction and never inferred
  /// from the handle value or from MPI_Comm_test_inter.  `free` uses MPI_Comm_free; `disconnect`
  /// uses MPI_Comm_disconnect, which is the release operation of the two ends of a spawn or
  /// connect intercommunicator (MPI 4.1 §11.10.4 Releasing Connections: MPI_COMM_DISCONNECT is
  /// collective, waits for decoupled activities and enables the disconnected-process guarantee;
  /// MPI_COMM_FREE explicitly does not disconnect; §7.4.3 Communicator Destructors).
  enum class release_policy : unsigned char {
    free,        ///< MPI_Comm_free; every ordinary factory and adopt(MPI_Comm)
    disconnect,  ///< MPI_Comm_disconnect; spawn/spawn_multiple/parent/adopt_connected
  };

  /// Unique owning communicator.  Move-only.  It contains a private mpi_communicator view and
  /// exposes it by value through view()/an implicit conversion to mpi_communicator; it does not
  /// publicly derive from the view, so a mpi_communicator& cannot be bound to the owner and the
  /// owner's handle cannot be overwritten, sliced or deleted through a view.
  ///
  /// The destructor performs exactly one policy operation (MPI_Comm_free or
  /// MPI_Comm_disconnect), is noexcept, may block (both are collective), and abandons + counts
  /// the handle when MPI is no longer active.  free()/disconnect() are single-attempt and
  /// transactional: on an MPI error the object becomes terminal (the handle is counted as
  /// abandoned) and mpi_error is thrown; a collective teardown is not retried.  release() is
  /// the no-MPI-call escape that hands the raw handle to the caller.
  ///
  /// Move construction transfers ownership without an MPI call.  Move assignment to a live
  /// target is a teardown path exactly like free()/disconnect()/the destructor: it performs the
  /// target's policy operation (MPI_Comm_free or MPI_Comm_disconnect), which is collective over
  /// the communicator's group and may block.  Every rank must therefore reach an assignment over
  /// a live owner in the same way as destruction.  Like the destructor it is noexcept and
  /// abandons the old handle on an MPI error.
  class communicator {
  public:
    /// Invalid, owning nothing; no MPI call.
    communicator() noexcept = default;

    communicator(communicator&& other) noexcept
        : view_{other.view_}, policy_{other.policy_}, parent_{other.parent_} {
      other.reset_owner_state();
    }

    communicator& operator=(communicator&& other) noexcept {
      if (this != &other) {
        release_owner();
        view_ = other.view_;
        policy_ = other.policy_;
        parent_ = other.parent_;
        other.reset_owner_state();
      }
      return *this;
    }

    communicator(const communicator&) = delete;
    communicator& operator=(const communicator&) = delete;

    /// Frees or disconnects per the policy iff valid and MPI is active. Collective and
    /// potentially blocking. No MPI call after finalize.
    ~communicator() noexcept {
      release_owner();
    }

    /// \return the borrowed, non-owning view by value.  The result is a copied handle plus the
    /// metadata snapshot: it stays valid after this owner is moved, but only while the MPI
    /// object is live, so the (new) owner must outlive it.
    [[nodiscard]] mpi_communicator view() const noexcept {
      return view_;
    }

    /// Implicit borrow: pass an owner wherever a `mpi_communicator` or
    /// `const mpi_communicator&` is expected.  Returns a by-value copy of the borrowed view, not
    /// a reference into the movable owner, so it is unaffected by moving the owner.  The copy
    /// never transfers ownership and is invalidated by free()/disconnect()/release()/the
    /// owner's destruction/finalize (documented precondition, not checked).
    operator mpi_communicator() const noexcept {
      return view_;
    }

    /// \return true iff the owner currently holds a handle; not a liveness check for views
    [[nodiscard]] bool is_valid() const noexcept {
      return view_.is_valid();
    }

    /// \return non-owning native handle; safe after finalize
    [[nodiscard]] MPI_Comm native_handle() const noexcept {
      return view_.native_handle();
    }

    /// \return cached local size; requires a valid owner
    [[nodiscard]] int size() const {
      return view_.size();
    }

    /// \return cached local rank; requires a valid owner
    [[nodiscard]] int rank() const {
      return view_.rank();
    }

    /// \return cached MPI_Comm_test_inter result; requires a valid owner
    [[nodiscard]] bool is_inter() const {
      return view_.is_inter();
    }

    /// \return cached remote group size; requires a valid intercommunicator owner
    [[nodiscard]] int remote_size() const {
      return view_.remote_size();
    }

    /// \return cached largest legal tag value (MPI_TAG_UB, inclusive); no MPI call
    [[nodiscard]] int tag_ub() const {
      return view_.tag_ub();
    }

    /// MPI_Comm_remote_group; the owning wrapper of the remote group.
    [[nodiscard]] group remote_group() const {
      return view_.remote_group();
    }

    /// MPI_Comm_compare; both operands must be valid.
    [[nodiscard]] equality_type compare(const mpi_communicator& other) const {
      return view_.compare(other);
    }

    /// True iff both views are invalid or their handles compare MPI_IDENT.
    [[nodiscard]] bool operator==(const mpi_communicator& other) const {
      return view_ == other;
    }

    [[nodiscard]] bool operator!=(const mpi_communicator& other) const {
      return not(view_ == other);
    }

    /// MPI_Comm_get_info; the returned info owns a newly allocated native table.
    [[nodiscard]] info get_info() const {
      return view_.get_info();
    }

    /// MPI_Comm_set_info; collective and mutates the shared native communicator through the
    /// borrowed view.
    void set_info(const info& i) const {
      view_.set_info(i);
    }

    /// MPI_Abort; if it returns, throws mpi_error.
    void abort(int error_code) const {
      view_.abort(error_code);
    }

    /// \return the destructor policy fixed at construction
    [[nodiscard]] release_policy policy() const noexcept {
      return policy_;
    }

    /// Take ownership of a raw communicator handle (not null/world/self). Only call this for
    /// a handle whose ownership was transferred to the caller. The owner frees with
    /// MPI_Comm_free.
    [[nodiscard]] static communicator adopt(MPI_Comm h) {
      detail::require_mpi("mp1::communicator::adopt");
      if (h == MPI_COMM_NULL) [[unlikely]]
        throw_invalid("mp1::communicator::adopt", "MPI_COMM_NULL");
      if (h == MPI_COMM_WORLD or h == MPI_COMM_SELF) [[unlikely]] {
        throw_invalid("mp1::communicator::adopt",
                      "MPI_COMM_WORLD/MPI_COMM_SELF are not owned by mp1");
      }
      return take_result(h, release_policy::free);
    }

    /// Take ownership of a raw peer intercommunicator with disconnect policy: use this for a
    /// handle returned by raw MPI_Comm_spawn/MPI_Comm_connect/MPI_Comm_accept/MPI_Comm_join.
    /// Both ends of the connection must disconnect (MPI 4.1 §11.10.4 Releasing Connections);
    /// free() is rejected on the result and release() is the non-cooperating-peer escape.
    /// Do not wrap the MPI_Comm_get_parent handle with this: use mp1::parent() so the one-shot
    /// adoption flag is recorded (MPI 4.1 §11.8.2, p. 524).
    [[nodiscard]] static communicator adopt_connected(MPI_Comm h) {
      detail::require_mpi("mp1::communicator::adopt_connected");
      if (h == MPI_COMM_NULL) [[unlikely]]
        throw_invalid("mp1::communicator::adopt_connected", "MPI_COMM_NULL");
      if (h == MPI_COMM_WORLD or h == MPI_COMM_SELF) [[unlikely]] {
        throw_invalid("mp1::communicator::adopt_connected",
                      "MPI_COMM_WORLD/MPI_COMM_SELF are not owned by mp1");
      }
      return take_result(h, release_policy::disconnect);
    }

    [[nodiscard]] static communicator dup(const mpi_communicator& other) {
      require_factory_operand(other, "mp1::communicator::dup");
      MPI_Comm result{MPI_COMM_NULL};
      detail::check_mpi(MPI_Comm_dup(other.native_handle(), &result), "MPI_Comm_dup");
      return take_result(result, release_policy::free);
    }

    [[nodiscard]] static communicator dup_with_info(const mpi_communicator& other,
                                                    const info& i) {
      require_factory_operand(other, "mp1::communicator::dup_with_info");
      MPI_Comm result{MPI_COMM_NULL};
      detail::check_mpi(
          MPI_Comm_dup_with_info(other.native_handle(), i.native_handle(), &result),
          "MPI_Comm_dup_with_info");
      return take_result(result, release_policy::free);
    }

    /// MPI_Comm_idup.  Nonblocking duplicate: the returned request is a nonblocking collective
    /// request (request_kind::collective) and the duplicate is available only after
    /// wait/test/bulk completion.  On success the request enters
    /// request_state::result_pending and owns the MPI_Comm; transfer it with
    /// take_communicator(request&), or destroy the request to free it.  The source
    /// communicator must stay valid until the request completes, and every rank must issue the
    /// idup calls in the same order on the same source context.  The duplicate inherits the
    /// source error handler, exactly like MPI_Comm_dup (MPI 4.1 §7.4).
    [[nodiscard]] static request idup(const mpi_communicator& other) {
      require_factory_operand(other, "mp1::communicator::idup");
      auto payload = std::make_unique<detail::idup_payload>();
      MPI_Request handle{MPI_REQUEST_NULL};
      detail::check_mpi(MPI_Comm_idup(other.native_handle(), payload->slot(), &handle),
                        "MPI_Comm_idup");
      request result = request::from_collective(handle);
      detail::attach_payload(result, std::move(payload));
      return result;
    }

    /// MPI_Comm_idup_with_info.  As idup(), with the info hints associated with the new
    /// communicator.  The hints are captured at initiation (MPI 4.1 §7.4), so the info object
    /// may be destroyed after the call returns.
    [[nodiscard]] static request idup_with_info(const mpi_communicator& other,
                                                const info& hints) {
      require_factory_operand(other, "mp1::communicator::idup_with_info");
      auto payload = std::make_unique<detail::idup_payload>();
      MPI_Request handle{MPI_REQUEST_NULL};
      detail::check_mpi(MPI_Comm_idup_with_info(other.native_handle(), hints.native_handle(),
                                                payload->slot(), &handle),
                        "MPI_Comm_idup_with_info");
      request result = request::from_collective(handle);
      detail::attach_payload(result, std::move(payload));
      return result;
    }

    [[nodiscard]] static communicator split(const mpi_communicator& other, int color,
                                            int key = 0) {
      require_factory_operand(other, "mp1::communicator::split");
      if (color != MPI_UNDEFINED and color < 0) [[unlikely]] {
        detail::fail_as<std::invalid_argument>({"mp1::communicator::split"},
                                               std::source_location::current(),
                                               "color {} is negative", color);
      }
      MPI_Comm result{MPI_COMM_NULL};
      detail::check_mpi(MPI_Comm_split(other.native_handle(), color, key, &result),
                        "MPI_Comm_split");
      return take_result(result, release_policy::free);
    }

    [[nodiscard]] static communicator split_type(const mpi_communicator& other, int split_type,
                                                 int key = 0, const info& i = {}) {
      require_factory_operand(other, "mp1::communicator::split_type");
      if (split_type != MPI_UNDEFINED and split_type < 0) [[unlikely]] {
        detail::fail_as<std::invalid_argument>({"mp1::communicator::split_type"},
                                               std::source_location::current(),
                                               "split_type {} is negative", split_type);
      }
      MPI_Comm result{MPI_COMM_NULL};
      detail::check_mpi(MPI_Comm_split_type(other.native_handle(), split_type, key,
                                            i.native_handle(), &result),
                        "MPI_Comm_split_type");
      return take_result(result, release_policy::free);
    }

    [[nodiscard]] static communicator create(const mpi_communicator& other, const group& gr) {
      require_factory_operand(other, "mp1::communicator::create");
      if (not gr.is_valid()) [[unlikely]]
        throw_invalid("mp1::communicator::create", "invalid group");
      MPI_Comm result{MPI_COMM_NULL};
      detail::check_mpi(MPI_Comm_create(other.native_handle(), gr.native_handle(), &result),
                        "MPI_Comm_create");
      return take_result(result, release_policy::free);
    }

    [[nodiscard]] static communicator create_group(const mpi_communicator& other,
                                                   const group& gr, tag tag = {}) {
      require_factory_operand(other, "mp1::communicator::create_group");
      if (other.is_inter()) [[unlikely]] {
        throw_invalid("mp1::communicator::create_group",
                      "intercommunicators are not supported");
      }
      if (not gr.is_valid()) [[unlikely]]
        throw_invalid("mp1::communicator::create_group", "invalid group");
      // The group-creation tag does not conflict with message tags, but it is a tag in the
      // same legal range and must not be a wildcard (MPI 4.1 §7.4.4).
      if (tag == any_tag) [[unlikely]]
        detail::fail_as<std::invalid_argument>({"mp1::communicator::create_group"},
                                               std::source_location::current(),
                                               "MPI_ANY_TAG is not a group-creation tag");
      if (tag.value < 0) [[unlikely]]
        detail::fail_as<std::invalid_argument>({"mp1::communicator::create_group"},
                                               std::source_location::current(),
                                               "tag {} is negative", tag.value);
      if (tag.value > other.tag_ub()) [[unlikely]]
        detail::fail_as<std::invalid_argument>(
            {"mp1::communicator::create_group"}, std::source_location::current(),
            "tag {} exceeds MPI_TAG_UB {}", tag.value, other.tag_ub());
      MPI_Comm result{MPI_COMM_NULL};
      detail::check_mpi(
          MPI_Comm_create_group(other.native_handle(), gr.native_handle(), tag.value, &result),
          "MPI_Comm_create_group");
      return take_result(result, release_policy::free);
    }

    /// MPI_Comm_create_from_group.  Collective over the members of `gr` only (MPI 4.1 §7.4):
    /// every member must pass the same group and stringtag.  An empty group is a local
    /// operation and returns an invalid owner without an MPI creation call.  A non-member call
    /// is MPI-erroneous and deliberately not diagnosed locally; MPI is the authority.
    ///
    /// `hints` defaults to `mp1::info{}` (MPI_INFO_NULL) and `handler` defaults to
    /// `mp1::error_handler::are_fatal`, matching MPI's default initial error handler for
    /// communicators.  Under MPI 4.1 §9.3.1 the handler must be a predefined handler or one
    /// created by MPI_Comm_create_errhandler; MPI_ERRHANDLER_NULL is the null handle and is not
    /// a valid handler.  mp1::error_handler models only the predefined handlers, so a custom
    /// handler is installed with raw MPI or with MPI_Comm_set_errhandler on native_handle()
    /// after creation.  A library that has called install_recoverable_errors() passes
    /// `mp1::error_handler::returns` explicitly: these source-less factories do not inherit
    /// the world/self handler.
    [[nodiscard]] static communicator create_from_group(
        const group& gr, std::string_view stringtag, const info& hints = {},
        error_handler handler = error_handler::are_fatal) {
      const context ctx{"mp1::communicator::create_from_group"};
      if (not gr.is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid group");
      detail::require_mpi(ctx);
      const char* native_tag = detail::stringtag_text(stringtag, ctx);
      const MPI_Errhandler native_handler = detail::native_error_handler(ctx, handler);
      if (gr.is_empty())
        return communicator{};
      MPI_Comm result{MPI_COMM_NULL};
      detail::check_mpi(
          MPI_Comm_create_from_group(gr.native_handle(), native_tag, hints.native_handle(),
                                     native_handler, &result),
          "MPI_Comm_create_from_group");
      return take_result(result, release_policy::free);
    }

    /// MPI_Intercomm_create_from_groups.  Collective over the union of the local and remote
    /// groups (MPI 4.1 §7.6).  `local_group` and `remote_group` must be disjoint; within each
    /// group every process passes the same local group/leader and all processes pass the same
    /// stringtag.  Non-membership is MPI-erroneous and is deliberately left to MPI.
    ///
    /// `remote_group` and `remote_leader` are significant only at `local_leader`, so only the
    /// leader validates them; a non-leader may pass group{} and an arbitrary non-wildcard
    /// remote_leader value (wildcards are not permitted for either leader).  An
    /// empty local group is a local operation returning an invalid owner.  `hints` defaults to
    /// `mp1::info{}` and `handler` defaults to `mp1::error_handler::are_fatal`, with the MPI 4.1
    /// §9.3.1 validity rule described at create_from_group().  The result is released with
    /// MPI_Comm_free (release_policy::free), not MPI_Comm_disconnect.
    [[nodiscard]] static communicator intercomm_create_from_groups(
        const group& local_group, int local_leader, const group& remote_group,
        int remote_leader, std::string_view stringtag, const info& hints = {},
        error_handler handler = error_handler::are_fatal) {
      const context ctx{"mp1::communicator::intercomm_create_from_groups"};
      if (not local_group.is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid local group");
      detail::require_mpi(ctx);
      const char* native_tag = detail::stringtag_text(stringtag, ctx);
      const MPI_Errhandler native_handler = detail::native_error_handler(ctx, handler);
      if (local_group.is_empty())
        return communicator{};
      // Read the rank only to select the leader branch; a non-member is treated as a
      // non-leader and the erroneous call is left to MPI (no local throw).
      const int local_rank = local_group.rank();
      if (local_rank != MPI_UNDEFINED) {
        if (local_leader < 0 or local_leader >= local_group.size()) [[unlikely]] {
          detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                                 "local_leader {} is outside [0,{})",
                                                 local_leader, local_group.size());
        }
        if (local_rank == local_leader) {
          if (not remote_group.is_valid()) [[unlikely]]
            throw_invalid(ctx.operation, "invalid remote group");
          if (not remote_group.is_empty() and
              (remote_leader < 0 or remote_leader >= remote_group.size())) [[unlikely]] {
            detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                                   "remote_leader {} is outside [0,{})",
                                                   remote_leader, remote_group.size());
          }
        }
      }
      MPI_Comm result{MPI_COMM_NULL};
      detail::check_mpi(
          MPI_Intercomm_create_from_groups(
              local_group.native_handle(), local_leader, remote_group.native_handle(),
              remote_leader, native_tag, hints.native_handle(), native_handler, &result),
          "MPI_Intercomm_create_from_groups");
      return take_result(result, release_policy::free);
    }

    [[nodiscard]] static communicator merge(const mpi_communicator& inter, bool high) {
      require_factory_operand(inter, "mp1::communicator::merge");
      if (not inter.is_inter()) [[unlikely]] {
        throw_invalid("mp1::communicator::merge", "operand is not an intercommunicator");
      }
      MPI_Comm result{MPI_COMM_NULL};
      detail::check_mpi(MPI_Intercomm_merge(inter.native_handle(), high ? 1 : 0, &result),
                        "MPI_Intercomm_merge");
      return take_result(result, release_policy::free);
    }

    /// MPI_Intercomm_create; collective over the union of the local and remote groups
    /// (MPI 4.1 §7.6.2, p. 353).  `local` and `local_leader` are inputs of every local
    /// process and are validated on every local rank.  `peer` and `remote_leader` are
    /// "significant only at the local_leader", so they are validated (and only used) when
    /// `local.rank() == local_leader`; a non-leader may pass a default (MPI_COMM_NULL) peer
    /// and an arbitrary non-wildcard remote_leader value and must still enter the collective
    /// call.  Every process
    /// in a group must pass the same local/local_leader/tag; mp1 does not add a hidden
    /// agreement collective to check that (it is the application's MPI obligation).
    [[nodiscard]] static communicator intercomm_create(const mpi_communicator& local,
                                                       int local_leader,
                                                       const mpi_communicator& peer,
                                                       int remote_leader, tag tag = {}) {
      require_factory_operand(local, "mp1::communicator::intercomm_create");
      if (local.is_inter()) [[unlikely]] {
        throw_invalid("mp1::communicator::intercomm_create",
                      "local must be an intracommunicator");
      }
      if (local_leader < 0 or local_leader >= local.size()) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            {"mp1::communicator::intercomm_create"}, std::source_location::current(),
            "local_leader {} is outside [0,{})", local_leader, local.size());
      }
      if (tag == any_tag) [[unlikely]] {
        detail::fail_as<std::invalid_argument>({"mp1::communicator::intercomm_create"},
                                               std::source_location::current(),
                                               "MPI_ANY_TAG is not an intercommunicator tag");
      }
      if (tag.value < 0) [[unlikely]] {
        detail::fail_as<std::invalid_argument>({"mp1::communicator::intercomm_create"},
                                               std::source_location::current(),
                                               "tag {} is negative", tag.value);
      }
      if (tag.value > local.tag_ub()) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            {"mp1::communicator::intercomm_create"}, std::source_location::current(),
            "tag {} exceeds MPI_TAG_UB {}", tag.value, local.tag_ub());
      }
      if (local.rank() == local_leader) {
        // Leader-only arguments; MPI ignores them at every other local rank.
        require_factory_operand(peer, "mp1::communicator::intercomm_create");
        if (peer.is_inter()) [[unlikely]] {
          throw_invalid("mp1::communicator::intercomm_create",
                        "peer must be an intracommunicator");
        }
        if (remote_leader < 0 or remote_leader >= peer.size()) [[unlikely]] {
          detail::fail_as<std::invalid_argument>(
              {"mp1::communicator::intercomm_create"}, std::source_location::current(),
              "remote_leader {} is outside [0,{})", remote_leader, peer.size());
        }
      }
      MPI_Comm result{MPI_COMM_NULL};
      detail::check_mpi(
          MPI_Intercomm_create(local.native_handle(), local_leader, peer.native_handle(),
                               remote_leader, tag.value, &result),
          "MPI_Intercomm_create");
      return take_result(result, release_policy::free);
    }

    /// MPI_Comm_free; invalid is a no-op.  Only a `free`-policy owner may call this: on a
    /// `disconnect`-policy owner (spawn/spawn_multiple/parent/adopt_connected) it throws
    /// std::logic_error before any MPI call because MPI_Comm_free does not disconnect and the
    /// peer's MPI_Comm_disconnect would block (MPI 4.1 §11.10.4, p. 544).  Single attempt:
    /// on an MPI error the handle is counted as abandoned and mpi_error is thrown.
    void free() {
      if (not is_valid())
        return;
      detail::require_mpi_checked("mp1::communicator::free");
      if (policy_ != release_policy::free) [[unlikely]] {
        detail::fail_with<std::logic_error>(
            "mp1::communicator::free", std::source_location::current(),
            "this owner has disconnect policy; call disconnect() (both ends of a "
            "spawned/connected intercommunicator must disconnect) or release() for a "
            "non-cooperating peer");
      }
      if (native_handle() == MPI_COMM_WORLD or native_handle() == MPI_COMM_SELF) [[unlikely]] {
        detail::fail_with<std::logic_error>("mp1::communicator::free",
                                            std::source_location::current(),
                                            "MPI_COMM_WORLD/MPI_COMM_SELF cannot be freed");
      }
      MPI_Comm handle = native_handle();
      const int rc = MPI_Comm_free(&handle);
      reset_owner_state();
      if (rc != MPI_SUCCESS) [[unlikely]] {
        detail::note_abandoned_handle();
        detail::check_mpi(rc, "MPI_Comm_free");
      }
    }

    /// MPI_Comm_disconnect; owner-only, collective, waits for decoupled activities and is the
    /// last MPI operation on the handle (MPI 4.1 §11.10.4, p. 544).  Available on every valid
    /// owner except MPI_COMM_WORLD/MPI_COMM_SELF; invalid is a no-op.  Single attempt: on an
    /// MPI error the handle is counted as abandoned, the one-shot parent adoption flag is
    /// cleared (exactly as in release_owner()), and mpi_error is thrown; the collective teardown
    /// is not retried.
    void disconnect() {
      if (not is_valid())
        return;
      detail::require_mpi_checked("mp1::communicator::disconnect");
      if (native_handle() == MPI_COMM_WORLD or native_handle() == MPI_COMM_SELF) [[unlikely]] {
        detail::fail_with<std::logic_error>(
            "mp1::communicator::disconnect", std::source_location::current(),
            "MPI_COMM_WORLD/MPI_COMM_SELF cannot be disconnected");
      }
      MPI_Comm handle = native_handle();
      const bool was_parent = parent_;
      const int rc = MPI_Comm_disconnect(&handle);
      reset_owner_state();
      // Clear the process-global one-shot flag on every terminal path, including a failed
      // MPI_Comm_disconnect: the owner is terminal either way, so leaving the flag set would
      // reject every later parent() adoption for a handle no mp1 owner holds (release_owner()
      // clears it before inspecting rc for the same reason).
      if (was_parent)
        detail::parent_adopted().store(false);
      if (rc != MPI_SUCCESS) [[unlikely]] {
        detail::note_abandoned_handle();
        detail::check_mpi(rc, "MPI_Comm_disconnect");
      }
    }

    /// Detach the raw handle and give up ownership without any MPI call (MPI 4.1 §11.10.4: the
    /// caller becomes responsible for releasing it, and must choose the operation that matches
    /// the peer).  The object becomes invalid; a parent owner's one-shot adoption flag stays set
    /// so the same parent handle cannot be adopted twice.
    [[nodiscard]] MPI_Comm release() noexcept {
      const MPI_Comm handle = native_handle();
      // The one-shot parent adoption flag deliberately stays set after release().
      reset_owner_state();
      return handle;
    }

  private:
    struct parent_handle_t {
      explicit parent_handle_t() = default;
    };

    communicator(mpi_communicator view, release_policy policy) noexcept
        : view_{view}, policy_{policy} {
    }

    communicator(mpi_communicator view, release_policy policy, parent_handle_t) noexcept
        : view_{view}, policy_{policy}, parent_{true} {
    }

    static void require_factory_operand(const mpi_communicator& other, const context& ctx) {
      detail::require_valid(other, ctx);
      detail::require_mpi(ctx);
    }

    [[nodiscard]] static communicator take_result(MPI_Comm handle, release_policy policy) {
      if (handle == MPI_COMM_NULL)
        return communicator{};
      try {
        return communicator{mpi_communicator{handle}, policy};
      } catch (...) {
        // The handle was produced by a collective factory.  A rank-local free/disconnect on
        // this local metadata-caching error path would deadlock the peers, so leave the handle
        // to process cleanup and account for it as abandoned.
        detail::note_abandoned_handle();
        throw;
      }
    }

    /// No MPI call: reset the handle, metadata, policy and parent provenance.
    void reset_owner_state() noexcept {
      view_ = mpi_communicator{};
      policy_ = release_policy::free;
      parent_ = false;
    }

    void release_owner() noexcept {
      if (not is_valid()) {
        reset_owner_state();
        return;
      }
      MPI_Comm handle = native_handle();
      const release_policy policy = policy_;
      const bool was_parent = parent_;
      reset_owner_state();
      if (not detail::mpi_active()) [[unlikely]] {
        if (was_parent)
          detail::parent_adopted().store(false);
        detail::note_abandoned_handle();
        return;
      }
      const int rc = policy == release_policy::disconnect ? MPI_Comm_disconnect(&handle)
                                                          : MPI_Comm_free(&handle);
      if (was_parent)
        detail::parent_adopted().store(false);
      if (rc != MPI_SUCCESS) [[unlikely]]
        detail::note_abandoned_handle();
    }

    mpi_communicator view_;
    release_policy policy_{release_policy::free};
    bool parent_{false};

    friend communicator parent();
    friend struct detail::comm_factory;
  };

  namespace detail {

    // Single construction point for the owning communicator result of take_communicator().
    struct comm_factory {
      [[nodiscard]] static communicator make(MPI_Comm handle, release_policy policy) {
        return communicator::take_result(handle, policy);
      }
    };

  }  // namespace detail

  /// Transfer the completed duplicate out of an MPI_Comm_idup request (MPI 4.1 §7.4).
  ///
  /// Requires request::has_result(): a request with no retained result (every non-idup
  /// request, and an idup request whose result was already taken) is std::invalid_argument;
  /// an idup request that has not completed is std::logic_error.  On success the request
  /// becomes freed and the returned owner releases the duplicate with MPI_Comm_free.  If
  /// building the communicator view fails, the handle is counted as abandoned and mpi_error /
  /// std::runtime_error propagates; the request is already freed.  take_communicator() does
  /// not wait: complete the request first so its status stays visible.
  [[nodiscard]] inline communicator take_communicator(request& r) {
    if (not detail::request_access::retains_result(r)) [[unlikely]]
      throw_invalid("mp1::take_communicator", "request has no communicator result");
    if (r.state() != request_state::result_pending) [[unlikely]] {
      detail::fail_with<std::logic_error>("mp1::take_communicator",
                                          std::source_location::current(),
                                          "the request has not completed");
    }

    MPI_Comm handle = detail::request_access::payload(r)->release_comm();
    detail::request_access::clear_payload(r);
    detail::request_access::set_state(r, request_state::freed);

    if (handle == MPI_COMM_NULL)
      return communicator{};
    return detail::comm_factory::make(handle, release_policy::free);
  }

  namespace detail {

    /// The one validity precondition used by every operation taking a communicator.
    inline void require_valid(const mpi_communicator& c, const context& ctx) {
      if (not c.is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid communicator");
    }

    /// Process-global one-shot parent adoption flag; trivially destructible, atomic so
    /// concurrent parent() calls cannot both claim the single MPI_Comm_get_parent handle
    /// (MPI 4.1 §11.8.2 advice to users: a second MPI_Comm_get_parent returns the same
    /// intercommunicator, and one disconnect invalidates every other reference).
    inline std::atomic<bool>& parent_adopted() noexcept {
      static std::atomic<bool> adopted{false};
      return adopted;
    }

    /// Cached metadata view of MPI_COMM_WORLD.  The handle and its size/rank/inter flag
    /// are process-lifetime constants; the view is constructed on first use while MPI is active
    /// and is copied by value afterwards, so steady-state comm_world()/comm_self() calls make no
    /// MPI call.  The view owns no handle and has a trivial destructor, so it is safe to copy or
    /// to leave alive after MPI_Finalize.  As with any mpi_communicator snapshot, the first use
    /// must happen during the active lifetime being queried; a finalize/re-init cycle that
    /// changed the process set would need a new process (MPI 4.1 §11.2, §11.3).
    inline const mpi_communicator& world_view() {
      static const mpi_communicator view{MPI_COMM_WORLD};
      return view;
    }

    inline const mpi_communicator& self_view() {
      static const mpi_communicator view{MPI_COMM_SELF};
      return view;
    }

  }  // namespace detail

  /// Non-owning view of MPI_COMM_WORLD; throws std::runtime_error when MPI is not active.
  /// The metadata is cached once, so repeated calls issue no MPI call.  Returns the
  /// process-lifetime borrowed view by reference: `auto world = comm_world()` still copies it,
  /// and nothing may be written through it.
  [[nodiscard]] inline const mpi_communicator& comm_world() {
    detail::require_mpi("mp1::comm_world");
    return detail::world_view();
  }

  /// Non-owning view of MPI_COMM_SELF; throws std::runtime_error when MPI is not active.
  /// The metadata is cached once, so repeated calls issue no MPI call.  Returns the
  /// process-lifetime borrowed view by reference (see comm_world()).
  [[nodiscard]] inline const mpi_communicator& comm_self() {
    detail::require_mpi("mp1::comm_self");
    return detail::self_view();
  }

  /// Always-available invalid view; no MPI call.  Returns MPI_COMM_NULL by value: it is a
  /// null sentinel, not a process-lifetime singleton like comm_world()/comm_self().
  [[nodiscard]] inline mpi_communicator comm_null() noexcept {
    return mpi_communicator{};
  }

  /// MPI_Comm_set_errhandler with a predefined handler (MPI 4.1 §9.3.1).  Non-collective,
  /// process-local; the communicator is logically mutated although the view is passed by
  /// const reference.  Requires active MPI and a valid communicator.
  inline void set_error_handler(const mpi_communicator& comm, error_handler handler) {
    const context ctx{"mp1::set_error_handler"};
    detail::require_valid(comm, ctx);
    detail::require_mpi(ctx);
    detail::check_mpi(MPI_Comm_set_errhandler(comm.native_handle(),
                                              detail::native_error_handler(ctx, handler)),
                      ctx);
  }

  /// MPI_Comm_get_errhandler plus classification of the predefined handlers.  MPI 4.1 §9.3
  /// makes the returned reference a new error-handler object, so it is freed with
  /// MPI_Errhandler_free before returning (always available, Table 11.1).  A user-created
  /// handler installed by raw MPI yields std::nullopt (best-effort: MPI does not define
  /// handle equality).
  [[nodiscard]] inline std::optional<error_handler> get_error_handler(
      const mpi_communicator& comm) {
    detail::require_valid(comm, "mp1::get_error_handler");
    detail::require_mpi("mp1::get_error_handler");
    MPI_Errhandler handler{MPI_ERRHANDLER_NULL};
    detail::check_mpi(MPI_Comm_get_errhandler(comm.native_handle(), &handler),
                      "MPI_Comm_get_errhandler");
    const std::optional<error_handler> result = detail::classify_error_handler(handler);
    if (handler != MPI_ERRHANDLER_NULL)
      detail::check_mpi(MPI_Errhandler_free(&handler), "MPI_Errhandler_free");
    return result;
  }

  /// Set MPI_COMM_WORLD and MPI_COMM_SELF to MPI_ERRORS_RETURN.  Both are set because the
  /// installed implementations disagree about which handler governs errors that are not
  /// associated with a communicator (MPI 4.1 §2.8 raises those on MPI_COMM_SELF).  Requires
  /// active MPI; non-collective, process-local and idempotent.  It overwrites any handler the
  /// application installed on WORLD/SELF; new communicators inherit WORLD's handler, while
  /// windows and files do not inherit (§9.3).
  inline void install_recoverable_errors() {
    set_error_handler(comm_world(), error_handler::returns);
    set_error_handler(comm_self(), error_handler::returns);
  }

  /// MPI_Comm_get_parent as an owning, one-shot communicator with disconnect policy.  Returns
  /// an invalid owner when there is no parent.  Both ends of the spawn connection must
  /// disconnect (MPI 4.1 §11.10.4); free() is rejected on the result.  Throws std::logic_error
  /// if a previous parent owner has not been disconnected yet and std::runtime_error after
  /// finalize.  MPI_Comm_get_parent is the only source: do not wrap its handle with adopt.
  ///
  /// The one-shot claim is a process-global atomic compare-exchange, so two threads calling
  /// parent() concurrently produce exactly one owner (the loser throws std::logic_error) even
  /// at MPI_THREAD_MULTIPLE.  The claim is taken before the owner is constructed and released
  /// again if the construction throws.
  [[nodiscard]] inline communicator parent() {
    detail::require_mpi("mp1::parent");
    MPI_Comm handle{MPI_COMM_NULL};
    detail::check_mpi(MPI_Comm_get_parent(&handle), "MPI_Comm_get_parent");
    if (handle == MPI_COMM_NULL)
      return communicator{};
    bool expected = false;
    if (not detail::parent_adopted().compare_exchange_strong(expected, true)) [[unlikely]]
      detail::fail_with<std::logic_error>("mp1::parent", std::source_location::current(),
                                          "the parent intercommunicator was already adopted; "
                                          "disconnect the previous owner first");
    try {
      return communicator{mpi_communicator{handle}, release_policy::disconnect,
                          communicator::parent_handle_t{}};
    } catch (...) {
      detail::parent_adopted().store(false);
      detail::note_abandoned_handle();
      throw;
    }
  }

  /// Largest legal tag value of a communicator (MPI_TAG_UB, inclusive) from the cached
  /// metadata; no MPI call and never below tag_ub_min.  The bound is fixed when the view is
  /// constructed, so a later MPI_Comm_set_attr on the native communicator is not observed.
  [[nodiscard]] inline int tag_ub(const mpi_communicator& c) {
    return c.tag_ub();
  }

  /// MPI_Comm_group for a communicator (local group on an intercommunicator).
  [[nodiscard]] inline group get_group(const mpi_communicator& c) {
    detail::require_valid(c, "mp1::get_group");
    detail::require_mpi("mp1::get_group");
    MPI_Group handle{MPI_GROUP_NULL};
    detail::check_mpi(MPI_Comm_group(c.native_handle(), &handle), "MPI_Comm_group");
    return detail::group_factory::make(handle);
  }

}  // namespace mp1

#endif
