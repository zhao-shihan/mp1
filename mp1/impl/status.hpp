#if !(defined MP1_IMPL_STATUS_HPP)

#define MP1_IMPL_STATUS_HPP

#include "mp1/impl/datatype.hpp"
#include "mp1/impl/detail/mpi_call.hpp"

#include <compare>
#include <stdexcept>

// status::get_count<T>()/get_elements<T>() need the native_datatype<T>() definition to be
// link-complete for a TU that includes only status.hpp. Including datatype.hpp (which defines
// it) is the natural dependency; the declaration cannot be left dangling.
namespace mp1 {

  namespace detail {
    struct status_access;
  }  // namespace detail

  /// Where the wrapped MPI_Status came from.
  ///
  /// native: produced by MPI for a point-to-point or file completion.
  /// normalized: the wrapper status used for collective, RMA and buffer-flush completions
  /// (source/tag are MPI_ANY_SOURCE/MPI_ANY_TAG, count/elements are 0, no MPI call is made by
  /// the accessors; the MPI_ERROR field is preserved for MPI_ERR_IN_STATUS array completions).
  /// The default/empty status is also normalized.  make_proc_null() is the source exception: a
  /// synthesized MPI_PROC_NULL status has source MPI_PROC_NULL and tag MPI_ANY_TAG (MPI 4.1
  /// 3.10), count/elements 0 and MPI_ERROR MPI_SUCCESS.
  enum class status_kind : unsigned char { native, normalized };

  /// Point-to-point tag value (MPI 4.1 §3.2.3).
  ///
  /// An aggregate with a public `value`; default-construction is tag 0, the MPI default tag.
  /// All six comparisons are value-based.  There is no implicit conversion from `int`, so a
  /// literal tag is spelled `mp1::tag{7}` and a tag is passed to a `status` query as
  /// `s.tag().value` only where a raw MPI integer is required.
  ///
  /// A tag is legal for sending when `0 <= value` and `value <= tag_ub(comm)`; receiving and
  /// probing additionally accept the wildcard `any_tag`.
  struct tag {
    int value{};
    friend constexpr auto operator<=>(tag, tag) = default;
    friend constexpr bool operator==(tag, tag) = default;
  };

  /// MPI_ANY_TAG wildcard: matches any tag in a receive or a probe; never legal for a send.
  inline constexpr tag any_tag{MPI_ANY_TAG};

  /// The smallest MPI_TAG_UB value guaranteed by MPI 4.1 §3.2.3 (32767 inclusive).  Used as
  /// the cached fallback when an implementation does not expose the MPI_TAG_UB attribute.
  inline constexpr tag tag_ub_min{32767};

  /// Status of a completed MPI operation.
  ///
  /// Status content is tracked by provenance, never inferred from MPI_SOURCE/MPI_TAG/MPI_ERROR
  /// field values: a conforming provider may leave the fields of a file or send status at any
  /// values, including the ones that look like an MPI empty status. The MPI empty status is
  /// defined in MPI 4.1 3.7.3: SOURCE=MPI_ANY_SOURCE, TAG=MPI_ANY_TAG, ERROR=MPI_SUCCESS, and
  /// is internally configured so that MPI_GET_COUNT/MPI_GET_ELEMENTS return 0 and
  /// MPI_TEST_CANCELLED returns false.
  ///
  /// Provenance table (one row per constructor in detail::status_access):
  ///
  /// | constructor        | MPI call that produced it                    | count | cancel |
  /// |--------------------|----------------------------------------------|-------|--------|
  /// | default            | none (wrapper normalized)                    | no    | no     |
  /// | make_normalized    | collective/RMA/flush completion              | no    | no     |
  /// | make/make_receive  | p2p receive, probe/iprobe/mprobe, matched    | yes   | yes    |
  /// |                    | receive, combined send-receive receive side  |       |        |
  /// |                    | (a real message)                             |       |        |
  /// | make_proc_null     | MPI_PROC_NULL receive (blocking,             | no    | no     |
  /// |                    | nonblocking, persistent or matched),         |       |        |
  /// |                    | probe/iprobe, and the receive side of a      |       |        |
  /// |                    | combined send-receive (blocking or           |       |        |
  /// |                    | nonblocking) (MPI 4.1 3.10 synthesizes the   |       |        |
  /// |                    | standard status)                             |       |        |
  /// | make_file          | file data access (MPI 4.1 14.4.1)            | yes   | yes    |
  /// | make_send          | p2p send completion; fields undefined except | no    | yes    |
  /// |                    | MPI_ERROR for IN_STATUS; MPI_Test_cancelled  |       |        |
  /// | make_pending       | MPI_ERR_PENDING slot of a multiple completion | no    | no     |
  ///
  /// Per-kind validity (MPI 4.1): point-to-point receive statuses carry source/tag/count
  /// (3.2.5); point-to-point send status fields are undefined except MPI_ERROR for
  /// MPI_ERR_IN_STATUS and MPI_Test_cancelled (3.7.3); nonblocking collective completions
  /// have undefined SOURCE/TAG (6.12); request-based RMA has all query fields undefined
  /// except MPI_ERROR for MPI_ERR_IN_STATUS, see 12.3.5; file operations define count/elements
  /// and allow MPI_Test_cancelled, but not source/tag (14.4.1). The wrapper normalizes the
  /// undefined source/tag of a file, send or pending status to ANY and starts from ERROR==
  /// MPI_SUCCESS unless the completion returned MPI_ERR_IN_STATUS, see 3.2.5.
  ///
  /// A native status query (get_count, get_elements, is_cancelled) requires active MPI, except
  /// that is_cancelled answers false for an empty native status after MPI_Finalize; a
  /// normalized status is fully answerable without MPI.
  class status {
  public:
    /// Wrapper empty status: MPI_SOURCE=MPI_ANY_SOURCE, tag=any_tag, MPI_ERROR=MPI_SUCCESS.
    /// These are the MPI empty-status field values; the native status is not internally
    /// configured, so it must not be passed to an MPI status query (see native_handle()).
    status() noexcept : s_{} {
      s_.MPI_SOURCE = MPI_ANY_SOURCE;
      s_.MPI_TAG = any_tag.value;
      s_.MPI_ERROR = MPI_SUCCESS;
    }

    /// \return provenance of the wrapped status; no MPI call
    [[nodiscard]] status_kind kind() const noexcept {
      return kind_;
    }

    /// \return source rank of the message; no MPI call.  MPI_ANY_SOURCE for a normalized
    /// collective/RMA/flush status and for a send, file or PENDING status, whose SOURCE field
    /// is undefined; MPI_PROC_NULL for a receive or probe whose peer is MPI_PROC_NULL (MPI 4.1
    /// 3.10).
    [[nodiscard]] int source() const noexcept {
      return s_.MPI_SOURCE;
    }

    /// \return tag of the message; no MPI call. any_tag for the wrapper's normalized status
    /// and for a send or PENDING status, whose TAG field is undefined.  The source rank stays
    /// an `int`; only the tag is wrapped because it has a wildcard and a communicator-dependent
    /// upper bound.
    [[nodiscard]] mp1::tag tag() const noexcept {
      return mp1::tag{s_.MPI_TAG};
    }

    /// \return MPI_ERROR field; meaningful only for MPI_ERR_IN_STATUS array completions
    /// (MPI 4.1 3.2.5, 3.7.3). A PENDING status carries MPI_ERR_PENDING here.
    [[nodiscard]] int error() const noexcept {
      return s_.MPI_ERROR;
    }

    /// \return true if the associated operation was cancelled; false without an MPI call for a
    /// normalized or PENDING status, which cannot be queried with MPI_Test_cancelled, and for an
    /// empty native status when MPI is not active; requires active MPI otherwise.
    ///
    /// A cancelled receive also reports MPI_ANY_SOURCE/MPI_ANY_TAG/MPI_SUCCESS, so it is
    /// indistinguishable from an empty native status by field values; MPI_Test_cancelled is
    /// the only way to tell them apart. Provenance distinguishes the wrapper's normalized
    /// empty status (which must never be passed to MPI_Test_cancelled) from a native status.
    [[nodiscard]] bool is_cancelled() const {
      if (not cancel_allowed_)
        return false;
      // After MPI_Finalize no status query is possible. A native status whose fields are the
      // empty ones is the only case the wrapper can answer without MPI; field values are never
      // used to infer a count.
      if (empty() and not detail::mpi_active())
        return false;
      detail::require_mpi("mp1::status::is_cancelled");
      int flag{0};
      detail::check_mpi(MPI_Test_cancelled(&s_, &flag), {"MPI_Test_cancelled"});
      return flag != 0;
    }

    /// \return number of T elements in the message (MPI_Get_count_c; MPI 4.1 3.2.5);
    /// MPI_UNDEFINED on a type mismatch; 0 without an MPI call for a status whose operation
    /// family does not define a count (send, PENDING, normalized)
    template<class T>
    [[nodiscard]] count_t get_count() const {
      if (not count_defined_)
        return 0;
      detail::require_mpi("mp1::status::get_count");
      MPI_Count count{0};
      detail::check_mpi(MPI_Get_count_c(&s_, native_datatype<T>(), &count),
                        {"MPI_Get_count_c"});
      return detail::checked_count(count, "mp1::status::get_count");
    }

    /// \return number of layout repetitions in the message (MPI_Get_count_c on l's datatype),
    /// not T elements; 0 without an MPI call for a status whose operation family does not
    /// define a count (send, PENDING, normalized)
    template<class T>
    [[nodiscard]] count_t get_count(const layout<T>& l) const {
      if (not count_defined_)
        return 0;
      if (not l.is_valid())
        throw_invalid("mp1::status::get_count", "invalid layout");
      detail::require_mpi("mp1::status::get_count");
      MPI_Count count{0};
      detail::check_mpi(MPI_Get_count_c(&s_, l.native_handle(), &count), {"MPI_Get_count_c"});
      return detail::checked_count(count, "mp1::status::get_count");
    }

    /// \return number of basic elements in the message (MPI_Get_elements_c); 0 without an MPI
    /// call for a status whose operation family does not define a count (send, PENDING,
    /// normalized)
    template<class T>
    [[nodiscard]] count_t get_elements() const {
      if (not count_defined_)
        return 0;
      detail::require_mpi("mp1::status::get_elements");
      MPI_Count count{0};
      detail::check_mpi(MPI_Get_elements_c(&s_, native_datatype<T>(), &count),
                        {"MPI_Get_elements_c"});
      return detail::checked_count(count, "mp1::status::get_elements");
    }

    /// \return number of basic elements in the message (MPI_Get_elements_c on l's datatype);
    /// 0 without an MPI call for a status whose operation family does not define a count
    /// (send, PENDING, normalized)
    template<class T>
    [[nodiscard]] count_t get_elements(const layout<T>& l) const {
      if (not count_defined_)
        return 0;
      if (not l.is_valid())
        throw_invalid("mp1::status::get_elements", "invalid layout");
      detail::require_mpi("mp1::status::get_elements");
      MPI_Count count{0};
      detail::check_mpi(MPI_Get_elements_c(&s_, l.native_handle(), &count),
                        {"MPI_Get_elements_c"});
      return detail::checked_count(count, "mp1::status::get_elements");
    }

    /// \return non-owning view of the wrapped MPI_Status, valid for this object's lifetime.
    ///
    /// For a normalized status this is the wrapper's synthetic status, not an MPI empty
    /// status: it is not internally configured, so it must not be passed to MPI_Get_count,
    /// MPI_Get_elements, MPI_Test_cancelled or any other MPI status query. The same applies to
    /// a PENDING status, of which only error() is meaningful.
    [[nodiscard]] const MPI_Status& native_handle() const noexcept {
      return s_;
    }

  private:
    [[nodiscard]] bool empty() const noexcept {
      return s_.MPI_SOURCE == MPI_ANY_SOURCE and s_.MPI_TAG == any_tag.value and
             s_.MPI_ERROR == MPI_SUCCESS;
    }

    MPI_Status s_{};
    // Provenance bits; field values are never used to infer count/cancellation. A status the
    // wrapper did not receive from a count-defining MPI operation must not be handed to a
    // count or cancellation query.
    status_kind kind_{status_kind::normalized};
    bool count_defined_{false};
    bool cancel_allowed_{false};

    friend class request;
    friend struct detail::status_access;
  };

  namespace detail {

    struct status_access {
      /// Native status of a count-defining, cancellable MPI operation: p2p receive, probe and
      /// matched/composed receive, where SOURCE/TAG/count are meaningful (MPI 4.1 3.2.5).
      static status make_receive(const MPI_Status& native) noexcept {
        status result;
        result.s_ = native;
        result.kind_ = status_kind::native;
        result.count_defined_ = true;
        result.cancel_allowed_ = true;
        return result;
      }

      /// Backwards-compatible name for make_receive (native receive/probe status).
      static status make(const MPI_Status& native) noexcept {
        return make_receive(native);
      }

      /// Status of an operation MPI ignores because the peer is MPI_PROC_NULL (MPI 4.1 3.10):
      /// source = MPI_PROC_NULL, tag = MPI_ANY_TAG, MPI_ERROR = MPI_SUCCESS, count = 0. This
      /// covers blocking, nonblocking, persistent and matched receives, probe/iprobe, and the
      /// receive side of a combined send-receive (blocking or nonblocking). The provider may
      /// leave SOURCE at MPI_ANY_SOURCE for some nonblocking/persistent receives or probes
      /// (pinned MPICH 4.3.2 does so for a persistent receive initialized with MPI_PROC_NULL
      /// and for MPI_Iprobe(MPI_PROC_NULL)), so the wrapper synthesizes the standard envelope
      /// instead of forwarding it. The count is defined to be 0, so this status needs no MPI
      /// count query; the raw handle is synthetic and must not be passed to an MPI status
      /// query (see native_handle()).
      static status make_proc_null() noexcept {
        status result;
        result.s_.MPI_SOURCE = MPI_PROC_NULL;
        result.s_.MPI_TAG = any_tag.value;
        result.s_.MPI_ERROR = MPI_SUCCESS;
        result.kind_ = status_kind::normalized;
        result.count_defined_ = false;
        result.cancel_allowed_ = false;
        return result;
      }

      // Collective/RMA/flush completions return the wrapper normalized status: source/tag are
      // undefined for those families, count/elements are not meaningful, and MPI_Test_cancelled
      // must never be called on them. MPI_ERROR is copied only when the caller knows that the
      // completion returned MPI_ERR_IN_STATUS, see 3.2.5; otherwise it is undefined and forced
      // to MPI_SUCCESS in the normalized status.
      static status make_normalized(const MPI_Status& native,
                                    bool error_defined = false) noexcept {
        status result;
        if (error_defined)
          result.s_.MPI_ERROR = native.MPI_ERROR;
        result.kind_ = status_kind::normalized;
        return result;
      }

      /// File data-access status: count/elements and MPI_Test_cancelled are defined, SOURCE/
      /// TAG are not (MPI 4.1 14.4.1) and are normalized to ANY. MPI_ERROR is copied only when
      /// the completion returned MPI_ERR_IN_STATUS.
      static status make_file(const MPI_Status& native, bool error_defined) noexcept {
        status result;
        // Copy the whole status first: MPI may keep the count in implementation-specific
        // fields, so normalizing the envelope must not discard them.
        result.s_ = native;
        result.s_.MPI_SOURCE = MPI_ANY_SOURCE;
        result.s_.MPI_TAG = any_tag.value;
        // MPI_ERROR is undefined unless the completion returned MPI_ERR_IN_STATUS, see 3.2.5.
        result.s_.MPI_ERROR = error_defined ? native.MPI_ERROR : MPI_SUCCESS;
        result.kind_ = status_kind::native;
        result.count_defined_ = true;
        result.cancel_allowed_ = true;
        return result;
      }

      /// Point-to-point send completion: every field except MPI_ERROR for IN_STATUS and the
      /// MPI_Test_cancelled query is undefined (MPI 4.1 3.7.3). Count/elements are therefore
      /// not queryable; SOURCE/TAG are normalized to ANY.
      static status make_send(const MPI_Status& native, bool error_defined) noexcept {
        status result;
        result.s_ = native;
        result.s_.MPI_SOURCE = MPI_ANY_SOURCE;
        result.s_.MPI_TAG = any_tag.value;
        // MPI_ERROR is undefined unless the completion returned MPI_ERR_IN_STATUS, see 3.7.3.
        result.s_.MPI_ERROR = error_defined ? native.MPI_ERROR : MPI_SUCCESS;
        result.kind_ = status_kind::native;
        result.count_defined_ = false;
        result.cancel_allowed_ = true;
        return result;
      }

      /// MPI_ERR_PENDING slot of a multiple completion (MPI 4.1 3.7.5): the operation has
      /// neither failed nor completed, so no count/cancellation query is meaningful. Only
      /// error() (MPI_ERR_PENDING) is carried; source/tag are normalized to ANY.
      static status make_pending(const MPI_Status& native) noexcept {
        status result;
        result.s_ = native;
        result.s_.MPI_SOURCE = MPI_ANY_SOURCE;
        result.s_.MPI_TAG = any_tag.value;
        result.s_.MPI_ERROR = native.MPI_ERROR;
        result.kind_ = status_kind::native;
        result.count_defined_ = false;
        result.cancel_allowed_ = false;
        return result;
      }

      static MPI_Status& native(status& s) noexcept {
        return s.s_;
      }

      static const MPI_Status& native(const status& s) noexcept {
        return s.s_;
      }
    };

  }  // namespace detail

}  // namespace mp1

#endif
