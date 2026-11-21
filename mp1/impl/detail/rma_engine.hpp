#if !(defined MP1_DETAIL_RMA_ENGINE_HPP)

#define MP1_DETAIL_RMA_ENGINE_HPP

// Shared implementation of the RMA data-movement families.  One descriptor preparation path
// serves the blocking and request-based forms and every origin/result/target shape
// combination; rma.hpp only classifies the span/region form and forwards.  There is no
// staging, no automatic flush and no automatic epoch management here: every function makes
// exactly one MPI data-movement call.  A request initiation additionally duplicates the
// layouts it passes to MPI (MPI_Type_dup, the request-ownership contract below); those are
// metadata calls, not data-movement or epoch calls.
//
// Form and the size rule.  The origin and result are spans or regions and the target
// is a count or a layout, so the size equality the old pointer API encoded in its call shape
// is an explicit check before the MPI call:
//   * origin.size()/result.size() must equal target_count (flat target) or target.size()
//     (layout target); a region contributes map().size(), a span contributes span.size();
//   * the both-flat origin/result pair of get_accumulate/rget_accumulate/fetch_and_op/
//     compare_and_swap must have disjoint byte ranges (MPI 4.1 §12.3.4);
//   * MPI_PROC_NULL skips every data-movement check (the whole call is a normalized no-op),
//     and a null/empty origin under MPI_NO_OP skips the origin size check because MPI ignores
//     the origin arguments (§12.3.4); an empty origin has no bytes, so it is not charged the
//     target extent by the both-flat overlap check either;
//   * target_count is never inferred from a buffer size or ignored: a mismatch throws
//     std::invalid_argument before MPI.
//
// Request ownership.  The four request families (rput/rget/raccumulate/rget_accumulate)
// duplicate every layout they pass to the native request (a region origin/result map and the
// target layout) into a request payload, so the caller's region object and layout may die as
// soon as the call returns; the payload is attached to the
// request and freed on completion.  The data pointers are copied, never owned.  The blocking
// families borrow the layout for the duration of the call.  The duplicate is created only
// after the size checks pass, so a rejected call still makes no MPI call.
//
// The typed boundary is enforced here rather than at the MPI call:
//   * accumulate/raccumulate/get_accumulate/rget_accumulate require a predefined or
//     homogeneous-derived element datatype (all basic components the same predefined datatype,
//     MPI 4.1 §12.3.4) and a predefined operator valid for that basic component; the
//     operator is checked against basic_component_t<T>, and an untrusted custom/manual
//     annotation is validated once against its committed datatype before the first call;
//   * fetch_and_op requires a predefined datatype and operator (§12.3.4 Fetch and Op);
//   * compare_and_swap requires a predefined C++ datatype in the C integer, Logical or Byte
//     categories (§12.3.4 Compare and Swap and §6.9.2 Predefined Reduction Operations);
//   * MPI_PROC_NULL is a valid target rank for the ten RMA communication calls (§12.3
//     Communication Calls); lock/unlock/flush keep requiring a non-negative window rank
//     (§12.5.3 Lock, §12.5.4 Flush and Sync).
// Rejections are std::invalid_argument and happen before the MPI call.

#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/operator.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/vocabulary.hpp"
#include "mp1/impl/window.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace mp1::detail {

  // A non-null, suitably aligned address used when a zero-element buffer was passed as null.
  // MPI ignores the buffer for a zero count, but passing null unconditionally is not portable.
  [[nodiscard]] inline void* rma_dummy_buffer() noexcept {
    static std::max_align_t storage{};
    return &storage;
  }

  inline void require_rma_window(const mpi_window& w, const context& ctx) {
    require_valid(w, ctx);
    require_mpi(ctx);
  }

  // MPI 4.1 §12.3 (Communication Calls): "MPI_PROC_NULL is a valid target rank in all MPI RMA
  // communication calls."  lock/unlock/flush document their rank as a non-negative rank in
  // the window group, so they keep using require_rma_peer_rank.
  inline void require_rma_data_rank(const mpi_window& w, int rank, const context& ctx) {
    if (rank == MPI_PROC_NULL)
      return;
    const int size = w.group_size();
    if (rank < 0 or rank >= size) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "target rank {} is outside [0,{})", rank, size);
    }
  }

  inline void require_rma_peer_rank(const mpi_window& w, int rank, const context& ctx) {
    const int size = w.group_size();
    if (rank < 0 or rank >= size) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(
          ctx, std::source_location::current(),
          "target rank {} is outside [0,{}); MPI_PROC_NULL is not accepted by this call", rank,
          size);
    }
  }

  inline void require_rma_target_disp(const mpi_window& w, MPI_Aint target_disp,
                                      const context& ctx) {
    if (w.create_flavor() != MPI_WIN_FLAVOR_DYNAMIC and target_disp < 0) [[unlikely]] {
      throw_invalid(ctx.operation, "negative target_disp on a non-dynamic window");
    }
  }

  // MPI 4.1 §12.3.4 (Accumulate): "The parameter target_datatype must not specify
  // overlapping entries"; §12.3.2 (Get): "The origin_datatype may not specify overlapping
  // entries in the origin buffer."  Only a proven overlap is rejected; the layout factories
  // record `disjoint` when they can prove it and `unknown` otherwise, and `unknown` is
  // accepted (documented).
  template<class T>
  inline void require_non_overlapping_layout(const layout<T>& value, const context& ctx,
                                             const char* position) {
    if (value.has_overlapping_entries()) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(
          ctx, std::source_location::current(),
          "{} layout has overlapping type-map entries; MPI 4.1 §5.1.11 and §12.3.4 make a "
          "datatype that overlaps a buffer updated by the operation erroneous",
          position);
    }
  }

  // fetch_and_op and compare_and_swap still require one predefined MPI datatype (MPI 4.1
  // §12.3.4 (Fetch and Op, Compare and Swap)); the accumulating families accept a
  // homogeneous-derived datatype and use the compile-time basic-component trait plus
  // require_rma_custom_homogeneous<T>() instead.  Built-in element types are decided at
  // compile time by the static_assert in each family; a custom MP1_DATATYPE_TRAITS mapping
  // is checked against its committed handle before the MPI call.
  template<class T>
  inline rma_datatype_category require_rma_atomic_datatype(const context& ctx) {
    return require_predefined_rma_datatype<T>(ctx);
  }

  template<class T>
  inline void require_rma_cas_datatype(const context& ctx) {
    const rma_datatype_category category = require_rma_atomic_datatype<T>(ctx);
    if (category != rma_datatype_category::c_integer and
        category != rma_datatype_category::logical and category != rma_datatype_category::byte)
        [[unlikely]] {
      detail::fail_as<std::invalid_argument>(
          ctx, std::source_location::current(),
          "MPI_Compare_and_swap requires a predefined C++ datatype in the C integer, Logical "
          "or Byte category (MPI 4.1 §12.3.4 Compare and Swap); the element datatype is {}",
          rma_datatype_category_name(category));
    }
  }

  template<class T>
  inline MPI_Op require_rma_op(MPI_Op op, bool allow_no_op, const context& ctx) {
    return validate_raw_rma_op<T>(op, allow_no_op, ctx);
  }

  // ---- mandatory one-time validation of an untrusted custom/manual annotation ---------------
  //
  // MP1_DATATYPE_BASIC(T,U) and a manually written struct_builder<T>::component_types are
  // compile-time contracts: mp1 did not synthesize native_datatype<T>() from the listed C++
  // declarations, so the committed datatype is walked once with MPI 4.1 §5.1.13
  // envelope/contents and every positive-multiplicity basic component must equal
  // native_datatype<U>().  This is the only runtime homogeneity check: the trusted
  // pair/tuple/array/MP1_REFLECTION datatypes are built by mp1 from the same declarations and
  // are classified at compile time.  Rejecting here keeps a wrong annotation (in particular a
  // heterogeneous derived MPI_REPLACE) away from MPICH's MPIR_DATATYPE_IS_PREDEFINED abort.

  struct rma_datatype_contents {
    MPI_Count integers{0};
    MPI_Count addresses{0};
    MPI_Count large_counts{0};
    MPI_Count datatypes{0};
    int combiner{MPI_COMBINER_NAMED};
  };

  [[nodiscard]] inline rma_datatype_contents rma_datatype_envelope(MPI_Datatype type) {
    // MPI_DATATYPE_NULL has no envelope: MPI_Type_get_envelope_c is erroneous for it and
    // aborts under the default error handler.  The one-time validator treats that annotation
    // as invalid instead of asking the provider about the null handle.
    if (type == MPI_DATATYPE_NULL) [[unlikely]]
      throw_invalid("custom datatype validation", "MPI_DATATYPE_NULL is not a valid datatype");
    require_mpi("mp1: custom datatype validation");
    rma_datatype_contents value;
    check_mpi(MPI_Type_get_envelope_c(type, &value.integers, &value.addresses,
                                      &value.large_counts, &value.datatypes, &value.combiner),
              "MPI_Type_get_envelope_c(custom datatype)");
    return value;
  }

  // Owns the new derived datatype objects returned in MPI_Type_get_contents_c's datatypes
  // array.  Named predefined handles are borrowed constants and are never freed; an unnamed
  // predefined (F90) handle is also left alone.  A repeated handle is freed only once.
  class rma_child_datatypes {
  public:
    rma_child_datatypes() = default;
    rma_child_datatypes(const rma_child_datatypes&) = delete;
    rma_child_datatypes& operator=(const rma_child_datatypes&) = delete;

    ~rma_child_datatypes() {
      for (MPI_Datatype& type : types_) {
        if (type != MPI_DATATYPE_NULL) {
          (void)MPI_Type_free(&type);
          type = MPI_DATATYPE_NULL;
        }
      }
    }

    void own(MPI_Datatype type) {
      if (type == MPI_DATATYPE_NULL)
        return;
      for (MPI_Datatype existing : types_) {
        if (existing == type)
          return;
      }
      types_.push_back(type);
    }

  private:
    std::vector<MPI_Datatype> types_;
  };

  // True for a handle returned by MPI_Type_get_contents_c that must not be freed (a named
  // predefined datatype, an unnamed predefined F90 datatype, or the predefined unnamed
  // value/index pair of MPI_Type_get_value_index); every other returned handle is a new
  // derived object owned by the caller (MPI 4.1 §5.1.13, §6.9.4).  The value/index pair "is
  // not a duplicate" and "cannot be freed", so treating it as owned would free a predefined
  // handle.  A NULL handle is reported as borrowed too: it owns nothing, and the walk fails
  // closed on it before any MPI call.
  [[nodiscard]] inline bool rma_datatype_is_borrowed(MPI_Datatype type) {
    if (type == MPI_DATATYPE_NULL)
      return true;
    const rma_datatype_contents env = rma_datatype_envelope(type);
    switch (env.combiner) {
      case MPI_COMBINER_NAMED:
      case MPI_COMBINER_F90_REAL:
      case MPI_COMBINER_F90_COMPLEX:
      case MPI_COMBINER_F90_INTEGER:
      case MPI_COMBINER_VALUE_INDEX:
        return true;
      default:
        return false;
    }
  }

  // Datatype nesting deeper than this is rejected as unverifiable instead of risking the walk's
  // stack.  No mp1-built or realistic user datatype comes close.
  inline constexpr int rma_validation_max_depth = 64;

  [[nodiscard]] inline bool rma_walk_basic_components(MPI_Datatype type, MPI_Datatype expected,
                                                      bool& any_component, bool& unsupported,
                                                      int depth);

  [[nodiscard]] inline bool rma_visit_component(MPI_Datatype type, MPI_Count multiplicity,
                                                MPI_Datatype expected, bool& any_component,
                                                bool& unsupported, int depth) {
    // A zero-multiplicity child contributes no basic components (and may have a different
    // basic type without affecting homogeneity).  MPI 3.0 removed the MPI_LB/MPI_UB marker
    // constants, so a committed datatype cannot contain them; explicit bounds arrive as a
    // MPI_COMBINER_RESIZED node, which this walk decodes.  The child handle is freed by the
    // caller's RAII guard whether or not it is visited.
    if (multiplicity <= 0)
      return true;
    return rma_walk_basic_components(type, expected, any_component, unsupported, depth + 1);
  }

  [[nodiscard]] inline bool rma_walk_basic_components(MPI_Datatype type, MPI_Datatype expected,
                                                      bool& any_component, bool& unsupported,
                                                      int depth) {
    // A null handle is not a datatype: reject it locally instead of passing it to
    // MPI_Type_get_envelope_c, where the default error handler would abort.  The same guard
    // covers a null expected/basic-component handle.
    if (type == MPI_DATATYPE_NULL or expected == MPI_DATATYPE_NULL) {
      unsupported = true;
      return false;
    }
    if (depth > rma_validation_max_depth) {
      unsupported = true;
      return false;
    }
    const rma_datatype_contents env = rma_datatype_envelope(type);
    if (env.combiner == MPI_COMBINER_NAMED) {
      // A named predefined handle is one basic component of that predefined type.  The
      // operator is applied to elements of the predefined type (§12.3.4), so the leaf must be
      // exactly native_datatype<U>() -- a named multi-component pair type such as MPI_2INT is
      // not the modeled basic type and is rejected.
      any_component = true;
      return type == expected;
    }
    if (env.combiner == MPI_COMBINER_F90_REAL or env.combiner == MPI_COMBINER_F90_COMPLEX or
        env.combiner == MPI_COMBINER_F90_INTEGER) {
      unsupported = true;
      return false;
    }
    if (env.combiner == MPI_COMBINER_VALUE_INDEX) {
      // A predefined unnamed value/index pair (§6.9.4): it is not a derived datatype this
      // homogeneity walk can decode, and it is borrowed, so fail closed before asking for
      // its contents.
      unsupported = true;
      return false;
    }

    const MPI_Count num_integers = env.integers;
    const MPI_Count num_addresses = env.addresses;
    const MPI_Count num_large_counts = env.large_counts;
    const MPI_Count num_datatypes = env.datatypes;
    std::vector<int> integers(static_cast<std::size_t>(num_integers));
    std::vector<MPI_Aint> addresses(static_cast<std::size_t>(num_addresses));
    std::vector<MPI_Count> large_counts(static_cast<std::size_t>(num_large_counts));
    std::vector<MPI_Datatype> datatypes(static_cast<std::size_t>(num_datatypes));
    int dummy_integer = 0;
    MPI_Aint dummy_address = 0;
    MPI_Count dummy_large_count = 0;
    MPI_Datatype dummy_datatype = MPI_DATATYPE_NULL;
    check_mpi(MPI_Type_get_contents_c(
                  type, num_integers, num_addresses, num_large_counts, num_datatypes,
                  integers.empty() ? &dummy_integer : integers.data(),
                  addresses.empty() ? &dummy_address : addresses.data(),
                  large_counts.empty() ? &dummy_large_count : large_counts.data(),
                  datatypes.empty() ? &dummy_datatype : datatypes.data()),
              "MPI_Type_get_contents_c(custom datatype)");

    rma_child_datatypes children;
    for (MPI_Datatype child : datatypes) {
      if (child == MPI_DATATYPE_NULL) {
        // A provider or a malformed committed datatype can hand back a null child.  It is not
        // queryable and not freeable; fail the walk closed without an MPI metadata call.
        unsupported = true;
        continue;
      }
      if (not rma_datatype_is_borrowed(child))
        children.own(child);
    }

    // Every supported combiner describes one child datatype (STRUCT: one per entry) and a
    // multiplicity that is the number of basic components it contributes; the field layout is
    // the MPI 4.1 §5.1.13 table.  The large-count (MPI-4 *_c) constructors return their
    // count/blocklength/displacement arguments in array_of_large_counts instead of
    // array_of_integers/array_of_addresses, so the same index sequence is read from whichever
    // array the envelope says is populated (num_large_counts > 0).  A combiner mp1 cannot
    // decode (DARRAY, F90_*, VALUE_INDEX, ...) is rejected as unverifiable.
    const bool has_large_counts = num_large_counts > 0;
    const MPI_Count count_fields = has_large_counts ? num_large_counts : num_integers;
    const auto count_at = [&](MPI_Count index) -> MPI_Count {
      return has_large_counts
                 ? large_counts[static_cast<std::size_t>(index)]
                 : static_cast<MPI_Count>(integers[static_cast<std::size_t>(index)]);
    };
    switch (env.combiner) {
      case MPI_COMBINER_DUP: {
        if (num_datatypes < 1)
          break;
        return rma_visit_component(datatypes[0], 1, expected, any_component, unsupported,
                                   depth);
      }
      case MPI_COMBINER_CONTIGUOUS: {
        if (count_fields < 1 or num_datatypes < 1)
          break;
        return rma_visit_component(datatypes[0], count_at(0), expected, any_component,
                                   unsupported, depth);
      }
      case MPI_COMBINER_VECTOR:
      case MPI_COMBINER_HVECTOR: {
        if (count_fields < 2 or num_datatypes < 1)
          break;
        return rma_visit_component(datatypes[0], count_at(0) * count_at(1), expected,
                                   any_component, unsupported, depth);
      }
      case MPI_COMBINER_INDEXED:
      case MPI_COMBINER_HINDEXED: {
        if (count_fields < 1 or num_datatypes < 1)
          break;
        const MPI_Count count = count_at(0);
        if (count < 0 or count_fields < count + 1)
          break;
        MPI_Count total = 0;
        for (MPI_Count i = 0; i < count; ++i)
          total += count_at(1 + i);
        return rma_visit_component(datatypes[0], total, expected, any_component, unsupported,
                                   depth);
      }
      case MPI_COMBINER_INDEXED_BLOCK:
      case MPI_COMBINER_HINDEXED_BLOCK: {
        if (count_fields < 2 or num_datatypes < 1)
          break;
        return rma_visit_component(datatypes[0], count_at(0) * count_at(1), expected,
                                   any_component, unsupported, depth);
      }
      case MPI_COMBINER_STRUCT: {
        // int-count struct: integers = {count, blocklengths}, addresses = displacements.
        // struct_c: large_counts = {count, blocklengths, displacements}.  The count and the
        // blocklengths use the same indices in both forms.
        if (count_fields < 1)
          break;
        const MPI_Count count = count_at(0);
        if (count < 0 or count_fields < count + 1 or num_datatypes < count)
          break;
        bool homogeneous = true;
        for (MPI_Count i = 0; i < count; ++i) {
          const MPI_Datatype child = datatypes[static_cast<std::size_t>(i)];
          const MPI_Count blocklength = count_at(1 + i);
          if (not rma_visit_component(child, blocklength, expected, any_component, unsupported,
                                      depth))
            homogeneous = false;
        }
        return homogeneous;
      }
      case MPI_COMBINER_SUBARRAY: {
        // int-count subarray: integers = {ndims, sizes, subsizes, starts, order}.
        // subarray_c: integers = {ndims, order}; large_counts = {sizes, subsizes, starts}.
        if (num_integers < 2 or num_datatypes < 1)
          break;
        const MPI_Count ndims = integers[0];
        if (ndims < 0)
          break;
        MPI_Count total = 1;
        if (has_large_counts) {
          if (num_large_counts < 3 * ndims)
            break;
          for (MPI_Count i = 0; i < ndims; ++i)
            total *= large_counts[static_cast<std::size_t>(ndims + i)];
        } else {
          if (num_integers < 3 * ndims + 2)
            break;
          for (MPI_Count i = 0; i < ndims; ++i)
            total *= integers[static_cast<std::size_t>(ndims + 1 + i)];
        }
        return rma_visit_component(datatypes[0], total, expected, any_component, unsupported,
                                   depth);
      }
      case MPI_COMBINER_RESIZED: {
        if (num_datatypes < 1)
          break;
        return rma_visit_component(datatypes[0], 1, expected, any_component, unsupported,
                                   depth);
      }
      default:
        break;
    }
    unsupported = true;
    return false;
  }

  // True iff native_datatype<T>() has at least one basic component and every basic component is
  // native_datatype<basic_component_t<T>>().  Used only for an untrusted custom/manual
  // annotation; the walk frees every temporary derived handle before returning.
  template<class T>
  [[nodiscard]] inline bool rma_committed_datatype_is_homogeneous() {
    using U = std::remove_cv_t<mp1::basic_component_t<T>>;
    static_assert(not std::is_void_v<U>,
                  "mp1: an untrusted custom datatype must have a basic component");
    // The annotation can be wrong (or a custom factory can fail): a null committed handle or a
    // null basic component is rejected before any MPI metadata call.
    const MPI_Datatype committed = native_datatype<T>();
    const MPI_Datatype expected = native_datatype<U>();
    if (committed == MPI_DATATYPE_NULL or expected == MPI_DATATYPE_NULL)
      return false;
    bool any_component = false;
    bool unsupported = false;
    const bool homogeneous =
        rma_walk_basic_components(committed, expected, any_component, unsupported, 0);
    return homogeneous and any_component and not unsupported;
  }

  // The mandatory one-time contract check for MP1_DATATYPE_BASIC(T,U) / manual
  // struct_builder<T>::component_types.  It is a no-op (zero runtime cost) for predefined and
  // trusted library-generated datatypes.
  template<class T>
  inline void require_rma_custom_homogeneous(const context& ctx) {
    if constexpr (component_info<std::remove_cv_t<T>>::runtime_validate) {
      static const bool homogeneous = rma_committed_datatype_is_homogeneous<T>();
      if (not homogeneous) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "custom/manual homogeneous annotation for {} claims basic component {}, but the "
            "committed datatype is not a derived datatype whose basic components are all that "
            "predefined type (or it cannot be validated); rejecting before the accumulating "
            "MPI call (MPICH aborts in MPIR_DATATYPE_IS_PREDEFINED for a heterogeneous "
            "REPLACE)",
            datatype_type_name<std::remove_cv_t<T>>(),
            datatype_type_name<std::remove_cv_t<mp1::basic_component_t<T>>>());
      }
    }
  }

  // ---- disjoint origin/result byte ranges ------------------------------------------------
  // MPI 4.1 §12.3.4 (Get Accumulate, Fetch and Op, and Compare and Swap) require
  // the origin and result buffers (origin_addr and result_addr) to be disjoint.  The both-flat
  // form uses one predefined T and a known element count, so the accessed byte ranges are exact
  // and any intersection is rejected before the MPI call.  In a mixed flat/layout or
  // layout/layout pair the layout objects' addresses are not available here, so overlap across
  // the distinct buffers/layout objects remains the caller's responsibility (documented in
  // rma.hpp).  A null origin (MPI_NO_OP) or a zero element count has no origin bytes and is
  // not an overlap.

  [[nodiscard]] inline std::size_t rma_flat_byte_extent(count_t elements,
                                                        std::size_t element_size) noexcept {
    const std::size_t count = static_cast<std::size_t>(elements);
    const std::size_t max = std::numeric_limits<std::size_t>::max();
    if (element_size != 0 and count > max / element_size)
      return max;
    return count * element_size;
  }

  [[nodiscard]] inline bool rma_byte_ranges_overlap(const void* first, const void* second,
                                                    std::size_t bytes) noexcept {
    if (bytes == 0)
      return false;
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(first);
    const std::uintptr_t b = reinterpret_cast<std::uintptr_t>(second);
    return a <= b ? b - a < bytes : a - b < bytes;
  }

  template<class T>
  inline void require_disjoint_flat_origin_result(const T* origin, const T* result,
                                                  count_t elements, const context& ctx) {
    if (origin == nullptr or result == nullptr or elements <= 0)
      return;
    if (rma_byte_ranges_overlap(origin, result, rma_flat_byte_extent(elements, sizeof(T))))
        [[unlikely]] {
      throw_invalid(
          ctx.operation,
          "origin and result buffers overlap; MPI 4.1 §12.3.4 requires them to be disjoint");
    }
  }

  // Compile-time CAS category check. A custom MP1_DATATYPE_TRAITS mapping that claims
  // predefinedness is classified at run time (category `invalid` statically) and re-checked by
  // require_rma_cas_datatype before the call.
  template<class T>
  inline constexpr bool rma_cas_datatype_v = [] {
    using U = std::remove_cv_t<T>;
    if constexpr (not is_predefined_datatype_v<U>) {
      return false;
    } else {
      constexpr rma_datatype_category category = static_rma_category<U>();
      return category == rma_datatype_category::c_integer or
             category == rma_datatype_category::logical or
             category == rma_datatype_category::byte or
             category == rma_datatype_category::invalid;
    }
  }();

  // ---- MPI_PROC_NULL no-op calls -----------------------------------------------------------
  //
  // MPI ignores every other argument of a communication call whose target rank is
  // MPI_PROC_NULL, but the wrapper must not rely on provider leniency for a null buffer or a
  // null layout.  Each no-op branch issues exactly one normalized MPI call with valid
  // descriptors: MPI_BYTE, count 0 and a dummy buffer for put/get, MPI_BYTE + the family's
  // most permissive valid op (MPI_REPLACE for accumulate, MPI_NO_OP for get_accumulate and
  // fetch_and_op) and MPI_BYTE for compare_and_swap.  The epoch must still be closed by the
  // application (MPI 4.1 §12.3).

  inline void rma_noop_put(const mpi_window& w) {
    check_mpi(MPI_Put_c(rma_dummy_buffer(), 0, MPI_BYTE, MPI_PROC_NULL, 0, 0, MPI_BYTE,
                        w.native_handle()),
              "MPI_Put_c");
  }

  inline void rma_noop_get(const mpi_window& w) {
    check_mpi(MPI_Get_c(rma_dummy_buffer(), 0, MPI_BYTE, MPI_PROC_NULL, 0, 0, MPI_BYTE,
                        w.native_handle()),
              "MPI_Get_c");
  }

  inline void rma_noop_accumulate(const mpi_window& w) {
    check_mpi(MPI_Accumulate_c(rma_dummy_buffer(), 0, MPI_BYTE, MPI_PROC_NULL, 0, 0, MPI_BYTE,
                               MPI_REPLACE, w.native_handle()),
              "MPI_Accumulate_c");
  }

  inline void rma_noop_get_accumulate(const mpi_window& w) {
    check_mpi(
        MPI_Get_accumulate_c(rma_dummy_buffer(), 0, MPI_BYTE, rma_dummy_buffer(), 0, MPI_BYTE,
                             MPI_PROC_NULL, 0, 0, MPI_BYTE, MPI_NO_OP, w.native_handle()),
        "MPI_Get_accumulate_c");
  }

  [[nodiscard]] inline request rma_noop_rput(const mpi_window& w) {
    MPI_Request native{MPI_REQUEST_NULL};
    check_mpi(MPI_Rput_c(rma_dummy_buffer(), 0, MPI_BYTE, MPI_PROC_NULL, 0, 0, MPI_BYTE,
                         w.native_handle(), &native),
              "MPI_Rput_c");
    return request::from_rma(native);
  }

  [[nodiscard]] inline request rma_noop_rget(const mpi_window& w) {
    MPI_Request native{MPI_REQUEST_NULL};
    check_mpi(MPI_Rget_c(rma_dummy_buffer(), 0, MPI_BYTE, MPI_PROC_NULL, 0, 0, MPI_BYTE,
                         w.native_handle(), &native),
              "MPI_Rget_c");
    return request::from_rma(native);
  }

  [[nodiscard]] inline request rma_noop_raccumulate(const mpi_window& w) {
    MPI_Request native{MPI_REQUEST_NULL};
    check_mpi(MPI_Raccumulate_c(rma_dummy_buffer(), 0, MPI_BYTE, MPI_PROC_NULL, 0, 0, MPI_BYTE,
                                MPI_REPLACE, w.native_handle(), &native),
              "MPI_Raccumulate_c");
    return request::from_rma(native);
  }

  [[nodiscard]] inline request rma_noop_rget_accumulate(const mpi_window& w) {
    MPI_Request native{MPI_REQUEST_NULL};
    check_mpi(MPI_Rget_accumulate_c(rma_dummy_buffer(), 0, MPI_BYTE, rma_dummy_buffer(), 0,
                                    MPI_BYTE, MPI_PROC_NULL, 0, 0, MPI_BYTE, MPI_NO_OP,
                                    w.native_handle(), &native),
              "MPI_Rget_accumulate_c");
    return request::from_rma(native);
  }

  // ---- descriptors ------------------------------------------------------------------------
  // type/count are exactly what the MPI call receives; elements is the number of T elements
  // described and is used only for local size matching.
  struct rma_origin_view {
    const void* data{nullptr};
    MPI_Datatype type{MPI_DATATYPE_NULL};
    MPI_Count count{0};
    count_t elements{0};
  };

  struct rma_result_view {
    void* data{nullptr};
    MPI_Datatype type{MPI_DATATYPE_NULL};
    MPI_Count count{0};
    count_t elements{0};
  };

  struct rma_target_view {
    int rank{0};
    MPI_Aint disp{0};
    MPI_Datatype type{MPI_DATATYPE_NULL};
    MPI_Count count{0};
    count_t elements{0};
  };

  // ---- caller-side origin/result form descriptors -----------------------------------------
  //
  // The buffer forms replace the pointer+count/layout spellings with two shapes:
  //   * std::span<T>/std::span<const T>  native T array; `elements` is span.size() and the MPI
  //                                      datatype is native_datatype<T>() with an element
  //                                      count;
  //   * region<T>/region<const T>        one occurrence of the caller's layout; `elements` is
  //                                      map().size() and the MPI datatype is the layout with
  //                                      count 1.
  // `elements` is the size fact that the size rule checks against the resolved target element
  // count (target_count for a flat target, target.size() for a layout target).  These
  // descriptors are built by the thin public adapters in rma.hpp; every equality and the
  // both-flat byte-range check are implemented once, in the prepare_* functions below, never
  // per overload.
  template<class T>
  struct rma_origin_buffer {
    const T* data{nullptr};
    const layout<T>* map{nullptr};
    count_t elements{0};
  };

  template<class T>
  struct rma_result_buffer {
    T* data{nullptr};
    const layout<T>* map{nullptr};
    count_t elements{0};
  };

  // Shape classification for the public `requires` clauses (rma_origin /
  // rma_result are detail refinements).  A read-only origin accepts any region (mutable or
  // const data is never written) or a span of T/const T.  A writable result accepts a mutable
  // region<T> or a span<T>; span<const T>/region<const T> are rejected.
  template<class B>
  inline constexpr bool rma_span_v = false;
  template<class T, std::size_t Extent>
  inline constexpr bool rma_span_v<std::span<T, Extent>> = true;

  template<class B>
  struct rma_element {
    using type = void;
  };
  template<class T, std::size_t Extent>
  struct rma_element<std::span<T, Extent>> {
    using type = std::remove_cv_t<T>;
  };
  template<class T>
  struct rma_element<region<T>> {
    using type = std::remove_cv_t<T>;
  };
  template<class B>
  using rma_element_t = typename rma_element<std::remove_cvref_t<B>>::type;

  template<class B>
  concept rma_origin =
      is_region_v<std::remove_cvref_t<B>> || rma_span_v<std::remove_cvref_t<B>>;

  template<class U>
  inline constexpr bool rma_result_v = false;
  template<class T>
  inline constexpr bool rma_result_v<region<T>> = not std::is_const_v<T>;
  template<class T, std::size_t Extent>
  inline constexpr bool rma_result_v<std::span<T, Extent>> = not std::is_const_v<T>;

  template<class B>
  concept rma_result = rma_result_v<std::remove_cvref_t<B>>;

  // target_layout == nullptr selects the flat form (count elements of T); otherwise the
  // layout describes exactly one target occurrence and its size() is the element count.
  // target_is_updated is true for put/accumulate/get_accumulate (the target buffer is updated
  // and therefore may not have overlapping type-map entries) and false for get (the target is
  // read-only, so an overlapping target datatype is not erroneous under MPI 4.1 §5.1.11).
  //
  // The window is validated once at the RMA entry (require_rma_window), before the
  // MPI_PROC_NULL no-op branch and before any family-specific check; this helper only
  // prepares the descriptor and must not validate it again.
  template<class T>
  [[nodiscard]] rma_target_view prepare_rma_target(const mpi_window& w, int target_rank,
                                                   MPI_Aint target_disp, count_t target_count,
                                                   const layout<T>* target_layout,
                                                   bool target_is_updated, const context& ctx) {
    require_rma_data_rank(w, target_rank, ctx);
    require_rma_target_disp(w, target_disp, ctx);

    rma_target_view view;
    view.rank = target_rank;
    view.disp = target_disp;
    if (target_layout != nullptr) {
      if (not target_layout->is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid target layout");
      if (target_is_updated)
        require_non_overlapping_layout(*target_layout, ctx, "target");
      view.type = target_layout->native_handle();
      view.count = 1;
      view.elements = target_layout->size();
    } else {
      if (target_count < 0) [[unlikely]]
        throw_invalid(ctx.operation, "negative target count");
      const MPI_Count count = checked_mpi_count(target_count, ctx);
      view.type = datatype_info_of<T>().type;
      view.count = count;
      view.elements = target_count;
    }
    return view;
  }

  // Size rule: the origin buffer size must equal the target element count before the MPI call.
  // `allow_ignored` is the MPI_NO_OP case (§12.3.4: "the origin_addr, origin_count, and
  // origin_datatype arguments are ignored").  Every MPI_NO_OP call is origin-ignored, whatever
  // the caller supplied: a non-empty origin is not size-checked and not inspected, and the wire
  // carries the dummy origin with count 0 and MPI_BYTE, so a provider that still looks at the
  // ignored argument sees an addressable empty buffer.  This makes the standard atomic-read
  // idiom (a real origin buffer, possibly aliasing the result) legal.  A null pointer with a
  // nonzero size stays rejected for every active operator.  The dominant message names both
  // counts so a mismatch is diagnosable.
  template<class T>
  [[nodiscard]] rma_origin_view prepare_rma_origin(const rma_origin_buffer<T>& origin,
                                                   count_t target_elements, bool allow_ignored,
                                                   const context& ctx) {
    rma_origin_view view;
    view.elements = target_elements;
    if (allow_ignored) {
      view.type = MPI_BYTE;
      view.count = 0;
      view.data = rma_dummy_buffer();
      return view;
    }
    if (origin.data == nullptr and origin.elements > 0) [[unlikely]] {
      throw_invalid(ctx.operation, "null origin buffer");
    }
    if (origin.map != nullptr) {
      if (not origin.map->is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid origin layout");
      if (origin.map->size() != target_elements) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "origin layout has {} elements, target has {}",
                                               origin.map->size(), target_elements);
      }
      view.type = origin.map->native_handle();
      view.count = 1;
    } else {
      if (origin.elements != target_elements) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "origin span has {} elements, target has {}",
                                               origin.elements, target_elements);
      }
      view.type = datatype_info_of<T>().type;
      view.count = static_cast<MPI_Count>(target_elements);
    }
    view.data =
        origin.data != nullptr ? static_cast<const void*>(origin.data) : rma_dummy_buffer();
    return view;
  }

  // Size rule: the result buffer size must equal the target element count (get, get_accumulate
  // and rget_accumulate).  The result is written even for MPI_NO_OP, so it has no ignored case.
  // The result buffer is updated by the operation, so an overlapping result layout is rejected
  // (MPI 4.1 §5.1.11, §12.3.2, §12.3.4 (Get Accumulate)).
  template<class T>
  [[nodiscard]] rma_result_view prepare_rma_result(const rma_result_buffer<T>& result,
                                                   count_t target_elements,
                                                   const context& ctx) {
    rma_result_view view;
    view.elements = target_elements;
    if (result.data == nullptr and target_elements > 0) [[unlikely]]
      throw_invalid(ctx.operation, "null result buffer");
    if (result.map != nullptr) {
      if (not result.map->is_valid()) [[unlikely]]
        throw_invalid(ctx.operation, "invalid result layout");
      require_non_overlapping_layout(*result.map, ctx, "result");
      if (result.map->size() != target_elements) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "result layout has {} elements, target has {}",
                                               result.map->size(), target_elements);
      }
      view.type = result.map->native_handle();
      view.count = 1;
    } else {
      if (result.elements != target_elements) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "result span has {} elements, target has {}",
                                               result.elements, target_elements);
      }
      view.type = datatype_info_of<T>().type;
      view.count = static_cast<MPI_Count>(target_elements);
    }
    view.data = result.data != nullptr ? static_cast<void*>(result.data) : rma_dummy_buffer();
    return view;
  }

  // ---- blocking families ----

  template<class T>
  void rma_put(const mpi_window& w, const rma_origin_buffer<T>& origin, int target_rank,
               MPI_Aint target_disp, count_t target_count, const layout<T>* target_layout,
               const context& ctx) {
    require_rma_window(w, ctx);
    if (target_rank == MPI_PROC_NULL) {
      rma_noop_put(w);
      return;
    }
    const rma_target_view target = prepare_rma_target<T>(
        w, target_rank, target_disp, target_count, target_layout, true, ctx);
    const rma_origin_view source = prepare_rma_origin(origin, target.elements, false, ctx);
    check_mpi(MPI_Put_c(source.data, source.count, source.type, target.rank, target.disp,
                        target.count, target.type, w.native_handle()),
              "MPI_Put_c");
  }

  template<class T>
  void rma_get(const mpi_window& w, const rma_result_buffer<T>& origin, int target_rank,
               MPI_Aint target_disp, count_t target_count, const layout<T>* target_layout,
               const context& ctx) {
    require_rma_window(w, ctx);
    if (target_rank == MPI_PROC_NULL) {
      rma_noop_get(w);
      return;
    }
    // The get target datatype is read-only, so an overlapping target map is legal (§12.3.2).
    const rma_target_view target = prepare_rma_target<T>(
        w, target_rank, target_disp, target_count, target_layout, false, ctx);
    // MPI calls the get output the origin buffer; the wrapper names the position result.
    const rma_result_view result_buffer = prepare_rma_result(origin, target.elements, ctx);
    check_mpi(MPI_Get_c(result_buffer.data, result_buffer.count, result_buffer.type,
                        target.rank, target.disp, target.count, target.type, w.native_handle()),
              "MPI_Get_c");
  }

  template<class T>
  void rma_accumulate(const mpi_window& w, const rma_origin_buffer<T>& origin, int target_rank,
                      MPI_Aint target_disp, count_t target_count,
                      const layout<T>* target_layout, MPI_Op op, const context& ctx) {
    // T is compile-time gated to a predefined or homogeneous-derived element type, and the op
    // was already constructed as predefined_op<basic_component_t<T>>.  Re-checking the family
    // and the category against the basic component (not the aggregate T) keeps MPI_NO_OP out
    // of accumulate even for MPI_PROC_NULL, and lets a homogeneous-derived T reach the
    // accumulate call (MPI 4.1 §12.3.4).
    require_rma_window(w, ctx);
    require_rma_op<mp1::basic_component_t<T>>(op, false, ctx);
    require_rma_custom_homogeneous<T>(ctx);
    if (target_rank == MPI_PROC_NULL) {
      rma_noop_accumulate(w);
      return;
    }
    const rma_target_view target = prepare_rma_target<T>(
        w, target_rank, target_disp, target_count, target_layout, true, ctx);
    const rma_origin_view source = prepare_rma_origin(origin, target.elements, false, ctx);
    check_mpi(MPI_Accumulate_c(source.data, source.count, source.type, target.rank, target.disp,
                               target.count, target.type, op, w.native_handle()),
              "MPI_Accumulate_c");
  }

  template<class T>
  void rma_get_accumulate(const mpi_window& w, const rma_origin_buffer<T>& origin,
                          const rma_result_buffer<T>& result, int target_rank,
                          MPI_Aint target_disp, count_t target_count,
                          const layout<T>* target_layout, MPI_Op op, const context& ctx) {
    require_rma_window(w, ctx);
    require_rma_op<mp1::basic_component_t<T>>(op, true, ctx);
    require_rma_custom_homogeneous<T>(ctx);
    if (target_rank == MPI_PROC_NULL) {
      rma_noop_get_accumulate(w);
      return;
    }
    const rma_target_view target = prepare_rma_target<T>(
        w, target_rank, target_disp, target_count, target_layout, true, ctx);
    const rma_origin_view source =
        prepare_rma_origin(origin, target.elements, op == MPI_NO_OP, ctx);
    const rma_result_view destination = prepare_rma_result(result, target.elements, ctx);
    // Only the both-flat pair has exact locally knowable byte ranges; any mixed or
    // layout/layout pair leaves cross-object overlap to the caller (see the block comment on
    // the disjointness helpers above).  MPI_NO_OP ignores every origin argument, so the caller
    // may use a real origin buffer that aliases the result (the standard atomic-read idiom);
    // the wrapper passes the dummy origin for that call.
    const bool origin_ignored = op == MPI_NO_OP;
    if (not origin_ignored and origin.map == nullptr and result.map == nullptr) {
      require_disjoint_flat_origin_result(origin.data, result.data, target.elements, ctx);
    }
    check_mpi(
        MPI_Get_accumulate_c(source.data, source.count, source.type, destination.data,
                             destination.count, destination.type, target.rank, target.disp,
                             target.count, target.type, op, w.native_handle()),
        "MPI_Get_accumulate_c");
  }

  // ---- request-based families (passive-target epochs only) ----

  // ---- request-owned layout duplicates -----------------------------------------------------
  //
  // An RMA request plan duplicates every layout at initiation, so
  // the caller's `region` object and layout may die as soon as the call returns while the data
  // buffers may not.  MPI itself retains a datatype used by an outstanding communication
  // (MPI 4.1 §5.1.9), but mp1 owns an explicit duplicate of each layout it passes to the
  // native request instead of borrowing the caller's handle:
  //   * a blocking call borrows the layout for the call (no duplicate);
  //   * the four request families capture the origin/result region data pointer and duplicate
  //     its map through `capture_region` (the plan hook), and duplicate the target layout,
  //     then pass the duplicates to the native request;
  //   * the payload is attached to the request and its destructor frees the duplicates when
  //     the request completes (wait/test) or is dropped (which waits first); an initiation
  //     that throws before a request exists frees them at scope exit.  The data pointers are
  //     copied, never owned;
  //   * the payload is only created after the size checks, so a locally rejected call still
  //     makes no MPI call at all; an all-flat request has no payload.
  template<class T>
  class rma_request_payload final : public request_payload {
  public:
    /// `origin` is the origin-role descriptor (nullptr when the call has no origin region);
    /// `result` is the result-role descriptor (nullptr when the call has no result region);
    /// `target_layout` is the layout target, or nullptr for the flat target.
    rma_request_payload(const rma_origin_buffer<T>* origin, const rma_result_buffer<T>* result,
                        const layout<T>* target_layout) {
      if (origin != nullptr and origin->map != nullptr)
        origin_.emplace(capture_region(region<const T>{origin->data, *origin->map}));
      if (result != nullptr and result->map != nullptr)
        result_.emplace(capture_region(region<T>{result->data, *result->map}));
      if (target_layout != nullptr)
        target_.emplace(*target_layout);
    }

    void complete(const status&) override {
    }

    /// The duplicate handle for a position, or the borrowed (flat) handle when that position
    /// has no layout.
    [[nodiscard]] MPI_Datatype origin_type(MPI_Datatype flat) const noexcept {
      return origin_ ? origin_->map.native_handle() : flat;
    }

    [[nodiscard]] MPI_Datatype result_type(MPI_Datatype flat) const noexcept {
      return result_ ? result_->map.native_handle() : flat;
    }

    [[nodiscard]] MPI_Datatype target_type(MPI_Datatype flat) const noexcept {
      return target_ ? target_->native_handle() : flat;
    }

  private:
    std::optional<region_capture<const T>> origin_;
    std::optional<region_capture<T>> result_;
    std::optional<layout<T>> target_;
  };

  /// Own the layouts an initiation passes to the native RMA request and re-point the prepared
  /// views at the duplicates.  `origin`/`result` are the caller descriptors of the matching
  /// role (the other is nullptr), and `origin_view`/`result_view`/`target_view` are the
  /// prepared views to patch.  Returns nullptr (no allocation) when the shape is all-flat.
  /// Must run after prepare_* so every local rejection still happens before the first MPI
  /// call.
  template<class T>
  [[nodiscard]] std::unique_ptr<rma_request_payload<T>> own_rma_request_layouts(
      const rma_origin_buffer<T>* origin, const rma_result_buffer<T>* result,
      const layout<T>* target_layout, rma_origin_view* origin_view,
      rma_result_view* result_view, rma_target_view* target_view) {
    if ((origin == nullptr or origin->map == nullptr) and
        (result == nullptr or result->map == nullptr) and target_layout == nullptr)
      return nullptr;
    auto payload = std::make_unique<rma_request_payload<T>>(origin, result, target_layout);
    if (origin_view != nullptr)
      origin_view->type = payload->origin_type(origin_view->type);
    if (result_view != nullptr)
      result_view->type = payload->result_type(result_view->type);
    if (target_view != nullptr)
      target_view->type = payload->target_type(target_view->type);
    return payload;
  }

  // The four request families share this initiation body (the file layer's "shared helper +
  // per-call lambda" idiom): prepare the target and the origin/result views from the caller
  // descriptors, duplicate the layouts the native request retains, issue the family's native
  // MPI call, attach the payload and return the owned request.  `origin`/`result` are null
  // for a family without that role, `origin_ignored` is the MPI_NO_OP case, and
  // `mpi_entry` keeps the native routine name in the mpi_error context as before.
  //
  // The window was validated once at the entry (require_rma_window), before the
  // MPI_PROC_NULL branch and the family-specific op checks.  Every other local check happens
  // here before `call`: a locally rejected initiation therefore still makes no MPI call.  The
  // payload is attached only after the native call succeeds, so a failed initiation frees the
  // duplicates at scope exit and reports no request.
  template<class T, class Call>
  [[nodiscard]] request rma_request_transfer(
      const mpi_window& w, const context& ctx, int target_rank, MPI_Aint target_disp,
      count_t target_count, const layout<T>* target_layout, const rma_origin_buffer<T>* origin,
      const rma_result_buffer<T>* result, bool target_is_updated, bool origin_ignored,
      const char* mpi_entry, Call&& call) {
    rma_target_view target = prepare_rma_target<T>(w, target_rank, target_disp, target_count,
                                                   target_layout, target_is_updated, ctx);
    rma_origin_view source;
    rma_result_view destination;
    if (origin != nullptr)
      source = prepare_rma_origin(*origin, target.elements, origin_ignored, ctx);
    if (result != nullptr)
      destination = prepare_rma_result(*result, target.elements, ctx);
    // Only the both-flat get_accumulate pair has exact locally knowable byte ranges; a family
    // without both roles, an ignored origin, or a layout in either position leaves
    // cross-object overlap to the caller (see the block comment on the disjointness helpers).
    if (origin != nullptr and result != nullptr and not origin_ignored and
        origin->map == nullptr and result->map == nullptr) {
      require_disjoint_flat_origin_result(origin->data, result->data, target.elements, ctx);
    }
    auto payload = own_rma_request_layouts<T>(
        origin, result, target_layout, origin != nullptr ? &source : nullptr,
        result != nullptr ? &destination : nullptr, &target);
    MPI_Request native{MPI_REQUEST_NULL};
    check_mpi(std::forward<Call>(call)(source, destination, target, &native), mpi_entry);
    request request = request::from_rma(native);
    attach_payload(request, std::move(payload));
    return request;
  }

  template<class T>
  [[nodiscard]] request rma_rput(const mpi_window& w, const rma_origin_buffer<T>& origin,
                                 int target_rank, MPI_Aint target_disp, count_t target_count,
                                 const layout<T>* target_layout, const context& ctx) {
    require_rma_window(w, ctx);
    if (target_rank == MPI_PROC_NULL)
      return rma_noop_rput(w);
    return rma_request_transfer<T>(w, ctx, target_rank, target_disp, target_count,
                                   target_layout, &origin, nullptr, true, false, "MPI_Rput_c",
                                   [&w](const rma_origin_view& source, const rma_result_view&,
                                        const rma_target_view& target, MPI_Request* native) {
                                     return MPI_Rput_c(source.data, source.count, source.type,
                                                       target.rank, target.disp, target.count,
                                                       target.type, w.native_handle(), native);
                                   });
  }

  template<class T>
  [[nodiscard]] request rma_rget(const mpi_window& w, const rma_result_buffer<T>& origin,
                                 int target_rank, MPI_Aint target_disp, count_t target_count,
                                 const layout<T>* target_layout, const context& ctx) {
    require_rma_window(w, ctx);
    if (target_rank == MPI_PROC_NULL)
      return rma_noop_rget(w);
    return rma_request_transfer<T>(
        w, ctx, target_rank, target_disp, target_count, target_layout, nullptr, &origin, false,
        false, "MPI_Rget_c",
        [&w](const rma_origin_view&, const rma_result_view& destination,
             const rma_target_view& target, MPI_Request* native) {
          return MPI_Rget_c(destination.data, destination.count, destination.type, target.rank,
                            target.disp, target.count, target.type, w.native_handle(), native);
        });
  }

  template<class T>
  [[nodiscard]] request rma_raccumulate(const mpi_window& w, const rma_origin_buffer<T>& origin,
                                        int target_rank, MPI_Aint target_disp,
                                        count_t target_count, const layout<T>* target_layout,
                                        MPI_Op op, const context& ctx) {
    require_rma_window(w, ctx);
    require_rma_op<mp1::basic_component_t<T>>(op, false, ctx);
    require_rma_custom_homogeneous<T>(ctx);
    if (target_rank == MPI_PROC_NULL)
      return rma_noop_raccumulate(w);
    return rma_request_transfer<T>(
        w, ctx, target_rank, target_disp, target_count, target_layout, &origin, nullptr, true,
        false, "MPI_Raccumulate_c",
        [&w, op](const rma_origin_view& source, const rma_result_view&,
                 const rma_target_view& target, MPI_Request* native) {
          return MPI_Raccumulate_c(source.data, source.count, source.type, target.rank,
                                   target.disp, target.count, target.type, op,
                                   w.native_handle(), native);
        });
  }

  template<class T>
  [[nodiscard]] request rma_rget_accumulate(const mpi_window& w,
                                            const rma_origin_buffer<T>& origin,
                                            const rma_result_buffer<T>& result, int target_rank,
                                            MPI_Aint target_disp, count_t target_count,
                                            const layout<T>* target_layout, MPI_Op op,
                                            const context& ctx) {
    require_rma_window(w, ctx);
    require_rma_op<mp1::basic_component_t<T>>(op, true, ctx);
    require_rma_custom_homogeneous<T>(ctx);
    if (target_rank == MPI_PROC_NULL)
      return rma_noop_rget_accumulate(w);
    // MPI_NO_OP ignores every origin argument, so a real origin may alias the result (the
    // standard atomic-read idiom); the shared helper normalizes the ignored origin.
    return rma_request_transfer<T>(
        w, ctx, target_rank, target_disp, target_count, target_layout, &origin, &result, true,
        op == MPI_NO_OP, "MPI_Rget_accumulate_c",
        [&w, op](const rma_origin_view& source, const rma_result_view& destination,
                 const rma_target_view& target, MPI_Request* native) {
          return MPI_Rget_accumulate_c(source.data, source.count, source.type, destination.data,
                                       destination.count, destination.type, target.rank,
                                       target.disp, target.count, target.type, op,
                                       w.native_handle(), native);
        });
  }

  // ---- single-element atomics ----

  template<class T>
  void rma_compare_and_swap(const mpi_window& w, const T* origin, const T* compare, T* result,
                            int target_rank, MPI_Aint target_disp, const context& ctx) {
    static_assert(rma_cas_datatype_v<T>,
                  "mp1: MPI_Compare_and_swap requires a predefined C++ datatype in the C "
                  "integer, Logical or Byte category (MPI 4.1 §12.3.4 Compare and Swap); "
                  "double/float/complex, char/wchar_t and derived structs are not permitted");
    require_rma_window(w, ctx);
    if (target_rank == MPI_PROC_NULL) {
      check_mpi(MPI_Compare_and_swap(rma_dummy_buffer(), rma_dummy_buffer(), rma_dummy_buffer(),
                                     MPI_BYTE, MPI_PROC_NULL, 0, w.native_handle()),
                "MPI_Compare_and_swap");
      return;
    }
    require_rma_cas_datatype<T>(ctx);
    const rma_target_view target =
        prepare_rma_target<T>(w, target_rank, target_disp, 1, nullptr, true, ctx);
    if (origin == nullptr) [[unlikely]]
      throw_invalid(ctx.operation, "null origin buffer");
    if (compare == nullptr) [[unlikely]]
      throw_invalid(ctx.operation, "null compare buffer");
    if (result == nullptr) [[unlikely]]
      throw_invalid(ctx.operation, "null result buffer");
    // §12.3.4 (Compare and Swap) makes only origin and result disjoint; compare_addr is a
    // separate read-only input and may alias either of them.
    require_disjoint_flat_origin_result(origin, result, 1, ctx);
    check_mpi(MPI_Compare_and_swap(origin, compare, result, target.type, target.rank,
                                   target.disp, w.native_handle()),
              "MPI_Compare_and_swap");
  }

  template<class T>
  void rma_fetch_and_op(const mpi_window& w, const T* origin, T* result, int target_rank,
                        MPI_Aint target_disp, MPI_Op op, const context& ctx) {
    static_assert(is_predefined_datatype_v<T>,
                  "mp1: MPI_Fetch_and_op requires a predefined datatype (MPI 4.1 §12.3.4 "
                  "Fetch and Op); derived structs and reflected aggregates are not permitted");
    require_rma_window(w, ctx);
    if (target_rank == MPI_PROC_NULL) {
      check_mpi(MPI_Fetch_and_op(rma_dummy_buffer(), rma_dummy_buffer(), MPI_BYTE,
                                 MPI_PROC_NULL, 0, MPI_NO_OP, w.native_handle()),
                "MPI_Fetch_and_op");
      return;
    }
    require_rma_atomic_datatype<T>(ctx);
    require_rma_op<T>(op, true, ctx);
    const rma_target_view target =
        prepare_rma_target<T>(w, target_rank, target_disp, 1, nullptr, true, ctx);
    // MPI_NO_OP ignores the origin arguments (§12.3.4), so a null origin is the atomic-read
    // spelling and a real origin (even one aliasing the result) is equally legal: normalize
    // every ignored origin to the dummy buffer exactly like get_accumulate/rget_accumulate.
    if (origin == nullptr and op != MPI_NO_OP) [[unlikely]]
      throw_invalid(ctx.operation, "null origin buffer");
    if (result == nullptr) [[unlikely]]
      throw_invalid(ctx.operation, "null result buffer");
    const bool origin_ignored = op == MPI_NO_OP;
    if (not origin_ignored)
      require_disjoint_flat_origin_result(origin, result, 1, ctx);
    const void* source = origin_ignored ? static_cast<const void*>(rma_dummy_buffer())
                                        : static_cast<const void*>(origin);
    check_mpi(MPI_Fetch_and_op(source, result, target.type, target.rank, target.disp, op,
                               w.native_handle()),
              "MPI_Fetch_and_op");
  }

}  // namespace mp1::detail

#endif
