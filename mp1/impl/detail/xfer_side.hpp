#if !(defined MP1_DETAIL_XFER_SIDE_HPP)

#define MP1_DETAIL_XFER_SIDE_HPP

// One direction of a variable collective: the shared transfer plan.  It owns the wire element
// counts and extent displacements, the typed staging image, one pack scratch buffer, and
// (copy-on-need) the layout metadata that completion or restart reads; it borrows the
// caller's data buffer and descriptors.  Blocking, nonblocking and persistent forms all use
// this one plan object.
//
// Wire invariant (MPI 4.1 5.1.11 and 6.5-6.8).  Every used typed position of every family,
// lifetime and role transfers
//
//     ( buffer: a contiguous T-array of n elements, count: n, datatype: native_datatype<T>() )
//
// and nothing else:
//  * the wire type is a function of T only.  MPI_BYTE appears iff native_datatype<T>() is
//    MPI_BYTE, i.e. std::byte or a custom MP1_DATATYPE_TRAITS that maps T to it;
//  * the wire count is an element count of T, never a byte count.  A count repeats a datatype
//    at its extent stride (5.1.11), so a byte count is the wrong image whenever
//    size(native_datatype<T>()) != extent(native_datatype<T>()) -- and the library enforces
//    extent(native_datatype<T>()) == sizeof(T), so "n T elements" and "n wire elements"
//    always agree;
//  * a displacement is an extent unit of native_datatype<T>() (6.5/6.6/6.8), which is the
//    public element-displacement unit because the native extent is the C++ object size.
//    No true_lb is added to a pointer or a displacement: MPI applies the datatype's own
//    lower bound.
//
// There are exactly two lowerings, and both produce the same wire type and element count:
//
//  * DIRECT -- the block already is a contiguous array of T in the caller's buffer: a dense
//    counts/pointer+count block, or a layout block recorded as a native-T repetition
//    (single block: is_base_repetition; many blocks: is_identity_repetition).  The block is
//    passed as (ptr, n, native_datatype<T>()) with no copy and no datatype duplication.
//  * STAGED -- any other caller map.  The mapped positions are packed into one contiguous
//    T[n] staging image with MPI_Pack_c + MPI_Unpack_c, the collective transfers that image
//    natively, and completion unpacks only the mapped positions back into the caller's
//    buffer.  Holes and canaries are never written (5.1.11 "Advice to implementors": a
//    receive cannot change storage outside the entries of the communication buffer).  A
//    staged direction owns one scratch buffer sized once at construction; the packed
//    position only sizes that scratch, it is never a wire count (5.2 makes MPI_Pack_size_c
//    an upper bound and the packing-unit format implementation-defined, so the
//    Pack_c -> Unpack_c round trip is the portable conversion between the layout image and
//    the contiguous native T image).
//
// The plan does not know root roles or collective family names and never calls a
// per-block-type all-to-all engine.  It is move-only and is never moved after MPI has seen
// its arrays: a nonblocking or persistent adapter constructs the plan, binds its payload-owned
// metadata, and only then issues the native call.  Its count/displacement/staging arrays are
// inline (small_vector, K = xfer_inline_blocks) for a typical communicator, so a dense
// blocking direction neither allocates nor converts through a heap vector.
//
// Restricted-side disjointness (5.1.11, 6.5, 6.6, 6.7): a direct block i touches native
// elements [displs[i], displs[i] + n_i), so two direct blocks alias exactly when their
// element-index intervals intersect -- exact even when size < extent.  A staged block is
// checked against its exact caller byte map (the packed interval for a dense block, the
// decoded basic-entry intervals otherwise).  The normal monotone displacement arrays are
// proven disjoint in one O(P) allocation-free pass; only out-of-order arrays are sorted, into
// an inline small_vector.

#include "mp1/impl/detail/buffer.hpp"
#include "mp1/impl/detail/layout_lowering.hpp"
#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/detail/vector.hpp"
#include "mp1/impl/layout.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>


namespace mp1::detail {

  // Inline capacity of the plan's count/displacement/offset arrays.  A communicator up to this
  // size never allocates for a dense blocking direction; a larger one allocates once, at
  // construction, because MPI needs the arrays.  K = 16 keeps the per-plan footprint small
  // (3 * 16 * 8 = 384 bytes of inline storage) while covering the 2- and 4-rank suite and
  // typical small communicators.
  inline constexpr std::size_t xfer_inline_blocks = 16;

  // A wire count is an MPI_Count and a displacement is an MPI_Aint; the public element count
  // and displacement are count_t.  The plan never type-puns, so the wider object type must
  // represent every count_t value.
  static_assert(std::numeric_limits<MPI_Count>::is_signed,
                "mp1: MPI_Count must be a signed integer type (MPI 4.1 4.1.2)");
  static_assert(static_cast<unsigned long long>(std::numeric_limits<MPI_Count>::max()) >=
                    static_cast<unsigned long long>(std::numeric_limits<count_t>::max()),
                "mp1: MPI_Count cannot represent every mp1::count_t value");
  static_assert(static_cast<unsigned long long>(std::numeric_limits<MPI_Aint>::max()) >=
                    static_cast<unsigned long long>(std::numeric_limits<count_t>::max()),
                "mp1: MPI_Aint cannot represent every mp1::count_t displacement");

  // ---------------------------------------------------------------------------------------
  // Validated facts of the base datatype of T. The reduction flat predicate and the scalar
  // variable-collective block both consume datatype_info_of<T>(); there is no second cached
  // copy of geometry in this adapter.
  // ---------------------------------------------------------------------------------------

  template<class T>
  [[nodiscard]] const datatype_info& base_datatype() {
    return datatype_info_of<T>();
  }

  [[nodiscard]] inline std::size_t xfer_size(MPI_Count value, const context& ctx) {
    if (value < 0) [[unlikely]] {
      detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                             "negative staging size");
    }
    if (static_cast<unsigned long long>(value) >
        static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) [[unlikely]] {
      detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                             "staging size does not fit std::size_t");
    }
    return static_cast<std::size_t>(value);
  }

  [[nodiscard]] inline MPI_Aint xfer_aint(MPI_Count value, const context& ctx) {
    if (value < static_cast<MPI_Count>(std::numeric_limits<MPI_Aint>::min()) or
        value > static_cast<MPI_Count>(std::numeric_limits<MPI_Aint>::max())) [[unlikely]] {
      detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                             "displacement does not fit MPI_Aint");
    }
    return static_cast<MPI_Aint>(value);
  }

  // A public displacement is in T elements; MPI multiplies it by extent(native_datatype<T>()),
  // which the library pins to the C++ object size.  The value itself is therefore the wire
  // displacement.
  [[nodiscard]] inline MPI_Aint xfer_element_displacement(count_t elements,
                                                          const context& ctx) {
    if (elements < 0) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "negative displacement {}", elements);
    }
    return xfer_aint(static_cast<MPI_Count>(elements), ctx);
  }

  // A dense (direct) displacement is passed to MPI unchanged, so MPI forms the block address as
  // base + d_i * extent itself (MPI 4.1 6.5/6.6/6.8).  The library still validates that the
  // implied byte offset is representable, exactly as the pre-typed-wire dense conversion did:
  // a displacement whose byte address overflows MPI_Aint is a caller error, and handing it to
  // MPI would be a wild address instead of a local std::overflow_error.
  [[nodiscard]] inline MPI_Aint xfer_dense_displacement(count_t elements, count_t extent,
                                                        const context& ctx) {
    const MPI_Aint displacement = xfer_element_displacement(elements, ctx);
    if (extent < 0) [[unlikely]] {
      detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                             "native datatype has a negative extent");
    }
    const count_t bytes = checked_count_mul(elements, extent, ctx);
    (void)xfer_aint(static_cast<MPI_Count>(bytes), ctx);
    return displacement;
  }

  // The byte shift of a staged block base from the caller's base pointer: d_i * extent.
  [[nodiscard]] inline count_t xfer_caller_byte_offset(count_t elements, count_t extent,
                                                       const context& ctx) {
    if (elements < 0) [[unlikely]] {
      detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                             "negative displacement {}", elements);
    }
    return checked_count_mul(elements, extent, ctx);
  }

  [[nodiscard]] inline MPI_Aint xfer_add_aint(MPI_Aint left, MPI_Aint right,
                                              const context& ctx) {
    if ((right > 0 and left > std::numeric_limits<MPI_Aint>::max() - right) or
        (right < 0 and left < std::numeric_limits<MPI_Aint>::min() - right)) [[unlikely]] {
      detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                             "displacement overflow");
    }
    return left + right;
  }

  [[nodiscard]] inline MPI_Count xfer_add_count(MPI_Count left, MPI_Count right,
                                                const context& ctx) {
    if ((right > 0 and left > std::numeric_limits<MPI_Count>::max() - right) or
        (right < 0 and left < std::numeric_limits<MPI_Count>::min() - right)) [[unlikely]] {
      detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                             "element count overflow");
    }
    return left + right;
  }

  [[nodiscard]] inline MPI_Count checked_bytes(MPI_Count elements, MPI_Count element_size,
                                               const context& ctx) {
    if (elements < 0) [[unlikely]]
      throw_invalid(ctx.operation, "negative count");
    if (element_size < 0) [[unlikely]]
      detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                             "negative type size");
    if (elements != 0 and element_size != 0 and
        elements > std::numeric_limits<MPI_Count>::max() / element_size) [[unlikely]] {
      detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                             "byte count overflow");
    }
    return elements * element_size;
  }

  [[nodiscard]] inline MPI_Count pack_capacity(MPI_Count count, MPI_Datatype type,
                                               const context& ctx) {
    MPI_Count size{0};
    check_mpi(MPI_Pack_size_c(count, type, MPI_COMM_SELF, &size), ctx);
    return size;
  }

  // MPI_Pack_size_c is only an upper bound (MPI 4.1 5.2), so a conforming implementation may
  // report a position larger than MPI_Type_size_c; the extra bytes are the implementation's
  // packing-unit format and are consumed by the matching MPI_Unpack_c.  The scratch buffer is
  // sized from MPI_Pack_size_c at construction, so the one local invariant is that the
  // reported position fits that allocation.
  [[nodiscard]] inline MPI_Count checked_pack_position(MPI_Count position, MPI_Count capacity,
                                                       const context& ctx) {
    if (position < 0 or position > capacity) [[unlikely]] {
      detail::fail_as<std::runtime_error>(
          ctx, std::source_location::current(),
          "MPI_Pack_c reported {} packed bytes with a staging capacity of {}",
          static_cast<long long>(position), static_cast<long long>(capacity));
    }
    return position;
  }

  // ---------------------------------------------------------------------------------------
  // Uninitialized move-only element staging.  `std::vector<bool>` cannot be used for a
  // reduction stage (its data() is deleted) and value-initialization would memset a buffer
  // that is always overwritten before it is read.  `new T[n]` default-initializes without
  // the zero fill; a reduction stage is filled by MPI_Pack_c/MPI_Unpack_c or by the native
  // reduction before any read.  The one exception is a reduction whose callback datatype
  // covers a complete T object (an mp1-managed functor): its stage slots are explicitly
  // zero-filled first, because bytes outside the layout's type map are part of the reduction
  // image and must not be transmitted with indeterminate values.  This is the storage
  // contract for staged reductions.  A staged variable direction requires T to be
  // default-constructible for the same reason; the dense spelling does not.
  // ---------------------------------------------------------------------------------------

  template<class T>
  class stage_buffer {
  public:
    stage_buffer() noexcept = default;
    stage_buffer(const stage_buffer&) = delete;
    stage_buffer& operator=(const stage_buffer&) = delete;
    stage_buffer(stage_buffer&&) noexcept = default;
    stage_buffer& operator=(stage_buffer&&) noexcept = default;
    ~stage_buffer() = default;

    void resize(std::size_t elements) {
      if (elements == size_)
        return;
      // Deliberately no value-initialization: staging is fully written before it is read.
      std::unique_ptr<T[]> next;
      if (elements != 0)
        next.reset(new T[elements]);
      data_ = std::move(next);
      size_ = elements;
    }

    [[nodiscard]] T* data() noexcept {
      return data_.get();
    }

    [[nodiscard]] const T* data() const noexcept {
      return data_.get();
    }

    [[nodiscard]] std::size_t size() const noexcept {
      return size_;
    }

    [[nodiscard]] bool empty() const noexcept {
      return size_ == 0;
    }

  private:
    std::unique_ptr<T[]> data_;
    std::size_t size_{0};
  };

  // ---------------------------------------------------------------------------------------
  // Shared pack/unpack helpers.  Every staged path (variable collectives and reductions) uses
  // these; there is no int MPI_Pack anywhere in the tree.  The `capacity` argument is the
  // MPI_Pack_size_c allocation and is passed to MPI_Pack_c as its outsize.  A position larger
  // than the supplied capacity is a local error.  A zero-byte block never reaches
  // MPI_Pack_c/MPI_Unpack_c: Open MPI rejects a null outbuf with outsize 0 and MPICH rejects
  // a null inbuf with incount > 0, so the only portable action is to skip the call
  // (MPI 4.1 5.1.8 makes the type map empty).
  // ---------------------------------------------------------------------------------------

  template<class T>
  [[nodiscard]] MPI_Count pack_block(const T* src, const layout<T>& l, void* dst,
                                     MPI_Count capacity, MPI_Comm comm, const context& ctx) {
    if (not l.is_valid()) [[unlikely]]
      throw_invalid(ctx.operation, "invalid layout");
    const MPI_Count exact = l.byte_size();
    if (exact == 0)
      return 0;
    const T* base = staged_ptr(src, l, ctx);
    if (capacity < exact) [[unlikely]] {
      detail::fail_with<std::runtime_error>(
          ctx.operation, std::source_location::current(),
          "staging buffer is smaller than the exact packed size");
    }
    MPI_Count position{0};
    check_mpi(MPI_Pack_c(base, 1, l.native_handle(), dst, capacity, &position, comm), ctx);
    return checked_pack_position(position, capacity, ctx);
  }

  template<class T>
  [[nodiscard]] MPI_Count pack_block(const T* src, const layout<T>& l, MPI_Count repetitions,
                                     void* dst, MPI_Count capacity, MPI_Comm comm,
                                     const context& ctx) {
    if (not l.is_valid()) [[unlikely]]
      throw_invalid(ctx.operation, "invalid layout");
    if (repetitions < 0) [[unlikely]] {
      throw_invalid(ctx.operation, "negative block repetition count");
    }
    // `capacity` is a byte capacity from MPI_Pack_size_c, so the exact lower bound is
    // repetitions * MPI_Type_size_c(l.native_handle()) == repetitions * l.byte_size() bytes.
    // l.size() is the element count, not the block's byte size (they differ whenever the
    // block has padding or holes).
    const MPI_Count exact =
        checked_bytes(repetitions, static_cast<MPI_Count>(l.byte_size()), ctx);
    if (exact == 0)
      return 0;
    const T* base = staged_ptr(src, l, ctx);
    if (capacity < exact) [[unlikely]] {
      detail::fail_with<std::runtime_error>(
          ctx.operation, std::source_location::current(),
          "staging buffer is smaller than the exact packed size");
    }
    MPI_Count position{0};
    check_mpi(MPI_Pack_c(base, repetitions, l.native_handle(), dst, capacity, &position, comm),
              ctx);
    return checked_pack_position(position, capacity, ctx);
  }

  // Pack `repetitions` copies of `block` starting at consecutive byte_extent strides from
  // `src`, then unpack the reported packed bytes (which may include a packing-unit header) into
  // a flat base-T array at `flat_offset`.  `scratch`/`capacity` are the caller-owned pack
  // buffer sized once at construction; this overload never allocates and never calls
  // MPI_Pack_size_c.
  //
  // The pack->unpack round trip is NOT redundant: MPI 4.1 5.1.8 makes the packing-unit format
  // implementation-defined (the rationale explicitly allows an implementation to prepend a
  // header, e.g. sender-architecture information), so a packing unit must not be reinterpreted
  // as the native object image of the flat T array.  MPI_Unpack is the portable conversion into
  // the native representation.
  template<class T>
  void stage_pack_blocks(const T* src, const layout<T>& block, count_t repetitions,
                         stage_buffer<T>& flat, count_t flat_offset,
                         stage_buffer<std::byte>& scratch, MPI_Count capacity,
                         const context& ctx) {
    if (repetitions == 0)
      return;
    if (not block.is_valid()) [[unlikely]]
      throw_invalid(ctx.operation, "invalid block layout");
    if (block.byte_size() == 0)
      return;
    const MPI_Count packed = pack_block(src, block, static_cast<MPI_Count>(repetitions),
                                        scratch.data(), capacity, MPI_COMM_SELF, ctx);
    const MPI_Count elements = checked_bytes(static_cast<MPI_Count>(repetitions),
                                             static_cast<MPI_Count>(block.size()), ctx);
    const std::size_t offset = static_cast<std::size_t>(flat_offset);
    if (flat.size() < offset + static_cast<std::size_t>(elements)) [[unlikely]] {
      detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                             "staging vector too small");
    }
    MPI_Count position{0};
    check_mpi(MPI_Unpack_c(scratch.data(), packed, &position, flat.data() + offset, elements,
                           native_datatype<T>(), MPI_COMM_SELF),
              ctx);
    if (position != packed) [[unlikely]] {
      detail::fail_with<std::runtime_error>(ctx.operation, std::source_location::current(),
                                            "flat staging consumed the wrong byte count");
    }
  }

  // Pack a flat base-T range into `scratch`, then unpack `repetitions` copies of `block` into
  // the caller's buffer at consecutive byte_extent strides.  MPI_Unpack writes only the type
  // map's entries, so the caller's holes and padding are preserved (MPI 4.1 5.1.11 advice).
  template<class T>
  void stage_unpack_blocks(const stage_buffer<T>& flat, count_t flat_offset, T* dst,
                           const layout<T>& block, count_t repetitions,
                           stage_buffer<std::byte>& scratch, MPI_Count capacity,
                           const context& ctx) {
    if (repetitions == 0)
      return;
    if (not block.is_valid()) [[unlikely]]
      throw_invalid(ctx.operation, "invalid block layout");
    if (block.byte_size() == 0)
      return;
    const MPI_Count elements = checked_bytes(static_cast<MPI_Count>(repetitions),
                                             static_cast<MPI_Count>(block.size()), ctx);
    const std::size_t offset = static_cast<std::size_t>(flat_offset);
    if (flat.size() < offset + static_cast<std::size_t>(elements)) [[unlikely]] {
      detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                             "staging vector too small");
    }
    MPI_Count position{0};
    check_mpi(MPI_Pack_c(flat.data() + offset, elements, native_datatype<T>(), scratch.data(),
                         capacity, &position, MPI_COMM_SELF),
              ctx);
    position = checked_pack_position(position, capacity, ctx);
    MPI_Count unpacked{0};
    check_mpi(
        MPI_Unpack_c(scratch.data(), position, &unpacked, dst,
                     static_cast<MPI_Count>(repetitions), block.native_handle(), MPI_COMM_SELF),
        ctx);
    if (unpacked != position) [[unlikely]] {
      detail::fail_with<std::runtime_error>(ctx.operation, std::source_location::current(),
                                            "layout staging consumed the wrong byte count");
    }
  }

  // Allocating spellings, kept as thin wrappers so the reduction call sites are unchanged.
  template<class T>
  void stage_pack_blocks(const T* src, const layout<T>& block, count_t repetitions,
                         stage_buffer<T>& flat, count_t flat_offset, const context& ctx) {
    if (repetitions == 0 or not block.is_valid() or block.byte_size() == 0)
      return;
    const MPI_Count capacity =
        pack_capacity(static_cast<MPI_Count>(repetitions), block.native_handle(), ctx);
    stage_buffer<std::byte> scratch;
    scratch.resize(xfer_size(capacity, ctx));
    stage_pack_blocks(src, block, repetitions, flat, flat_offset, scratch, capacity, ctx);
  }

  template<class T>
  void stage_unpack_blocks(const stage_buffer<T>& flat, count_t flat_offset, T* dst,
                           const layout<T>& block, count_t repetitions, const context& ctx) {
    if (repetitions == 0 or not block.is_valid() or block.byte_size() == 0)
      return;
    const MPI_Count elements = checked_bytes(static_cast<MPI_Count>(repetitions),
                                             static_cast<MPI_Count>(block.size()), ctx);
    const MPI_Count capacity = pack_capacity(elements, native_datatype<T>(), ctx);
    stage_buffer<std::byte> scratch;
    scratch.resize(xfer_size(capacity, ctx));
    stage_unpack_blocks(flat, flat_offset, dst, block, repetitions, scratch, capacity, ctx);
  }

  // ---------------------------------------------------------------------------------------
  // Restricted-side interval merge.  Two caller blocks alias exactly when their basic-entry
  // byte intervals intersect; a proven overlap is a caller error (MPI 4.1 5.1.11 makes a
  // datatype that updates one location twice erroneous, and 6.5/6.6/6.7/6.8 make the root's
  // gatherv receive / scatterv send / allgatherv receive the restricted sides).
  // ---------------------------------------------------------------------------------------

  struct xfer_interval {
    MPI_Aint start{0};
    MPI_Aint end{0};
    std::size_t block{0};
  };

  // True when the intervals, in caller order, are already known disjoint: every non-empty
  // block starts at or after the end of everything before it.  This is a proof, not a failed
  // search, so accepting on it never misses a real alias.  No allocation, O(P).
  [[nodiscard]] inline bool xfer_intervals_monotone(const xfer_interval* first,
                                                    std::size_t count) noexcept {
    MPI_Aint covered_end = 0;
    bool have_covered = false;
    for (std::size_t i = 0; i < count; ++i) {
      const xfer_interval& current = first[i];
      if (have_covered and current.start < covered_end)
        return false;
      covered_end = current.end;
      have_covered = true;
    }
    return true;
  }

  // Sort and merge the intervals after the monotone fast path failed (an out-of-order but
  // disjoint caller layout).  Inline for the normal block count; only a very large or very
  // fragmentary map allocates.
  inline void xfer_require_disjoint_intervals(
      small_vector<xfer_interval, xfer_inline_blocks>& intervals, const context& ctx) {
    if (xfer_intervals_monotone(intervals.data(), intervals.size()))
      return;
    std::sort(intervals.data(), intervals.data() + intervals.size(),
              [](const xfer_interval& left, const xfer_interval& right) {
                return left.start < right.start;
              });
    MPI_Aint covered_end = 0;
    bool have_covered = false;
    std::size_t covered_block = 0;
    for (std::size_t i = 0; i < intervals.size(); ++i) {
      const xfer_interval& current = intervals[i];
      if (have_covered and current.start < covered_end) [[unlikely]] {
        if (current.block == covered_block) {
          detail::fail_as<std::invalid_argument>(
              ctx, std::source_location::current(),
              "receive layout {} addresses the same location more than once (MPI 4.1 5.1.11); "
              "the restricted side must not read or write a location twice",
              current.block);
        }
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "blocks {} and {} overlap in the caller buffer; the restricted side must not read "
            "or write a location twice (make the block layouts or displacements disjoint)",
            covered_block, current.block);
      }
      if (not have_covered or current.end > covered_end) {
        covered_end = current.end;
        covered_block = current.block;
        have_covered = true;
      }
    }
  }

  // The direction of one plan and the staging-prime policy its in-place receive needs.  They
  // live here because the plan is the lowest layer that names a direction (the collective
  // engine includes this header, not the other way round).
  enum class transfer_direction : unsigned char { send, receive };
  enum class prime_policy : unsigned char { none, may_prime };

  // ---------------------------------------------------------------------------------------
  // One direction of a variable collective.
  //
  // Metadata ownership: a blocking plan borrows; a nonblocking plan copies the receive
  // descriptors only when its completion unpacks a staged receive; a persistent plan copies
  // every used direction because prepare_start re-reads the type map after init (MPI 4.1 6.13:
  // the datatype arrays must not be modified until free).  The copies live in
  // `owned_layouts`, shared between the two directions when they refer to the same caller
  // array, so one array costs one set of MPI_Type_dup calls.
  // ---------------------------------------------------------------------------------------

  template<class T>
  struct owned_layouts {
    std::optional<layout<T>> single;
    std::vector<layout<T>> many;
  };

  template<class T>
  class xfer_plan {
  public:
    xfer_plan() noexcept = default;
    xfer_plan(const xfer_plan&) = delete;
    xfer_plan& operator=(const xfer_plan&) = delete;
    xfer_plan(xfer_plan&&) noexcept = default;
    xfer_plan& operator=(xfer_plan&&) noexcept = default;

    // A direction MPI ignores (non-receiving remote role, non-root receive, no-op role).  The
    // buffer is a per-position dummy and the count is zero; the native call must still be made.
    // The direction selects the dummy slot, so an ignored send and an ignored receive never
    // alias even when both sides are absent at the same time (MPI 4.1 §6.5/§6.6; some
    // implementations reject aliased buffers even at count 0).
    [[nodiscard]] static xfer_plan absent(transfer_direction direction) noexcept {
      xfer_plan side;
      side.receiving_ = direction == transfer_direction::receive;
      return side;
    }

    // The MPI_IN_PLACE sentinel in the position the operation defines.  Only the explicit
    // in-place overloads construct this, and only for an intracommunicator root/all-rank role.
    [[nodiscard]] static xfer_plan in_place() noexcept {
      xfer_plan side;
      side.mode_ = mode::in_place;
      return side;
    }

    // `elements` repetitions of native_datatype<T>() at the caller's pointer: the
    // pointer+count single-block dense form, whose scalar spelling is elements == 1.  Direct
    // by construction -- the wire is (base, elements, native_datatype<T>()) and the count
    // repeats at the extent stride (MPI 4.1 5.1.11), which is correct for any T whose extent
    // is sizeof(T) (enforced, including padded and derived layouts of the object).
    [[nodiscard]] static xfer_plan one_base(const T* base, count_t elements,
                                            transfer_direction direction) {
      if (elements < 0) [[unlikely]] {
        throw_invalid("variable collective", "negative scalar block element count");
      }
      const datatype_info& info = base_datatype<T>();
      xfer_plan side;
      side.receiving_ = direction == transfer_direction::receive;
      side.base_ = base;
      side.base_form_ = true;
      side.n_ = 1;
      side.counts_.resize(1);
      // A zero-byte native datatype has an empty type signature, so count N and count 0
      // describe the same no-data message (MPI 4.1 §5.1.11).  Normalize the wire count to 0:
      // a positive count of a zero-size type is legal but MPICH has been recorded to abort on
      // it (SIGFPE) in the variable collectives, and the normalized form is equally correct on
      // every provider.  The caller's element count is still validated for non-negativity.
      side.counts_[0] = info.size == 0 ? 0 : static_cast<MPI_Count>(elements);
      side.total_ = side.counts_[0];
      if (elements != 0 and info.size != 0 and base == nullptr) [[unlikely]] {
        throw_invalid("variable collective", "null pointer for a non-empty scalar block");
      }
      side.mode_ = mode::direct_one;
      // An empty block (n * size == 0) is normalized to the dummy pointer.
      side.direct_ptr_ = (elements == 0 or info.size == 0) ? nullptr : base;
      return side;
    }

    // One caller layout block (`l` describes exactly one block).  The layout is a borrow; the
    // plan never stores the layout object address, only the element pointer.
    //
    // A layout recorded as a native-T repetition (is_base_repetition) is direct: the pointer
    // is shifted by the repetition offset and the count is the layout's element count.  Any
    // other map stages to a contiguous T[n] image so the wire stays a native-T array.
    [[nodiscard]] static xfer_plan one(const T* base, const layout<T>& l,
                                       transfer_direction direction) {
      require_relative(l, "mp1: variable collective: send/receive layout");
      if (direction == transfer_direction::receive)
        require_receive_layout(l, "mp1: variable collective: receive layout");
      xfer_plan side;
      side.receiving_ = direction == transfer_direction::receive;
      side.base_ = base;
      side.borrowed_one_ = std::addressof(l);
      side.n_ = 1;
      side.counts_.resize(1);
      // Same zero-byte normalization as one_base: an empty type signature is a no-data block,
      // so keep the wire count at 0 instead of forwarding a positive count of a zero-size
      // derived type.
      side.counts_[0] = l.byte_size() == 0 ? 0 : static_cast<MPI_Count>(l.size());
      side.total_ = side.counts_[0];
      if (l.byte_size() != 0 and base == nullptr) [[unlikely]] {
        throw_invalid("variable collective", "null pointer for a non-empty layout");
      }
      if (l.is_base_repetition()) {
        side.mode_ = mode::direct_one;
        side.direct_ptr_ = l.byte_size() == 0
                               ? nullptr
                               : repetition_ptr(base, l, "mp1: variable collective: block");
        return side;
      }
      side.mode_ = mode::staged_one;
      side.size_staging_one();
      if (direction == transfer_direction::send)
        side.pack_staged_one("mp1: variable collective: block pack");
      return side;
    }

    // `n` caller layout blocks.  `caller_displs` is the public element displacement of block
    // i (extent units of native_datatype<T>()); nullptr/empty means all-zero.  The direction
    // is direct iff every block is the identity native-T repetition; otherwise the whole
    // direction stages (one buffer, one pointer per role).  `may_prime` reserves the scratch
    // bound of a prime (the send-stage step of an in-place local contribution).
    [[nodiscard]] static xfer_plan many_layouts(const T* base, std::span<const layout<T>> ls,
                                                std::span<const count_t> caller_displs,
                                                std::size_t n, transfer_direction direction,
                                                prime_policy priming) {
      if (n > ls.size()) [[unlikely]] {
        throw_invalid("variable collective", "fewer layouts than the required block count");
      }
      if (not caller_displs.empty() and caller_displs.size() != n) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            {"variable collective"}, std::source_location::current(),
            "displacement array length {} does not match the required {}", caller_displs.size(),
            n);
      }
      const datatype_info& info = base_datatype<T>();
      xfer_plan side;
      side.receiving_ = direction == transfer_direction::receive;
      side.base_ = base;
      side.borrowed_many_ = ls;
      side.n_ = n;
      side.counts_.resize(n);
      MPI_Count total{0};
      bool all_direct = true;
      for (std::size_t i = 0; i < n; ++i) {
        const layout<T>& block = ls[i];
        require_relative(block, "mp1: variable collective: layout in a block array");
        if (direction == transfer_direction::receive)
          require_receive_layout(block, "mp1: variable collective: receive layout");
        if (block.byte_size() != 0 and base == nullptr) [[unlikely]] {
          throw_invalid("variable collective", "null base pointer for a non-empty block");
        }
        // A zero-byte block has an empty type signature: count N and count 0 are the same
        // no-data block, so normalize the wire count to 0 (see one_base).
        side.counts_[i] = block.byte_size() == 0 ? 0 : static_cast<MPI_Count>(block.size());
        total =
            xfer_add_count(total, side.counts_[i], "mp1: variable collective: element count");
        all_direct =
            all_direct and is_identity_repetition(block, "mp1: variable collective: block");
      }
      side.total_ = total;
      side.displs_.resize(n);
      if (all_direct) {
        // Every block is the flat array of its size() native datatypes at the caller's
        // pointer, so displs[i] is literally the public element displacement.
        for (std::size_t i = 0; i < n; ++i) {
          side.displs_[i] =
              xfer_dense_displacement(caller_element_displacement(caller_displs, i),
                                      info.extent, "mp1: variable collective: displacement");
        }
        side.mode_ = mode::direct_many;
        return side;
      }
      // Staged: the wire displacements are element offsets into the staging image, while the
      // caller's block bases are byte offsets d_i * extent from the caller's pointer.
      side.mode_ = mode::staged_many;
      side.caller_byte_offsets_.resize(n);
      MPI_Count offset{0};
      for (std::size_t i = 0; i < n; ++i) {
        const count_t displacement = caller_element_displacement(caller_displs, i);
        side.displs_[i] = xfer_aint(offset, "mp1: variable collective: staging displacement");
        side.caller_byte_offsets_[i] = xfer_caller_byte_offset(
            displacement, info.extent, "mp1: variable collective: block byte offset");
        offset =
            xfer_add_count(offset, side.counts_[i], "mp1: variable collective: staging size");
      }
      if (total > static_cast<MPI_Count>(std::numeric_limits<count_t>::max())) [[unlikely]] {
        detail::fail_with<std::overflow_error>(
            "variable collective", std::source_location::current(),
            "staging element count does not fit mp1::count_t");
      }
      side.staging_.resize(xfer_size(total, "mp1: variable collective"));
      side.size_staging_many(priming);
      if (direction == transfer_direction::send)
        side.pack_staged_many("mp1: variable collective: block pack");
      return side;
    }

    // `n` dense blocks.  Block i is exactly layout<T>::contiguous(counts[i]) at the public
    // element displacement displs[i], so it is a native-T repetition: the wire is
    // (base, counts[i], native_datatype<T>()) and displs[i] is passed unchanged as an extent
    // displacement.  Any T with a native datatype is accepted -- no `packed` gate and no
    // size == extent gate: the count repeats at extent(native_datatype<T>()) == sizeof(T),
    // which is the exact image of the caller's T array even for a padded or derived T.
    [[nodiscard]] static xfer_plan many_counts(const T* base, std::span<const count_t> values,
                                               std::span<const count_t> caller_displs,
                                               std::size_t n, transfer_direction direction) {
      if (values.size() != n) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            {"variable collective"}, std::source_location::current(),
            "counts length {} does not match the required {}", values.size(), n);
      }
      if (caller_displs.size() != n) [[unlikely]] {
        detail::fail_as<std::invalid_argument>(
            {"variable collective"}, std::source_location::current(),
            "displacement array length {} does not match the required {}", caller_displs.size(),
            n);
      }
      const datatype_info& info = base_datatype<T>();
      xfer_plan side;
      side.receiving_ = direction == transfer_direction::receive;
      side.base_ = base;
      side.base_form_ = true;
      side.n_ = n;
      side.counts_.resize(n);
      side.displs_.resize(n);
      MPI_Count total{0};
      for (std::size_t i = 0; i < n; ++i) {
        const count_t value = values[i];
        if (value < 0) [[unlikely]] {
          detail::fail_as<std::invalid_argument>(
              {"variable collective"}, std::source_location::current(),
              "negative count {} in counts block {}", value, i);
        }
        // A zero-size native datatype addresses no byte: normalize every count to 0 for the
        // wire (the type signature is empty, so the message is empty either way).  The
        // caller's count is still checked for negativity and the span fit is handled by the
        // builder above; see require_variable_blocks_fit.
        side.counts_[i] = info.size == 0 ? 0 : static_cast<MPI_Count>(value);
        if (value != 0 and info.size != 0 and base == nullptr) [[unlikely]] {
          throw_invalid("variable collective", "null base pointer for a non-empty block");
        }
        total =
            xfer_add_count(total, side.counts_[i], "mp1: variable collective: element count");
      }
      for (std::size_t i = 0; i < n; ++i) {
        side.displs_[i] = xfer_dense_displacement(caller_displs[i], info.extent,
                                                  "mp1: variable collective: displacement");
      }
      side.total_ = total;
      side.mode_ = mode::direct_many;
      return side;
    }

    [[nodiscard]] const void* ptr() const noexcept {
      if (mode_ == mode::in_place)
        return MPI_IN_PLACE;
      if (mode_ == mode::absent)
        return dummy_buffer(slot());
      if (mode_ == mode::direct_one)
        return direct_ptr_ == nullptr ? dummy_buffer(slot()) : direct_ptr_;
      if (mode_ == mode::staged_one or mode_ == mode::staged_many)
        return staging_.empty() ? dummy_buffer(slot())
                                : static_cast<const void*>(staging_.data());
      return total_ == 0 or base_datatype<T>().size == 0 ? dummy_buffer(slot())
                                                         : static_cast<const void*>(base_);
    }

    // Mutable view for the receive position of a native collective.  The caller declared the
    // receive buffer non-const in the public overload; the adapter stores it as const so the
    // same object can represent either direction.
    [[nodiscard]] void* mutable_ptr() const noexcept {
      return const_cast<void*>(ptr());
    }

    // Scalar count of a single-block direction.  A many-block direction is described by
    // counts()/displs() instead.
    [[nodiscard]] MPI_Count count() const noexcept {
      if (mode_ == mode::absent or mode_ == mode::in_place)
        return 0;
      if (mode_ == mode::direct_many or mode_ == mode::staged_many)
        return total_;
      return counts_.empty() ? 0 : counts_[0];
    }

    [[nodiscard]] const MPI_Count* counts() const noexcept {
      return (mode_ == mode::direct_many or mode_ == mode::staged_many) ? counts_.data()
                                                                        : nullptr;
    }

    [[nodiscard]] const MPI_Aint* displs() const noexcept {
      return (mode_ == mode::direct_many or mode_ == mode::staged_many) ? displs_.data()
                                                                        : nullptr;
    }

    // -------------------------------------------------------------------------------------
    // Metadata ownership (copy-on-need).  A plan that borrowed its descriptors can be given
    // payload-owned copies after construction, before the native call; the accessors then
    // read the owned copies and the borrow is dropped.  Direct directions never need this.
    // -------------------------------------------------------------------------------------

    // A base-datatype block (one_base) is described by native_datatype<T>() and never has a
    // caller layout, even when it stages through the one-block modes (MPI 4.1 5.1.8 makes
    // the base type map the datatype's own map).
    [[nodiscard]] bool uses_layout() const noexcept {
      return not base_form_ and (mode_ == mode::direct_one or mode_ == mode::staged_one or
                                 mode_ == mode::direct_many or mode_ == mode::staged_many);
    }

    // The completion of a nonblocking direction reads the layout only when a staged receive is
    // unpacked into the caller buffer; a send or direct direction never touches it again.
    [[nodiscard]] bool needs_metadata_at_completion() const noexcept {
      return not base_form_ and receiving_ and
             (mode_ == mode::staged_one or mode_ == mode::staged_many);
    }

    // A persistent start re-packs a staged send and re-primes a staged in-place receive from
    // the caller buffer; both read the type map after *_init returns.  A direct direction
    // never reads a layout after construction.
    [[nodiscard]] bool needs_metadata_at_restart() const noexcept {
      return not base_form_ and (mode_ == mode::staged_one or mode_ == mode::staged_many);
    }

    [[nodiscard]] std::shared_ptr<const owned_layouts<T>> metadata() const noexcept {
      return owned_;
    }

    // One copy set (one MPI_Type_dup per block) of the borrowed descriptors, owned by a
    // request payload.  A moved-in input already has metadata and is not copied again.
    [[nodiscard]] std::shared_ptr<owned_layouts<T>> copy_borrowed_metadata() const {
      if (not uses_layout()) [[unlikely]] {
        detail::fail_with<std::logic_error>(
            "xfer_plan", std::source_location::current(),
            "a direct or scalar direction has no borrowed metadata");
      }
      auto hold = std::make_shared<owned_layouts<T>>();
      if (mode_ == mode::direct_one or mode_ == mode::staged_one) {
        hold->single.emplace(single_layout());
      } else {
        hold->many.reserve(n_);
        for (std::size_t i = 0; i < n_; ++i)
          hold->many.push_back(layout_at(i));
      }
      return hold;
    }

    // Bind already-owned metadata (moved input or a direction that shares the other
    // direction's array).  Never throws; the plan is not moved after this point.
    void bind_metadata(std::shared_ptr<const owned_layouts<T>> metadata) noexcept {
      owned_ = std::move(metadata);
    }

    void clear_borrowed_metadata() noexcept {
      borrowed_one_ = nullptr;
      borrowed_many_ = std::span<const layout<T>>{};
    }

    // True when the two directions were constructed from the same caller array/object, so one
    // copy set serves both (dedup of MPI_Type_dup for alltoallv's send/recv sharing).
    [[nodiscard]] bool shares_borrowed_metadata(const xfer_plan& other) const noexcept {
      if (not uses_layout() or not other.uses_layout())
        return false;
      if (mode_ == mode::direct_one or mode_ == mode::staged_one) {
        return (other.mode_ == mode::direct_one or other.mode_ == mode::staged_one) and
               borrowed_one_ != nullptr and borrowed_one_ == other.borrowed_one_;
      }
      if (mode_ == mode::direct_many or mode_ == mode::staged_many) {
        return (other.mode_ == mode::direct_many or other.mode_ == mode::staged_many) and
               borrowed_many_.data() == other.borrowed_many_.data() and
               borrowed_many_.size() == other.borrowed_many_.size();
      }
      return false;
    }

    [[nodiscard]] const layout<T>& single_layout() const {
      if (owned_ != nullptr) {
        if (not owned_->single.has_value()) [[unlikely]]
          detail::fail_with<std::logic_error>("xfer_plan", std::source_location::current(),
                                              "owned metadata has no single layout");
        return *owned_->single;
      }
      if (borrowed_one_ == nullptr) [[unlikely]]
        detail::fail_with<std::logic_error>("xfer_plan", std::source_location::current(),
                                            "single layout is not available");
      return *borrowed_one_;
    }

    [[nodiscard]] const layout<T>& layout_at(std::size_t index) const {
      if (owned_ != nullptr) {
        if (index >= owned_->many.size()) [[unlikely]]
          detail::fail_with<std::logic_error>("xfer_plan", std::source_location::current(),
                                              "owned metadata has no such block");
        return owned_->many[index];
      }
      if (index >= borrowed_many_.size()) [[unlikely]]
        detail::fail_with<std::logic_error>("xfer_plan", std::source_location::current(),
                                            "block layout is not available");
      return borrowed_many_[index];
    }

    // Wire element count of one block, independent of the spelling: a dense count, a layout's
    // element count, or a scalar block's element count.  The collective self-block checks
    // compare these (the element count is the type-signature unit, MPI 4.1 5.1.11/6.1).  A
    // single-block direction uses index 0.
    [[nodiscard]] MPI_Count block_elements(std::size_t index) const noexcept {
      const bool many = mode_ == mode::direct_many or mode_ == mode::staged_many;
      if (many)
        return index < n_ ? counts_[index] : 0;
      return counts_.empty() ? 0 : counts_[0];
    }

    // Restricted-side alias check.  MPI 4.1 5.1.11 defines the type map as the set of
    // locations a datatype addresses; 6.5 forbids the gatherv root receive from writing one
    // location twice and 6.7 inherits that rule for allgatherv.  6.6 is the read-side analogue
    // for scatterv's root send.  6.8 states no verbatim write-once sentence for alltoallv, but
    // mp1 keeps the same receive-side check there as the documented safety contract.
    //
    // A direct block i touches exactly the native elements [displs[i], displs[i] + n_i), so
    // element-interval disjointness is exact for the whole direction even when
    // size < extent.  A staged block contributes its exact caller byte map: the packed
    // interval for a dense block, the decoded basic-entry intervals for a non-dense one, and
    // the conservative true-extent bound only for a constructor the decoder does not model.
    // The monotone caller arrays are proven in one allocation-free O(P) pass; only
    // out-of-order arrays are sorted.  Zero-size blocks contribute nothing.
    void require_disjoint_blocks(const context& ctx) const {
      if (mode_ == mode::direct_many) {
        if (total_ == 0 or base_datatype<T>().size == 0)
          return;
        MPI_Aint covered_end = 0;
        bool have_covered = false;
        bool monotone = true;
        for (std::size_t i = 0; i < n_; ++i) {
          if (counts_[i] == 0)
            continue;
          const MPI_Aint start = displs_[i];
          const MPI_Aint end = xfer_add_aint(start, xfer_aint(counts_[i], ctx), ctx);
          if (have_covered and start < covered_end) {
            monotone = false;
            break;
          }
          covered_end = end;
          have_covered = true;
        }
        if (monotone)
          return;
        small_vector<xfer_interval, xfer_inline_blocks> intervals;
        intervals.resize(n_);
        std::size_t used = 0;
        for (std::size_t i = 0; i < n_; ++i) {
          if (counts_[i] == 0)
            continue;
          const MPI_Aint start = displs_[i];
          intervals[used++] =
              xfer_interval{start, xfer_add_aint(start, xfer_aint(counts_[i], ctx), ctx), i};
        }
        intervals.resize(used);
        xfer_require_disjoint_intervals(intervals, ctx);
        return;
      }
      if (mode_ != mode::staged_many)
        return;
      small_vector<xfer_interval, xfer_inline_blocks> intervals;
      std::size_t used = 0;
      const auto append = [&](MPI_Aint start, MPI_Aint end, std::size_t block) {
        if (used == intervals.size())
          intervals.resize(intervals.empty() ? xfer_inline_blocks : intervals.size() * 2);
        intervals[used++] = xfer_interval{start, end, block};
      };
      for (std::size_t i = 0; i < n_; ++i) {
        if (counts_[i] == 0)
          continue;
        const layout<T>& block = layout_at(i);
        const MPI_Aint block_base =
            xfer_aint(static_cast<MPI_Count>(caller_byte_offsets_[i]), ctx);
        if (block.dense()) {
          const MPI_Aint start = xfer_add_aint(
              block_base, xfer_aint(static_cast<MPI_Count>(block.true_byte_lower_bound()), ctx),
              ctx);
          const MPI_Aint span = xfer_aint(static_cast<MPI_Count>(block.byte_size()), ctx);
          if (span != 0)
            append(start, xfer_add_aint(start, span, ctx), i);
          continue;
        }
        std::vector<type_map_interval> map;
        if (decode_type_map(block.native_handle(), map, ctx)) {
          for (const type_map_interval& entry : map) {
            if (entry.end <= entry.begin)
              continue;
            append(xfer_add_aint(block_base, entry.begin, ctx),
                   xfer_add_aint(block_base, entry.end, ctx), i);
          }
          continue;
        }
        // Conservative superset for an unmodelled constructor (DARRAY, VALUE_INDEX, ...).
        const count_t span = block.true_byte_extent();
        if (span == 0)
          continue;
        const MPI_Aint start = xfer_add_aint(
            block_base, xfer_aint(static_cast<MPI_Count>(block.true_byte_lower_bound()), ctx),
            ctx);
        append(start, xfer_add_aint(start, xfer_aint(static_cast<MPI_Count>(span), ctx), ctx),
               i);
      }
      intervals.resize(used);
      xfer_require_disjoint_intervals(intervals, ctx);
    }

    // Fill one already-present block of a staged receive direction.  Used by the in-place
    // forms whose local contribution is already in the caller's receive buffer (gatherv root,
    // allgatherv local block, alltoallv whole buffer).  No-op on a direct direction: with the
    // direct receive MPI writes the caller's buffer itself, so there is nothing to prime.
    void prime_block(std::size_t index, const context& ctx) {
      if (mode_ != mode::staged_many or not receiving_)
        return;
      if (index >= n_) [[unlikely]]
        throw_invalid(ctx.operation, "block index out of range");
      prime_block_ = index;
      prime_all_ = false;
      primed_ = true;
      pack_staged_block(index, ctx);
    }

    void prime_all(const context& ctx) {
      if (mode_ != mode::staged_many or not receiving_)
        return;
      prime_block_ = 0;
      prime_all_ = true;
      primed_ = true;
      for (std::size_t i = 0; i < n_; ++i)
        pack_staged_block(i, ctx);
    }

    // Persistent-cycle reset: re-pack a staged send from the caller's (possibly modified)
    // buffer and re-prime a staged in-place receive from the caller's receive buffer.  Local
    // only (MPI_Pack_c/MPI_Unpack_c into the existing staging and scratch buffers); never a
    // collective call and never an allocation.
    void prepare_start(const context& ctx) {
      finished_ = false;
      if (mode_ == mode::staged_one) {
        if (not receiving_)
          pack_staged_one(ctx);
        return;
      }
      if (mode_ != mode::staged_many)
        return;
      if (not receiving_) {
        pack_staged_many(ctx);
        return;
      }
      if (not primed_)
        return;
      if (prime_all_)
        prime_all(ctx);
      else if (prime_block_ < n_)
        pack_staged_block(prime_block_, ctx);
    }

    // Completion of a staged receive direction: unpack every block's mapped positions into
    // the caller's buffer.  The other wire parameters are fixed at construction, so completion
    // only converts the native staging image back to the caller's type map.  A request payload
    // calls this from complete(); a blocking adapter calls it after the native call.
    void finish() {
      if (not receiving_ or (mode_ != mode::staged_one and mode_ != mode::staged_many))
        return;
      if (finished_)
        return;
      finished_ = true;
      if (mode_ == mode::staged_one) {
        if (counts_[0] == 0 or staging_.empty())
          return;
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): MPI C API
        stage_unpack_blocks(staging_, 0, const_cast<T*>(base_), single_layout(), 1, scratch_,
                            scratch_capacity_, "mp1: variable collective: block unpack");
        return;
      }
      for (std::size_t i = 0; i < n_; ++i) {
        if (counts_[i] == 0)
          continue;
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast): MPI C API
        T* block_base = offset_bytes(const_cast<T*>(base_), caller_byte_offsets_[i]);
        stage_unpack_blocks(staging_, static_cast<count_t>(displs_[i]), block_base,
                            layout_at(i), 1, scratch_, scratch_capacity_,
                            "mp1: variable collective: block unpack");
      }
    }

  private:
    enum class mode : unsigned char {
      absent,
      in_place,
      direct_one,
      staged_one,
      direct_many,
      staged_many
    };

    // The dummy substitute is per position so an ignored send and an ignored receive never
    // alias: MPICH's fixed-collective parameter checks reject aliased buffers even at count 0.
    [[nodiscard]] buffer_slot slot() const noexcept {
      return receiving_ ? buffer_slot::recv : buffer_slot::send;
    }

    // The public displacement array is optional for the typed spelling: an empty span means
    // all-zero offsets.
    [[nodiscard]] static count_t caller_element_displacement(std::span<const count_t> values,
                                                             std::size_t index) noexcept {
      return (values.empty() or index >= values.size()) ? 0 : values[index];
    }

    // The staging image is exactly one contiguous T slot per mapped element and the scratch is
    // one shared pack buffer; both are sized once, at construction.
    void size_staging_one() {
      staging_.resize(xfer_size(static_cast<MPI_Count>(counts_[0]),
                                "mp1: variable collective: staging image"));
      // A single-block send packs the caller layout; a single-block receive packs the staging
      // image back at completion.  Priming is a many-block operation (prime_block), so a
      // single receive needs only the native bound.
      const MPI_Count capacity =
          receiving_ ? pack_capacity(counts_[0], native_datatype<T>(), "MPI_Pack_size_c")
                     : pack_capacity(1, single_layout().native_handle(), "MPI_Pack_size_c");
      scratch_capacity_ = capacity;
      scratch_.resize(xfer_size(capacity, "mp1: variable collective: scratch"));
    }

    void size_staging_many(prime_policy priming) {
      MPI_Count capacity{0};
      for (std::size_t i = 0; i < n_; ++i) {
        if (counts_[i] == 0)
          continue;
        const layout<T>& block = layout_at(i);
        // A send (or an in-place prime) packs one caller layout occurrence; completion of a
        // receive packs n_i native elements.  Both bounds are upper bounds of the same
        // message, so share one scratch sized for the larger.
        if (receiving_) {
          capacity = std::max(
              capacity, pack_capacity(counts_[i], native_datatype<T>(), "MPI_Pack_size_c"));
          if (priming == prime_policy::may_prime) {
            capacity =
                std::max(capacity, pack_capacity(1, block.native_handle(), "MPI_Pack_size_c"));
          }
        } else {
          capacity =
              std::max(capacity, pack_capacity(1, block.native_handle(), "MPI_Pack_size_c"));
        }
      }
      scratch_capacity_ = capacity;
      scratch_.resize(xfer_size(capacity, "mp1: variable collective: scratch"));
    }

    void pack_staged_one(const context& ctx) {
      if (counts_.empty() or counts_[0] == 0)
        return;
      stage_pack_blocks(base_, single_layout(), 1, staging_, 0, scratch_, scratch_capacity_,
                        ctx);
    }

    void pack_staged_many(const context& ctx) {
      for (std::size_t i = 0; i < n_; ++i)
        pack_staged_block(i, ctx);
    }

    // Fill one staging slot from the caller's block at d_i * extent.  Used by a staged send, a
    // persistent start and an in-place prime (the caller's contribution is already in the
    // receive buffer, so the staging image must be refreshed from it before the call).
    void pack_staged_block(std::size_t index, const context& ctx) {
      if (counts_[index] == 0)
        return;
      const T* block_base = offset_bytes(base_, caller_byte_offsets_[index]);
      stage_pack_blocks(block_base, layout_at(index), 1, staging_,
                        static_cast<count_t>(displs_[index]), scratch_, scratch_capacity_, ctx);
    }

    mode mode_{mode::absent};
    bool receiving_{false};
    bool finished_{false};
    bool base_form_{false};
    bool primed_{false};
    bool prime_all_{false};
    std::size_t prime_block_{0};
    const T* base_{nullptr};
    const layout<T>* borrowed_one_{nullptr};
    std::span<const layout<T>> borrowed_many_{};
    std::shared_ptr<const owned_layouts<T>> owned_;
    const T* direct_ptr_{nullptr};
    std::size_t n_{0};
    MPI_Count total_{0};
    // Wire form of the direction: element counts and extent-unit displacements.  Inline for a
    // typical communicator, so a dense blocking direction never allocates.
    small_vector<MPI_Count, xfer_inline_blocks> counts_;
    small_vector<MPI_Aint, xfer_inline_blocks> displs_;
    // Staged only: byte offset d_i * extent(native_datatype<T>()) of each caller block from
    // the caller's base pointer.
    small_vector<count_t, xfer_inline_blocks> caller_byte_offsets_;
    // Staged only: the exact contiguous T image handed to MPI and the one shared pack buffer.
    stage_buffer<T> staging_;
    stage_buffer<std::byte> scratch_;
    MPI_Count scratch_capacity_{0};
  };

}  // namespace mp1::detail

#endif
