#if !(defined MP1_DETAIL_LAYOUT_LOWERING_HPP)

#define MP1_DETAIL_LAYOUT_LOWERING_HPP

// The one module allowed to convert (layout, pointer) into a lowered pointer/count/datatype.
//
// Rules (MPI 4.1 5.1.6, 5.1.8, 5.1.11):
//  * A layout is applied at the caller's pointer with count 1 and its own handle whenever its
//    map is packed or unpacked (staging, p2p, reductions).
//  * The typed direct lowering replaces (caller_base, 1, layout) by
//    (p + shift, size(), native_datatype<T>()), where shift is
//    layout.true_lb - true_lb(native_datatype<T>()): the layout's map is size() copies of the
//    base datatype at the base extent stride, so a count repeats the map at exactly the
//    caller's stride (5.1.11).  It is legal only when the recorded base_repetition fact
//    holds; applying only layout.true_lb would double-apply the base datatype's own true
//    lower bound.  The many-block form needs the stricter identity_repetition fact (zero
//    shift and byte_extent == size * base extent), because a displacement array cannot
//    express a per-block sub-extent shift.
//  * The packed byte lowering of a layout is (p + true_lb, MPI_BYTE, byte_size), legal only
//    when the recorded packed fact holds.  It is a layout-facts helper; the typed variable
//    wire lowering does not use it, so its size/extent arithmetic cannot leak into a
//    collective lowering.
//  * A staged pack/unpack uses the caller's pointer unchanged and the layout's own handle;
//    the round trip through native_datatype<T>() is the portable layout -> flat T image
//    conversion.
//
// No other header may use byte_lower_bound()/true_byte_lower_bound()/byte_extent()/
// true_byte_extent() in pointer arithmetic.

#include "mp1/impl/detail/buffer.hpp"
#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/layout.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>


namespace mp1::detail {

  [[nodiscard]] inline MPI_Count lowering_mpi_count(count_t value, const context& ctx) {
    if (value < 0) [[unlikely]] {
      throw_invalid(ctx.operation, "negative element count");
    }
    return static_cast<MPI_Count>(value);
  }

  // A pointer to the packed byte image of a fully packed layout.
  template<class T>
  struct packed_view {
    const T* ptr;
    MPI_Count bytes;
  };

  // A repeated-base lowering: (ptr, count, base) replaces (caller_base, 1, layout).
  template<class T>
  struct repetition_view {
    const T* ptr;
    MPI_Count count;
    MPI_Datatype base;
  };

  // A flat-array lowering: (ptr, count, base) addresses the layout's repetitions at the
  // native datatype's extent stride starting at the caller's pointer.
  template<class T>
  struct identity_view {
    const T* ptr;
    MPI_Count count;
    MPI_Datatype base;
  };

  template<class T>
  void require_relative(const layout<T>& l, const context& ctx) {
    if (not l.is_valid()) [[unlikely]] {
      throw_invalid(ctx.operation, "invalid layout");
    }
  }

  // ---------------------------------------------------------------------------------------
  // Exact type map of one layout occurrence.
  //
  // MPI 4.1 §5.1.11 makes the type map authoritative for which memory locations a datatype
  // addresses, and §6.5/§6.6/§6.7/§6.8 require the root's gatherv receive, the root's scatterv
  // send, the allgatherv receive and the alltoallv receive to touch no location twice.  Two
  // blocks alias exactly when their basic-entry byte intervals intersect.
  //
  // A bounding [true_lb, true_lb + true_extent) interval is only a superset of a type map: a
  // subarray tile selecting rows of a row-major matrix has a true extent spanning the
  // unselected columns between the selected rows, so distinct self-placed tiles can appear to
  // overlap although their maps are disjoint.  Decode the map with
  // MPI_Type_get_envelope_c/MPI_Type_get_contents_c (§5.1.13) and compare the exact intervals.
  //
  // The decode covers every constructor the public layout factories emit plus the standard
  // derived-type combiners a custom native datatype can carry.  A constructor this decoder
  // does not model (DARRAY, VALUE_INDEX, large-count SUBARRAY) returns false; the caller then
  // falls back to the conservative true-extent interval, which may over-reject a legal
  // interleaving but never accepts a real alias.
  // ---------------------------------------------------------------------------------------

  struct type_map_interval {
    MPI_Aint begin{0};
    MPI_Aint end{0};
  };

  namespace type_map_detail {

    [[nodiscard]] inline MPI_Aint checked_aint(MPI_Count value, const context& ctx) {
      if (value < static_cast<MPI_Count>(std::numeric_limits<MPI_Aint>::min()) or
          value > static_cast<MPI_Count>(std::numeric_limits<MPI_Aint>::max())) [[unlikely]] {
        detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                               "type-map byte offset does not fit MPI_Aint");
      }
      return static_cast<MPI_Aint>(value);
    }

    [[nodiscard]] inline std::size_t checked_size(MPI_Count value, const context& ctx) {
      if (value < 0 or static_cast<unsigned long long>(value) >
                           static_cast<unsigned long long>(
                               std::numeric_limits<std::size_t>::max())) [[unlikely]] {
        detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                               "type-map entry count does not fit std::size_t");
      }
      return static_cast<std::size_t>(value);
    }

    [[nodiscard]] inline MPI_Aint checked_add(MPI_Aint left, MPI_Aint right,
                                              const context& ctx) {
      if ((right > 0 and left > std::numeric_limits<MPI_Aint>::max() - right) or
          (right < 0 and left < std::numeric_limits<MPI_Aint>::min() - right)) [[unlikely]] {
        detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                               "type-map byte offset overflow");
      }
      return left + right;
    }

    [[nodiscard]] inline MPI_Count checked_product(MPI_Count left, MPI_Count right,
                                                   const context& ctx) {
      if (left < 0 or right < 0) [[unlikely]] {
        detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                               "negative type-map geometry");
      }
      if (left != 0 and right > std::numeric_limits<MPI_Count>::max() / left) [[unlikely]] {
        detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                               "type-map geometry overflow");
      }
      return left * right;
    }

    [[nodiscard]] inline MPI_Count checked_sum(MPI_Count left, MPI_Count right,
                                               const context& ctx) {
      if ((right > 0 and left > std::numeric_limits<MPI_Count>::max() - right) or
          (right < 0 and left < std::numeric_limits<MPI_Count>::min() - right)) [[unlikely]] {
        detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                               "type-map geometry overflow");
      }
      return left + right;
    }

    [[nodiscard]] inline MPI_Aint checked_mul(MPI_Count left, MPI_Count right,
                                              const context& ctx) {
      return checked_aint(checked_product(left, right, ctx), ctx);
    }

    [[nodiscard]] inline MPI_Count element_extent(MPI_Datatype type, const context& ctx) {
      MPI_Count lower_bound{0};
      MPI_Count extent{0};
      check_mpi(MPI_Type_get_extent_c(type, &lower_bound, &extent), ctx);
      if (extent < 0) [[unlikely]] {
        detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                               "negative datatype extent");
      }
      return extent;
    }

    // A constituent returned by MPI_TYPE_GET_CONTENTS is a new derived object the caller must
    // free, except for a predefined datatype. A named or F90 predefined handle, and the
    // predefined unnamed pair of the value-index query (combiner
    // MPI_COMBINER_VALUE_INDEX), are borrowed constants: the pair "is not a duplicate" and
    // "cannot be freed" (MPI 4.1 §6.9.4 §5.1.13), so freeing one is erroneous.
    [[nodiscard]] inline bool is_freeable(MPI_Datatype type, const context& ctx) {
      MPI_Count num_ints{0};
      MPI_Count num_addrs{0};
      MPI_Count num_large{0};
      MPI_Count num_types{0};
      int combiner = MPI_COMBINER_NAMED;
      check_mpi(MPI_Type_get_envelope_c(type, &num_ints, &num_addrs, &num_large, &num_types,
                                        &combiner),
                ctx);
      return combiner != MPI_COMBINER_NAMED and combiner != MPI_COMBINER_F90_REAL and
             combiner != MPI_COMBINER_F90_COMPLEX and combiner != MPI_COMBINER_F90_INTEGER and
             combiner != MPI_COMBINER_VALUE_INDEX;
    }

    class handle_owner {
    public:
      handle_owner() = default;
      handle_owner(const handle_owner&) = delete;
      handle_owner& operator=(const handle_owner&) = delete;
      ~handle_owner() {
        for (MPI_Datatype& handle : handles_) {
          if (handle != MPI_DATATYPE_NULL)
            (void)MPI_Type_free(&handle);
        }
      }

      void add(MPI_Datatype handle) {
        handles_.push_back(handle);
      }

    private:
      std::vector<MPI_Datatype> handles_;
    };

  }  // namespace type_map_detail

  // Decode `root` (one occurrence, i.e. layout count 1) into basic-entry byte intervals
  // relative to the datatype pointer.  Returns false for an unmodelled constructor; the output
  // is then unspecified and the caller must fall back to the conservative bound.
  [[nodiscard]] inline bool decode_type_map(MPI_Datatype root,
                                            std::vector<type_map_interval>& intervals,
                                            const context& ctx) {
    intervals.clear();
    struct frame {
      MPI_Datatype type{MPI_DATATYPE_NULL};
      MPI_Aint base{0};
    };
    std::vector<frame> pending;
    pending.push_back(frame{root, 0});
    type_map_detail::handle_owner owned;

    while (not pending.empty()) {
      const frame current = pending.back();
      pending.pop_back();
      if (current.type == MPI_DATATYPE_NULL)
        return false;

      MPI_Count num_ints{0};
      MPI_Count num_addrs{0};
      MPI_Count num_large{0};
      MPI_Count num_types{0};
      int combiner = MPI_COMBINER_NAMED;
      check_mpi(MPI_Type_get_envelope_c(current.type, &num_ints, &num_addrs, &num_large,
                                        &num_types, &combiner),
                ctx);

      // A named or F90 datatype is a leaf: its whole size is one basic entry.
      if (combiner == MPI_COMBINER_NAMED or combiner == MPI_COMBINER_F90_REAL or
          combiner == MPI_COMBINER_F90_COMPLEX or combiner == MPI_COMBINER_F90_INTEGER) {
        MPI_Count size{0};
        check_mpi(MPI_Type_size_c(current.type, &size), ctx);
        if (size > 0) {
          const MPI_Aint end = type_map_detail::checked_add(
              current.base, type_map_detail::checked_aint(size, ctx), ctx);
          intervals.push_back(type_map_interval{current.base, end});
        }
        continue;
      }
      // MPI_COMBINER_VALUE_INDEX is a predefined unnamed value/index pair, not a derived map
      // this walk models. Reject it as unverifiable before asking for its contents: the
      // handle is borrowed (is_freeable above keeps it out of handle_owner) and the caller
      // falls back to the conservative extent interval.
      if (combiner == MPI_COMBINER_VALUE_INDEX)
        return false;

      const std::size_t int_count = type_map_detail::checked_size(num_ints, ctx);
      const std::size_t addr_count = type_map_detail::checked_size(num_addrs, ctx);
      const std::size_t large_count = type_map_detail::checked_size(num_large, ctx);
      const std::size_t type_count = type_map_detail::checked_size(num_types, ctx);
      std::vector<int> ints(int_count == 0 ? 1 : int_count);
      std::vector<MPI_Aint> addrs(addr_count == 0 ? 1 : addr_count);
      std::vector<MPI_Count> large(large_count == 0 ? 1 : large_count);
      std::vector<MPI_Datatype> types(type_count == 0 ? 1 : type_count, MPI_DATATYPE_NULL);
      check_mpi(MPI_Type_get_contents_c(current.type, num_ints, num_addrs, num_large, num_types,
                                        ints.data(), addrs.data(), large.data(), types.data()),
                ctx);
      for (std::size_t i = 0; i < type_count; ++i) {
        if (types[i] != MPI_DATATYPE_NULL and type_map_detail::is_freeable(types[i], ctx))
          owned.add(types[i]);
      }

      // The large-count constructors put their MPI_Count arguments in array_of_large_counts;
      // the original int constructors put the same arguments in array_of_integers (§5.1.13).
      const auto argument = [&](std::size_t large_index, std::size_t int_index) -> MPI_Count {
        if (large_index < large_count)
          return large[large_index];
        if (int_index < int_count)
          return static_cast<MPI_Count>(ints[int_index]);
        throw_invalid(ctx.operation, "datatype constructor arguments are missing");
      };
      const auto push_child = [&](MPI_Datatype type, MPI_Aint base) {
        pending.push_back(frame{type, base});
      };
      const auto entry_count = [&](MPI_Count count) -> std::size_t {
        const auto entries = type_map_detail::checked_size(count, ctx);
        if (entries > (std::numeric_limits<std::size_t>::max() - 1) / 2) [[unlikely]] {
          detail::fail_with<std::overflow_error>(ctx.operation, std::source_location::current(),
                                                 "type-map entry count is too large");
        }
        return entries;
      };

      switch (combiner) {
        case MPI_COMBINER_DUP:
        case MPI_COMBINER_RESIZED: {
          // Markers change the extent only; the type map is the child's map at offset 0.
          if (num_types != 1)
            return false;
          push_child(types[0], current.base);
          break;
        }
        case MPI_COMBINER_CONTIGUOUS: {
          if (num_types != 1)
            return false;
          const MPI_Count count = argument(0, 0);
          if (count < 0)
            return false;
          const MPI_Count extent = type_map_detail::element_extent(types[0], ctx);
          const MPI_Aint step = type_map_detail::checked_aint(extent, ctx);
          MPI_Aint entry = current.base;
          for (MPI_Count i = 0; i < count; ++i) {
            push_child(types[0], entry);
            entry = type_map_detail::checked_add(entry, step, ctx);
          }
          break;
        }
        case MPI_COMBINER_VECTOR: {
          if (num_types != 1)
            return false;
          const MPI_Count count = argument(0, 0);
          const MPI_Count blocklength = argument(1, 1);
          const MPI_Count stride = argument(2, 2);
          if (count < 0 or blocklength < 0 or stride < 0)
            return false;
          const MPI_Count extent = type_map_detail::element_extent(types[0], ctx);
          const MPI_Aint step = type_map_detail::checked_aint(extent, ctx);
          const MPI_Aint block_step = type_map_detail::checked_mul(stride, extent, ctx);
          MPI_Aint block = current.base;
          for (MPI_Count i = 0; i < count; ++i) {
            MPI_Aint entry = block;
            for (MPI_Count j = 0; j < blocklength; ++j) {
              push_child(types[0], entry);
              entry = type_map_detail::checked_add(entry, step, ctx);
            }
            block = type_map_detail::checked_add(block, block_step, ctx);
          }
          break;
        }
        case MPI_COMBINER_HVECTOR: {
          if (num_types != 1)
            return false;
          const MPI_Count count = argument(0, 0);
          const MPI_Count blocklength = argument(1, 1);
          MPI_Aint stride{0};
          if (num_large >= 3) {
            stride = type_map_detail::checked_aint(large[2], ctx);
          } else if (num_addrs >= 1) {
            stride = addrs[0];
          } else {
            return false;
          }
          if (count < 0 or blocklength < 0)
            return false;
          const MPI_Count extent = type_map_detail::element_extent(types[0], ctx);
          const MPI_Aint step = type_map_detail::checked_aint(extent, ctx);
          MPI_Aint block = current.base;
          for (MPI_Count i = 0; i < count; ++i) {
            MPI_Aint entry = block;
            for (MPI_Count j = 0; j < blocklength; ++j) {
              push_child(types[0], entry);
              entry = type_map_detail::checked_add(entry, step, ctx);
            }
            block = type_map_detail::checked_add(block, stride, ctx);
          }
          break;
        }
        case MPI_COMBINER_INDEXED: {
          if (num_types != 1)
            return false;
          const std::size_t entries = entry_count(argument(0, 0));
          const bool large_form = large_count != 0;
          if ((large_form ? large_count : int_count) != 2 * entries + 1)
            return false;
          const MPI_Count extent = type_map_detail::element_extent(types[0], ctx);
          const MPI_Aint step = type_map_detail::checked_aint(extent, ctx);
          for (std::size_t e = 0; e < entries; ++e) {
            const MPI_Count blocklength =
                large_form ? large[1 + e] : static_cast<MPI_Count>(ints[1 + e]);
            const MPI_Count displacement = large_form
                                               ? large[1 + entries + e]
                                               : static_cast<MPI_Count>(ints[1 + entries + e]);
            MPI_Aint entry = type_map_detail::checked_add(
                current.base, type_map_detail::checked_mul(displacement, extent, ctx), ctx);
            for (MPI_Count j = 0; j < blocklength; ++j) {
              push_child(types[0], entry);
              entry = type_map_detail::checked_add(entry, step, ctx);
            }
          }
          break;
        }
        case MPI_COMBINER_HINDEXED: {
          if (num_types != 1)
            return false;
          const std::size_t entries = entry_count(argument(0, 0));
          // The int constructor returns count+1 ints plus count addresses; the large-count
          // constructor returns 2*count+1 large counts (count, blocklengths, displacements).
          if (large_count != 0) {
            if (large_count != entries + 1 and large_count != 2 * entries + 1)
              return false;
          } else if (int_count != entries + 1) {
            return false;
          }
          const bool displacements_in_large = large_count == 2 * entries + 1;
          if (not displacements_in_large and addr_count < entries)
            return false;
          const MPI_Count extent = type_map_detail::element_extent(types[0], ctx);
          const MPI_Aint step = type_map_detail::checked_aint(extent, ctx);
          for (std::size_t e = 0; e < entries; ++e) {
            const MPI_Count blocklength =
                large_count != 0 ? large[1 + e] : static_cast<MPI_Count>(ints[1 + e]);
            const MPI_Aint displacement =
                displacements_in_large
                    ? type_map_detail::checked_aint(large[1 + entries + e], ctx)
                    : addrs[e];
            MPI_Aint entry = type_map_detail::checked_add(current.base, displacement, ctx);
            for (MPI_Count j = 0; j < blocklength; ++j) {
              push_child(types[0], entry);
              entry = type_map_detail::checked_add(entry, step, ctx);
            }
          }
          break;
        }
        case MPI_COMBINER_INDEXED_BLOCK: {
          if (num_types != 1)
            return false;
          const std::size_t entries = entry_count(argument(0, 0));
          const MPI_Count blocklength = argument(1, 1);
          const bool large_form = large_count != 0;
          if ((large_form ? large_count : int_count) != entries + 2)
            return false;
          const MPI_Count extent = type_map_detail::element_extent(types[0], ctx);
          const MPI_Aint step = type_map_detail::checked_aint(extent, ctx);
          for (std::size_t e = 0; e < entries; ++e) {
            const MPI_Count displacement =
                large_form ? large[2 + e] : static_cast<MPI_Count>(ints[2 + e]);
            MPI_Aint entry = type_map_detail::checked_add(
                current.base, type_map_detail::checked_mul(displacement, extent, ctx), ctx);
            for (MPI_Count j = 0; j < blocklength; ++j) {
              push_child(types[0], entry);
              entry = type_map_detail::checked_add(entry, step, ctx);
            }
          }
          break;
        }
        case MPI_COMBINER_HINDEXED_BLOCK: {
          if (num_types != 1)
            return false;
          const std::size_t entries = entry_count(argument(0, 0));
          const MPI_Count blocklength = argument(1, 1);
          // Int constructor: two ints plus count addresses.  Large-count constructor:
          // count+2 large counts (count, blocklength, displacements).
          if (large_count != 0) {
            if (large_count != entries + 2)
              return false;
          } else if (int_count != 2 or addr_count < entries) {
            return false;
          }
          const MPI_Count extent = type_map_detail::element_extent(types[0], ctx);
          const MPI_Aint step = type_map_detail::checked_aint(extent, ctx);
          for (std::size_t e = 0; e < entries; ++e) {
            const MPI_Aint displacement =
                large_count != 0 ? type_map_detail::checked_aint(large[2 + e], ctx) : addrs[e];
            MPI_Aint entry = type_map_detail::checked_add(current.base, displacement, ctx);
            for (MPI_Count j = 0; j < blocklength; ++j) {
              push_child(types[0], entry);
              entry = type_map_detail::checked_add(entry, step, ctx);
            }
          }
          break;
        }
        case MPI_COMBINER_STRUCT: {
          const std::size_t entries = entry_count(argument(0, 0));
          if (type_count < entries)
            return false;
          // Int constructor: count+1 ints plus count addresses.  Large-count constructor:
          // 2*count+1 large counts (count, blocklengths, displacements).  Both forms keep
          // the block byte displacements in the constructor's displacement array.
          if (large_count != 0) {
            if (large_count != entries + 1 and large_count != 2 * entries + 1)
              return false;
          } else if (int_count != entries + 1) {
            return false;
          }
          const bool displacements_in_large = large_count == 2 * entries + 1;
          if (not displacements_in_large and addr_count < entries)
            return false;
          for (std::size_t e = 0; e < entries; ++e) {
            const MPI_Count blocklength =
                large_count != 0 ? large[1 + e] : static_cast<MPI_Count>(ints[1 + e]);
            const MPI_Aint displacement =
                displacements_in_large
                    ? type_map_detail::checked_aint(large[1 + entries + e], ctx)
                    : addrs[e];
            const MPI_Count extent = type_map_detail::element_extent(types[e], ctx);
            const MPI_Aint step = type_map_detail::checked_aint(extent, ctx);
            MPI_Aint entry = type_map_detail::checked_add(current.base, displacement, ctx);
            for (MPI_Count j = 0; j < blocklength; ++j) {
              push_child(types[e], entry);
              entry = type_map_detail::checked_add(entry, step, ctx);
            }
          }
          break;
        }
        case MPI_COMBINER_SUBARRAY: {
          if (num_types != 1 or int_count < 2)
            return false;
          const int ndims = ints[0];
          if (ndims < 1)
            return false;
          const std::size_t rank = static_cast<std::size_t>(ndims);
          std::vector<MPI_Count> sizes(rank);
          std::vector<MPI_Count> subsizes(rank);
          std::vector<MPI_Count> starts(rank);
          int order = MPI_ORDER_C;
          if (large_count != 0) {
            // MPI_Type_create_subarray_c: ndims and order are the two ints; the sizes,
            // subsizes and starts are 3*ndims large counts.
            if (int_count != 2 or large_count != 3 * rank)
              return false;
            order = ints[1];
            for (std::size_t d = 0; d < rank; ++d) {
              sizes[d] = large[d];
              subsizes[d] = large[rank + d];
              starts[d] = large[2 * rank + d];
              if (sizes[d] < 0 or subsizes[d] < 0 or starts[d] < 0)
                return false;
            }
          } else {
            if (int_count != 3 * rank + 2)
              return false;
            order = ints[1 + 3 * rank];
            for (std::size_t d = 0; d < rank; ++d) {
              sizes[d] = ints[1 + d];
              subsizes[d] = ints[1 + rank + d];
              starts[d] = ints[1 + 2 * rank + d];
              if (sizes[d] < 0 or subsizes[d] < 0 or starts[d] < 0)
                return false;
            }
          }
          // MPI_ORDER_C stores the last dimension contiguously, MPI_ORDER_FORTRAN the first.
          // The traversal order only reorders the type map; the set of selected positions is
          // the same, so one stride assignment per memory order is enough.
          std::vector<MPI_Count> strides(rank);
          if (order == MPI_ORDER_C) {
            MPI_Count acc{1};
            for (std::size_t d = rank; d > 0; --d) {
              strides[d - 1] = acc;
              acc = type_map_detail::checked_product(acc, sizes[d - 1], ctx);
            }
          } else if (order == MPI_ORDER_FORTRAN) {
            MPI_Count acc{1};
            for (std::size_t d = 0; d < rank; ++d) {
              strides[d] = acc;
              acc = type_map_detail::checked_product(acc, sizes[d], ctx);
            }
          } else {
            return false;
          }
          bool empty = false;
          for (std::size_t d = 0; d < rank; ++d)
            empty = empty or subsizes[d] == 0;
          if (not empty) {
            const MPI_Count extent = type_map_detail::element_extent(types[0], ctx);
            std::vector<MPI_Count> position(starts);
            bool done = false;
            while (not done) {
              MPI_Count linear{0};
              for (std::size_t d = 0; d < rank; ++d) {
                linear = type_map_detail::checked_sum(
                    linear, type_map_detail::checked_product(position[d], strides[d], ctx),
                    ctx);
              }
              push_child(types[0], type_map_detail::checked_add(
                                       current.base,
                                       type_map_detail::checked_mul(linear, extent, ctx), ctx));
              std::size_t d = rank;
              while (d > 0) {
                --d;
                position[d] = type_map_detail::checked_sum(position[d], 1, ctx);
                const MPI_Count end = type_map_detail::checked_sum(starts[d], subsizes[d], ctx);
                if (position[d] < end)
                  break;
                position[d] = starts[d];
                if (d == 0)
                  done = true;
              }
            }
          }
          break;
        }
        default:
          return false;
      }
    }
    return true;
  }

  // A receive datatype places the received bytes at the type map's locations. MPI 4.1 §5.1.11
  // makes a receive with a datatype that writes a location more than once erroneous; a proven
  // overlap is diagnosed locally, while `unknown` stays the caller's responsibility (MPI
  // cannot check it either).
  template<class T>
  void require_disjoint_layout(const layout<T>& l, const context& ctx,
                               std::string_view side = {}) {
    require_relative(l, ctx);
    if (l.has_overlapping_entries()) [[unlikely]] {
      if (side.empty()) {
        throw_invalid(ctx.operation,
                      "type-map entries overlap; a location written or read more than once is "
                      "erroneous (MPI 4.1 5.1.11)");
      }
      // Keep one `mp1: <operation>` prefix: the side label belongs to the message, never to
      // the operation field.
      detail::fail_as<std::invalid_argument>(
          ctx, std::source_location::current(),
          "{}: type-map entries overlap; a location written or read more than once is "
          "erroneous (MPI 4.1 5.1.11)",
          side);
    }
  }

  template<class T>
  void require_receive_layout(const layout<T>& l, const context& ctx) {
    require_relative(l, ctx);
    if (l.has_overlapping_entries()) [[unlikely]] {
      throw_invalid(ctx.operation, "receive layout has overlapping type-map entries");
    }
  }

  // Layout-facts helper: the packed byte image (p + true_lb, byte_size) of a layout.  It is
  // used by the layout-fact gates, NOT by the collective wire lowering (which is typed and
  // never multiplies or shifts by size/extent); keeping them separate is what makes the
  // size != extent confusion structurally unavailable.
  template<class T>
  [[nodiscard]] count_t packed_offset(const layout<T>& l, const context& ctx) {
    if (not l.is_valid()) [[unlikely]] {
      throw_invalid(ctx.operation, "invalid layout");
    }
    if (not l.dense()) [[unlikely]] {
      detail::fail_with<std::logic_error>(ctx.operation, std::source_location::current(),
                                          "packed lowering requires a packed layout");
    }
    return l.byte_size() == 0 ? 0 : l.true_byte_lower_bound();
  }

  template<class T>
  [[nodiscard]] const T* packed_ptr(const T* p, const layout<T>& l, const context& ctx) {
    const count_t offset = packed_offset(l, ctx);
    if (l.byte_size() != 0 and p == nullptr) [[unlikely]] {
      throw_invalid(ctx.operation, "null pointer for a non-empty packed layout");
    }
    return offset == 0 ? p : offset_bytes(p, offset);
  }

  template<class T>
  [[nodiscard]] packed_view<T> packed_view_of(const T* p, const layout<T>& l,
                                              const context& ctx) {
    return packed_view<T>{packed_ptr(p, l, ctx), lowering_mpi_count(l.byte_size(), ctx)};
  }

  template<class T>
  [[nodiscard]] count_t repetition_shift(const layout<T>& l, const context& ctx) {
    if (not l.is_valid()) [[unlikely]] {
      throw_invalid(ctx.operation, "invalid layout");
    }
    if (not l.is_base_repetition()) [[unlikely]] {
      detail::fail_with<std::logic_error>(
          ctx.operation, std::source_location::current(),
          "repetition lowering requires the base_repetition fact");
    }
    const datatype_info& base = datatype_info_of<T>();
    return checked_count_sub(l.true_byte_lower_bound(), base.true_lb, ctx);
  }

  template<class T>
  // A zero-byte layout addresses no location, so a null base pointer is legal even when its
  // repetition count size() is positive (for example an empty native type repeated by a
  // vector).  Only a layout that addresses bytes requires a non-null base.
  [[nodiscard]] const T* repetition_ptr(const T* p, const layout<T>& l, const context& ctx) {
    const count_t shift = repetition_shift(l, ctx);
    if (l.byte_size() != 0 and p == nullptr) [[unlikely]] {
      throw_invalid(ctx.operation, "null pointer for a non-empty layout");
    }
    if (p == nullptr or l.size() == 0 or shift == 0)
      return p;
    return offset_bytes(p, shift);
  }

  template<class T>
  [[nodiscard]] T* repetition_ptr(T* p, const layout<T>& l, const context& ctx) {
    const count_t shift = repetition_shift(l, ctx);
    if (l.byte_size() != 0 and p == nullptr) [[unlikely]] {
      throw_invalid(ctx.operation, "null pointer for a non-empty layout");
    }
    if (p == nullptr or l.size() == 0 or shift == 0)
      return p;
    return offset_bytes(p, shift);
  }

  template<class T>
  [[nodiscard]] repetition_view<T> repetition_view_of(const T* p, const layout<T>& l,
                                                      const context& ctx) {
    const T* ptr = repetition_ptr(p, l, ctx);
    return repetition_view<T>{ptr, lowering_mpi_count(l.size(), ctx),
                              datatype_info_of<T>().type};
  }

  // Identity repetition: the layout is the flat array of its size() native datatypes at the
  // caller's pointer, so the reduce_scatter[_block] send side can be passed unchanged.
  template<class T>
  [[nodiscard]] bool is_identity_repetition(const layout<T>& l, const context& ctx) {
    if (not l.is_valid())
      return false;
    const datatype_info& base = datatype_info_of<T>();
    if (not l.is_base_repetition() or l.true_byte_lower_bound() != base.true_lb)
      return false;
    return l.byte_extent() == checked_count_mul(l.size(), base.extent, ctx);
  }

  template<class T>
  [[nodiscard]] identity_view<T> identity_view_of(const T* p, const layout<T>& l,
                                                  const context& ctx) {
    if (not is_identity_repetition(l, ctx)) [[unlikely]] {
      detail::fail_with<std::logic_error>(ctx.operation, std::source_location::current(),
                                          "identity lowering requires an identity repetition");
    }
    if (l.size() != 0 and p == nullptr) [[unlikely]] {
      throw_invalid(ctx.operation, "null pointer for a non-empty layout");
    }
    return identity_view<T>{p, lowering_mpi_count(l.size(), ctx), datatype_info_of<T>().type};
  }

  // Staged pack/unpack always uses the caller's pointer with the layout's own handle. The
  // pointer is not shifted; MPI applies the map's displacements itself.
  template<class T>
  [[nodiscard]] const T* staged_ptr(const T* p, const layout<T>& l, const context& ctx) {
    require_relative(l, ctx);
    if (l.byte_size() != 0 and p == nullptr) [[unlikely]] {
      throw_invalid(ctx.operation, "null pointer for a non-empty layout");
    }
    return p;
  }

  template<class T>
  [[nodiscard]] T* staged_ptr(T* p, const layout<T>& l, const context& ctx) {
    require_relative(l, ctx);
    if (l.byte_size() != 0 and p == nullptr) [[unlikely]] {
      throw_invalid(ctx.operation, "null pointer for a non-empty layout");
    }
    return p;
  }

}  // namespace mp1::detail

#endif
