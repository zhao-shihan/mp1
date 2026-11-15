#if !(defined MP1_DETAIL_BUFFER_HPP)

#define MP1_DETAIL_BUFFER_HPP

#include "mp1/impl/detail/concepts.hpp"
#include "mp1/impl/detail/mpi_call.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/status.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>


namespace mp1::detail {

  // ---------------------------------------------------------------------------------------
  // The one buffer descriptor used by the p2p engine. A datatype is always borrowed.
  // ---------------------------------------------------------------------------------------

  struct buffer_view {
    const void* data;
    MPI_Count count;
    MPI_Datatype type;
  };

  struct mutable_buffer_view {
    void* data;
    MPI_Count count;
    MPI_Datatype type;
  };

  // ---------------------------------------------------------------------------------------
  // Container classification for the p2p container forms.
  //
  // C++20 ranges classify the containers (the eleven container_traits specializations are
  // gone):
  //
  //   * an owning contiguous range is a contiguous container (std::vector, std::array,
  //     std::basic_string); std::span/string_view are borrowed, a built-in array is not a
  //     class, and vector<bool> is not contiguous;
  //   * a forward, non-contiguous range with lvalue element references is a sequence container
  //     (std::list, std::forward_list, std::deque and the flat associative containers
  //     set/multiset/unordered_set/unordered_multiset).  The map family has a keyed pair
  //     element and is not a flat sequence.
  //
  // The public p2p overloads spell their constraints through the detail/concepts.hpp
  // predicates (element_buffer, element_sequence).  The names below are the classification
  // the zero-copy/staging send policy and the sequence-container receive path still use.
  // ---------------------------------------------------------------------------------------

  template<class R>
  concept contiguous_container =
      std::ranges::contiguous_range<R> && std::ranges::sized_range<R> && std::is_class_v<R> &&
      !std::ranges::borrowed_range<R> && requires(R& r) { r.max_size(); };

  template<class R>
  concept resizable_contiguous_container =
      contiguous_container<R> && requires(R& r, std::size_t n) { r.resize(n); };

  template<class R>
  concept flat_sequence_container =
      std::ranges::forward_range<R> && !std::ranges::contiguous_range<R> &&
      std::is_lvalue_reference_v<std::ranges::range_reference_t<R>> &&
      (!associative_like<R> || requires {
        requires std::same_as<std::ranges::range_value_t<R>, typename R::key_type>;
      });

  template<class T>
  inline constexpr bool is_contiguous_container_v =
      contiguous_container<std::remove_cv_t<std::remove_reference_t<T>>>;

  template<class T>
  inline constexpr bool is_sequence_container_v =
      flat_sequence_container<std::remove_cv_t<std::remove_reference_t<T>>>;

  template<class T>
  inline constexpr bool is_contiguous_resizable_v =
      resizable_contiguous_container<std::remove_cv_t<std::remove_reference_t<T>>>;

  template<class T>
  using container_value_t =
      std::ranges::range_value_t<std::remove_cv_t<std::remove_reference_t<T>>>;

  template<class T>
  inline constexpr bool is_associative_sequence =
      is_sequence_container_v<T> and
      associative_like<std::remove_cv_t<std::remove_reference_t<T>>>;

  // A scalar is a single value with a datatype mapping.  It is deliberately NOT restricted to
  // trivially copyable types: std::pair/std::tuple have datatype_traits specializations but
  // are not trivially copyable in libstdc++.  MPI moves the object according to its registered
  // datatype, so the C++ copy traits are not the relevant contract.
  template<class T>
  inline constexpr bool is_scalar_data_v = element_value<T>;

  /// The datatype of one container element.  The contiguous-container forms describe the
  /// caller's array with this handle; they never copy or reinterpret the elements, so C++
  /// trivial copyability is not the contract.  A type without a datatype mapping is rejected
  /// here (once, at the container-form boundary) with the same actionable wording as
  /// native_datatype<T>(); only a predefined or reflected mapping is accepted.
  template<class T>
  [[nodiscard]] inline MPI_Datatype container_element_datatype() {
    static_assert(has_datatype_v<T>,
                  "mp1: p2p container element type has no MPI datatype mapping: no "
                  "datatype_traits<T> specialization and no MP1_REFLECTION for the element "
                  "type (MP1_DATATYPE_TRAITS provides a predefined mapping)");
    if constexpr (has_datatype_v<T>) {
      return datatype_info_of<T>().type;
    } else {
      return MPI_DATATYPE_NULL;
    }
  }

  // The container counts are signed element counts: std::ssize() is the length without the
  // size_t -> count_t cast, and the signed check is part of the conversion.
  template<class R>
  [[nodiscard]] inline MPI_Count container_mpi_count(
      const R& data, const context& ctx,
      std::source_location where = std::source_location::current()) {
    return checked_mpi_count(std::ssize(data), ctx, where);
  }

  // ---------------------------------------------------------------------------------------
  // Small checked conversions and pointer helpers.
  // ---------------------------------------------------------------------------------------

  [[nodiscard]] inline MPI_Count buffer_to_mpi_count(
      std::size_t size, const context& ctx,
      std::source_location where = std::source_location::current()) {
    if (not std::in_range<MPI_Count>(size))
      detail::fail_as<std::overflow_error>(ctx, where, "element count does not fit MPI_Count");
    return static_cast<MPI_Count>(size);
  }

  [[nodiscard]] inline MPI_Count buffer_iterator_count(
      std::ptrdiff_t distance, const context& ctx,
      std::source_location where = std::source_location::current()) {
    if (not std::in_range<std::size_t>(distance))
      detail::fail_as<std::invalid_argument>(ctx, where, "negative iterator range");
    return buffer_to_mpi_count(static_cast<std::size_t>(distance), ctx, where);
  }

  /// The two data positions of a collective.  The dummy substitute is per slot so an
  /// ignored send and an ignored receive (or two empty layouts) never alias: MPICH's
  /// fixed-collective parameter checks reject aliased buffers even when the count is zero.
  enum class buffer_slot : unsigned char { send, recv };

  /// A stable, suitably aligned non-null address for MPI_PROC_NULL paths where MPI ignores
  /// the buffer but some implementations still reject a null pointer. It is never
  /// dereferenced.  Keep the zero-argument form (slot send) for consumers that do not
  /// distinguish the positions.
  alignas(std::max_align_t) inline unsigned char dummy_buffer_storage[2][sizeof(
      std::max_align_t)]{};

  [[nodiscard]] inline void* dummy_buffer(buffer_slot slot) noexcept {
    return dummy_buffer_storage[static_cast<unsigned>(slot)];
  }

  [[nodiscard]] inline void* dummy_buffer() noexcept {
    return dummy_buffer(buffer_slot::send);
  }

  template<class T>
  [[nodiscard]] const T* offset_bytes(const T* p, count_t bytes) noexcept {
    if (p == nullptr)
      return nullptr;
    return reinterpret_cast<const T*>(reinterpret_cast<const char*>(p) + bytes);
  }

  template<class T>
  [[nodiscard]] T* offset_bytes(T* p, count_t bytes) noexcept {
    if (p == nullptr)
      return nullptr;
    return reinterpret_cast<T*>(reinterpret_cast<char*>(p) + bytes);
  }

  // ---------------------------------------------------------------------------------------
  // Scalar and pointer buffers.
  // ---------------------------------------------------------------------------------------

  template<class T>
  requires element_value<T>
  [[nodiscard]] inline buffer_view make_send_buffer(const T& data) {
    return {std::addressof(data), 1, datatype_info_of<T>().type};
  }

  template<class T>
  requires element_value<T>
  [[nodiscard]] inline mutable_buffer_view make_recv_buffer(T& data) {
    return {std::addressof(data), 1, datatype_info_of<T>().type};
  }

  /// pointer + element count. `ignored` is true for MPI_PROC_NULL, where MPI ignores the
  /// buffer and mp1 must not require a non-null pointer. A null pointer with a zero count is
  /// an empty position and is normalized to the per-slot dummy: MPI never dereferences it,
  /// and some providers reject a null buffer even for zero elements.
  template<class T>
  [[nodiscard]] inline buffer_view make_send_buffer(const T* data, count_t count,
                                                    bool ignored = false) {
    const MPI_Count mpi_count = checked_mpi_count(count, "mp1::send buffer");
    if (data == nullptr) {
      if (count != 0 and not ignored) {
        throw_invalid("mp1::send buffer", "null data pointer with a positive count");
      }
      data = static_cast<const T*>(dummy_buffer(buffer_slot::send));
    }
    return {data, mpi_count, datatype_info_of<T>().type};
  }

  template<class T>
  [[nodiscard]] inline mutable_buffer_view make_recv_buffer(T* data, count_t capacity,
                                                            bool ignored = false) {
    const MPI_Count mpi_count = checked_mpi_count(capacity, "mp1::recv buffer");
    if (data == nullptr) {
      if (capacity != 0 and not ignored) {
        throw_invalid("mp1::recv buffer", "null data pointer with a positive capacity");
      }
      data = static_cast<T*>(dummy_buffer(buffer_slot::recv));
    }
    return {data, mpi_count, datatype_info_of<T>().type};
  }

  /// pointer + layout (count 1, the layout's datatype). Null is legal iff byte_size() == 0.
  template<class T>
  [[nodiscard]] inline buffer_view make_send_buffer(const T* data, const layout<T>& l,
                                                    bool ignored = false) {
    if (not l.is_valid())
      throw_invalid("mp1::send buffer", "invalid layout");
    if (data == nullptr and l.byte_size() != 0) {
      if (not ignored) {
        throw_invalid("mp1::send buffer", "null data pointer with a non-empty layout");
      }
      data = static_cast<const T*>(dummy_buffer());
    }
    return {data, 1, l.native_handle()};
  }

  template<class T>
  [[nodiscard]] inline mutable_buffer_view make_recv_buffer(T* data, const layout<T>& l,
                                                            bool ignored = false) {
    if (not l.is_valid())
      throw_invalid("mp1::recv buffer", "invalid layout");
    if (data == nullptr and l.byte_size() != 0) {
      if (not ignored) {
        throw_invalid("mp1::recv buffer", "null data pointer with a non-empty layout");
      }
      data = static_cast<T*>(dummy_buffer());
    }
    return {data, 1, l.native_handle()};
  }

  // ---------------------------------------------------------------------------------------
  // Contiguous containers (vector/array/string): data() + size, element datatype.
  //
  // The transfer uses the element's MPI datatype; mp1 never copies or reinterprets the
  // elements itself, so the only element contract is a datatype mapping (has_datatype_v), not
  // C++ trivial copyability.  A reflected std::pair/std::tuple (not trivially copyable in
  // libstdc++ C++17) is therefore accepted exactly like the scalar and iterator forms.
  // ---------------------------------------------------------------------------------------

  template<class C>
  requires element_buffer<C>
  [[nodiscard]] inline buffer_view make_send_buffer(const C& data) {
    using value_type = std::ranges::range_value_t<C>;
    return {std::ranges::data(data), container_mpi_count(data, "mp1::send buffer"),
            container_element_datatype<value_type>()};
  }

  template<class C>
  requires element_buffer<C>
  [[nodiscard]] inline mutable_buffer_view make_recv_buffer(C& data) {
    using value_type = std::ranges::range_value_t<C>;
    return {std::ranges::data(data), container_mpi_count(data, "mp1::recv buffer"),
            container_element_datatype<value_type>()};
  }

  // Diagnostic: element_buffer/resizable_element_buffer already exclude a container whose
  // element has no MPI datatype mapping, so without these candidates the call fails as "no
  // matching function" and the actionable element type is lost. Each overload is selected
  // only for that excluded case and repeats the earlier static_assert wording.
  template<class C>
  requires contiguous_container<C> && (!element<std::ranges::range_value_t<C>>)
  [[nodiscard]] inline buffer_view make_send_buffer(const C&) {
    static_assert(element<std::ranges::range_value_t<C>>,
                  "mp1: p2p container element type has no MPI datatype mapping: no "
                  "datatype_traits<T> specialization and no MP1_REFLECTION for the element "
                  "type (MP1_DATATYPE_TRAITS provides a predefined mapping)");
    return {};
  }

  template<class C>
  requires contiguous_container<C> && (!element<std::ranges::range_value_t<C>>)
  [[nodiscard]] inline mutable_buffer_view make_recv_buffer(C&) {
    static_assert(element<std::ranges::range_value_t<C>>,
                  "mp1: p2p container element type has no MPI datatype mapping: no "
                  "datatype_traits<T> specialization and no MP1_REFLECTION for the element "
                  "type (MP1_DATATYPE_TRAITS provides a predefined mapping)");
    return {};
  }

  template<class C>
  requires resizable_contiguous_container<C> && (!element<std::ranges::range_value_t<C>>)
  [[nodiscard]] inline mutable_buffer_view make_recv_buffer(C&, count_t) {
    static_assert(element<std::ranges::range_value_t<C>>,
                  "mp1: p2p container element type has no MPI datatype mapping: no "
                  "datatype_traits<T> specialization and no MP1_REFLECTION for the element "
                  "type (MP1_DATATYPE_TRAITS provides a predefined mapping)");
    return {};
  }

  /// Resize a contiguous resizable container to the receive capacity, then describe it.
  template<class C>
  requires resizable_element_buffer<C>
  [[nodiscard]] inline mutable_buffer_view make_recv_buffer(C& data, count_t capacity) {
    using value_type = std::ranges::range_value_t<C>;
    static_assert(std::is_default_constructible_v<value_type>,
                  "mp1: capacity receive requires a default-constructible container element "
                  "(the container is resized before MPI writes into it)");
    if (capacity < 0)
      throw_invalid("mp1::recv buffer", "negative container capacity");
    const std::size_t size = static_cast<std::size_t>(capacity);
    if (size > data.max_size())
      throw_invalid("mp1::recv buffer", "container capacity exceeds max_size");
    data.resize(size);
    return {data.data(), checked_mpi_count(capacity, "mp1::recv buffer"),
            container_element_datatype<value_type>()};
  }

  // ---------------------------------------------------------------------------------------
  // Iterator ranges (send only; iterator-pair receives are not in 1.0).
  //
  // A statically contiguous range (raw pointer, std::vector/array/string iterator) is
  // described by pointer + count with no copy and no datatype constructor.  A non-contiguous
  // forward range of a trivially copyable value type is staged into one owned contiguous
  // buffer (one native call, one local copy, no per-element MPI call).
  //
  // A non-contiguous range of any other value type is described without copying a value:
  // MPI 4.1 §5.1.12 requires buf = MPI_BOTTOM, count = 1 and a datatype whose displacements
  // are the absolute addresses of the elements, so an internal hindexed map over
  // native_datatype<T>() is built and passed as {MPI_BOTTOM, 1, map}.  The wire stays typed.
  // That path borrows the caller's elements; there is no owned copy, so they must stay alive
  // and unmodified until the send completes.
  // ---------------------------------------------------------------------------------------

  /// Absolute-address MPI datatype over one occurrence of each element of a non-contiguous
  /// range, for the MPI 4.1 §5.1.12 MPI_BOTTOM form.
  ///
  /// Construction walks the range once with MPI_Get_address, merges mechanically contiguous
  /// elements into hindexed blocks, and commits one hindexed datatype over the native element
  /// datatype.  The displacement of each block is the absolute address of its first element;
  /// the map is only ever used with buf = MPI_BOTTOM and count = 1.  Contiguous-run detection
  /// uses MPI_Aint_add (MPI 4.1 §5.1.5, never intrinsic address arithmetic) and is only an
  /// optimization: an address that the procedure cannot express simply ends the run, and the
  /// exact MPI_Get_address values are authoritative.
  ///
  /// The element datatype extent must equal sizeof(T): the C++ elements are spaced by
  /// sizeof(T), and a datatype whose map repeats at a different extent would address bytes
  /// outside the object.  The map is movable, never copyable, and frees its handle through the
  /// owning-handle policy (no MPI call after finalize).
  template<class T>
  class absolute_range_map {
  public:
    absolute_range_map() noexcept = default;

    template<std::forward_iterator It>
    absolute_range_map(It first, It last, const context& ctx) {
      const detail::datatype_info& base = detail::datatype_info_of<T>();
      element_type_ = base.type;
      if (first == last or base.size == 0)
        return;
      if (base.extent != static_cast<count_t>(sizeof(T))) {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "the element datatype extent ({}) must equal "
                                               "sizeof(T) ({}) for the absolute-address "
                                               "sequence map",
                                               base.extent, static_cast<count_t>(sizeof(T)));
      }

      std::vector<int> blocklengths;
      std::vector<MPI_Aint> displacements;
      count_t run_length{0};
      MPI_Aint run_start{0};
      MPI_Aint expected{0};
      bool have_run = false;
      for (It current = first; current != last; ++current) {
        MPI_Aint address{0};
        detail::check_mpi(MPI_Get_address(std::addressof(*current), &address), ctx.operation);
        if (not have_run) {
          run_start = address;
          run_length = 1;
          have_run = true;
        } else if (address == expected) {
          if (run_length == static_cast<count_t>(std::numeric_limits<int>::max())) {
            detail::fail_as<std::invalid_argument>(
                ctx, std::source_location::current(),
                "more than INT_MAX contiguous elements in one block");
          }
          ++run_length;
        } else {
          append_run(blocklengths, displacements, run_start, run_length, ctx);
          run_start = address;
          run_length = 1;
        }
        // The next element of this run starts one element extent after its first element.
        // This is the sanctioned address arithmetic (MPI 4.1 §5.1.5); if the address space
        // cannot represent the sum, the comparison below simply fails and the run ends, so
        // the map stays correct either way.
        expected = detail::checked_address_add(
            run_start, detail::checked_count_mul(run_length, base.extent, ctx), ctx);
      }
      if (have_run)
        append_run(blocklengths, displacements, run_start, run_length, ctx);

      MPI_Datatype type = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_create_hindexed(static_cast<int>(blocklengths.size()), blocklengths.data(),
                                   displacements.data(), base.type, &type),
          ctx.operation);
      try {
        detail::check_mpi(MPI_Type_commit(&type), ctx.operation);
      } catch (...) {
        if (type != MPI_DATATYPE_NULL)
          (void)MPI_Type_free(&type);
        throw;
      }
      type_ = type;
    }

    absolute_range_map(const absolute_range_map&) = delete;
    absolute_range_map& operator=(const absolute_range_map&) = delete;

    absolute_range_map(absolute_range_map&& other) noexcept
        : type_{std::exchange(other.type_, MPI_DATATYPE_NULL)},
          element_type_{other.element_type_} {
    }

    absolute_range_map& operator=(absolute_range_map&& other) noexcept {
      if (this != std::addressof(other)) {
        release();
        type_ = std::exchange(other.type_, MPI_DATATYPE_NULL);
        element_type_ = other.element_type_;
      }
      return *this;
    }

    ~absolute_range_map() {
      release();
    }

    /// The §5.1.12 buffer form: MPI_BOTTOM, count 1, absolute-displacement datatype.  For an
    /// empty range (or a zero-size element datatype) the view is a count-0 dummy buffer.
    [[nodiscard]] buffer_view view() const noexcept {
      if (type_ == MPI_DATATYPE_NULL)
        return {detail::dummy_buffer(buffer_slot::send), 0, element_type_};
      return {MPI_BOTTOM, 1, type_};
    }

  private:
    static void append_run(std::vector<int>& blocklengths, std::vector<MPI_Aint>& displacements,
                           MPI_Aint start, count_t length, const context& ctx) {
      if (blocklengths.size() >= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "more than INT_MAX blocks in the absolute map");
      }
      blocklengths.push_back(static_cast<int>(length));
      displacements.push_back(start);
    }

    void release() noexcept {
      detail::release_handle(type_, MPI_DATATYPE_NULL, true, &detail::free_datatype);
    }

    MPI_Datatype type_{MPI_DATATYPE_NULL};
    MPI_Datatype element_type_{MPI_DATATYPE_NULL};
  };

  template<class It>
  class iterator_send_buffer {
  public:
    using value_type = std::iter_value_t<It>;
    static_assert(is_scalar_data_v<value_type>,
                  "mp1: p2p iterator value_type must be a scalar data type");
    iterator_send_buffer(It first, It last) {
      const std::ptrdiff_t distance = std::distance(first, last);
      count_ = buffer_iterator_count(distance, "mp1::send iterator range");
      if (distance == 0) {
        view_ = buffer_view{detail::dummy_buffer(buffer_slot::send), 0,
                            datatype_info_of<value_type>().type};
        return;
      }
      if constexpr (std::contiguous_iterator<It>) {
        static_assert(std::is_lvalue_reference_v<std::iter_reference_t<It>>,
                      "mp1: a contiguous iterator must dereference to an lvalue");
        // Zero-copy: the caller's range is one contiguous T array; MPI may read it until the
        // send completes (MPI 4.1 §3.7.2 send-buffer rule).
        view_ =
            buffer_view{std::addressof(*first), count_, datatype_info_of<value_type>().type};
      } else if constexpr (std::is_trivially_copyable_v<value_type>) {
        // One local staging copy; the buffer is owned by this object, which the nonblocking
        // payload keeps alive until completion (MPI 4.1 §3.7.2).
        const std::size_t n = static_cast<std::size_t>(distance);
        stage_.reset(new value_type[n]);
        std::copy(first, last, stage_.get());
        view_ = buffer_view{stage_.get(), count_, datatype_info_of<value_type>().type};
      } else {
        static_assert(std::is_lvalue_reference_v<std::iter_reference_t<It>>,
                      "mp1: the absolute-address iterator form requires an lvalue element");
        // No value semantics are invented and no byte image is built: the typed absolute
        // addresses describe the caller's objects. The nonblocking payload keeps this buffer
        // alive until completion; a blocking call owns it on the stack.
        map_.emplace(first, last, context{"mp1::send buffer"});
        view_ = map_->view();
      }
    }

    iterator_send_buffer(const iterator_send_buffer&) = delete;
    iterator_send_buffer& operator=(const iterator_send_buffer&) = delete;
    iterator_send_buffer(iterator_send_buffer&&) noexcept = default;
    iterator_send_buffer& operator=(iterator_send_buffer&&) noexcept = default;
    ~iterator_send_buffer() noexcept = default;

    [[nodiscard]] buffer_view view() const noexcept {
      return view_;
    }

  private:
    std::optional<absolute_range_map<value_type>> map_;
    std::unique_ptr<value_type[]> stage_;
    MPI_Count count_{0};
    buffer_view view_{nullptr, 0, MPI_DATATYPE_NULL};
  };

  // ---------------------------------------------------------------------------------------
  // Sequence-container receive assignment.
  // ---------------------------------------------------------------------------------------

  /// Assign a received flat buffer back into a sequence container. Associative containers
  /// have no assign(first,last) with a hint; clear+insert is the correct form.
  template<class C>
  inline void assign_sequence(C& data, const std::vector<container_value_t<C>>& values) {
    if constexpr (is_associative_sequence<C>) {
      data.clear();
      data.insert(values.begin(), values.end());
    } else {
      data.assign(values.begin(), values.end());
    }
  }

}  // namespace mp1::detail

#endif
