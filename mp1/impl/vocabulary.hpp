#if !(defined MP1_IMPL_VOCABULARY_HPP)

#define MP1_IMPL_VOCABULARY_HPP

// The 1.0 vocabulary header.
//
// The vocabulary defines the four value types (`capacity`, `region<T>`,
// `blocks`, the `in_place_t` name), the `count_t` alias they are built on, and the checked
// pointer+count -> span conversion.  `in_place_t` itself stays defined in collective.hpp (the
// 16-header baseline counts it there and its first consumers are the collective overloads);
// this header declares it so detail/concepts.hpp can name the sentinel before the collective
// surface is included.
//
// The header is deliberately MPI-free and light: it is included by fwd.hpp and
// detail/concepts.hpp, so it must not pull a heavy standard header into the base of the
// include graph.  `<format>` is not needed here (span_from composes its diagnostic with
// std::string) and `<ranges>` lives in detail/concepts.hpp, which owns the buffer_element
// aliases.
//
// detail/concepts.hpp adds the datatype trait through datatype.hpp.  `region` names `layout`
// only through the forward declaration; its constructor and accessors can be defined against
// an incomplete type, and the plan hook that *copies* the layout is a template instantiated
// where layout.hpp is complete.
//
// The transitional aliases are gone: `block_count`, `counts`, `displacements`,
// `layouts<T>`, `layout_view<T>` and `make_layout_span` are not part of the API.  `blocks`
// is the dense descriptor and `std::span<const layout<T>>` + `std::span<const count_t>` the
// typed one.

#include <cstddef>
#include <source_location>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>


namespace mp1 {

  // Element count used by every public API.  Kept in the vocabulary header (moved out of
  // fwd.hpp) because capacity, blocks and region are defined in terms of it.
  using count_t = std::ptrdiff_t;

  /// Explicit empty message: a zero-count span of bytes.
  ///
  /// Sending it sends a message whose type signature contributes no elements (MPI 4.1 §3.2.4),
  /// so any zero-count receive matches, and receiving it discards a zero-count message.  It is
  /// an ordinary zero-length buffer, not a special API path: no function inspects it, no
  /// collective treats it specially, and no MPI call is required to construct it.  Spelled
  /// `mp1::empty_data` because `empty` is a common member name and this is message data, not a
  /// layout or a container.
  inline constexpr std::span<std::byte> empty_data{};

  // Upper bound of a resizable-container `recv`/`irecv`.  It is a distinct type so the third
  // argument can never be silently read as a source rank: `recv(c, v, 1)` has exactly one
  // meaning (auto-size from source 1, default tag 0), while the capacity is spelled
  // `recv(c, v, capacity{1}, 0)`.  The value is an upper bound, not an exact count: the
  // container is resized to the received count and a longer message is a truncation error
  // (MPI 4.1 §3.2.4).
  struct capacity {
    explicit constexpr capacity(count_t value) noexcept : value_{value} {
    }

    [[nodiscard]] constexpr count_t value() const noexcept {
      return value_;
    }

  private:
    count_t value_;
  };

  // Defined in collective.hpp, which counts it; declared here so concepts.hpp can name the
  // sentinel without including the collective surface.
  struct in_place_t;

  template<class T>
  class layout;

  /// One occurrence of a committed layout over T at data.
  ///
  /// Non-owning: two words, the data pointer and a reference to the layout.  It is a class,
  /// not an aggregate, because the deleted rvalue-layout constructor is the only way to make
  /// `region{p, layout<T>{...}}` a compile error.  Copy/move construction is fine (copying a
  /// borrow); assignment is deleted (reference member); there is no default constructor.
  ///
  /// The lifetime guard is local, not a proof: the deleted constructor rejects a temporary
  /// layout in the same full-expression, but a `region` returned from a function whose local
  /// layout has died still compiles.  The "layout must outlive a blocking call" rule remains
  /// the caller's obligation; nonblocking/persistent initiation duplicates the layout so the
  /// caller's layout may die after the call returns.
  template<class T>
  class region {
  public:
    region(T* data, const layout<std::remove_const_t<T>>& map) noexcept
        : data_{data}, map_{map} {
    }
    region(T*, const layout<std::remove_const_t<T>>&&) = delete;  // no dangling temporary

    region(const region&) = default;  // two words; copying a borrow is fine
    region(region&&) = default;
    region& operator=(const region&) = delete;  // reference member
    region& operator=(region&&) = delete;

    [[nodiscard]] T* data() const noexcept {
      return data_;
    }
    [[nodiscard]] const layout<std::remove_const_t<T>>& map() const noexcept {
      return map_;
    }

  private:
    T* data_;
    const layout<std::remove_const_t<T>>& map_;
  };

  // CTAD deduces T from the pointer argument.  `region{nullptr, l}` therefore does NOT
  // compile (T cannot be deduced from nullptr); use `region<int>{nullptr, l}` when a null
  // data pointer is intended.  A rvalue layout is rejected by the deleted constructor when T
  // is explicit or deducible.
  template<class T>
  region(T*, const layout<std::remove_const_t<T>>&) -> region<T>;

  template<class T>
  inline constexpr bool is_region_v = false;
  template<class T>
  inline constexpr bool is_region_v<region<T>> = true;

  template<class R>
  struct region_element {
    using type = void;
  };
  template<class T>
  struct region_element<region<T>> {
    using type = T;
  };
  template<class R>
  using region_element_t = typename region_element<R>::type;

  // buffer_element/buffer_element_t live in detail/concepts.hpp: the range_value_t mapping
  // needs <ranges>, which this foundational header must not include.

  /// One (counts, displacements) pair for the four variable families.
  ///
  /// Non-owning: two `std::span<const count_t>`.  It is an aggregate so designated
  /// initializers name the two MPI arrays; C++20 requires the designators in declaration
  /// order and the two members have the same type, so a swapped *value* pair still compiles
  /// (the wrapper's `counts.size() == displacements.size()` check is the runtime guard).  An
  /// empty `blocks{}` is the zero-block descriptor.  A temporary vector bound by the spans is
  /// safe within a blocking full-expression; nonblocking/persistent initiation copies both
  /// arrays, and a *stored* blocks object must outlive the arrays its spans refer to.
  struct blocks {
    std::span<const count_t> counts;
    std::span<const count_t> displacements;
  };

  /// The checked pointer + count -> span conversion.  The current pointer+count forms
  /// take a signed count_t and reject a negative count before MPI; std::span's
  /// (pointer, size_type) constructor silently converts a negative count_t to a huge
  /// size_t, so this helper is the supported spelling for a raw buffer.  A negative count
  /// throws std::invalid_argument.  A null pointer with a positive count also throws
  /// std::invalid_argument: std::span's (pointer, size) constructor requires a non-null
  /// pointer when size is nonzero (a non-empty span over null violates its precondition,
  /// and MPI would receive a null buffer with a positive count).  (nullptr, 0) is an empty
  /// span, and a null pointer with a zero count stays null.
  template<class T>
  [[nodiscard]] inline std::span<T> span_from(T* data, count_t count) {
    const std::source_location where = std::source_location::current();
    if (count < 0) {
      // vocabulary.hpp is the MPI-free base of the include graph, so it cannot include
      // error.hpp (whose MPI boundary pulls in <mpi.h>) and deliberately does not include
      // <format> either: the same check-site message is composed from std::string and
      // std::to_string, keeping the foundational header light.
      throw std::invalid_argument{"mp1: mp1::span_from (" + std::string{where.file_name()} +
                                  ":" + std::to_string(where.line()) + "): count " +
                                  std::to_string(static_cast<long long>(count)) +
                                  " is negative"};
    }
    if (data == nullptr and count > 0) {
      throw std::invalid_argument{"mp1: mp1::span_from (" + std::string{where.file_name()} +
                                  ":" + std::to_string(where.line()) +
                                  "): null pointer with positive count " +
                                  std::to_string(static_cast<long long>(count))};
    }
    return std::span<T>{data, static_cast<std::size_t>(count)};
  }

  namespace detail {

    // Plan hook (the request copies the layout).
    //
    // A blocking call borrows a `region` directly.  A nonblocking/persistent initiation
    // captures the data pointer (a value, never owned) and duplicates the layout through
    // layout's copy constructor (MPI_Type_dup), so the caller's `region` object and layout
    // may die when the call returns while the data buffer may not.  The RMA request payload
    // (`detail/rma_engine.hpp`) uses this hook; a blocking call never does.
    template<class T>
    struct region_capture {
      T* data;
      layout<std::remove_const_t<T>> map;  // owning duplicate
    };

    template<class T>
    [[nodiscard]] region_capture<T> capture_region(region<T> buffer) {
      return region_capture<T>{buffer.data(), buffer.map()};
    }

  }  // namespace detail

}  // namespace mp1

#endif
