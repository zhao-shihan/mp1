#if !(defined MP1_IMPL_OPERATOR_HPP)

#define MP1_IMPL_OPERATOR_HPP

#include "mp1/impl/datatype.hpp"
#include "mp1/impl/detail/mpi_call.hpp"

#include <algorithm>
#include <array>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace mp1 {

  namespace detail {

    // The reduction family of a built-in functor.  Each functor carries the kind as a
    // one-line static constexpr member, which replaces the reduction_functor_family partial
    // specializations; the std functors that cannot carry a member are mapped by the one
    // consteval predicate below.
    enum class reduction_kind : unsigned char {
      none,
      sum,
      product,
      maximum,
      minimum,
      bit_and,
      bit_or,
      bit_xor,
      logical_and,
      logical_or,
      logical_xor
    };

    // One consteval mapping from a functor class template to its reduction family.  An
    // explicit member wins (the mp1 functors and user functors that opt in); the standard
    // functors are recognized structurally because they cannot be modified.
    template<template<class> class Family, class T>
    [[nodiscard]] consteval reduction_kind reduction_kind_of() noexcept {
      if constexpr (requires { Family<T>::kind; }) {
        return Family<T>::kind;
      } else if constexpr (std::same_as<Family<T>, std::plus<T>>) {
        return reduction_kind::sum;
      } else if constexpr (std::same_as<Family<T>, std::multiplies<T>>) {
        return reduction_kind::product;
      } else if constexpr (std::same_as<Family<T>, std::bit_and<T>>) {
        return reduction_kind::bit_and;
      } else if constexpr (std::same_as<Family<T>, std::bit_or<T>>) {
        return reduction_kind::bit_or;
      } else if constexpr (std::same_as<Family<T>, std::bit_xor<T>>) {
        return reduction_kind::bit_xor;
      } else if constexpr (std::same_as<Family<T>, std::logical_and<T>>) {
        return reduction_kind::logical_and;
      } else if constexpr (std::same_as<Family<T>, std::logical_or<T>>) {
        return reduction_kind::logical_or;
      } else {
        return reduction_kind::none;
      }
    }

    // The predefined-operation kinds. The first eleven enumerators stay aligned with
    // reduction_kind because the collective-functor and raw-RMA-handle classifications share
    // one metadata table; the four RMA-only kinds (REPLACE/NO_OP/MAXLOC/MINLOC) follow.
    enum class predefined_op_kind : unsigned char {
      none = 0,
      sum,
      product,
      maximum,
      minimum,
      bit_and,
      bit_or,
      bit_xor,
      logical_and,
      logical_or,
      logical_xor,
      replace,
      no_op,
      maxloc,
      minloc
    };

    static_assert(static_cast<unsigned char>(reduction_kind::none) ==
                          static_cast<unsigned char>(predefined_op_kind::none) and
                      static_cast<unsigned char>(reduction_kind::sum) ==
                          static_cast<unsigned char>(predefined_op_kind::sum) and
                      static_cast<unsigned char>(reduction_kind::product) ==
                          static_cast<unsigned char>(predefined_op_kind::product) and
                      static_cast<unsigned char>(reduction_kind::maximum) ==
                          static_cast<unsigned char>(predefined_op_kind::maximum) and
                      static_cast<unsigned char>(reduction_kind::minimum) ==
                          static_cast<unsigned char>(predefined_op_kind::minimum) and
                      static_cast<unsigned char>(reduction_kind::bit_and) ==
                          static_cast<unsigned char>(predefined_op_kind::bit_and) and
                      static_cast<unsigned char>(reduction_kind::bit_or) ==
                          static_cast<unsigned char>(predefined_op_kind::bit_or) and
                      static_cast<unsigned char>(reduction_kind::bit_xor) ==
                          static_cast<unsigned char>(predefined_op_kind::bit_xor) and
                      static_cast<unsigned char>(reduction_kind::logical_and) ==
                          static_cast<unsigned char>(predefined_op_kind::logical_and) and
                      static_cast<unsigned char>(reduction_kind::logical_or) ==
                          static_cast<unsigned char>(predefined_op_kind::logical_or) and
                      static_cast<unsigned char>(reduction_kind::logical_xor) ==
                          static_cast<unsigned char>(predefined_op_kind::logical_xor),
                  "reduction_kind and predefined_op_kind must share the same first eleven "
                  "enumerator values: one metadata table is indexed by both");

    // One consteval row per predefined operation. It replaces the four parallel chains that
    // spelled the same fifteen-row MPI 4.1 §6.9.2 table (name, collective-functor gate,
    // RMA-functor gate, raw-RMA-handle gate); the native handle mapping stays a function
    // because Open MPI's MPI_Op handles are not constant expressions before C++26.
    struct predefined_op_descriptor {
      const char* name;
      std::uint16_t raw_rma_categories;
      std::uint16_t rma_functor_categories;
      std::uint16_t collective_categories;
    };

    template<rma_datatype_category... Categories>
    [[nodiscard]] consteval std::uint16_t category_mask() noexcept {
      return ((std::uint16_t{1} << static_cast<unsigned>(Categories)) | ... | std::uint16_t{0});
    }

    inline constexpr std::uint16_t kCInteger =
        category_mask<rma_datatype_category::c_integer>();
    inline constexpr std::uint16_t kFortranInteger =
        category_mask<rma_datatype_category::fortran_integer>();
    inline constexpr std::uint16_t kFloating = category_mask<rma_datatype_category::floating>();
    inline constexpr std::uint16_t kLogical = category_mask<rma_datatype_category::logical>();
    inline constexpr std::uint16_t kComplex = category_mask<rma_datatype_category::complex>();
    inline constexpr std::uint16_t kByte = category_mask<rma_datatype_category::byte>();
    inline constexpr std::uint16_t kCharacter =
        category_mask<rma_datatype_category::character>();
    inline constexpr std::uint16_t kMultiLanguage =
        category_mask<rma_datatype_category::multi_language>();

    [[nodiscard]] consteval std::array<predefined_op_descriptor, 15>
    make_predefined_op_table() noexcept {
      return {{
          // none: never mapped; the diagnostic spelling matches the old fallback.
          {"an unknown/user-defined MPI_Op", 0, 0, 0},
          // SUM/PROD: §6.9.2 C integer, Fortran integer, floating, complex, multi-language.
          // Collectives map only C integer and floating (no complex; the callback stays).
          {"MPI_SUM", kCInteger | kFortranInteger | kFloating | kComplex | kMultiLanguage,
           kCInteger | kFloating | kComplex, kCInteger | kFloating},
          {"MPI_PROD", kCInteger | kFortranInteger | kFloating | kComplex | kMultiLanguage,
           kCInteger | kFloating | kComplex, kCInteger | kFloating},
          // MAX/MIN: no complex; collectives keep the callback for floating-point NaN
          // behaviour, so only C integers map there.
          {"MPI_MAX", kCInteger | kFortranInteger | kFloating | kMultiLanguage,
           kCInteger | kFloating, kCInteger},
          {"MPI_MIN", kCInteger | kFortranInteger | kFloating | kMultiLanguage,
           kCInteger | kFloating, kCInteger},
          // Bitwise: C integer, Fortran integer, Byte, multi-language.
          {"MPI_BAND", kCInteger | kFortranInteger | kByte | kMultiLanguage, kCInteger | kByte,
           kCInteger | kByte},
          {"MPI_BOR", kCInteger | kFortranInteger | kByte | kMultiLanguage, kCInteger | kByte,
           kCInteger | kByte},
          {"MPI_BXOR", kCInteger | kFortranInteger | kByte | kMultiLanguage, kCInteger | kByte,
           kCInteger | kByte},
          // Logical: the collective result is not specified as 0/1 for C integers, so
          // collectives never map them; RMA has no callback and accepts only Logical.
          {"MPI_LAND", kCInteger | kLogical, kLogical, 0},
          {"MPI_LOR", kCInteger | kLogical, kLogical, 0},
          {"MPI_LXOR", kCInteger | kLogical, kLogical, 0},
          // REPLACE/NO_OP: defined for every predefined category (§12.3.4).
          {"MPI_REPLACE",
           kCInteger | kFortranInteger | kFloating | kLogical | kComplex | kByte | kCharacter |
               kMultiLanguage,
           0, 0},
          {"MPI_NO_OP",
           kCInteger | kFortranInteger | kFloating | kLogical | kComplex | kByte | kCharacter |
               kMultiLanguage,
           0, 0},
          // MAXLOC/MINLOC need an MPI_2* pair datatype, which mp1 does not model.
          {"MPI_MAXLOC", 0, 0, 0},
          {"MPI_MINLOC", 0, 0, 0},
      }};
    }

    inline constexpr auto predefined_ops = make_predefined_op_table();

    [[nodiscard]] constexpr std::uint16_t category_bit(
        rma_datatype_category category) noexcept {
      const unsigned index = static_cast<unsigned>(category);
      return index < 10 ? static_cast<std::uint16_t>(std::uint16_t{1} << index)
                        : std::uint16_t{0};
    }

    [[nodiscard]] consteval std::array<const char*, 10> make_rma_category_names() noexcept {
      return {{"a derived datatype", "a C integer", "a Fortran integer",
               "a floating-point type", "a logical type", "a complex type", "a byte type",
               "a character type", "a multi-language integer type",
               "an unsupported predefined datatype"}};
    }

    inline constexpr auto rma_category_names = make_rma_category_names();

  }  // namespace detail

  // User opt-in tag for a commutative user-defined reduction functor.  The built-in functors
  // below deliberately do not inherit it: their commutativity depends on the operand type and
  // on the exact C++ operation (see the op_traits specializations after the primary template),
  // so a blanket tag would tell MPI that a non-commutative reduction may be reordered.
  struct commutative_op {};

  template<class T>
  struct max {
    static constexpr detail::reduction_kind kind = detail::reduction_kind::maximum;

    T operator()(const T& x, const T& y) const noexcept(noexcept(x < y)) {
      return (x < y) ? y : x;
    }
  };

  template<class T>
  struct min {
    static constexpr detail::reduction_kind kind = detail::reduction_kind::minimum;

    T operator()(const T& x, const T& y) const noexcept(noexcept(y < x)) {
      return (y < x) ? y : x;
    }
  };

  template<class T>
  struct logical_xor {
    static constexpr detail::reduction_kind kind = detail::reduction_kind::logical_xor;

    T operator()(const T& x, const T& y) const
        noexcept(noexcept(static_cast<T>(static_cast<bool>(x) != static_cast<bool>(y)))) {
      return static_cast<T>(static_cast<bool>(x) != static_cast<bool>(y));
    }
  };

  template<class F>
  struct op_traits {
    static constexpr bool is_commutative = std::is_base_of_v<commutative_op, F>;
  };

  // The remaining mp1 functors (max, min, logical_xor) are commutative only for the
  // operation/type pairs where the C++ operation is actually commutative.  In particular
  // max/min on floating-point values are not commutative under the callback
  // (x < y) ? y : x when an operand is NaN, and a user-defined operator< for a class type is
  // not assumed commutative.  The trait is used only for the MPI_Op_create commute promise;
  // the predefined mapping above is independent of it.
  template<class T>
  struct op_traits<max<T>> {
    static constexpr bool is_commutative = std::is_integral_v<std::remove_cv_t<T>>;
  };

  template<class T>
  struct op_traits<min<T>> {
    static constexpr bool is_commutative = std::is_integral_v<std::remove_cv_t<T>>;
  };

  template<class T>
  struct op_traits<logical_xor<T>> {
    static constexpr bool is_commutative = std::is_arithmetic_v<std::remove_cv_t<T>>;
  };

  // Standard-library functors are only known to be commutative for the element types whose
  // built-in operation is commutative.  In particular std::plus<std::string> and
  // std::plus<MyType> stay non-commutative, and std::minus/std::divides are never marked.
  template<class T>
  struct op_traits<std::plus<T>> {
    static constexpr bool is_commutative = std::is_arithmetic_v<T>;
  };

  template<class T>
  struct op_traits<std::multiplies<T>> {
    static constexpr bool is_commutative = std::is_arithmetic_v<T>;
  };

  // std::complex addition and multiplication are commutative in both components.  The plain
  // arithmetic specialization above excludes them; name the two operations explicitly so the
  // public truth table retains the answer the removed duplicate mp1 functors provided.  Complex
  // SUM/PROD is mapped only in RMA, so a collective complex reduction (which has no
  // predefined mapping) is the callback-side consumer of this trait.
  template<class T>
  struct op_traits<std::plus<std::complex<T>>> {
    static constexpr bool is_commutative = std::is_floating_point_v<T>;
  };

  template<class T>
  struct op_traits<std::multiplies<std::complex<T>>> {
    static constexpr bool is_commutative = std::is_floating_point_v<T>;
  };

  template<class T>
  struct op_traits<std::bit_and<T>> {
    static constexpr bool is_commutative = std::is_integral_v<std::remove_cv_t<T>> or
                                           std::is_same_v<std::remove_cv_t<T>, std::byte>;
  };

  template<class T>
  struct op_traits<std::bit_or<T>> {
    static constexpr bool is_commutative = std::is_integral_v<std::remove_cv_t<T>> or
                                           std::is_same_v<std::remove_cv_t<T>, std::byte>;
  };

  template<class T>
  struct op_traits<std::bit_xor<T>> {
    static constexpr bool is_commutative = std::is_integral_v<std::remove_cv_t<T>> or
                                           std::is_same_v<std::remove_cv_t<T>, std::byte>;
  };

  template<class T>
  struct op_traits<std::logical_and<T>> {
    static constexpr bool is_commutative = std::is_arithmetic_v<T>;
  };

  template<class T>
  struct op_traits<std::logical_or<T>> {
    static constexpr bool is_commutative = std::is_arithmetic_v<T>;
  };

  // Equality is commutative for arithmetic values.  These functors are never mapped to a
  // predefined MPI operation; the trait only affects the MPI_Op_create commute flag.
  template<class T>
  struct op_traits<std::equal_to<T>> {
    static constexpr bool is_commutative = std::is_arithmetic_v<T>;
  };

  template<class T>
  struct op_traits<std::not_equal_to<T>> {
    static constexpr bool is_commutative = std::is_arithmetic_v<T>;
  };

  namespace detail {

    // ---------------------------------------------------------------------------------------
    // Operator adapter.
    //
    // Every reduction family accepts either a native MPI_Op or a functor.  This layer is the
    // single place that decides how a functor is represented:
    //
    //   * native MPI_Op               -> validated and forwarded unchanged;
    //   * safely mappable built-in    -> the corresponding predefined MPI_Op; no
    //                                    MPI_Op_create, and the implementation may use its
    //                                    optimized predefined reduction algorithms;
    //   * everything else             -> the existing stateless callback cache (mpi_op).
    //
    // RMA differs: MPI only permits predefined operators there (MPI 4.1 §12.3.4), so
    // predefined_op<T> rejects an unmappable functor at compile time and an unknown raw
    // handle before the RMA call instead of silently creating or forwarding an MPI_Op.
    // ---------------------------------------------------------------------------------------

    template<class F>
    struct reduction_functor_traits {
      static constexpr reduction_kind kind = reduction_kind::none;
      using operand_type = void;
    };

    template<template<class> class Family, class T>
    struct reduction_functor_traits<Family<T>> {
      static constexpr reduction_kind kind = reduction_kind_of<Family, T>();
      using operand_type = T;
    };

    // The mapped operations are selected from the MPI datatype category, not from a C++ type
    // trait: MPI 4.1 §6.9.2 defines SUM/PROD/MAX/MIN/BAND/BOR/BXOR per predefined datatype
    // category (C integer, Fortran integer, floating, complex, byte, logical, character,
    // multi-language), and the category of a C++ element type is the category of
    // native_datatype<T>(). In particular char and wchar_t are character (not C integer) and
    // are excluded from MPI_SUM/MAX/bitwise, while char16_t/char32_t map to a uint_leastN_t C
    // integer datatype and are included. An enum's category is its underlying integer
    // category, but its C++ operators are user-defined, so enums stay on the callback path;
    // std::byte is the one enum that is mapped, for the Byte-category bitwise operations.
    template<class T>
    inline constexpr bool has_builtin_operator_v =
        not std::is_enum_v<std::remove_cv_t<T>> or
        std::is_same_v<std::remove_cv_t<T>, std::byte>;

    template<class T>
    inline constexpr rma_datatype_category operand_category_v =
        static_rma_category<std::remove_cv_t<T>>();

    // The functor must be declared for exactly the element type being reduced (or be one of
    // the transparent std::plus<void>-style functors).  A functor declared for another type
    // can change the arithmetic -- for example std::plus<int> truncates double operands
    // between steps -- so it must not be replaced by a predefined operation on T.
    template<class T, class F>
    inline constexpr bool has_matching_operand_v = [] {
      using traits = reduction_functor_traits<F>;
      if constexpr (traits::kind == reduction_kind::none) {
        return false;
      } else {
        using operand = std::remove_cv_t<typename traits::operand_type>;
        return std::is_void_v<operand> or std::is_same_v<operand, T>;
      }
    }();

    // The collective-functor gate reads its category mask from the one metadata table above.
    //
    // Collective reductions keep the callback for floating-point MAX/MIN: MPI_MAX/MPI_MIN are
    // unspecified when an operand is NaN, so mp1 does not change that behaviour.  The logical
    // kinds are never mapped here: with MPI_INT some implementations return the integer
    // operand itself (for example 3 instead of 1), which differs from the 0/1 a C++ logical
    // functor returns.  Character and Byte are not MAX/MIN categories.
    [[nodiscard]] consteval bool predefined_reduction_defined_for(
        reduction_kind kind, rma_datatype_category category) noexcept {
      // The first eleven enumerators of reduction_kind index the metadata table directly.
      const auto index = static_cast<std::size_t>(kind);
      return index < predefined_ops.size() and
             (predefined_ops[index].collective_categories & category_bit(category)) != 0;
    }

    // RMA has no callback fallback: MPI only permits predefined operators (MPI 4.1 §12.3.4
    // Accumulate/Get Accumulate/Fetch and Op).  SUM/PROD accept C integer, floating, complex
    // and the multi-language types; MAX/MIN accept C integer and floating (float NaN
    // behaviour is MPI's, unspecified); bitwise ops accept C integer and Byte; logical ops
    // accept C integer and Logical (bool maps to MPI_CXX_BOOL).  Character element types are
    // predefined but are not in any reduction category (§6.9.3), so only
    // MPI_REPLACE/MPI_NO_OP can reach them through a raw handle.
    //
    // The functor gate is stricter than the raw-handle gate: a functor is mapped only when
    // the exact operands are in the §6.9.2 row, while a raw handle is forwarded for every
    // category the row defines.
    [[nodiscard]] consteval bool rma_predefined_defined_for(
        reduction_kind kind, rma_datatype_category category) noexcept {
      const auto index = static_cast<std::size_t>(kind);
      return index < predefined_ops.size() and
             (predefined_ops[index].rma_functor_categories & category_bit(category)) != 0;
    }

    // The one native-handle mapping.  It cannot be consteval: Open MPI defines MPI_SUM & co.
    // as a void* -> MPI_Op cast, which is not a constant expression before C++26.  The
    // logical handles are reachable only from the RMA path (the collective logical kinds are
    // never mapped), so the caller selects them explicitly; REPLACE/NO_OP/MAXLOC/MINLOC are
    // reachable only through a raw MPI_Op handle.
    [[nodiscard]] constexpr MPI_Op predefined_handle_of(predefined_op_kind kind,
                                                        bool include_logical) noexcept {
      switch (kind) {
        case predefined_op_kind::sum:
          return MPI_SUM;
        case predefined_op_kind::product:
          return MPI_PROD;
        case predefined_op_kind::maximum:
          return MPI_MAX;
        case predefined_op_kind::minimum:
          return MPI_MIN;
        case predefined_op_kind::bit_and:
          return MPI_BAND;
        case predefined_op_kind::bit_or:
          return MPI_BOR;
        case predefined_op_kind::bit_xor:
          return MPI_BXOR;
        case predefined_op_kind::logical_and:
          return include_logical ? MPI_LAND : MPI_OP_NULL;
        case predefined_op_kind::logical_or:
          return include_logical ? MPI_LOR : MPI_OP_NULL;
        case predefined_op_kind::logical_xor:
          return include_logical ? MPI_LXOR : MPI_OP_NULL;
        case predefined_op_kind::replace:
          return MPI_REPLACE;
        case predefined_op_kind::no_op:
          return MPI_NO_OP;
        case predefined_op_kind::maxloc:
          return MPI_MAXLOC;
        case predefined_op_kind::minloc:
          return MPI_MINLOC;
        case predefined_op_kind::none:
          return MPI_OP_NULL;
      }
      return MPI_OP_NULL;
    }

    template<class T, class F>
    [[nodiscard]] consteval bool is_predefined_reduction_functor() noexcept {
      using U = std::remove_cv_t<T>;
      if constexpr (not has_matching_operand_v<T, F> or not has_builtin_operator_v<U>) {
        return false;
      } else {
        return predefined_reduction_defined_for(reduction_functor_traits<F>::kind,
                                                operand_category_v<U>);
      }
    }

    template<class T, class F>
    inline constexpr bool is_predefined_reduction_functor_v =
        is_predefined_reduction_functor<T, F>();

    template<class T, class F>
    [[nodiscard]] consteval bool is_predefined_rma_functor() noexcept {
      using U = std::remove_cv_t<T>;
      if constexpr (not has_matching_operand_v<T, F> or not has_builtin_operator_v<U>) {
        return false;
      } else {
        return rma_predefined_defined_for(reduction_functor_traits<F>::kind,
                                          operand_category_v<U>);
      }
    }

    template<class T, class F>
    inline constexpr bool is_predefined_rma_functor_v = is_predefined_rma_functor<T, F>();

    template<class T, class F>
    [[nodiscard]] constexpr MPI_Op predefined_mpi_op() noexcept {
      constexpr reduction_kind kind = reduction_functor_traits<F>::kind;
      // Only the ten shared reduction kinds map here; an out-of-range user-supplied `kind`
      // stays unmapped exactly as the old reduction_kind switch left it.
      if constexpr (not has_matching_operand_v<T, F> or
                    static_cast<unsigned char>(kind) >
                        static_cast<unsigned char>(reduction_kind::logical_xor))
        return MPI_OP_NULL;
      else
        return predefined_handle_of(static_cast<predefined_op_kind>(kind), false);
    }

    // Self-guarding RMA mapping: returns MPI_OP_NULL unless is_predefined_rma_functor_v is
    // true, so a caller that ignores the predicate can never obtain a real handle for an
    // invalid (functor, T) pair.
    template<class T, class F>
    [[nodiscard]] constexpr MPI_Op predefined_rma_mpi_op() noexcept {
      if constexpr (not is_predefined_rma_functor<T, F>())
        return MPI_OP_NULL;
      else
        return predefined_handle_of(
            static_cast<predefined_op_kind>(reduction_functor_traits<F>::kind), true);
    }

    // ---------------------------------------------------------------------------------------
    // The predefined-operation metadata for RMA.
    //
    // MPI_Op_commutative is never used as a predefinedness query: Open MPI reports it true
    // for MPI_REPLACE/MPI_NO_OP and MPICH false.  Handle equality against the standard handles
    // is the portable test (the handles are not constexpr on Open MPI, hence a runtime
    // function).  A user-created or vendor-specific handle is `none` and is rejected before
    // any MPI call.  Classification iterates the one metadata table above, so the names, the
    // accepted categories and the handle mapping cannot drift apart.
    // ---------------------------------------------------------------------------------------

    [[nodiscard]] inline predefined_op_kind classify_predefined_op(MPI_Op op) noexcept {
      for (std::size_t index = 1; index < predefined_ops.size(); ++index) {
        const auto kind = static_cast<predefined_op_kind>(index);
        if (op == predefined_handle_of(kind, true))
          return kind;
      }
      return predefined_op_kind::none;
    }

    [[nodiscard]] inline const char* predefined_op_name(predefined_op_kind kind) noexcept {
      const auto index = static_cast<std::size_t>(kind);
      return index < predefined_ops.size() ? predefined_ops[index].name : "an unknown MPI_Op";
    }

    [[nodiscard]] inline const char* rma_datatype_category_name(
        rma_datatype_category category) noexcept {
      const auto index = static_cast<std::size_t>(category);
      return index < rma_category_names.size() ? rma_category_names[index]
                                               : "an unknown datatype category";
    }

    // MPI 4.1 §6.9.2 (the predefined reduction table) plus MPI_REPLACE/MPI_NO_OP, which
    // §12.3.4 defines for the accumulating RMA families only.  MAXLOC/MINLOC are always
    // rejected: mp1 has no MPI_2* pair datatype, so the typed API cannot express them and a
    // raw handle cannot be checked for the required pair signature.
    [[nodiscard]] constexpr bool op_defined_for(predefined_op_kind kind,
                                                rma_datatype_category category) noexcept {
      const auto index = static_cast<std::size_t>(kind);
      return index < predefined_ops.size() and
             (predefined_ops[index].raw_rma_categories & category_bit(category)) != 0;
    }

    // Validate a raw MPI_Op for an RMA accumulating family and T.  Throws
    // std::invalid_argument before any MPI call.  The family-level NO_OP gate is separate
    // (allow_no_op), because the handle is also valid for get_accumulate/fetch_and_op.
    template<class T>
    [[nodiscard]] inline MPI_Op validate_raw_rma_op(MPI_Op op, bool allow_no_op,
                                                    const context& ctx) {
      if (op == MPI_OP_NULL) {
        throw_invalid(ctx.operation, "MPI_OP_NULL is not a valid RMA operator");
      }
      const predefined_op_kind kind = classify_predefined_op(op);
      if (kind == predefined_op_kind::none) {
        throw_invalid(
            ctx.operation,
            "only a predefined MPI_Op is allowed for RMA (MPI 4.1 §12.3.4 Accumulate/Get "
            "Accumulate/Fetch and Op); user-created and unknown handles cannot be used");
      }
      if (kind == predefined_op_kind::maxloc or kind == predefined_op_kind::minloc) {
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "{} requires an MPI_2* pair datatype, which mp1 does not model (MPI 4.1 §6.9.4 "
            "MINLOC and MAXLOC)",
            predefined_op_name(kind));
      }
      if (kind == predefined_op_kind::no_op and not allow_no_op) {
        throw_invalid(ctx.operation,
                      "MPI_NO_OP is allowed only in MPI_Get_accumulate, MPI_Rget_accumulate "
                      "and MPI_Fetch_and_op (MPI 4.1 §12.3.4 Get Accumulate)");
      }
      const rma_datatype_category category = rma_datatype_category_of<T>();
      if (not op_defined_for(kind, category)) {
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "{} is not defined for {} (MPI 4.1 §6.9.2 Predefined Reduction Operations); use a "
            "matching predefined operation or the native handle() escape for raw MPI",
            predefined_op_name(kind), rma_datatype_category_name(category));
      }
      return op;
    }

    // Reject a datatype that the typed single-element RMA atomic boundary (fetch_and_op,
    // compare_and_swap) cannot accept.  Built-in element types are decided at compile time; a
    // custom MP1_DATATYPE_TRAITS mapping is checked once against its committed handle.  This
    // catches a custom trait that claims predefinedness for a derived handle before the RMA
    // call reaches MPI.
    template<class T>
    [[nodiscard]] inline rma_datatype_category require_predefined_rma_datatype(
        const context& ctx) {
      const rma_datatype_category category = rma_datatype_category_of<T>();
      if (category == rma_datatype_category::derived or
          category == rma_datatype_category::invalid) {
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "the element datatype is {}, not one predefined MPI datatype; MPI 4.1 §12.3.4 "
            "requires a predefined datatype for the single-element atomic RMA families "
            "(fetch_and_op and compare_and_swap)",
            rma_datatype_category_name(category));
      }
      return category;
    }

    // The transparent standard functors (std::plus<>, std::bit_and<>, ...) cannot be
    // classified without the element type; decide from T.  std::plus<MyType> is handled by
    // the op_traits specialization above (and is not marked commutative).  One consteval
    // predicate replaces the ten partial specializations.
    template<class T, class F>
    [[nodiscard]] consteval bool is_transparent_std_commutative() noexcept {
      if constexpr (std::same_as<F, std::plus<void>> or
                    std::same_as<F, std::multiplies<void>> or
                    std::same_as<F, std::equal_to<void>> or
                    std::same_as<F, std::not_equal_to<void>> or
                    std::same_as<F, std::logical_and<void>> or
                    std::same_as<F, std::logical_or<void>>) {
        return std::is_arithmetic_v<T>;
      } else if constexpr (std::same_as<F, std::bit_and<void>> or
                           std::same_as<F, std::bit_or<void>> or
                           std::same_as<F, std::bit_xor<void>>) {
        return std::is_integral_v<T>;
      } else {
        return false;
      }
    }

    template<class T, class F>
    inline constexpr bool is_transparent_std_commutative_v =
        is_transparent_std_commutative<T, F>();

    template<class T, class F>
    inline constexpr bool is_commutative_reduction_v =
        op_traits<F>::is_commutative or is_transparent_std_commutative_v<T, F>;

    // ---------------------------------------------------------------------------------------
    // Nothrow functor gate for the callback path.
    //
    // MPI 4.1 §6.9.5 calls a C function from inside the MPI library; a C++ exception escaping
    // that function terminates the process and cannot be reported to the caller.  A functor
    // used on the managed callback path must therefore be nothrow-invocable for T.
    //
    // The standard-library binary functors are conditionally noexcept in the operation but
    // their operator() itself is not declared noexcept (the standard does not require it).
    // Checking std::is_nothrow_invocable_r_v on them would reject every standard functor, so
    // the operation is re-formed here with an explicit conditional noexcept.  A functor
    // declared for another operand type is not re-formed; it must satisfy the direct
    // invocability check.  User and mp1 functors always use the direct check.
    // ---------------------------------------------------------------------------------------
#define MP1_DEFINE_BINARY_EXPRESSION(NAME, EXPRESSION)                    \
  struct NAME##_expression {                                              \
    template<class T>                                                     \
    [[nodiscard]] constexpr auto operator()(const T& x, const T& y) const \
        noexcept(noexcept(EXPRESSION)) -> decltype(EXPRESSION) {          \
      return EXPRESSION;                                                  \
    }                                                                     \
  }

    MP1_DEFINE_BINARY_EXPRESSION(plus, x + y);
    MP1_DEFINE_BINARY_EXPRESSION(minus, x - y);
    MP1_DEFINE_BINARY_EXPRESSION(multiplies, x* y);
    MP1_DEFINE_BINARY_EXPRESSION(divides, x / y);
    MP1_DEFINE_BINARY_EXPRESSION(modulus, x % y);
    MP1_DEFINE_BINARY_EXPRESSION(bit_and, x& y);
    MP1_DEFINE_BINARY_EXPRESSION(bit_or, x | y);
    MP1_DEFINE_BINARY_EXPRESSION(bit_xor, x ^ y);
    MP1_DEFINE_BINARY_EXPRESSION(logical_and, x and y);
    MP1_DEFINE_BINARY_EXPRESSION(logical_or, x or y);
    MP1_DEFINE_BINARY_EXPRESSION(equal_to, x == y);
    MP1_DEFINE_BINARY_EXPRESSION(not_equal_to, x != y);
    MP1_DEFINE_BINARY_EXPRESSION(less, x < y);
    MP1_DEFINE_BINARY_EXPRESSION(less_equal, x <= y);
    MP1_DEFINE_BINARY_EXPRESSION(greater, x > y);
    MP1_DEFINE_BINARY_EXPRESSION(greater_equal, x >= y);

#undef MP1_DEFINE_BINARY_EXPRESSION

    template<class F, class T>
    struct std_functor_expression {
      static constexpr bool matched = false;
      using adapter = void;
    };

#define MP1_DEFINE_STD_EXPRESSION(STD_TEMPLATE, ADAPTER)                               \
  template<class U, class T>                                                           \
  struct std_functor_expression<STD_TEMPLATE<U>, T> {                                  \
    static constexpr bool matched =                                                    \
        std::is_void_v<U> or std::is_same_v<std::remove_cv_t<U>, std::remove_cv_t<T>>; \
    using adapter = ADAPTER;                                                           \
  }

    MP1_DEFINE_STD_EXPRESSION(std::plus, plus_expression);
    MP1_DEFINE_STD_EXPRESSION(std::minus, minus_expression);
    MP1_DEFINE_STD_EXPRESSION(std::multiplies, multiplies_expression);
    MP1_DEFINE_STD_EXPRESSION(std::divides, divides_expression);
    MP1_DEFINE_STD_EXPRESSION(std::modulus, modulus_expression);
    MP1_DEFINE_STD_EXPRESSION(std::bit_and, bit_and_expression);
    MP1_DEFINE_STD_EXPRESSION(std::bit_or, bit_or_expression);
    MP1_DEFINE_STD_EXPRESSION(std::bit_xor, bit_xor_expression);
    MP1_DEFINE_STD_EXPRESSION(std::logical_and, logical_and_expression);
    MP1_DEFINE_STD_EXPRESSION(std::logical_or, logical_or_expression);
    MP1_DEFINE_STD_EXPRESSION(std::equal_to, equal_to_expression);
    MP1_DEFINE_STD_EXPRESSION(std::not_equal_to, not_equal_to_expression);
    MP1_DEFINE_STD_EXPRESSION(std::less, less_expression);
    MP1_DEFINE_STD_EXPRESSION(std::less_equal, less_equal_expression);
    MP1_DEFINE_STD_EXPRESSION(std::greater, greater_expression);
    MP1_DEFINE_STD_EXPRESSION(std::greater_equal, greater_equal_expression);

#undef MP1_DEFINE_STD_EXPRESSION

    template<class T, class F>
    inline constexpr bool is_nothrow_reduction_functor_v = [] {
      using expression = std_functor_expression<F, T>;
      if constexpr (expression::matched) {
        return std::is_nothrow_invocable_r_v<T, typename expression::adapter, const T&,
                                             const T&>;
      } else {
        return std::is_nothrow_invocable_r_v<T, const F&, const T&, const T&>;
      }
    }();

    // ---------------------------------------------------------------------------------------
    // The mapped-byte plan of the managed callback image.
    //
    // The callback must update only the bytes that are part of the native type map.  If it
    // writes a complete T object (for example through T copy assignment), an in-place
    // reduction whose MPI implementation hands the callback the caller's buffer directly
    // would overwrite the caller's padding and unlisted members -- exactly the full-byte
    // overwrite this plan avoids.  The plan is the set of contiguous mapped byte runs of
    // native_datatype<T>(), computed once from MPI_Type_get_contents_c (never inside the
    // callback, which may not call MPI).  The callback copies only those runs from the
    // functor result into the output object.
    //
    // A datatype whose map cannot be decoded from the standard combiners, or whose map
    // overlaps itself, cannot be the callback image of a managed reduction: the exact
    // mapped-byte write is impossible, and mp1 rejects it with std::invalid_argument before
    // the reduction rather than silently falling back to a full-object overwrite.
    // ---------------------------------------------------------------------------------------
    struct image_byte_run {
      count_t offset{0};
      count_t length{0};
    };

    inline void append_image_byte_run(std::vector<image_byte_run>& runs, count_t offset,
                                      count_t length) {
      if (length == 0)
        return;
      if (not runs.empty()) {
        const count_t back_end =
            checked_count_add(runs.back().offset, runs.back().length, {"mp1: reduction image"});
        if (back_end == offset) {
          runs.back().length =
              checked_count_add(runs.back().length, length, {"mp1: reduction image"});
          return;
        }
      }
      runs.push_back(image_byte_run{offset, length});
    }

    [[nodiscard]] inline count_t image_child_extent(MPI_Datatype type, const context& ctx) {
      MPI_Count lower_bound = 0;
      MPI_Count extent = 0;
      check_mpi(MPI_Type_get_extent_c(type, &lower_bound, &extent),
                {"MPI_Type_get_extent_c(reduction image child)"});
      const count_t result = checked_count(extent, ctx);
      if (result < 0)
        detail::fail_as<std::invalid_argument>(ctx, std::source_location::current(),
                                               "native datatype child extent {} is negative",
                                               result);
      return result;
    }

    // MPI_Type_get_contents_c returns a new derived datatype object for every datatype
    // argument; the caller must free it.  A named predefined child is the constant handle and
    // must not be freed.  The mapped-byte plan is built once per T, but the child handles are
    // released as soon as their runs are copied so a failed construction leaves no handle
    // behind.
    inline void release_image_child(MPI_Datatype type) noexcept {
      if (type == MPI_DATATYPE_NULL)
        return;
      MPI_Count num_integers{0};
      MPI_Count num_addresses{0};
      MPI_Count num_large_counts{0};
      MPI_Count num_datatypes{0};
      int combiner = MPI_COMBINER_NAMED;
      if (MPI_Type_get_envelope_c(type, &num_integers, &num_addresses, &num_large_counts,
                                  &num_datatypes, &combiner) != MPI_SUCCESS)
        return;
      if (combiner != MPI_COMBINER_NAMED and num_datatypes > 0) {
        MPI_Datatype owned = type;
        (void)MPI_Type_free(&owned);
      }
    }

    // Append the mapped byte runs of `type` displaced by `base`.  The recursion covers the
    // constructors the library itself produces for reflected aggregates (named, resized,
    // struct) plus the common derived constructors a custom datatype_traits mapping can use.
    //
    // MPI 4.1 5.1.13 specifies the decode arrays only for datatypes built with the
    // non-large-count constructors.  The _c constructors are decoded differently by the
    // providers (both Open MPI and MPICH put the count, blocklengths and displacements of
    // MPI_Type_create_struct_c into array_of_large_counts).  The envelope counts select the
    // one layout that fits: the standard arrays when num_large_counts is zero, otherwise the
    // provider large-count layout.  A layout that does not fit is rejected rather than
    // guessed.
    inline void collect_image_byte_runs(MPI_Datatype type, count_t base,
                                        std::vector<image_byte_run>& runs, const context& ctx) {
      MPI_Count num_integers{0};
      MPI_Count num_addresses{0};
      MPI_Count num_large_counts{0};
      MPI_Count num_datatypes{0};
      int combiner = MPI_COMBINER_NAMED;
      check_mpi(MPI_Type_get_envelope_c(type, &num_integers, &num_addresses, &num_large_counts,
                                        &num_datatypes, &combiner),
                {"MPI_Type_get_envelope_c(reduction image)"});
      if (combiner == MPI_COMBINER_NAMED) {
        MPI_Count size = 0;
        check_mpi(MPI_Type_size_c(type, &size), {"MPI_Type_size_c(reduction image leaf)"});
        append_image_byte_run(runs, base, checked_count(size, ctx));
        return;
      }
      const count_t integer_count = checked_count(num_integers, ctx);
      const count_t address_count = checked_count(num_addresses, ctx);
      const count_t large_count = checked_count(num_large_counts, ctx);
      const count_t datatype_count = checked_count(num_datatypes, ctx);
      if (datatype_count == 0)
        detail::fail_as<std::invalid_argument>(
            ctx, std::source_location::current(),
            "native datatype combiner {} has no child datatype", combiner);
      std::vector<int> integers(static_cast<std::size_t>(integer_count));
      std::vector<MPI_Aint> addresses(static_cast<std::size_t>(address_count));
      std::vector<MPI_Count> large_counts(static_cast<std::size_t>(large_count));
      std::vector<MPI_Datatype> types(static_cast<std::size_t>(datatype_count));
      check_mpi(MPI_Type_get_contents_c(type, num_integers, num_addresses, num_large_counts,
                                        num_datatypes, integers.data(), addresses.data(),
                                        large_counts.data(), types.data()),
                {"MPI_Type_get_contents_c(reduction image)"});

      // Each array_of_datatypes entry must be released after its last use.  `released`
      // makes the exception path idempotent and never frees a predefined handle.
      std::vector<bool> released(types.size(), false);
      const auto release = [&](count_t index) {
        if (not released[static_cast<std::size_t>(index)]) {
          release_image_child(types[static_cast<std::size_t>(index)]);
          released[static_cast<std::size_t>(index)] = true;
        }
      };
      const auto require = [&](bool condition, const char* what) {
        if (not condition)
          detail::fail_as<std::invalid_argument>(
              ctx, std::source_location::current(),
              "native datatype combiner {} cannot be decoded: {} (integers {}, addresses "
              "{}, large counts {}, datatypes {})",
              combiner, what, integer_count, address_count, large_count, datatype_count);
      };
      const auto i = [&](count_t index) { return checked_count(integers[index], ctx); };
      const auto l = [&](count_t index) { return checked_count(large_counts[index], ctx); };
      const auto a = [&](count_t index) { return checked_count(addresses[index], ctx); };
      const auto repeat = [&](MPI_Datatype child, count_t displacement, count_t blocklength) {
        require(blocklength >= 0, "negative blocklength");
        const count_t extent = image_child_extent(child, ctx);
        for (count_t copy = 0; copy < blocklength; ++copy) {
          const count_t offset =
              checked_count_add(displacement, checked_count_mul(copy, extent, ctx), ctx);
          collect_image_byte_runs(child, checked_count_add(base, offset, ctx), runs, ctx);
        }
      };

      try {
        switch (combiner) {
          case MPI_COMBINER_DUP:
            collect_image_byte_runs(types[0], base, runs, ctx);
            release(0);
            return;
          case MPI_COMBINER_CONTIGUOUS: {
            const count_t count = large_count > 0 ? l(0) : i(0);
            repeat(types[0], 0, count);
            release(0);
            return;
          }
          case MPI_COMBINER_VECTOR: {
            count_t count{0};
            count_t blocklength{0};
            count_t stride{0};
            if (large_count > 0) {
              require(large_count >= 3, "vector large-count arguments are incomplete");
              count = l(0);
              blocklength = l(1);
              stride = l(2);
            } else {
              require(integer_count >= 3, "vector integer arguments are incomplete");
              count = i(0);
              blocklength = i(1);
              stride = i(2);
            }
            for (count_t block = 0; block < count; ++block)
              repeat(types[0], checked_count_mul(block, stride, ctx), blocklength);
            release(0);
            return;
          }
          case MPI_COMBINER_HVECTOR: {
            if (large_count > 0) {
              require(large_count >= 3, "hvector large-count arguments are incomplete");
              const count_t count = l(0);
              const count_t blocklength = l(1);
              const count_t stride = l(2);
              for (count_t block = 0; block < count; ++block)
                repeat(types[0], checked_count_mul(block, stride, ctx), blocklength);
            } else {
              require(integer_count >= 2 and address_count >= 1,
                      "hvector arguments are incomplete");
              const count_t count = i(0);
              const count_t blocklength = i(1);
              const count_t stride = a(0);
              for (count_t block = 0; block < count; ++block)
                repeat(types[0], checked_count_mul(block, stride, ctx), blocklength);
            }
            release(0);
            return;
          }
          case MPI_COMBINER_INDEXED: {
            if (large_count > 0) {
              require(large_count >= 1, "indexed large-count arguments are incomplete");
              const count_t count = l(0);
              require(large_count >= 1 + 2 * count,
                      "indexed large-count arguments are incomplete");
              for (count_t block = 0; block < count; ++block)
                repeat(types[0], l(1 + count + block), l(1 + block));
            } else {
              require(integer_count >= 1, "indexed arguments are incomplete");
              const count_t count = i(0);
              require(integer_count >= 1 + 2 * count, "indexed arguments are incomplete");
              for (count_t block = 0; block < count; ++block)
                repeat(types[0], i(1 + count + block), i(1 + block));
            }
            release(0);
            return;
          }
          case MPI_COMBINER_HINDEXED: {
            if (large_count > 0) {
              require(large_count >= 1, "hindexed large-count arguments are incomplete");
              const count_t count = l(0);
              require(large_count >= 1 + 2 * count,
                      "hindexed large-count arguments are incomplete");
              for (count_t block = 0; block < count; ++block)
                repeat(types[0], l(1 + count + block), l(1 + block));
            } else {
              require(integer_count >= 1, "hindexed arguments are incomplete");
              const count_t count = i(0);
              require(integer_count >= 1 + count and address_count >= count,
                      "hindexed arguments are incomplete");
              for (count_t block = 0; block < count; ++block)
                repeat(types[0], a(block), i(1 + block));
            }
            release(0);
            return;
          }
          case MPI_COMBINER_INDEXED_BLOCK: {
            count_t count{0};
            count_t blocklength{0};
            if (large_count > 0) {
              require(large_count >= 2, "indexed-block large-count arguments are incomplete");
              count = l(0);
              blocklength = l(1);
              require(large_count >= 2 + count,
                      "indexed-block large-count arguments are incomplete");
            } else {
              require(integer_count >= 2, "indexed-block arguments are incomplete");
              count = i(0);
              blocklength = i(1);
              require(integer_count >= 2 + count, "indexed-block arguments are incomplete");
            }
            for (count_t block = 0; block < count; ++block)
              repeat(types[0], large_count > 0 ? l(2 + block) : i(2 + block), blocklength);
            release(0);
            return;
          }
          case MPI_COMBINER_HINDEXED_BLOCK: {
            count_t count{0};
            count_t blocklength{0};
            if (large_count > 0) {
              require(large_count >= 2, "hindexed-block large-count arguments are incomplete");
              count = l(0);
              blocklength = l(1);
              require(large_count >= 2 + count,
                      "hindexed-block large-count arguments are incomplete");
            } else {
              require(integer_count >= 2, "hindexed-block arguments are incomplete");
              count = i(0);
              blocklength = i(1);
              require(address_count >= count, "hindexed-block arguments are incomplete");
            }
            for (count_t block = 0; block < count; ++block)
              repeat(types[0], large_count > 0 ? l(2 + block) : a(block), blocklength);
            release(0);
            return;
          }
          case MPI_COMBINER_STRUCT: {
            const count_t count = datatype_count;
            if (large_count >= 1 + 2 * count) {
              // Provider large-count decode: count, blocklengths, displacements, all in
              // array_of_large_counts (Open MPI 6 and MPICH 4/5).
              for (count_t block = 0; block < count; ++block) {
                repeat(types[block], l(1 + count + block), l(1 + block));
                release(block);
              }
            } else if (large_count >= 1 + count and address_count >= count) {
              // Large-count blocklengths with MPI_Aint displacements.
              for (count_t block = 0; block < count; ++block) {
                repeat(types[block], a(block), l(1 + block));
                release(block);
              }
            } else if (integer_count >= 1 + count and address_count >= count) {
              // Standard MPI_Type_create_struct decode (MPI 4.1 5.1.13).
              for (count_t block = 0; block < count; ++block) {
                repeat(types[block], a(block), i(1 + block));
                release(block);
              }
            } else {
              require(false, "struct arguments do not match a known decode layout");
            }
            return;
          }
          case MPI_COMBINER_RESIZED:
            // MPI_Type_create_resized changes only the lower bound and extent, never the type
            // map, so the child's runs keep their displacements.
            collect_image_byte_runs(types[0], base, runs, ctx);
            release(0);
            return;
          default:
            detail::fail_as<std::invalid_argument>(
                ctx, std::source_location::current(),
                "native datatype combiner {} cannot be decoded into a mapped-byte plan; a "
                "managed reduction functor requires a native datatype built from named, "
                "duplicated, contiguous, vector, indexed or struct constructors",
                combiner);
        }
      } catch (...) {
        for (count_t index = 0; index < datatype_count; ++index)
          release(index);
        throw;
      }
    }

    template<class T>
    [[nodiscard]] const std::vector<image_byte_run>& reduction_image_byte_runs() {
      require_mpi("mp1: reduction callback datatype construction");
      static const std::vector<image_byte_run> runs = [] {
        const context ctx{"mp1: reduction callback datatype"};
        const datatype_info& native = datatype_info_of<T>();
        std::vector<image_byte_run> collected;
        collect_image_byte_runs(native.type, 0, collected, ctx);
        std::sort(collected.begin(), collected.end(),
                  [](const image_byte_run& left, const image_byte_run& right) {
                    return left.offset < right.offset;
                  });
        std::vector<image_byte_run> merged;
        count_t mapped_bytes{0};
        for (const image_byte_run& run : collected) {
          if (run.length == 0)
            continue;
          const count_t object_size = static_cast<count_t>(sizeof(T));
          const count_t end =
              checked_count_add(run.offset, run.length, {"mp1: reduction callback datatype"});
          if (run.offset < 0 or end > object_size)
            detail::fail_as<std::invalid_argument>(
                ctx, std::source_location::current(),
                "native datatype maps bytes [{}, {}) outside one T object of size {}",
                run.offset, end, object_size);
          if (not merged.empty()) {
            const count_t back_end =
                checked_count_add(merged.back().offset, merged.back().length,
                                  {"mp1: reduction callback datatype"});
            if (run.offset < back_end)
              detail::fail_as<std::invalid_argument>(
                  ctx, std::source_location::current(),
                  "native datatype type map overlaps at byte {}; an overlapping receive is "
                  "erroneous for a reduction (MPI 4.1 §6.9)",
                  run.offset);
            if (run.offset == back_end) {
              merged.back().length = checked_count_add(merged.back().length, run.length,
                                                       {"mp1: reduction callback datatype"});
              mapped_bytes = checked_count_add(mapped_bytes, run.length,
                                               {"mp1: reduction callback datatype"});
              continue;
            }
          }
          merged.push_back(run);
          mapped_bytes =
              checked_count_add(mapped_bytes, run.length, {"mp1: reduction callback datatype"});
        }
        if (mapped_bytes != native.size)
          detail::fail_as<std::invalid_argument>(
              ctx, std::source_location::current(),
              "native datatype size {} does not match the decoded mapped-byte total {}; the "
              "type map cannot be reproduced exactly",
              native.size, mapped_bytes);
        return merged;
      }();
      return runs;
    }

    // The native map is shifted down by this offset when the callback image is built, and the
    // buffer pointer passed to MPI is shifted up by the same amount.  The image's true lower
    // bound is therefore 0 while the typed leaves keep their C++ offsets relative to the real
    // T object.  The callback applies the inverse offset before its T indexing.  This is the
    // displacement/remap that lets a T with native true_lb > 0 keep the T[len] contract
    // without an MPI_BYTE prefix block.
    template<class T>
    [[nodiscard]] count_t reduction_image_buffer_offset() {
      require_mpi("mp1: reduction callback datatype construction");
      static const count_t value = datatype_info_of<T>().true_lb;
      return value;
    }

    // Copy only the mapped byte runs of `result` into `output`.  Bytes outside the runs keep
    // their previous value, so the caller's padding/unlisted members survive an in-place
    // reduction even when MPI hands the callback the caller's buffer.
    template<class T>
    void store_mapped_fields(T* output, const T& result,
                             const std::vector<image_byte_run>& runs) noexcept {
      auto* output_bytes = reinterpret_cast<unsigned char*>(output);
      const auto* result_bytes = reinterpret_cast<const unsigned char*>(std::addressof(result));
      for (const image_byte_run& run : runs)
        std::memcpy(output_bytes + run.offset, result_bytes + run.offset,
                    static_cast<std::size_t>(run.length));
    }

    // MPI reduction callback.  Its datatype is reduction_image_datatype<T>(): the native
    // typed map of T wrapped to extent sizeof(T), so each callback element is one complete T
    // object at the C++ stride and T[len] indexing addresses the listed fields at their C++
    // offsets.  Padding bytes and unlisted members are not in the type map and are never
    // transmitted; the callback must not inspect them.  T is trivially copyable (checked in
    // mpi_op) so the callback can treat the buffers as T arrays.  The functor is stateless per
    // the static_assert in mpi_op; no shared mutable state and no allocation per element.
    //
    // C++20 captureless lambdas are default-constructible (P0624R2), so the common stateless
    // functor is constructed directly in the callback and needs no shared state.  An empty
    // functor with a deleted default constructor keeps the old per-(T,F) slot, initialized
    // once by mpi_op(); MPI calls the callback only after that initialization.
    template<class T, class F>
    [[nodiscard]] const F*& reduction_functor_slot() noexcept {
      static const F* slot = nullptr;
      return slot;
    }

    template<class T, class F>
    // NOLINTNEXTLINE(readability-non-const-parameter): MPI_Op signature
    void reduction_callback(void* in, void* inout, int* len, MPI_Datatype*) {
      // MPI passes the image origin; the real T array starts buffer_offset bytes earlier.
      const count_t buffer_offset = reduction_image_buffer_offset<T>();
      const T* input = reinterpret_cast<const T*>(static_cast<const char*>(in) - buffer_offset);
      T* output = reinterpret_cast<T*>(static_cast<char*>(inout) - buffer_offset);
      // Initialized by mpi_op() before MPI_Op_create; the callback itself makes no MPI call.
      const std::vector<image_byte_run>& runs = reduction_image_byte_runs<T>();
      if constexpr (std::default_initializable<F>) {
        const F f{};
        for (int i = 0; i < *len; ++i) {
          const T result = f(input[i], output[i]);
          store_mapped_fields(output + i, result, runs);
        }
      } else {
        const F& f = *reduction_functor_slot<T, F>();
        for (int i = 0; i < *len; ++i) {
          const T result = f(input[i], output[i]);
          store_mapped_fields(output + i, result, runs);
        }
      }
    }

    // One process-lifetime MPI_Op per (T, decay_t<F>).  The functor object (not its type) is
    // captured on first use only when the functor is not default-initializable; stateless
    // functors, including captureless C++17/20 lambdas, otherwise default-construct.  The
    // handle is retained for the process lifetime: freeing it after MPI_Finalize is not
    // allowed, and the function-local static has trivial destruction so no destructor runs
    // after finalize.
    template<class T, class F>
    [[nodiscard]] MPI_Op mpi_op(const F& f) {
      static_assert(std::is_empty_v<F>,
                    "mp1: reduction functors must be stateless (empty) types");
      static_assert(std::is_trivially_copyable_v<T>,
                    "mp1: a managed reduction functor requires a trivially copyable element "
                    "type T; MPI transports only T's listed typed components, so the callback "
                    "can only treat the buffers as T arrays for a trivially copyable T");
      static_assert(is_nothrow_reduction_functor_v<T, F>,
                    "mp1: a reduction functor used on the callback path must be "
                    "nothrow-invocable for the element type; mark operator() noexcept "
                    "(an exception cannot unwind through the MPI C callback)");
      require_mpi("mp1: reduction operator creation");
      // Build the mapped-byte plan and the buffer offset before MPI_Op_create: the callback
      // must not perform the (allocating, MPI-querying) plan construction from inside MPI.
      (void)reduction_image_byte_runs<T>();
      (void)reduction_image_buffer_offset<T>();
      if constexpr (std::default_initializable<F>) {
        (void)f;  // the callback default-constructs the stateless functor
      } else {
        static const F stored = f;
        static const bool slot_ready = [] {
          reduction_functor_slot<T, F>() = &stored;
          return true;
        }();
        (void)slot_ready;
      }
      // A function-local static with trivial destruction: the handle is retained for the
      // process lifetime (freeing it after MPI_Finalize is not allowed) but there is no heap
      // allocation and no static destructor that could run after finalize.
      static const MPI_Op cached = [] {
        MPI_Op op = MPI_OP_NULL;
        check_mpi(MPI_Op_create(&reduction_callback<T, F>,
                                is_commutative_reduction_v<T, F> ? 1 : 0, &op),
                  "MPI_Op_create");
        return op;
      }();
      return cached;
    }

    // The callback datatype for an mp1-managed reduction functor: the native typed map of T,
    // shifted to true lower bound 0 and wrapped to a full-T element extent.
    //
    // MPI 4.1 §6.9.5 (User-Defined Reduction Operations) gives a user function
    // (invec, inoutvec, len, datatype) where invec is an array of len elements of `datatype`.
    // The image is MPI_Type_create_resized(shifted, 0, sizeof(T)), where `shifted` is the
    // native datatype when its true lower bound is 0, otherwise a one-block struct that places
    // the native datatype at displacement -true_lb.  Therefore:
    //
    //   * the type signature is exactly T's native typed components (the leaves of
    //     native_datatype<T>()); MPI 4.1 §3.3.2 representation conversion applies to those
    //     typed leaves on a heterogeneous run;
    //   * extent == sizeof(T), lower bound == 0 and true lower bound == 0, so element i starts
    //     at invec + i*sizeof(T);
    //   * the native true lower bound is applied on the buffer side: the MPI call adapters add
    //     reduction_image_buffer_offset<T>() to every callback-image buffer pointer, and the
    //     callback subtracts the same offset before indexing in/inout as T[len].  The mapped
    //     fields therefore land at their real C++ offsets even for a reflected T whose first
    //     listed member is at offset k > 0;
    //   * there is no MPI_BYTE prefix or suffix: padding and unlisted members are simply not
    //     part of the type map, are never transmitted, and a direct receive never writes them.
    //
    // The callback contract is therefore: complete T objects with one element per sizeof(T)
    // extent unit; the functor must read/write only the listed members and must not inspect
    // padding or unlisted members (their bytes in MPI's temporaries are unspecified).  A T
    // with alignof(T) > alignof(std::max_align_t) is not supported on this path: MPI may place
    // callback temporaries in storage with only fundamental alignment.
    //
    // reduce_scatter/reduce_scatter_block are the exception: MPICH 4.3.2 (the CI floor)
    // lays their user-op temporaries out at the image's true extent, so a T whose image has
    // unmapped bytes after the true extent is rejected there by
    // require_reduce_scatter_callback_image().
    //
    // The image is created once per T, committed, and retained for the process lifetime (the
    // same policy as the MPI_Op cache: freeing after MPI_Finalize is not allowed, and a static
    // destructor would run after finalize).  The cache is deliberately not destroyed at
    // process exit.  A failed construction frees the temporary handle before rethrowing, so a
    // later call retries instead of caching a broken handle.
    template<class T>
    [[nodiscard]] MPI_Datatype reduction_image_datatype() {
      require_mpi("mp1: reduction callback datatype construction");
      static const MPI_Datatype value = [] {
        const datatype_info& native = datatype_info_of<T>();
        MPI_Datatype shifted = MPI_DATATYPE_NULL;
        MPI_Datatype image = MPI_DATATYPE_NULL;
        try {
          MPI_Datatype source = native.type;
          if (native.true_lb > 0) {
            // Place the native typed map at displacement -true_lb so its first mapped byte is
            // at offset 0.  The image's true lower bound is 0 and the callback can receive an
            // origin; the caller-side buffer pointer is shifted up by true_lb (see
            // reduction_image_buffer_offset).  The block is typed: no byte component is added.
            MPI_Count blocklength = 1;
            MPI_Count displacement = -static_cast<MPI_Count>(native.true_lb);
            MPI_Datatype child = native.type;
            check_mpi(
                MPI_Type_create_struct_c(1, &blocklength, &displacement, &child, &shifted),
                "MPI_Type_create_struct_c");
            source = shifted;
          }
          check_mpi(
              MPI_Type_create_resized(source, 0, static_cast<MPI_Aint>(sizeof(T)), &image),
              "MPI_Type_create_resized");
          if (shifted != MPI_DATATYPE_NULL) {
            (void)MPI_Type_free(&shifted);
            shifted = MPI_DATATYPE_NULL;
          }
          MPI_Count size = 0;
          MPI_Count lower_bound = 0;
          MPI_Count extent = 0;
          MPI_Count true_lb = 0;
          MPI_Count true_extent = 0;
          check_mpi(MPI_Type_size_c(image, &size), {"MPI_Type_size_c(reduction image)"});
          check_mpi(MPI_Type_get_extent_c(image, &lower_bound, &extent),
                    {"MPI_Type_get_extent_c(reduction image)"});
          check_mpi(MPI_Type_get_true_extent_c(image, &true_lb, &true_extent),
                    {"MPI_Type_get_true_extent_c(reduction image)"});
          const count_t sizeof_t = static_cast<count_t>(sizeof(T));
          const count_t image_size = checked_count(size, {"mp1: reduction image size"});
          const count_t image_lb = checked_count(lower_bound, {"mp1: reduction image lb"});
          const count_t image_extent = checked_count(extent, {"mp1: reduction image extent"});
          const count_t image_true_lb =
              checked_count(true_lb, {"mp1: reduction image true lb"});
          const count_t image_true_extent =
              checked_count(true_extent, {"mp1: reduction image true extent"});
          const count_t true_upper = checked_count_add(image_true_lb, image_true_extent,
                                                       {"mp1: reduction image true extent"});
          if (image_lb != 0 or image_extent != sizeof_t or image_size != native.size or
              image_true_lb != 0 or image_true_extent != native.true_extent or
              true_upper > sizeof_t) {
            detail::fail_as<std::invalid_argument>(
                {"mp1: reduction callback datatype"}, std::source_location::current(),
                "MPI_Type_create_resized produced an inconsistent image: lb={}, extent={}, "
                "size={}, true_lb={}, true_extent={}; expected lb=0, extent={}, size={}, "
                "true_lb=0, true_extent={}",
                image_lb, image_extent, image_size, image_true_lb, image_true_extent, sizeof_t,
                native.size, native.true_extent);
          }
          check_mpi(MPI_Type_commit(&image), "MPI_Type_commit");
        } catch (...) {
          if (image != MPI_DATATYPE_NULL)
            (void)MPI_Type_free(&image);
          if (shifted != MPI_DATATYPE_NULL)
            (void)MPI_Type_free(&shifted);
          throw;
        }
        return image;
      }();
      return value;
    }

    // A resolved reduction operator: the borrowed MPI handle plus the datatype to pass to MPI.
    //
    // For a predefined functor or a raw MPI_Op handle (predefined or user-created) the
    // datatype is native_datatype<T>(): a caller-supplied callback receives the type map
    // exactly as the caller supplied it, including its true_lb, and must honor that map
    // (MPI 4.1 §6.9.5).  Only an mp1-managed functor callback is typed over T and indexes
    // in/inout as T[len], so it gets reduction_image_datatype<T>() (the native typed map at
    // extent sizeof(T), no MPI_BYTE component).  Resolving the datatype here gives every
    // reduction the same contract and keeps the native call adapters unchanged.
    struct reduction_operation {
      MPI_Op handle{MPI_OP_NULL};
      MPI_Datatype datatype{MPI_DATATYPE_NULL};
      // True when datatype is reduction_image_datatype<T>(): the element image is the native
      // typed map shifted to true lower bound 0, so staging zero-fills its full-T slots for
      // local determinism only; the zeroed padding/unlisted bytes are not part of the wire
      // image.
      bool callback_image{false};
      // Bytes added to every MPI buffer pointer for this operation: the native true lower
      // bound of T for the callback image, zero for the native/predefined/raw paths.  The MPI
      // call adapters apply it; the callback subtracts it again before T indexing.
      count_t buffer_offset{0};
    };

    // reduce_scatter and reduce_scatter_block are the two families where MPICH 4.3.2 (the CI
    // floor) does not lay the callback temporaries out at the datatype extent when the
    // image has unmapped bytes after its true extent (a leading true lower bound or trailing
    // padding).  A managed callback image has extent sizeof(T) by contract, so a T whose native
    // map does not span the whole object would give the callback a corrupt view on that
    // provider.  Reject the combination before MPI instead of silently returning wrong data;
    // the predefined mapping and a raw MPI_Op are unaffected.
    template<class T>
    void require_reduce_scatter_callback_image(const reduction_operation& operation,
                                               const context& ctx) {
      if (not operation.callback_image)
        return;
      const datatype_info& native = datatype_info_of<T>();
      if (native.true_lb != 0 or native.true_extent != static_cast<count_t>(sizeof(T))) {
        throw_invalid(
            ctx.operation,
            "mp1: managed reduce_scatter requires a callback image whose type map spans the "
            "whole T extent; native_datatype<T>() has a nonzero true lower bound or trailing "
            "unmapped bytes, which the floor provider cannot lay out at the callback stride. "
            "Use a predefined operation, a raw MPI_Op, or a T whose listed fields span the "
            "object");
      }
    }

    // Element-aware overload; the reductions call this form.
    template<class T, class Op>
    [[nodiscard]] reduction_operation reduction_op(const Op& op) {
      using F = std::decay_t<Op>;
      if constexpr (std::is_same_v<F, MPI_Op>) {
        if (op == MPI_OP_NULL) {
          throw_invalid("mp1", "MPI_OP_NULL is not a valid reduction operator");
        }
        return reduction_operation{op, native_datatype<T>(), false, 0};
      } else if constexpr (is_predefined_reduction_functor_v<T, F>) {
        return reduction_operation{predefined_mpi_op<T, F>(), native_datatype<T>(), false, 0};
      } else {
        return reduction_operation{mpi_op<T, F>(op), reduction_image_datatype<T>(), true,
                                   reduction_image_buffer_offset<T>()};
      }
    }

  }  // namespace detail

  /// Typed RMA operator: exactly one predefined MPI operation, known to be valid for the
  /// operand type T under the MPI 4.1 §6.9.2 (Predefined Reduction Operations) table.
  ///
  /// T is the **operand type U**, not necessarily the accumulating aggregate: for a homogeneous
  /// derived element type use `predefined_op<basic_component_t<T>>` (for example
  /// `predefined_op<int>` for `std::pair<int,int>`); `predefined_op<std::pair<int,int>>` is a
  /// compile-time error.  The accumulating families spell the parameter through
  /// `detail::accumulate_op_t<T>`, which resolves to `predefined_op<basic_component_t<T>>`.
  ///
  /// Construct implicitly from
  ///   * a built-in reduction functor with an exact predefined mapping
  ///     (`std::plus<T>`, `std::logical_and<bool>`, `std::bit_*<std::byte>`, ...); an
  ///     unmappable functor is a compile-time error; or
  ///   * a raw `MPI_Op` that compares equal to a standard predefined handle and is defined for
  ///     T's datatype category.  A user-created/unknown handle throws std::invalid_argument
  ///     before any MPI call.
  ///
  /// The type is a borrowed, trivially copyable value; it owns nothing and never frees a
  /// handle.  `MPI_NO_OP` is accepted here and rejected by accumulate/raccumulate (MPI 4.1
  /// §12.3.4 (Get Accumulate)); `MPI_REPLACE` is accepted by every accumulating family,
  /// including a homogeneous-derived element type (MPI 4.1 §12.3.4; the historical Annex B
  /// "predefined only" sentence is the withdrawn MPI 2.1 rule).
  template<class T>
  class predefined_op {
    static_assert(is_predefined_datatype_v<T>,
                  "mp1::predefined_op<T> requires a predefined operand type T: T is the MPI "
                  "operand, not an aggregate element type. The accumulating RMA families take "
                  "predefined_op<basic_component_t<T>> (for std::pair<int,int>, "
                  "predefined_op<int>); fetch_and_op and compare_and_swap require a predefined "
                  "datatype and have no derived fallback");

  public:
    using element_type = T;

    predefined_op() = delete;
    predefined_op(std::nullptr_t) = delete;

    // NOLINTNEXTLINE(google-explicit-constructor): preserves the existing RMA call syntax
    predefined_op(MPI_Op op)
        : op_{detail::validate_raw_rma_op<T>(op, true, "mp1::predefined_op")} {
    }

    // NOLINTNEXTLINE(google-explicit-constructor): preserves the existing RMA call syntax
    template<class F>
    requires(not std::same_as<std::remove_cvref_t<F>, MPI_Op>)
    predefined_op(const F&) : op_{detail::predefined_rma_mpi_op<T, std::remove_cvref_t<F>>()} {
      static_assert(detail::is_predefined_rma_functor_v<T, std::remove_cvref_t<F>>,
                    "mp1: RMA only accepts a predefined MPI operation or a built-in "
                    "reduction functor with an exact predefined mapping; user-defined "
                    "functors cannot be used with MPI RMA (MPI 4.1 §12.3.4)");
    }

    /// \return the borrowed native handle; no MPI call
    [[nodiscard]] MPI_Op native_handle() const noexcept {
      return op_;
    }

    /// \return true for MPI_NO_OP; the RMA call still enforces the family rule
    [[nodiscard]] bool is_no_op() const noexcept {
      return op_ == MPI_NO_OP;
    }

  private:
    MPI_Op op_{MPI_OP_NULL};
  };

  namespace detail {

    // Non-constructible diagnostic type used as the op parameter type after a failed
    // basic-component static_assert, so no predefined_op<void> or predefined_op<aggregate> is
    // ever formed for an invalid T.
    struct invalid_accumulate_op {
      invalid_accumulate_op() = delete;
      template<class... Args>
      invalid_accumulate_op(Args&&...) = delete;
    };

    // The checked operator type of the accumulating RMA families: exactly
    // predefined_op<basic_component_t<T>> for a valid T (predefined T or a homogeneous derived
    // T), and a compile-time error otherwise.  T is the element type of the origin/result/
    // target buffers, not necessarily the operator operand type: a homogeneous derived
    // std::pair<int,int> uses predefined_op<int>.
    template<class T, bool = is_homogeneous_datatype_v<T>>
    struct accumulate_op;

    template<class T>
    struct accumulate_op<T, true> {
      // The trait classifies C array members; the public accumulating surface takes a pointer
      // to one element type, so a top-level C array is not supported even though it is
      // homogeneous.
      static_assert(not std::is_array_v<std::remove_cv_t<T>>,
                    "mp1: a C array is not supported as the accumulating element type T; use "
                    "std::array<T, N> or a reflected struct with a C array member");
      using type = predefined_op<basic_component_t<T>>;
    };

    template<class T>
    struct accumulate_op<T, false> {
      // A C array classifies as a member type only; the public accumulating surface takes a
      // pointer to one element type, so a top-level C array is not supported.
      static_assert(not std::is_array_v<std::remove_cv_t<T>>,
                    "mp1: a C array is not supported as the accumulating element type T; use "
                    "std::array<T, N> or a reflected struct with a C array member");
      static_assert(component_info<T>::status != component_status::heterogeneous,
                    "mp1: RMA accumulating families require a predefined T or a T whose basic "
                    "components are all the same predefined datatype (MPI 4.1 §12.3.4); T "
                    "has mixed basic components");
      static_assert(component_info<T>::status != component_status::empty,
                    "mp1: RMA accumulating families require at least one basic component (MPI "
                    "4.1 §12.3.4); T has no basic components");
      static_assert(component_info<T>::status != component_status::unknown,
                    "mp1: RMA accumulating families cannot determine T's basic component; "
                    "provide MP1_DATATYPE_BASIC(T, U) for a custom homogeneous derived trait, "
                    "use a reflected struct/pair/tuple/array, or use native MPI");
      using type = invalid_accumulate_op;
    };

    template<class T>
    using accumulate_op_t = typename accumulate_op<std::remove_cv_t<T>>::type;

  }  // namespace detail

}  // namespace mp1

#endif
