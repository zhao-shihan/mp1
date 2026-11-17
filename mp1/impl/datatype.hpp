#if !(defined MP1_IMPL_DATATYPE_HPP)

#define MP1_IMPL_DATATYPE_HPP

#include "mp1/impl/detail/mpi_call.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>


namespace mp1 {

  namespace detail {

    // Internal datatype categories. Only basic_or_fixed_size_type is produced for supported
    // element types; unsupported_type marks the primary struct_builder fallback so diagnostics
    // point at the missing reflection/customization.
    struct basic_or_fixed_size_type {};
    struct unsupported_type {};

    // Tri-state overlap of a type map's positive-size entry intervals. `disjoint` is proven,
    // `overlap` is proven, and `unknown` is the conservative "not analyzed" state. Only a
    // proven overlap is rejected at an operation boundary; an unknown layout remains the
    // caller's responsibility, exactly as in native MPI (MPI 4.1 §5.1.11 leaves duplicate
    // displacements legal in the type map but a receive with overlapping entries is
    // erroneous).
    enum class overlap_state : unsigned char { disjoint, unknown, overlap };

    // Tri-state order of a type map's displacements. `monotone` is proven (every
    // displacement is nonnegative and nondecreasing in map order), `non_monotone` is proven,
    // and `unknown` marks a handle whose construction facts are not recorded (a custom
    // MP1_DATATYPE_TRAITS mapping of a derived datatype). MPI 4.1 §14.3.1 requires a derived
    // file-view etype to be monotone; only the proven violation is rejected locally.
    enum class type_map_order : unsigned char { monotone, non_monotone, unknown };

    // The C++ categories of the MPI 4.1 §6.9.2 (Predefined Reduction Operations) and
    // §12.3.4 (Compare and Swap) datatype groups, plus Character for MPI_CHAR/MPI_WCHAR.
    // The C-only and Fortran datatype variants (MPI_C_BOOL, MPI_C_*_COMPLEX, MPI_INTEGER,
    // ...) and the multi-language types are out of scope and classify as invalid.
    // The category is a property of the MPI basic datatype, not of a C++ type: enums follow
    // their underlying type, char16_t/char32_t follow the uint_leastN_t datatype they map
    // to, and a custom MP1_DATATYPE_TRAITS mapping is classified from the actual handle
    // (with MPI_Type_get_envelope, §5.1.13) before the first RMA call that needs it.
    enum class rma_datatype_category : unsigned char {
      derived,         // any derived datatype; not a basic predefined type
      c_integer,       // MPI 4.1 §6.9.2 "C integer" (the C++ integers use the same handles)
      floating_point,  // §6.9.2 "Floating point"
      logical,         // §6.9.2 "Logical" (MPI_CXX_BOOL for bool)
      complex,         // §6.9.2 "Complex" (MPI_CXX_*_COMPLEX for std::complex)
      byte,            // §6.9.2 "Byte" (MPI_BYTE)
      character,       // MPI_CHAR / MPI_WCHAR; not a reduction or CAS category (§6.9.3)
      invalid          // not a usable predefined C++ datatype (C-only or Fortran
                       // datatypes, multi-language types, vendor types, ...)
    };

    // Structural facts recorded by the construction algebra. Geometry (size/extent/true
    // extent) is descriptive and never selects a lowering; these facts do.
    struct datatype_facts {
      bool packed{true};
      overlap_state overlap{overlap_state::disjoint};
      // Map-order facts of MPI 4.1 §14.3.1. `last_displacement` is the final typemap
      // displacement and is meaningful only when `order == monotone` and the map is
      // non-empty.
      type_map_order order{type_map_order::monotone};
      count_t last_displacement{0};
    };

    // One-time validated facts of native_datatype<T>().
    struct datatype_info {
      MPI_Datatype type{MPI_DATATYPE_NULL};
      count_t size{0};
      count_t extent{0};
      count_t true_lb{0};
      count_t true_extent{0};
      bool packed{false};
      overlap_state overlap{overlap_state::unknown};
      // MPI 4.1 §14.3.1 etype facts. The first typemap displacement of a monotone map is
      // true_lb; `map_last` is the final one and is meaningful only when order is monotone.
      type_map_order order{type_map_order::unknown};
      count_t map_last{0};
    };

    template<class T>
    [[nodiscard]] const datatype_info& datatype_info_of();

    template<class T>
    [[nodiscard]] inline std::string datatype_type_name() {
#if defined(__GXX_RTTI) || defined(_CPPRTTI)
      return typeid(T).name();
#else
      return "T";
#endif
    }

    template<class>
    inline constexpr bool dependent_false = false;

    inline void free_datatype(MPI_Datatype& type) noexcept {
      (void)MPI_Type_free(&type);
    }

    template<class T>
    struct is_std_array : std::false_type {};

    template<class T, std::size_t N>
    struct is_std_array<std::array<T, N>> : std::true_type {
      using value_type = T;
      static constexpr std::size_t size = N;
    };

    // Compile-time member-type list.  std::pair/std::tuple/std::array and MP1_REFLECTION
    // expose their declared member types through struct_builder<T>::component_types; the
    // basic-component trait classifies that list.  MP1_REFLECTION_IMPL appends a trailing
    // void sentinel so the same X-macro dispatcher works for any member count, and
    // remove_void_t strips it (a data member cannot have type void).
    template<class... Ts>
    struct type_list {};

    template<class T, class List>
    struct prepend_type;

    template<class T, class... Ts>
    struct prepend_type<T, type_list<Ts...>> {
      using type = type_list<T, Ts...>;
    };

    template<class T, class List>
    using prepend_type_t = typename prepend_type<T, List>::type;

    template<class List>
    struct remove_void;

    template<>
    struct remove_void<type_list<>> {
      using type = type_list<>;
    };

    template<class... Ts>
    struct remove_void<type_list<void, Ts...>> {
      using type = type_list<Ts...>;
    };

    template<class T, class... Ts>
    struct remove_void<type_list<T, Ts...>> {
      using type = prepend_type_t<T, typename remove_void<type_list<Ts...>>::type>;
    };

    template<class List>
    using remove_void_t = typename remove_void<List>::type;

  }  // namespace detail

  template<class T>
  struct datatype_traits;

  template<class T>
  // NOLINTNEXTLINE(readability-redundant-declaration): self-contained header
  [[nodiscard]] MPI_Datatype native_datatype();

  //--------------------------------------------------------------------------
  // struct_builder: the single hook used by MP1_REFLECTION and by the
  // built-in std::pair/std::tuple/std::array specializations. The primary
  // template deliberately fails only when a datatype is requested, so a
  // reflected type can be declared before its MP1_REFLECTION specialization.
  //--------------------------------------------------------------------------

  template<class T>
  class struct_builder {
  public:
    using data_type_category = detail::unsupported_type;

    static MPI_Datatype get_datatype() {
      static_assert(detail::dependent_false<T>,
                    "mp1: no datatype_traits<T> specialization and no MP1_REFLECTION for T");
      return MPI_DATATYPE_NULL;
    }
  };

  template<class T>
  class base_struct_builder {
  public:
    using data_type_category = detail::basic_or_fixed_size_type;

    base_struct_builder(const base_struct_builder&) = delete;
    base_struct_builder& operator=(const base_struct_builder&) = delete;

    ~base_struct_builder() {
      detail::release_handle(type_, MPI_DATATYPE_NULL, true, &detail::free_datatype);
    }

    // One committed, resized MPI datatype per aggregate type. get_datatype() and
    // recorded_facts() share the same builder, so the map-order facts always describe the
    // returned handle.
    static MPI_Datatype get_datatype() {
      detail::require_mpi("mp1: reflected datatype construction");
      return instance().handle();
    }

    [[nodiscard]] static detail::datatype_facts recorded_facts() {
      detail::require_mpi("mp1: reflected datatype facts");
      return instance().facts_;
    }

  protected:
    base_struct_builder() = default;

    // One committed, resized MPI datatype. The block/type arrays are borrowed process-lifetime
    // handles for the reflected members; only type_ is owned by this builder.
    MPI_Datatype type_{MPI_DATATYPE_NULL};

    [[nodiscard]] MPI_Datatype handle() const noexcept {
      return type_;
    }

    void register_object(const T& object) {
      detail::check_mpi(MPI_Get_address(std::addressof(object), &base_),
                        {"MPI_Get_address(object)"});
    }

    // Scalars, C arrays and std::array are all contiguous members: register the first element
    // with the total element count. No sizeof-based address arithmetic is performed here.
    template<class M>
    void register_element(M& member) {
      using qualified = std::remove_cv_t<M>;
      if constexpr (detail::is_std_array<qualified>::value) {
        if constexpr (detail::is_std_array<qualified>::size > 0) {
          using element_type =
              std::remove_cv_t<typename detail::is_std_array<qualified>::value_type>;
          const detail::datatype_info& child = detail::datatype_info_of<element_type>();
          add_member(member.data(),
                     static_cast<MPI_Count>(detail::is_std_array<qualified>::size), child);
        }
      } else if constexpr (std::is_array_v<M>) {
        using element_type = std::remove_cv_t<std::remove_all_extents_t<M>>;
        constexpr std::size_t count = sizeof(M) / sizeof(element_type);
        if constexpr (count > 0) {
          const detail::datatype_info& child = detail::datatype_info_of<element_type>();
          add_member(std::addressof(member), static_cast<MPI_Count>(count), child);
        }
      } else {
        const detail::datatype_info& child = detail::datatype_info_of<qualified>();
        add_member(std::addressof(member), 1, child);
      }
    }

    void define_struct() {
      detail::require_mpi("mp1: reflected datatype construction");
      compute_map_facts();

      // MPI requires valid pointers even when the block count is zero; use dummies rather than
      // passing null vector data to a zero-member aggregate.
      const MPI_Count empty_block_length = 0;
      const MPI_Datatype empty_type = MPI_BYTE;
      const MPI_Count* block_lengths =
          block_lengths_.empty() ? &empty_block_length : block_lengths_.data();
      const MPI_Count* displacements =
          displacements_.empty() ? &empty_block_length : displacements_.data();
      const MPI_Datatype* data_types = data_types_.empty() ? &empty_type : data_types_.data();

      MPI_Datatype temporary = MPI_DATATYPE_NULL;
      detail::check_mpi(
          MPI_Type_create_struct_c(static_cast<MPI_Count>(block_lengths_.size()), block_lengths,
                                   displacements, data_types, &temporary),
          {"MPI_Type_create_struct_c"});
      try {
        detail::check_mpi(MPI_Type_commit(&temporary), {"MPI_Type_commit"});
        detail::check_mpi(
            MPI_Type_create_resized(temporary, 0, static_cast<MPI_Aint>(sizeof(T)), &type_),
            {"MPI_Type_create_resized"});
        detail::check_mpi(MPI_Type_commit(&type_), {"MPI_Type_commit"});
      } catch (...) {
        detail::release_handle(type_, MPI_DATATYPE_NULL, true, &detail::free_datatype);
        detail::release_handle(temporary, MPI_DATATYPE_NULL, true, &detail::free_datatype);
        throw;
      }
      detail::release_handle(temporary, MPI_DATATYPE_NULL, true, &detail::free_datatype);
    }

  private:
    static base_struct_builder& instance() {
      static struct_builder<T> value;
      return value;
    }

    void add_member(const void* address, MPI_Count block_length,
                    const detail::datatype_info& child) {
      MPI_Aint member_address{0};
      detail::check_mpi(MPI_Get_address(address, &member_address), {"MPI_Get_address(member)"});
      // MPI 4.1 §5.1.5: address arithmetic must use MPI_AINT_DIFF, never intrinsic `-`.
      // The member and the object are in the same sequential storage, so the difference is
      // defined; the checked helper still reports a non-representable one locally.
      const context ctx{"mp1: reflected datatype"};
      const MPI_Aint relative = detail::checked_address_diff(member_address, base_, ctx);
      if (not std::in_range<MPI_Count>(relative)) {
        detail::fail_as<std::overflow_error>(ctx, std::source_location::current(),
                                             "member displacement does not fit MPI_Count");
      }
      block_lengths_.push_back(block_length);
      displacements_.push_back(static_cast<MPI_Count>(relative));
      data_types_.push_back(child.type);
      child_sizes_.push_back(child.size);
      child_extents_.push_back(child.extent);
      child_true_lbs_.push_back(child.true_lb);
      child_packed_.push_back(child.packed);
      child_orders_.push_back(child.order);
      child_last_displacements_.push_back(child.map_last);
    }

    // Map-order coverage of MPI_Type_create_struct_c: members are listed in parameter order,
    // a blocklength b of a child is b copies at the child's extent stride, and the entries are
    // packed iff each child is packed, the copies tile, and every member starts where the
    // previous one ended. This is the base `packed` fact; it is computed from construction
    // arguments, never from size == true_extent.
    //
    // The same walk records the MPI 4.1 §14.3.1 etype map order: a monotone child at a
    // nonnegative member displacement contributes a monotone run (extent >= true extent, so
    // the copies do not step backwards), and the members must start at or after the previous
    // member's final map displacement. A child whose order is not recorded makes the whole
    // order unknown, never falsely monotone.
    void compute_map_facts() {
      count_t cursor{0};
      bool have_cursor = false;
      bool packed = true;
      detail::type_map_order order = detail::type_map_order::monotone;
      count_t last_displacement{0};
      bool have_entry = false;
      for (std::size_t i = 0; i < block_lengths_.size(); ++i) {
        const count_t blocklength =
            detail::checked_count(block_lengths_[i], "mp1: reflected datatype");
        const count_t child_size = child_sizes_[i];
        if (blocklength == 0 or child_size == 0)
          continue;
        const count_t displacement =
            detail::checked_count(displacements_[i], "mp1: reflected datatype");
        if (packed) {
          if (not child_packed_[i] or (blocklength > 1 and child_extents_[i] != child_size)) {
            packed = false;
          } else {
            const count_t start = detail::checked_count_add(displacement, child_true_lbs_[i],
                                                            "mp1: reflected datatype");
            if (have_cursor and start != cursor) {
              packed = false;
            } else {
              cursor = detail::checked_count_add(
                  have_cursor ? cursor : start,
                  detail::checked_count_mul(blocklength, child_size, "mp1: reflected datatype"),
                  "mp1: reflected datatype");
              have_cursor = true;
            }
          }
        }
        if (order == detail::type_map_order::monotone) {
          if (child_orders_[i] != detail::type_map_order::monotone) {
            order = child_orders_[i];
          } else {
            const count_t first = detail::checked_count_add(displacement, child_true_lbs_[i],
                                                            "mp1: reflected datatype");
            const count_t last = detail::checked_count_add(
                detail::checked_count_add(
                    displacement,
                    detail::checked_count_mul(blocklength - 1, child_extents_[i],
                                              "mp1: reflected datatype"),
                    "mp1: reflected datatype"),
                child_last_displacements_[i], "mp1: reflected datatype");
            if (first < 0 or (have_entry and first < last_displacement)) {
              order = detail::type_map_order::non_monotone;
            } else {
              last_displacement = last;
              have_entry = true;
            }
          }
        }
      }
      facts_.packed = packed;
      facts_.overlap =
          packed ? detail::overlap_state::disjoint : detail::overlap_state::unknown;
      facts_.order = order;
      facts_.last_displacement = last_displacement;
    }

    MPI_Aint base_{0};
    std::vector<MPI_Count> block_lengths_;
    std::vector<MPI_Count> displacements_;
    std::vector<MPI_Datatype> data_types_;
    std::vector<count_t> child_sizes_;
    std::vector<count_t> child_extents_;
    std::vector<count_t> child_true_lbs_;
    std::vector<bool> child_packed_;
    std::vector<detail::type_map_order> child_orders_;
    std::vector<count_t> child_last_displacements_;
    detail::datatype_facts facts_{};
  };

  //--------------------------------------------------------------------------
  // datatype_traits: the single customization point. Basic types return
  // predefined handles, enums borrow their underlying type, and everything
  // else is built by struct_builder (normally by MP1_REFLECTION).
  //--------------------------------------------------------------------------

  namespace detail {

    template<class T, bool = std::is_enum_v<T>>
    struct datatype_category {
      using type = typename struct_builder<T>::data_type_category;
    };

    template<class T>
    struct datatype_category<T, true> {
      using type = typename datatype_traits<std::underlying_type_t<T>>::data_type_category;
    };

  }  // namespace detail

  template<class T>
  struct datatype_traits {
    using data_type_category = typename detail::datatype_category<T>::type;

    // True iff get_datatype() names exactly one MPI predefined datatype. Basic scalar
    // specializations, char16_t/char32_t and the MP1_DATATYPE_TRAITS macro set it;
    // MP1_DATATYPE_TRAITS_PACKED maps a derived handle and deliberately leaves it false.
    // struct_builder (and therefore every MP1_REFLECTION type and built-in aggregate) produces
    // a derived committed datatype and keeps the false default. fetch_and_op and
    // compare_and_swap require this marker; the accumulating families accept a homogeneous
    // derived type and use basic_component_t<T> instead (MPI 4.1 §12.3.4).
    static constexpr bool is_predefined = [] {
      if constexpr (std::is_enum_v<T>) {
        return datatype_traits<std::underlying_type_t<T>>::is_predefined;
      } else {
        return false;
      }
    }();

    static MPI_Datatype get_datatype() {
      if constexpr (std::is_enum_v<T>) {
        return datatype_traits<std::underlying_type_t<T>>::get_datatype();
      } else {
        return struct_builder<T>::get_datatype();
      }
    }
  };

#define MP1_DATATYPE_TRAITS_IMPL(type, mpi_type)                 \
  template<>                                                     \
  struct datatype_traits<type> {                                 \
    using data_type_category = detail::basic_or_fixed_size_type; \
    static constexpr bool byte_contiguous = true;                \
    static constexpr bool is_predefined = true;                  \
    static MPI_Datatype get_datatype() {                         \
      return mpi_type;                                           \
    }                                                            \
  }

  MP1_DATATYPE_TRAITS_IMPL(char, MPI_CHAR);
  MP1_DATATYPE_TRAITS_IMPL(signed char, MPI_SIGNED_CHAR);
  MP1_DATATYPE_TRAITS_IMPL(unsigned char, MPI_UNSIGNED_CHAR);
  MP1_DATATYPE_TRAITS_IMPL(wchar_t, MPI_WCHAR);
  MP1_DATATYPE_TRAITS_IMPL(short, MPI_SHORT);
  MP1_DATATYPE_TRAITS_IMPL(unsigned short, MPI_UNSIGNED_SHORT);
  MP1_DATATYPE_TRAITS_IMPL(int, MPI_INT);
  MP1_DATATYPE_TRAITS_IMPL(unsigned int, MPI_UNSIGNED);
  MP1_DATATYPE_TRAITS_IMPL(long, MPI_LONG);
  MP1_DATATYPE_TRAITS_IMPL(unsigned long, MPI_UNSIGNED_LONG);
  MP1_DATATYPE_TRAITS_IMPL(long long, MPI_LONG_LONG);
  MP1_DATATYPE_TRAITS_IMPL(unsigned long long, MPI_UNSIGNED_LONG_LONG);
  MP1_DATATYPE_TRAITS_IMPL(bool, MPI_CXX_BOOL);
  MP1_DATATYPE_TRAITS_IMPL(float, MPI_FLOAT);
  MP1_DATATYPE_TRAITS_IMPL(double, MPI_DOUBLE);
  MP1_DATATYPE_TRAITS_IMPL(long double, MPI_LONG_DOUBLE);
  MP1_DATATYPE_TRAITS_IMPL(std::byte, MPI_BYTE);
  MP1_DATATYPE_TRAITS_IMPL(std::complex<float>, MPI_CXX_FLOAT_COMPLEX);
  MP1_DATATYPE_TRAITS_IMPL(std::complex<double>, MPI_CXX_DOUBLE_COMPLEX);
  MP1_DATATYPE_TRAITS_IMPL(std::complex<long double>, MPI_CXX_LONG_DOUBLE_COMPLEX);


  template<>
  struct datatype_traits<char16_t> {
    using data_type_category =
        typename datatype_traits<std::uint_least16_t>::data_type_category;
    static constexpr bool byte_contiguous = true;
    static constexpr bool is_predefined = true;

    static MPI_Datatype get_datatype() {
      return datatype_traits<std::uint_least16_t>::get_datatype();
    }
  };

  template<>
  struct datatype_traits<char32_t> {
    using data_type_category =
        typename datatype_traits<std::uint_least32_t>::data_type_category;
    static constexpr bool byte_contiguous = true;
    static constexpr bool is_predefined = true;

    static MPI_Datatype get_datatype() {
      return datatype_traits<std::uint_least32_t>::get_datatype();
    }
  };

  //--------------------------------------------------------------------------
  // Built-in reflected aggregates. These use the same owning builder as
  // MP1_REFLECTION and need no user macro.
  //--------------------------------------------------------------------------

  template<class T1, class T2>
  class struct_builder<std::pair<T1, T2>> : public base_struct_builder<std::pair<T1, T2>> {
  public:
    // The declared member types of the built datatype.  mp1 synthesizes the committed handle
    // from exactly these declarations, so the basic-component trait is authoritative
    // (component_types_trusted) and the accumulating families need no runtime walk.
    using component_types = detail::type_list<T1, T2>;
    static constexpr bool component_types_trusted = true;

    struct_builder() {
      std::pair<T1, T2> object{};
      this->register_object(object);
      this->register_element(object.first);
      this->register_element(object.second);
      this->define_struct();
    }
  };

  template<class... Ts>
  class struct_builder<std::tuple<Ts...>> : public base_struct_builder<std::tuple<Ts...>> {
  public:
    using component_types = detail::type_list<Ts...>;
    static constexpr bool component_types_trusted = true;

    struct_builder() {
      std::tuple<Ts...> object{};
      this->register_object(object);
      std::apply([this](auto&... members) { (this->register_element(members), ...); }, object);
      this->define_struct();
    }
  };

  template<class T, std::size_t N>
  class struct_builder<std::array<T, N>> : public base_struct_builder<std::array<T, N>> {
  public:
    // A zero-length std::array<>, tuple<> and reflected empty struct have no basic components
    // and are ignored inside a containing aggregate; as the top-level accumulating element
    // type they are compile-time errors.
    using component_types =
        std::conditional_t<(N > 0), detail::type_list<T>, detail::type_list<>>;
    static constexpr bool component_types_trusted = true;

    struct_builder() {
      std::array<T, N> object{};
      this->register_object(object);
      this->register_element(object);
      this->define_struct();
    }
  };

  template<class T, std::size_t N>
  class struct_builder<T[N]> : public base_struct_builder<T[N]> {
  public:
    struct_builder() {
      T object[N]{};
      this->register_object(object);
      this->register_element(object);
      this->define_struct();
    }
  };

  template<class T>
  [[nodiscard]] MPI_Datatype native_datatype() {
    return datatype_traits<T>::get_datatype();
  }

  namespace detail {

    // SFINAE markers used by datatype_info_of<T>(). A reflected built-in aggregate records
    // its map facts in base_struct_builder; a trait with an explicit byte_contiguous marker
    // (the built-in scalar traits and MP1_DATATYPE_TRAITS_PACKED) supplies the packed fact
    // directly; any other custom trait falls back to the MPI_COMBINER_NAMED classification.
    template<class T, class = void>
    struct has_recorded_datatype_facts : std::false_type {};

    template<class T>
    struct has_recorded_datatype_facts<
        T, std::void_t<decltype(struct_builder<T>::recorded_facts())>> : std::true_type {};

    template<class T, class = void>
    struct has_byte_contiguous : std::false_type {};

    template<class T>
    struct has_byte_contiguous<T, std::void_t<decltype(datatype_traits<T>::byte_contiguous)>>
        : std::true_type {
      static constexpr bool value = datatype_traits<T>::byte_contiguous;
    };

    [[noreturn]] inline void throw_invalid_base_geometry(const char* name, MPI_Count extent,
                                                         MPI_Count true_lb,
                                                         MPI_Count true_extent,
                                                         std::size_t sizeof_t) {
      detail::fail_as<std::invalid_argument>(
          {"mp1"}, std::source_location::current(),
          "native_datatype<{}>() does not describe one C++ object: MPI extent={}, true_lb={}, "
          "true_extent={}, sizeof(T)={}. MP1_DATATYPE_TRAITS requires "
          "extent(native_datatype<T>()) == sizeof(T) and 0 <= true_lb and true_lb + "
          "true_extent <= sizeof(T); padded members and a non-zero true_lb are legal.",
          name, static_cast<long long>(extent), static_cast<long long>(true_lb),
          static_cast<long long>(true_extent), sizeof_t);
    }

    // Typemap order of a committed handle that has no recorded construction facts. A named
    // datatype is one predefined basic-type entry at displacement 0 and is therefore
    // monotone; a derived handle is left unknown rather than claimed monotone (MPI 4.1
    // §14.3.1). This is one MPI call per T, never per use.
    [[nodiscard]] inline type_map_order classify_type_map_order(MPI_Datatype type) {
      require_mpi("mp1: custom datatype map classification");
      MPI_Count num_integers{0};
      MPI_Count num_addresses{0};
      MPI_Count num_large_counts{0};
      MPI_Count num_datatypes{0};
      int combiner{MPI_COMBINER_NAMED};
      check_mpi(MPI_Type_get_envelope_c(type, &num_integers, &num_addresses, &num_large_counts,
                                        &num_datatypes, &combiner),
                {"MPI_Type_get_envelope_c(native_datatype<T>)"});
      return combiner == MPI_COMBINER_NAMED ? type_map_order::monotone
                                            : type_map_order::unknown;
    }

    template<class T>
    [[nodiscard]] const datatype_info& datatype_info_of() {
      static const datatype_info value = [] {
        datatype_info info;
        info.type = native_datatype<T>();
        const std::string name = datatype_type_name<T>();
        // A custom trait can return MPI_DATATYPE_NULL, for example a failed constructor, a
        // disabled language binding, or a typo.  Every query below is erroneous for a null
        // handle and would abort under the default error handler; fail closed locally.
        if (info.type == MPI_DATATYPE_NULL)
          detail::fail_as<std::invalid_argument>(
              {"mp1"}, std::source_location::current(),
              "native_datatype<{}>() is MPI_DATATYPE_NULL; a datatype mapping must return a "
              "usable handle",
              name);
        MPI_Count size{0};
        MPI_Count lower_bound{0};
        MPI_Count extent{0};
        MPI_Count true_lb{0};
        MPI_Count true_extent{0};
        check_mpi(MPI_Type_size_c(info.type, &size), {"MPI_Type_size_c(native_datatype<T>)"});
        check_mpi(MPI_Type_get_extent_c(info.type, &lower_bound, &extent),
                  {"MPI_Type_get_extent_c(native_datatype<T>)"});
        check_mpi(MPI_Type_get_true_extent_c(info.type, &true_lb, &true_extent),
                  {"MPI_Type_get_true_extent_c(native_datatype<T>)"});
        info.size = checked_count(size, "native_datatype<T> size");
        info.extent = checked_count(extent, "native_datatype<T> extent");
        info.true_lb = checked_count(true_lb, "native_datatype<T> true lower bound");
        info.true_extent = checked_count(true_extent, "native_datatype<T> true extent");
        const count_t sizeof_t = static_cast<count_t>(sizeof(T));
        if (info.extent != sizeof_t)
          throw_invalid_base_geometry(name.c_str(), extent, true_lb, true_extent, sizeof(T));
        const count_t true_upper =
            checked_count_add(info.true_lb, info.true_extent, "native_datatype<T> true extent");
        if (info.true_lb < 0 or info.true_extent < 0 or true_upper > sizeof_t)
          throw_invalid_base_geometry(name.c_str(), extent, true_lb, true_extent, sizeof(T));

        if constexpr (std::is_enum_v<T>) {
          const datatype_info& underlying = datatype_info_of<std::underlying_type_t<T>>();
          info.packed = underlying.packed;
          info.overlap = underlying.overlap;
          info.order = underlying.order;
          info.map_last = underlying.map_last;
        } else if constexpr (has_recorded_datatype_facts<T>::value) {
          const datatype_facts facts = struct_builder<T>::recorded_facts();
          info.packed = facts.packed;
          info.overlap = facts.overlap;
          info.order = facts.order;
          info.map_last = facts.last_displacement;
        } else if constexpr (has_byte_contiguous<T>::value) {
          info.packed = datatype_traits<T>::byte_contiguous;
          info.overlap = info.packed ? overlap_state::disjoint : overlap_state::unknown;
          // The marker records a contiguous byte image, not the typemap order. A predefined
          // named handle is one entry; a custom derived handle stays unknown.
          info.order = classify_type_map_order(info.type);
        } else {
          // A custom trait with no opt-in marker: a predefined (MPI_COMBINER_NAMED) handle is
          // byte-contiguous by definition; any derived handle is conservative false. This is
          // one MPI call per T, never per use (MPI 4.1 §5.1.13).
          require_mpi("mp1: custom datatype classification");
          MPI_Count num_integers{0};
          MPI_Count num_addresses{0};
          MPI_Count num_large_counts{0};
          MPI_Count num_datatypes{0};
          int combiner = MPI_COMBINER_NAMED;
          check_mpi(MPI_Type_get_envelope_c(info.type, &num_integers, &num_addresses,
                                            &num_large_counts, &num_datatypes, &combiner),
                    {"MPI_Type_get_envelope_c(native_datatype<T>)"});
          info.packed = combiner == MPI_COMBINER_NAMED;
          info.overlap = info.packed ? overlap_state::disjoint : overlap_state::unknown;
          info.order = info.packed ? type_map_order::monotone : type_map_order::unknown;
        }
        return info;
      }();
      return value;
    }

    // True when datatype_traits<T> is an explicit user customization (the MP1_DATATYPE_TRAITS
    // and MP1_DATATYPE_TRAITS_PACKED macros).  An explicit mapping on an enum overrides the
    // underlying-type default, so the compile-time category must not be derived from the
    // underlying type; it is classified from the actual handle at run time instead.
    template<class T>
    struct explicit_datatype_traits : std::false_type {};

    // SFINAE access to datatype_traits<T>::is_predefined. A hand-written trait without the
    // marker is conservatively treated as derived, never as predefined.
    template<class T, class = void>
    struct predefined_datatype_marker : std::false_type {};

    template<class T>
    struct predefined_datatype_marker<T,
                                      std::void_t<decltype(datatype_traits<T>::is_predefined)>>
        : std::bool_constant<datatype_traits<T>::is_predefined> {};

    template<class T>
    struct is_std_complex : std::false_type {};

    template<class U>
    struct is_std_complex<std::complex<U>> : std::true_type {};

    // Compile-time category of the C++ element type. `invalid` means "the trait says this is
    // predefined but the C++ type does not name a known basic category" (a custom
    // MP1_DATATYPE_TRAITS mapping); the handle is classified at run time then.
    //
    // std::byte must be tested before std::is_enum_v: it is `enum class byte : unsigned char`,
    // but its MPI datatype is MPI_BYTE -- the §6.9.2 Byte category -- NOT the underlying
    // unsigned char (the C integer category).  Testing the enum branch first made
    // MPI_Byte + MPI_SUM/MAX/LAND reach MPI, which §6.9.2 does not define for Byte; the
    // order below is the fix for that hole.
    template<class T>
    [[nodiscard]] constexpr rma_datatype_category static_rma_category() noexcept {
      using U = std::remove_cv_t<T>;
      if constexpr (std::is_same_v<U, std::byte>) {
        return rma_datatype_category::byte;
      } else if constexpr (std::is_enum_v<U>) {
        // A plain enum borrows its underlying type's category.  An explicit macro mapping
        // overrides that default, so its handle is classified at run time (invalid) instead of
        // guessing the underlying category.
        if constexpr (explicit_datatype_traits<U>::value) {
          return rma_datatype_category::invalid;
        } else {
          return static_rma_category<std::underlying_type_t<U>>();
        }
      } else if constexpr (std::is_same_v<U, char> or std::is_same_v<U, wchar_t>) {
        return rma_datatype_category::character;
      } else if constexpr (std::is_same_v<U, bool>) {
        return rma_datatype_category::logical;
      } else if constexpr (std::is_same_v<U, char16_t> or std::is_same_v<U, char32_t>) {
        // Both map to a uint_leastN_t C integer datatype (datatype_traits specializations).
        return rma_datatype_category::c_integer;
      } else if constexpr (is_std_complex<U>::value) {
        return rma_datatype_category::complex;
      } else if constexpr (std::is_integral_v<U>) {
        return rma_datatype_category::c_integer;
      } else if constexpr (std::is_floating_point_v<U>) {
        return rma_datatype_category::floating_point;
      } else if constexpr (not predefined_datatype_marker<U>::value) {
        return rma_datatype_category::derived;
      } else {
        return rma_datatype_category::invalid;
      }
    }

    // Classify a committed datatype handle. Only a named datatype (MPI_COMBINER_NAMED) can be
    // a predefined basic type; the known basic handles are compared directly because MPI has
    // no "give me the basic type of this named handle" query. This is one MPI call per custom
    // T, cached by the caller, never per operation (MPI 4.1 §5.1.13 Type Envelopes and
    // Contents).
    [[nodiscard]] inline rma_datatype_category classify_predefined_datatype(MPI_Datatype type) {
      if (type == MPI_DATATYPE_NULL)
        return rma_datatype_category::invalid;
      MPI_Count num_integers{0};
      MPI_Count num_addresses{0};
      MPI_Count num_large_counts{0};
      MPI_Count num_datatypes{0};
      int combiner{MPI_COMBINER_NAMED};
      check_mpi(MPI_Type_get_envelope_c(type, &num_integers, &num_addresses, &num_large_counts,
                                        &num_datatypes, &combiner),
                {"MPI_Type_get_envelope_c(predefined datatype)"});
      if (combiner != MPI_COMBINER_NAMED)
        return rma_datatype_category::derived;

      if (type == MPI_BYTE)
        return rma_datatype_category::byte;
      if (type == MPI_CXX_BOOL)
        return rma_datatype_category::logical;
      if (type == MPI_CHAR or type == MPI_WCHAR)
        return rma_datatype_category::character;
      if (type == MPI_FLOAT or type == MPI_DOUBLE or type == MPI_LONG_DOUBLE)
        return rma_datatype_category::floating_point;
      if (type == MPI_CXX_FLOAT_COMPLEX or type == MPI_CXX_DOUBLE_COMPLEX or
          type == MPI_CXX_LONG_DOUBLE_COMPLEX)
        return rma_datatype_category::complex;
      if (type == MPI_INT or type == MPI_LONG or type == MPI_SHORT or
          type == MPI_UNSIGNED_SHORT or type == MPI_UNSIGNED or type == MPI_UNSIGNED_LONG or
          type == MPI_LONG_LONG_INT or type == MPI_UNSIGNED_LONG_LONG or
          type == MPI_SIGNED_CHAR or type == MPI_UNSIGNED_CHAR or type == MPI_INT8_T or
          type == MPI_INT16_T or type == MPI_INT32_T or type == MPI_INT64_T or
          type == MPI_UINT8_T or type == MPI_UINT16_T or type == MPI_UINT32_T or
          type == MPI_UINT64_T)
        return rma_datatype_category::c_integer;
      // A named handle that mp1 does not model: a C or Fortran variant where the C++ datatype
      // is the one mp1 maps (MPI_C_BOOL, MPI_C_*_COMPLEX, MPI_LOGICAL, MPI_REAL,
      // MPI_DOUBLE_PRECISION, MPI_COMPLEX, MPI_INTEGER), a multi-language datatype
      // (MPI_AINT, MPI_OFFSET, MPI_COUNT), or a vendor basic type.  It is a predefined
      // datatype, but not one the C++ typed surface classifies.
      return rma_datatype_category::invalid;
    }

  }  // namespace detail

  /// True iff native_datatype<T>() names exactly one MPI predefined datatype. Basic scalar
  /// specializations and MP1_DATATYPE_TRAITS set it; MP1_DATATYPE_TRAITS_PACKED maps a derived
  /// (byte-contiguous) handle and deliberately leaves it false, because byte-contiguity and
  /// predefinedness are independent facts. fetch_and_op and compare_and_swap require this
  /// marker at compile time and classify the committed handle at run time; the accumulating
  /// families use basic_component_t<T> (which may be a homogeneous derived type) instead
  /// (MPI 4.1 §12.3.4).
  template<class T>
  inline constexpr bool is_predefined_datatype_v =
      detail::predefined_datatype_marker<std::remove_cv_t<T>>::value;

  namespace detail {

    /// Category of T's MPI datatype. Built-in element types are classified at compile time; a
    /// custom MP1_DATATYPE_TRAITS mapping is classified once from its committed handle (one
    /// MPI_Type_get_envelope call per T), and an unknown/derived handle is rejected by the RMA
    /// boundary before the RMA call.
    template<class T>
    [[nodiscard]] rma_datatype_category rma_datatype_category_of() {
      constexpr rma_datatype_category category = static_rma_category<T>();
      if constexpr (category != rma_datatype_category::invalid) {
        return category;
      } else {
        static const rma_datatype_category value =
            classify_predefined_datatype(datatype_info_of<T>().type);
        return value;
      }
    }

  }  // namespace detail

  //--------------------------------------------------------------------------
  // Compile-time basic-component trait.
  //
  // The accumulating RMA families require a predefined datatype or a derived datatype all of
  // whose basic components are the same predefined datatype, and every datatype argument must
  // be built from that same predefined datatype (MPI 4.1 §12.3.4 Accumulate/Get
  // Accumulate).  MPI has no homogeneity query, so the trait classifies T at compile time:
  //
  //   * predefined T is its own basic component (U = T; existing source compatibility);
  //   * std::pair/std::tuple/std::array and MP1_REFLECTION expose their declared member types
  //     through struct_builder<T>::component_types and are combined recursively;
  //   * a C array member recurses into its element type (a C array is supported as a member,
  //     not as the top-level accumulating element type);
  //   * an opaque custom/manual trait opts in with MP1_DATATYPE_BASIC(T, U) (or a manual
  //     struct_builder<T>::component_types); the committed handle is then validated once at
  //     run time before the first accumulating call.
  //
  // Mixed (heterogeneous) and empty aggregates, and opaque custom types without an opt-in,
  // are compile-time errors in the accumulating families (detail::accumulate_op_t in
  // operator.hpp).  The classification of a predefined type is exact: MPI applies the
  // operator to elements of that predefined type (§12.3.4), which for a homogeneous-derived T
  // is exactly U.
  //--------------------------------------------------------------------------

  /// User opt-in for a custom non-predefined homogeneous derived datatype.  Specialize with
  /// `using type = U; static constexpr bool provided = true;` where U is a predefined type, or
  /// use MP1_DATATYPE_BASIC(T, U).  The primary template is never provided.
  template<class T, class = void>
  struct basic_component_traits {
    using type = void;
    static constexpr bool provided = false;
  };

  namespace detail {

    // Canonical identity used to compare C++ types that map to the same MPI basic datatype.
    // `opaque_basic<T>` marks a custom predefined type whose C++ type has no known built-in
    // mapping; it compares equal only to itself.  Enums follow their underlying type;
    // char16_t/char32_t map to the uint_leastN_t datatype they are constructed from
    // (datatype_traits specializations); std::byte stays std::byte (MPI_BYTE, the §6.9.2
    // Byte category) even though it is an enum.
    template<class T>
    struct opaque_basic {};

    template<class T>
    struct is_opaque_basic : std::false_type {};

    template<class T>
    struct is_opaque_basic<opaque_basic<T>> : std::true_type {};

    template<class T, class = void>
    struct identity_of {
      using type = opaque_basic<std::remove_cv_t<T>>;
    };

    template<class T>
    struct identity_of<T, std::enable_if_t<std::is_arithmetic_v<T>>> {
      using type = std::remove_cv_t<T>;
    };

    template<>
    struct identity_of<std::byte> {
      using type = std::byte;
    };

    template<>
    struct identity_of<char16_t> {
      using type = std::uint_least16_t;
    };

    template<>
    struct identity_of<char32_t> {
      using type = std::uint_least32_t;
    };

    template<class T>
    struct identity_of<std::complex<T>> {
      using type = std::complex<T>;
    };

    template<class T>
    struct identity_of<T, std::enable_if_t<std::is_enum_v<T>>> {
      using type = typename identity_of<std::underlying_type_t<T>>::type;
    };

    // True iff struct_builder<T> exposes declared member types (pair/tuple/array, a
    // MP1_REFLECTION type, or a manually written builder).  The primary struct_builder has no
    // component_types, so a plain custom trait is not an aggregate.
    template<class T, class = void>
    struct has_component_types : std::false_type {};

    template<class T>
    struct has_component_types<
        T, std::void_t<typename struct_builder<std::remove_cv_t<T>>::component_types>>
        : std::true_type {};

    // Library-internal trust marker: set only by the pair/tuple/array builders and by
    // MP1_REFLECTION, whose committed datatype is synthesized from the same declarations.
    // A manual builder must not set it; manual annotations are always untrusted and get the
    // one-time validation of require_rma_custom_homogeneous<T>().
    template<class T, class = void>
    struct component_types_trusted : std::false_type {};

    template<class T>
    struct component_types_trusted<
        T, std::void_t<decltype(struct_builder<std::remove_cv_t<T>>::component_types_trusted)>>
        : std::bool_constant<struct_builder<std::remove_cv_t<T>>::component_types_trusted> {};

    template<class T>
    inline constexpr bool component_types_trusted_v =
        component_types_trusted<std::remove_cv_t<T>>::value;

    // True iff the user supplied an opt-in basic_component_traits<T> specialization.
    template<class T, class = void>
    struct has_basic_component_traits : std::false_type {};

    template<class T>
    struct has_basic_component_traits<
        T, std::void_t<decltype(basic_component_traits<std::remove_cv_t<T>>::provided)>>
        : std::bool_constant<basic_component_traits<std::remove_cv_t<T>>::provided> {};

    enum class component_status : unsigned char { empty, homogeneous, heterogeneous, unknown };

    // One node of the classification.  `type` is the common operand type (void unless
    // homogeneous), `identity` the canonical comparison type, and `runtime_validate` true
    // only for an untrusted custom/manual annotation.
    template<component_status Status, class Type, class Identity, bool Validate>
    struct component_value {
      static constexpr component_status status = Status;
      using type = Type;
      using identity = Identity;
      static constexpr bool runtime_validate = Validate;
    };

    // Combine two member classifications.  Unknown absorbs everything (mp1 cannot prove the
    // component); heterogeneous is next.  Empty members contribute no basic components.  Two
    // homogeneous members are compatible when their C++ types or their canonical identities
    // agree; an opaque identity that differs from the other member's identity cannot be
    // proven and yields unknown, while two known different identities are heterogeneous.
    template<class Left, class Right>
    struct merge_component_values {
    private:
      static constexpr bool left_homogeneous = Left::status == component_status::homogeneous;
      static constexpr bool right_homogeneous = Right::status == component_status::homogeneous;
      static constexpr bool both_homogeneous = left_homogeneous and right_homogeneous;
      static constexpr bool one_homogeneous = left_homogeneous != right_homogeneous;
      static constexpr bool same_type =
          both_homogeneous and std::is_same_v<typename Left::type, typename Right::type>;
      static constexpr bool same_identity =
          both_homogeneous and
          std::is_same_v<typename Left::identity, typename Right::identity>;
      static constexpr bool opaque_identity =
          both_homogeneous and (is_opaque_basic<typename Left::identity>::value or
                                is_opaque_basic<typename Right::identity>::value);
      static constexpr bool unknown = Left::status == component_status::unknown or
                                      Right::status == component_status::unknown;
      static constexpr bool heterogeneous = Left::status == component_status::heterogeneous or
                                            Right::status == component_status::heterogeneous;
      static constexpr bool merged_homogeneous =
          both_homogeneous ? (same_type or (same_identity and not opaque_identity))
                           : one_homogeneous;
      // Two homogeneous members with different C++ types and different canonical identities
      // are either provably mixed (both identities are known) or unprovable (an opaque custom
      // predefined identity is involved).
      static constexpr bool opaque_mismatch =
          both_homogeneous and not same_type and not same_identity and opaque_identity;
      static constexpr bool known_mismatch =
          both_homogeneous and not same_type and not same_identity and not opaque_identity;
      using homogeneous_type =
          std::conditional_t<left_homogeneous, typename Left::type, typename Right::type>;
      using homogeneous_identity = std::conditional_t<left_homogeneous, typename Left::identity,
                                                      typename Right::identity>;

    public:
      static constexpr component_status status =
          (unknown or opaque_mismatch)        ? component_status::unknown
          : (heterogeneous or known_mismatch) ? component_status::heterogeneous
          : merged_homogeneous                ? component_status::homogeneous
                                              : component_status::empty;
      using type =
          std::conditional_t<status == component_status::homogeneous,
                             std::conditional_t<both_homogeneous and not same_type,
                                                typename Left::identity, homogeneous_type>,
                             void>;
      using identity = std::conditional_t<status == component_status::homogeneous,
                                          homogeneous_identity, void>;
      static constexpr bool runtime_validate =
          Left::runtime_validate or Right::runtime_validate;
    };

    template<class T>
    struct component_info;

    template<class List>
    struct component_list_info;

    template<>
    struct component_list_info<type_list<>>
        : component_value<component_status::empty, void, void, false> {};

    template<class Head, class... Tail>
    struct component_list_info<type_list<Head, Tail...>>
        : merge_component_values<component_info<std::remove_cv_t<Head>>,
                                 component_list_info<type_list<Tail...>>> {};

    enum class component_kind : unsigned char {
      predefined,
      c_array,
      aggregate,
      opt_in,
      opaque
    };

    template<class T>
    inline constexpr component_kind component_kind_of = [] {
      if constexpr (is_predefined_datatype_v<T>) {
        return component_kind::predefined;
      } else if constexpr (std::is_array_v<T>) {
        return component_kind::c_array;
      } else if constexpr (has_component_types<T>::value) {
        return component_kind::aggregate;
      } else if constexpr (has_basic_component_traits<T>::value) {
        return component_kind::opt_in;
      } else {
        return component_kind::opaque;
      }
    }();

    template<class T, component_kind Kind = component_kind_of<std::remove_cv_t<T>>>
    struct component_info_impl;

    template<class T>
    struct component_info_impl<T, component_kind::predefined>
        : component_value<component_status::homogeneous, std::remove_cv_t<T>,
                          typename identity_of<std::remove_cv_t<T>>::type, false> {};

    // A C array is classified through its element type; a C array as the top-level
    // accumulating element type is rejected separately (accumulate_op<T>).  The extent==0
    // guard is defensive only: GCC/Clang accept `int[0]` as an extension but report
    // std::is_array_v<int[0]> == false, so the zero-length spelling never reaches this
    // specialization as a supported member, and a data member of unknown bound is ill-formed.
    // std::array<T,0> is the supported empty spelling (see struct_builder<std::array<T,0>>).
    template<class T>
    struct component_info_impl<T, component_kind::c_array>
        : std::conditional_t<(std::extent_v<T> == 0),
                             component_value<component_status::empty, void, void, false>,
                             component_info<std::remove_cv_t<std::remove_extent_t<T>>>> {};

    template<class T>
    struct component_info_impl<T, component_kind::aggregate>
        : component_list_info<typename struct_builder<std::remove_cv_t<T>>::component_types> {
    private:
      using members =
          component_list_info<typename struct_builder<std::remove_cv_t<T>>::component_types>;

    public:
      // Only the library-generated pair/tuple/array/MP1_REFLECTION declarations are trusted;
      // everything else (a manual builder) is validated once at run time.
      static constexpr bool runtime_validate =
          (not component_types_trusted_v<T>) or members::runtime_validate;
    };

    template<class T>
    struct component_info_impl<T, component_kind::opt_in> {
    private:
      using component =
          std::remove_cv_t<typename basic_component_traits<std::remove_cv_t<T>>::type>;

    public:
      using type = component;
      using identity = typename identity_of<component>::type;
      static constexpr component_status status = component_status::homogeneous;
      // The committed handle is not synthesized by mp1 from the listed C++ members, so the
      // annotation is a contract validated once before the first accumulating MPI call.
      static constexpr bool runtime_validate = true;
      static_assert(is_predefined_datatype_v<component>,
                    "mp1: MP1_DATATYPE_BASIC(T, U) requires U to be a predefined type; U is "
                    "the operand of the accumulating RMA operator (MPI 4.1 §12.3.4)");
    };

    template<class T>
    struct component_info_impl<T, component_kind::opaque>
        : component_value<component_status::unknown, void, void, false> {};

    template<class T>
    struct component_info : component_info_impl<std::remove_cv_t<T>> {};

  }  // namespace detail

  /// Common basic component of T: T itself for a predefined T, the common predefined operand
  /// type for a homogeneous derived T (for example `int` for `std::pair<int,int>`), and void
  /// if T is mixed, empty, opaque or unsupported.  This is the operand type of the accumulating
  /// RMA operator.
  template<class T>
  using basic_component_t = typename detail::component_info<std::remove_cv_t<T>>::type;

  /// True iff T is predefined or a supported homogeneous derived type (MPI 4.1 §12.3.4).
  /// A C array member is classified; a C array as the top-level accumulating element type is
  /// not supported.
  template<class T>
  inline constexpr bool is_homogeneous_datatype_v =
      detail::component_info<std::remove_cv_t<T>>::status ==
      detail::component_status::homogeneous;

  // True when T has a datatype_traits category: a predefined datatype or a reflected aggregate
  // (including std::pair/std::array). This is the file/p2p data-type gate; unlike
  // std::is_trivially_copyable it accepts types MPI can move even when the C++ copy traits are
  // not trivial, and rejects types without any MPI datatype.
  template<class T>
  inline constexpr bool has_datatype_v =
      not std::is_same_v<typename datatype_traits<
                             std::remove_cv_t<std::remove_reference_t<T>>>::data_type_category,
                         detail::unsupported_type>;

}  // namespace mp1


// MP1_DATATYPE_TRAITS is the customization point for a basic type that maps T to exactly one
// MPI predefined datatype. It sets is_predefined (the compile-time marker consumed by
// is_predefined_datatype_v and the RMA gate); the RMA boundary still classifies the committed
// handle at run time, so a derived handle behind this macro is rejected before the first RMA
// call instead of being trusted. Like the reflection macros it opens namespace mp1 and must be
// used at global scope.
#define MP1_DATATYPE_TRAITS(type, mpi_type)                               \
  namespace mp1 {                                                         \
    template<>                                                            \
    struct datatype_traits<type> {                                        \
      using data_type_category = ::mp1::detail::basic_or_fixed_size_type; \
      static constexpr bool is_predefined = true;                         \
      static MPI_Datatype get_datatype() {                                \
        return mpi_type;                                                  \
      }                                                                   \
    };                                                                    \
    namespace detail {                                                    \
      template<>                                                          \
      struct explicit_datatype_traits<type> : std::true_type {};          \
    }                                                                     \
  }

// MP1_DATATYPE_TRAITS_PACKED is the opt-in form for a custom trait whose datatype is derived
// but byte-contiguous. It declares only byte_contiguous and deliberately does NOT set
// is_predefined: a derived handle is not one predefined datatype, so is_predefined_datatype_v
// stays false.  The accumulating RMA families need the additional basic-component annotation
// MP1_DATATYPE_BASIC below; put/get still accept the mixed type unchanged, and without either
// annotation a derived handle is conservatively classified as not packed and only ever takes a
// staging path.
#define MP1_DATATYPE_TRAITS_PACKED(type, mpi_type)                        \
  namespace mp1 {                                                         \
    template<>                                                            \
    struct datatype_traits<type> {                                        \
      using data_type_category = ::mp1::detail::basic_or_fixed_size_type; \
      static constexpr bool byte_contiguous = true;                       \
      static MPI_Datatype get_datatype() {                                \
        return mpi_type;                                                  \
      }                                                                   \
    };                                                                    \
    namespace detail {                                                    \
      template<>                                                          \
      struct explicit_datatype_traits<type> : std::true_type {};          \
    }                                                                     \
  }

// MP1_DATATYPE_BASIC(T, U) is the opt-in basic-component annotation for a custom
// NON-predefined datatype that mp1 cannot derive from C++ member types
// (MP1_DATATYPE_TRAITS_PACKED or a manual struct_builder).  It asserts that the committed
// native_datatype<T>() is a derived datatype all of whose basic components are the predefined
// datatype U (MPI 4.1 §12.3.4), so the accumulating families take `predefined_op<U>`.  The
// annotation is a contract: because mp1 did not synthesize the handle from the listed C++
// members, the committed datatype is validated once before the first accumulating MPI call and
// a wrong annotation is rejected before the data-movement call (a heterogeneous REPLACE would
// otherwise abort MPICH).  T and U must each be a single preprocessor argument (use a typedef
// for a type containing a comma), and the macro must be visible before the first accumulating
// instantiation of T (a C++ specialization declared later would be an ODR violation).
#define MP1_DATATYPE_BASIC(T, U)                                                        \
  namespace mp1 {                                                                       \
    template<>                                                                          \
    struct basic_component_traits<T> {                                                  \
      using type = U;                                                                   \
      static constexpr bool provided = true;                                            \
      static_assert(                                                                    \
          !::mp1::is_predefined_datatype_v<T>,                                          \
          "mp1: MP1_DATATYPE_BASIC is for a non-predefined custom trait; a predefined " \
          "type is its own basic component");                                           \
      static_assert(                                                                    \
          !::mp1::detail::has_component_types<T>::value,                                \
          "mp1: MP1_DATATYPE_BASIC is for a type without component_types; a reflected " \
          "or pair/tuple/array type already exposes its member types");                 \
      static_assert(::mp1::is_predefined_datatype_v<U>,                                 \
                    "mp1: MP1_DATATYPE_BASIC(T, U): U must be a predefined type");      \
    };                                                                                  \
  }

#undef MP1_DATATYPE_TRAITS_IMPL


// The reflective macros deliberately open namespace mp1, so they must be used at global scope.
// The member list is expanded by the preprocessor and is limited to 119 members
// (MP1_GET_NTH_ARG / MP1_FE_119 below). A 120..127-member list selects the empty
// MP1_REFLECTION_TOO_MANY_MEMBERS/MP1_REFLECTION_TOO_MANY_MEMBERS_TYPES fallbacks, so it
// neither produces an undeclared-member error per member nor a second diagnostic; the
// class-level MP1_REFLECTION_MEMBER_COUNT static_assert names the limit once. The dispatch
// table stops at 127 because the C++17 guaranteed macro-argument budget (256) is then
// exhausted: one direct argument + 127 members + 8 guards + 120 dispatch entries. There is no
// 256-member generation and no homogeneous-type shortcut.
#define MP1_GET_NTH_ARG(                                                                      \
    _0, _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, \
    _20, _21, _22, _23, _24, _25, _26, _27, _28, _29, _30, _31, _32, _33, _34, _35, _36, _37, \
    _38, _39, _40, _41, _42, _43, _44, _45, _46, _47, _48, _49, _50, _51, _52, _53, _54, _55, \
    _56, _57, _58, _59, _60, _61, _62, _63, _64, _65, _66, _67, _68, _69, _70, _71, _72, _73, \
    _74, _75, _76, _77, _78, _79, _80, _81, _82, _83, _84, _85, _86, _87, _88, _89, _90, _91, \
    _92, _93, _94, _95, _96, _97, _98, _99, _100, _101, _102, _103, _104, _105, _106, _107,   \
    _108, _109, _110, _111, _112, _113, _114, _115, _116, _117, _118, _119, _120, _121, _122, \
    _123, _124, _125, _126, _127, N, ...)                                                     \
  N

#define MP1_FE_0(MP1_CALL, ...)
#define MP1_FE_1(MP1_CALL, x) MP1_CALL(x)
#define MP1_FE_2(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_1(MP1_CALL, __VA_ARGS__)
#define MP1_FE_3(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_2(MP1_CALL, __VA_ARGS__)
#define MP1_FE_4(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_3(MP1_CALL, __VA_ARGS__)
#define MP1_FE_5(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_4(MP1_CALL, __VA_ARGS__)
#define MP1_FE_6(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_5(MP1_CALL, __VA_ARGS__)
#define MP1_FE_7(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_6(MP1_CALL, __VA_ARGS__)
#define MP1_FE_8(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_7(MP1_CALL, __VA_ARGS__)
#define MP1_FE_9(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_8(MP1_CALL, __VA_ARGS__)
#define MP1_FE_10(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_9(MP1_CALL, __VA_ARGS__)
#define MP1_FE_11(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_10(MP1_CALL, __VA_ARGS__)
#define MP1_FE_12(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_11(MP1_CALL, __VA_ARGS__)
#define MP1_FE_13(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_12(MP1_CALL, __VA_ARGS__)
#define MP1_FE_14(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_13(MP1_CALL, __VA_ARGS__)
#define MP1_FE_15(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_14(MP1_CALL, __VA_ARGS__)
#define MP1_FE_16(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_15(MP1_CALL, __VA_ARGS__)
#define MP1_FE_17(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_16(MP1_CALL, __VA_ARGS__)
#define MP1_FE_18(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_17(MP1_CALL, __VA_ARGS__)
#define MP1_FE_19(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_18(MP1_CALL, __VA_ARGS__)
#define MP1_FE_20(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_19(MP1_CALL, __VA_ARGS__)
#define MP1_FE_21(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_20(MP1_CALL, __VA_ARGS__)
#define MP1_FE_22(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_21(MP1_CALL, __VA_ARGS__)
#define MP1_FE_23(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_22(MP1_CALL, __VA_ARGS__)
#define MP1_FE_24(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_23(MP1_CALL, __VA_ARGS__)
#define MP1_FE_25(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_24(MP1_CALL, __VA_ARGS__)
#define MP1_FE_26(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_25(MP1_CALL, __VA_ARGS__)
#define MP1_FE_27(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_26(MP1_CALL, __VA_ARGS__)
#define MP1_FE_28(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_27(MP1_CALL, __VA_ARGS__)
#define MP1_FE_29(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_28(MP1_CALL, __VA_ARGS__)
#define MP1_FE_30(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_29(MP1_CALL, __VA_ARGS__)
#define MP1_FE_31(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_30(MP1_CALL, __VA_ARGS__)
#define MP1_FE_32(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_31(MP1_CALL, __VA_ARGS__)
#define MP1_FE_33(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_32(MP1_CALL, __VA_ARGS__)
#define MP1_FE_34(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_33(MP1_CALL, __VA_ARGS__)
#define MP1_FE_35(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_34(MP1_CALL, __VA_ARGS__)
#define MP1_FE_36(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_35(MP1_CALL, __VA_ARGS__)
#define MP1_FE_37(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_36(MP1_CALL, __VA_ARGS__)
#define MP1_FE_38(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_37(MP1_CALL, __VA_ARGS__)
#define MP1_FE_39(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_38(MP1_CALL, __VA_ARGS__)
#define MP1_FE_40(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_39(MP1_CALL, __VA_ARGS__)
#define MP1_FE_41(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_40(MP1_CALL, __VA_ARGS__)
#define MP1_FE_42(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_41(MP1_CALL, __VA_ARGS__)
#define MP1_FE_43(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_42(MP1_CALL, __VA_ARGS__)
#define MP1_FE_44(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_43(MP1_CALL, __VA_ARGS__)
#define MP1_FE_45(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_44(MP1_CALL, __VA_ARGS__)
#define MP1_FE_46(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_45(MP1_CALL, __VA_ARGS__)
#define MP1_FE_47(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_46(MP1_CALL, __VA_ARGS__)
#define MP1_FE_48(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_47(MP1_CALL, __VA_ARGS__)
#define MP1_FE_49(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_48(MP1_CALL, __VA_ARGS__)
#define MP1_FE_50(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_49(MP1_CALL, __VA_ARGS__)
#define MP1_FE_51(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_50(MP1_CALL, __VA_ARGS__)
#define MP1_FE_52(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_51(MP1_CALL, __VA_ARGS__)
#define MP1_FE_53(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_52(MP1_CALL, __VA_ARGS__)
#define MP1_FE_54(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_53(MP1_CALL, __VA_ARGS__)
#define MP1_FE_55(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_54(MP1_CALL, __VA_ARGS__)
#define MP1_FE_56(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_55(MP1_CALL, __VA_ARGS__)
#define MP1_FE_57(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_56(MP1_CALL, __VA_ARGS__)
#define MP1_FE_58(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_57(MP1_CALL, __VA_ARGS__)
#define MP1_FE_59(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_58(MP1_CALL, __VA_ARGS__)
#define MP1_FE_60(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_59(MP1_CALL, __VA_ARGS__)
#define MP1_FE_61(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_60(MP1_CALL, __VA_ARGS__)
#define MP1_FE_62(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_61(MP1_CALL, __VA_ARGS__)
#define MP1_FE_63(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_62(MP1_CALL, __VA_ARGS__)
#define MP1_FE_64(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_63(MP1_CALL, __VA_ARGS__)
#define MP1_FE_65(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_64(MP1_CALL, __VA_ARGS__)
#define MP1_FE_66(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_65(MP1_CALL, __VA_ARGS__)
#define MP1_FE_67(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_66(MP1_CALL, __VA_ARGS__)
#define MP1_FE_68(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_67(MP1_CALL, __VA_ARGS__)
#define MP1_FE_69(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_68(MP1_CALL, __VA_ARGS__)
#define MP1_FE_70(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_69(MP1_CALL, __VA_ARGS__)
#define MP1_FE_71(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_70(MP1_CALL, __VA_ARGS__)
#define MP1_FE_72(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_71(MP1_CALL, __VA_ARGS__)
#define MP1_FE_73(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_72(MP1_CALL, __VA_ARGS__)
#define MP1_FE_74(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_73(MP1_CALL, __VA_ARGS__)
#define MP1_FE_75(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_74(MP1_CALL, __VA_ARGS__)
#define MP1_FE_76(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_75(MP1_CALL, __VA_ARGS__)
#define MP1_FE_77(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_76(MP1_CALL, __VA_ARGS__)
#define MP1_FE_78(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_77(MP1_CALL, __VA_ARGS__)
#define MP1_FE_79(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_78(MP1_CALL, __VA_ARGS__)
#define MP1_FE_80(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_79(MP1_CALL, __VA_ARGS__)
#define MP1_FE_81(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_80(MP1_CALL, __VA_ARGS__)
#define MP1_FE_82(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_81(MP1_CALL, __VA_ARGS__)
#define MP1_FE_83(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_82(MP1_CALL, __VA_ARGS__)
#define MP1_FE_84(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_83(MP1_CALL, __VA_ARGS__)
#define MP1_FE_85(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_84(MP1_CALL, __VA_ARGS__)
#define MP1_FE_86(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_85(MP1_CALL, __VA_ARGS__)
#define MP1_FE_87(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_86(MP1_CALL, __VA_ARGS__)
#define MP1_FE_88(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_87(MP1_CALL, __VA_ARGS__)
#define MP1_FE_89(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_88(MP1_CALL, __VA_ARGS__)
#define MP1_FE_90(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_89(MP1_CALL, __VA_ARGS__)
#define MP1_FE_91(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_90(MP1_CALL, __VA_ARGS__)
#define MP1_FE_92(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_91(MP1_CALL, __VA_ARGS__)
#define MP1_FE_93(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_92(MP1_CALL, __VA_ARGS__)
#define MP1_FE_94(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_93(MP1_CALL, __VA_ARGS__)
#define MP1_FE_95(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_94(MP1_CALL, __VA_ARGS__)
#define MP1_FE_96(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_95(MP1_CALL, __VA_ARGS__)
#define MP1_FE_97(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_96(MP1_CALL, __VA_ARGS__)
#define MP1_FE_98(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_97(MP1_CALL, __VA_ARGS__)
#define MP1_FE_99(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_98(MP1_CALL, __VA_ARGS__)
#define MP1_FE_100(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_99(MP1_CALL, __VA_ARGS__)
#define MP1_FE_101(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_100(MP1_CALL, __VA_ARGS__)
#define MP1_FE_102(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_101(MP1_CALL, __VA_ARGS__)
#define MP1_FE_103(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_102(MP1_CALL, __VA_ARGS__)
#define MP1_FE_104(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_103(MP1_CALL, __VA_ARGS__)
#define MP1_FE_105(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_104(MP1_CALL, __VA_ARGS__)
#define MP1_FE_106(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_105(MP1_CALL, __VA_ARGS__)
#define MP1_FE_107(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_106(MP1_CALL, __VA_ARGS__)
#define MP1_FE_108(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_107(MP1_CALL, __VA_ARGS__)
#define MP1_FE_109(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_108(MP1_CALL, __VA_ARGS__)
#define MP1_FE_110(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_109(MP1_CALL, __VA_ARGS__)
#define MP1_FE_111(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_110(MP1_CALL, __VA_ARGS__)
#define MP1_FE_112(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_111(MP1_CALL, __VA_ARGS__)
#define MP1_FE_113(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_112(MP1_CALL, __VA_ARGS__)
#define MP1_FE_114(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_113(MP1_CALL, __VA_ARGS__)
#define MP1_FE_115(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_114(MP1_CALL, __VA_ARGS__)
#define MP1_FE_116(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_115(MP1_CALL, __VA_ARGS__)
#define MP1_FE_117(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_116(MP1_CALL, __VA_ARGS__)
#define MP1_FE_118(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_117(MP1_CALL, __VA_ARGS__)
#define MP1_FE_119(MP1_CALL, x, ...) MP1_CALL(x) MP1_FE_118(MP1_CALL, __VA_ARGS__)

// Selected instead of an MP1_FE_* dispatcher entry when a member list has 120..127 members.
// It expands to nothing so the constructor body stays syntactically valid and the single
// class-level MP1_REFLECTION_MEMBER_COUNT static_assert below is the only diagnostic.
#define MP1_REFLECTION_TOO_MANY_MEMBERS(MP1_CALL, ...)

// Type-context variant of the limit diagnostic.  It expands to nothing as well; for a
// 120..127-member list the member-type expansion is empty, and the class-level member-count
// static_assert names the limit without a duplicate diagnostic.
#define MP1_REFLECTION_TOO_MANY_MEMBERS_TYPES(MP1_CALL, ...)

// Shared dispatch table.  FALLBACK is selected for 120..127 members: the empty expansion in
// statement context (MP1_REFLECTION_TOO_MANY_MEMBERS) and in member-type-list context
// (MP1_REFLECTION_TOO_MANY_MEMBERS_TYPES); the class-level count assert reports the limit.
#define MP1_CALL_MACRO_X_FOR_EACH_IMPL(x, FALLBACK, ...)                                       \
  MP1_GET_NTH_ARG(                                                                             \
      "ignored", __VA_ARGS__, FALLBACK, FALLBACK, FALLBACK, FALLBACK, FALLBACK, FALLBACK,      \
      FALLBACK, FALLBACK, MP1_FE_119, MP1_FE_118, MP1_FE_117, MP1_FE_116, MP1_FE_115,          \
      MP1_FE_114, MP1_FE_113, MP1_FE_112, MP1_FE_111, MP1_FE_110, MP1_FE_109, MP1_FE_108,      \
      MP1_FE_107, MP1_FE_106, MP1_FE_105, MP1_FE_104, MP1_FE_103, MP1_FE_102, MP1_FE_101,      \
      MP1_FE_100, MP1_FE_99, MP1_FE_98, MP1_FE_97, MP1_FE_96, MP1_FE_95, MP1_FE_94, MP1_FE_93, \
      MP1_FE_92, MP1_FE_91, MP1_FE_90, MP1_FE_89, MP1_FE_88, MP1_FE_87, MP1_FE_86, MP1_FE_85,  \
      MP1_FE_84, MP1_FE_83, MP1_FE_82, MP1_FE_81, MP1_FE_80, MP1_FE_79, MP1_FE_78, MP1_FE_77,  \
      MP1_FE_76, MP1_FE_75, MP1_FE_74, MP1_FE_73, MP1_FE_72, MP1_FE_71, MP1_FE_70, MP1_FE_69,  \
      MP1_FE_68, MP1_FE_67, MP1_FE_66, MP1_FE_65, MP1_FE_64, MP1_FE_63, MP1_FE_62, MP1_FE_61,  \
      MP1_FE_60, MP1_FE_59, MP1_FE_58, MP1_FE_57, MP1_FE_56, MP1_FE_55, MP1_FE_54, MP1_FE_53,  \
      MP1_FE_52, MP1_FE_51, MP1_FE_50, MP1_FE_49, MP1_FE_48, MP1_FE_47, MP1_FE_46, MP1_FE_45,  \
      MP1_FE_44, MP1_FE_43, MP1_FE_42, MP1_FE_41, MP1_FE_40, MP1_FE_39, MP1_FE_38, MP1_FE_37,  \
      MP1_FE_36, MP1_FE_35, MP1_FE_34, MP1_FE_33, MP1_FE_32, MP1_FE_31, MP1_FE_30, MP1_FE_29,  \
      MP1_FE_28, MP1_FE_27, MP1_FE_26, MP1_FE_25, MP1_FE_24, MP1_FE_23, MP1_FE_22, MP1_FE_21,  \
      MP1_FE_20, MP1_FE_19, MP1_FE_18, MP1_FE_17, MP1_FE_16, MP1_FE_15, MP1_FE_14, MP1_FE_13,  \
      MP1_FE_12, MP1_FE_11, MP1_FE_10, MP1_FE_9, MP1_FE_8, MP1_FE_7, MP1_FE_6, MP1_FE_5,       \
      MP1_FE_4, MP1_FE_3, MP1_FE_2, MP1_FE_1, MP1_FE_0)(x, __VA_ARGS__)

#define MP1_CALL_MACRO_X_FOR_EACH(x, ...) \
  MP1_CALL_MACRO_X_FOR_EACH_IMPL(x, MP1_REFLECTION_TOO_MANY_MEMBERS, __VA_ARGS__)

#define MP1_CALL_MACRO_X_FOR_EACH_TYPES(x, ...) \
  MP1_CALL_MACRO_X_FOR_EACH_IMPL(x, MP1_REFLECTION_TOO_MANY_MEMBERS_TYPES, __VA_ARGS__)

// Number of trailing macro arguments, 0..127.  Passing "ignored" + k arguments + the numbers
// 127..-1 puts k in the MP1_GET_NTH_ARG N slot.
#define MP1_REFLECTION_MEMBER_COUNT(...)                                                      \
  MP1_GET_NTH_ARG("ignored", __VA_ARGS__, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118,   \
                  117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103,  \
                  102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85,  \
                  84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, \
                  65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, \
                  46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, \
                  27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  \
                  8, 7, 6, 5, 4, 3, 2, 1, 0, -1)

#define MP1_UNWRAP_IMPL(...) __VA_ARGS__
#define MP1_UNWRAP(...) MP1_UNWRAP_IMPL __VA_ARGS__

#define MP1_REFLECTION_MEMBER(member) this->register_element(object.member);

// The member-type extraction used by the basic-component trait.  `mp1_reflection_object` is the
// declared parameter name, and a parameter name is in scope in its own trailing return type, so
// decltype(mp1_reflection_object.member) yields the declared member type without specializing
// on STRUCT_TYPE inside the per-member macro.  The helper is declared but never defined or
// called (decltype is an unevaluated context), so it creates no ODR-use or link symbol.  The
// void sentinel after the expansion is stripped by detail::remove_void_t; a data member cannot
// have type void, so it cannot collide with a real member.
#define MP1_REFLECTION_COMPONENT(member) decltype(mp1_reflection_object.member),

#define MP1_REFLECTION_IMPL(TEMPLATE_HEADER, STRUCT_TYPE, MEMBER_LIST, ...)                  \
  namespace mp1 {                                                                            \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses): declaration context */                    \
    TEMPLATE_HEADER                                                                          \
    class struct_builder<STRUCT_TYPE> : public base_struct_builder<STRUCT_TYPE> {            \
    public:                                                                                  \
      static_assert(                                                                         \
          MP1_REFLECTION_MEMBER_COUNT(__VA_ARGS__) <= 119,                                   \
          "mp1: MP1_REFLECTION/MP1_REFLECTION_TEMPLATE supports at most 119 members; "       \
          "split the aggregate");                                                            \
      static auto component_types_impl(STRUCT_TYPE& mp1_reflection_object)                   \
          -> ::mp1::detail::remove_void_t<::mp1::detail::type_list<                          \
              MP1_CALL_MACRO_X_FOR_EACH_TYPES(MP1_REFLECTION_COMPONENT, __VA_ARGS__) void>>; \
      using component_types = decltype(component_types_impl(std::declval<STRUCT_TYPE&>()));  \
      static constexpr bool component_types_trusted = true;                                  \
      struct_builder() {                                                                     \
        STRUCT_TYPE object{};                                                                \
        this->register_object(object);                                                       \
        /* NOLINTNEXTLINE(bugprone-macro-parentheses): declaration context */                \
        MEMBER_LIST                                                                          \
        this->define_struct();                                                               \
      }                                                                                      \
    };                                                                                       \
  }

#define MP1_REFLECTION_EMPTY_IMPL(TEMPLATE_HEADER, STRUCT_TYPE)                   \
  namespace mp1 {                                                                 \
    /* NOLINTNEXTLINE(bugprone-macro-parentheses): declaration context */         \
    TEMPLATE_HEADER                                                               \
    class struct_builder<STRUCT_TYPE> : public base_struct_builder<STRUCT_TYPE> { \
    public:                                                                       \
      using component_types = ::mp1::detail::type_list<>;                         \
      static constexpr bool component_types_trusted = true;                       \
      struct_builder() {                                                          \
        STRUCT_TYPE object{};                                                     \
        this->register_object(object);                                            \
        this->define_struct();                                                    \
      }                                                                           \
    };                                                                            \
  }

#define MP1_REFLECTION(STRUCT, ...)                                                  \
  MP1_REFLECTION_IMPL(template<>, STRUCT,                                            \
                      MP1_CALL_MACRO_X_FOR_EACH(MP1_REFLECTION_MEMBER, __VA_ARGS__), \
                      __VA_ARGS__)

#define MP1_REFLECTION_EMPTY(STRUCT) MP1_REFLECTION_EMPTY_IMPL(template<>, STRUCT)

#define MP1_REFLECTION_TEMPLATE(TEMPLATE_PARAMS, STRUCT_TYPE, ...)                    \
  MP1_REFLECTION_IMPL(template<MP1_UNWRAP(TEMPLATE_PARAMS)>, MP1_UNWRAP(STRUCT_TYPE), \
                      MP1_CALL_MACRO_X_FOR_EACH(MP1_REFLECTION_MEMBER, __VA_ARGS__),  \
                      __VA_ARGS__)

#endif
