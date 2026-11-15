// Contract test for MPI-legal homogeneous-derived datatypes in the accumulating RMA families
// (MPI 4.1 §12.3.4 Accumulate/Get Accumulate; §12.3.5 request-based RMA; §12.7.1
// element-wise atomicity).
//
// The operator type of every accumulating overload is predefined_op<U>, where U is
// basic_component_t<T>: U = T for a predefined T and the common predefined operand for a
// homogeneous derived T (std::pair<int,int> uses predefined_op<int>).  Mixed and empty
// aggregates and opaque custom types are compile-time errors (test/compile_fail), so this file
// has no runtime test for them.  A custom MP1_DATATYPE_BASIC(T,U) annotation is a contract and
// is validated once against the committed datatype before the first accumulating MPI call; a
// wrong annotation must throw here, before the data-movement call, so MPICH's
// MPIR_DATATYPE_IS_PREDEFINED assertion is never reached (the same input
// makes Open MPI return MPI_ERR_TYPE when sent through raw MPI).
//
// The test runs at 2 and 4 ranks.  Most accumulating calls target the calling rank's own
// window segment, so target values are deterministic without inter-rank races; one extra case
// accumulates concurrently to rank 0 to exercise element-wise atomicity across ranks.
//
// accumulate_ops note (contract documentation, MPI 4.1 §12.2.1): the window info key defaults
// to same_op_no_op, so concurrent accumulates to the same basic element must use the same
// operator or MPI_NO_OP; atomicity is per basic element, never per C++ T object (§12.7.1).

#include "mp1/mp1.hpp"

#include <array>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#if defined(MP1_TEST_WRAP) && MP1_TEST_WRAP
#include "mpi_wrap.hpp"
#define MP1_HOMOGENEOUS_WRAP 1
#else
#define MP1_HOMOGENEOUS_WRAP 0
#endif


namespace {

  // ---- types under test -------------------------------------------------------------------

  enum class rma_enum : int { zero = 0, one = 1 };

  struct refl_pair {
    int a{0};
    int b{0};
  };

  bool operator==(const refl_pair& lhs, const refl_pair& rhs) {
    return lhs.a == rhs.a and lhs.b == rhs.b;
  }

  struct nested_refl {
    refl_pair inner{};
    int tail{0};
  };

  bool operator==(const nested_refl& lhs, const nested_refl& rhs) {
    return lhs.inner == rhs.inner and lhs.tail == rhs.tail;
  }

  struct array_member {
    int values[3]{};
  };

  bool operator==(const array_member& lhs, const array_member& rhs) {
    return lhs.values[0] == rhs.values[0] and lhs.values[1] == rhs.values[1] and
           lhs.values[2] == rhs.values[2];
  }

  // Two mapped int members followed by an unregistered tail: native_datatype<padded_pair>()
  // resizes the struct to sizeof(T) but its type map contains only the two ints, so the
  // accumulating call must leave the tail bytes untouched (MPI 4.1 §5.1.7/§5.1.8: the true
  // extent of the mapped type is 8; the 16-byte extent only sets the repetition stride).
  struct padded_pair {
    int a{0};
    int b{0};
    unsigned char tail[8]{};
  };

  // Mixed derived type used only by put/get (which have no homogeneity rule).
  struct mixed_value {
    int i{0};
    double d{0.};
  };


  // ---- custom datatypes for the untrusted-annotation cases ---------------------------------

  std::vector<MPI_Datatype>& custom_datatypes() {
    static std::vector<MPI_Datatype> types;
    return types;
  }

  // Packed pair annotation: a committed contiguous(2, MPI_INT) is homogeneous over MPI_INT.
  struct packed_pair {
    int a{0};
    int b{0};
  };

  bool operator==(const packed_pair& lhs, const packed_pair& rhs) {
    return lhs.a == rhs.a and lhs.b == rhs.b;
  }

  // The same contract built with the MPI-4 large-count constructor: the validator must read
  // the count from array_of_large_counts instead of array_of_integers.
  struct packed_pair_large {
    int a{0};
    int b{0};
  };

  bool operator==(const packed_pair_large& lhs, const packed_pair_large& rhs) {
    return lhs.a == rhs.a and lhs.b == rhs.b;
  }

  MPI_Datatype make_contiguous_two_ints() {
    static MPI_Datatype type = [] {
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_contiguous(2, MPI_INT, &created) == MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Type_commit(&created) == MPI_SUCCESS);
      custom_datatypes().push_back(created);
      return created;
    }();
    return type;
  }

  MPI_Datatype make_contiguous_two_ints_large() {
    static MPI_Datatype type = [] {
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_contiguous_c(2, MPI_INT, &created) == MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Type_commit(&created) == MPI_SUCCESS);
      custom_datatypes().push_back(created);
      return created;
    }();
    return type;
  }

  // A wrongly annotated type: the committed datatype is a heterogeneous {int,double} struct
  // resized to sizeof(T), but MP1_DATATYPE_BASIC claims the basic component int.
  struct wrong_mixed {
    int i{0};
    double d{0.};
  };

  MPI_Datatype make_wrong_mixed_datatype() {
    static MPI_Datatype type = [] {
      wrong_mixed object{};
      MPI_Aint base = 0;
      MPI_Aint d_address = 0;
      BOOST_REQUIRE(MPI_Get_address(&object, &base) == MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Get_address(&object.d, &d_address) == MPI_SUCCESS);
      int blocklengths[2] = {1, 1};
      MPI_Aint displacements[2] = {0, d_address - base};
      MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
      MPI_Datatype structure = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_struct(2, blocklengths, displacements, types, &structure) ==
                    MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Type_commit(&structure) == MPI_SUCCESS);
      MPI_Datatype resized = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_resized(structure, 0,
                                            static_cast<MPI_Aint>(sizeof(wrong_mixed)),
                                            &resized) == MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Type_commit(&resized) == MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Type_free(&structure) == MPI_SUCCESS);
      custom_datatypes().push_back(resized);
      return resized;
    }();
    return type;
  }

  // A wrongly annotated custom member inside a reflected aggregate: the outer member list looks
  // homogeneous over int, but the untrusted-member validation must propagate and reject it.
  struct outer_wrong {
    wrong_mixed inner{};
    int tail{0};
  };

  // ---- custom datatypes for the walk/combiner matrix ---------------------------------------
  //
  // Every type below is a distinct C++ struct (so its per-T one-time validation is independent)
  // whose committed datatype is built with a different MPI constructor.  The validator must
  // accept the homogeneous constructors and fail closed on the unmodeled ones.

  MPI_Datatype commit_and_own(MPI_Datatype type) {
    BOOST_REQUIRE(MPI_Type_commit(&type) == MPI_SUCCESS);
    custom_datatypes().push_back(type);
    return type;
  }

  // A committed child whose extent is smaller than sizeof(T) is resized to one C++ object; the
  // intermediate handle is freed and only the final handle is owned by the fixture.
  MPI_Datatype resize_committed_and_own(MPI_Datatype child, std::size_t size) {
    MPI_Datatype resized = MPI_DATATYPE_NULL;
    BOOST_REQUIRE(MPI_Type_create_resized(child, 0, static_cast<MPI_Aint>(size), &resized) ==
                  MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Type_commit(&resized) == MPI_SUCCESS);
    BOOST_REQUIRE(MPI_Type_free(&child) == MPI_SUCCESS);
    custom_datatypes().push_back(resized);
    return resized;
  }

  // Own a constructor result only when it is a new derived handle; a named/predefined or F90
  // handle is borrowed (MPI 4.1 §5.1.13) and must not be freed.
  MPI_Datatype own_if_derived(MPI_Datatype type) {
    if (type == MPI_DATATYPE_NULL)
      return type;
    MPI_Count num_integers = 0;
    MPI_Count num_addresses = 0;
    MPI_Count num_large_counts = 0;
    MPI_Count num_datatypes = 0;
    int combiner = MPI_COMBINER_NAMED;
    BOOST_REQUIRE(MPI_Type_get_envelope_c(type, &num_integers, &num_addresses,
                                          &num_large_counts, &num_datatypes,
                                          &combiner) == MPI_SUCCESS);
    if (combiner != MPI_COMBINER_NAMED)
      custom_datatypes().push_back(type);
    return type;
  }

#define MP1_TEST_WALK_PAIR(Name)                                       \
  struct Name {                                                        \
    int first{0};                                                      \
    int second{0};                                                     \
  };                                                                   \
  [[maybe_unused]] bool operator==(const Name& lhs, const Name& rhs) { \
    return lhs.first == rhs.first and lhs.second == rhs.second;        \
  }

  MP1_TEST_WALK_PAIR(dup_pair)
  MP1_TEST_WALK_PAIR(vector_pair)
  MP1_TEST_WALK_PAIR(hvector_pair)
  MP1_TEST_WALK_PAIR(indexed_pair)
  MP1_TEST_WALK_PAIR(hindexed_pair)
  MP1_TEST_WALK_PAIR(indexed_block_pair)
  MP1_TEST_WALK_PAIR(hindexed_block_pair)
  MP1_TEST_WALK_PAIR(struct_pair)
  MP1_TEST_WALK_PAIR(subarray_pair)
  MP1_TEST_WALK_PAIR(resized_pair)
  MP1_TEST_WALK_PAIR(zero_child_pair)
  MP1_TEST_WALK_PAIR(only_zero_pair)
  MP1_TEST_WALK_PAIR(cache_pair)
  MP1_TEST_WALK_PAIR(manual_pair_runtime)
  MP1_TEST_WALK_PAIR(darray_pair)

  // A non-byte-contiguous vector map: two ints at byte offsets 0 and 8, with the middle int
  // outside the type map.  The manual datatype_traits specialization declares no
  // byte_contiguous marker, which is the conservative spelling for a vector/indexed handle.
  struct gapped_vector {
    int first{0};
    int middle{0};
    int last{0};
  };

  bool operator==(const gapped_vector& lhs, const gapped_vector& rhs) {
    return lhs.first == rhs.first and lhs.middle == rhs.middle and lhs.last == rhs.last;
  }

  MPI_Datatype make_dup_pair() {
    static MPI_Datatype type = [] {
      MPI_Datatype duplicate = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_dup(MPI_INT, &duplicate) == MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Type_commit(&duplicate) == MPI_SUCCESS);
      return resize_committed_and_own(duplicate, sizeof(dup_pair));
    }();
    return type;
  }

  MPI_Datatype make_vector_pair() {
    static MPI_Datatype type = [] {
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_vector(2, 1, 1, MPI_INT, &created) == MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  MPI_Datatype make_hvector_pair() {
    static MPI_Datatype type = [] {
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_hvector(2, 1, static_cast<MPI_Aint>(sizeof(int)), MPI_INT,
                                            &created) == MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  MPI_Datatype make_indexed_pair() {
    static MPI_Datatype type = [] {
      const int blocklengths[2] = {1, 1};
      const int displacements[2] = {0, 1};
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_indexed(2, blocklengths, displacements, MPI_INT, &created) ==
                    MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  MPI_Datatype make_hindexed_pair() {
    static MPI_Datatype type = [] {
      const int blocklengths[2] = {1, 1};
      const MPI_Aint displacements[2] = {0, static_cast<MPI_Aint>(sizeof(int))};
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_hindexed(2, blocklengths, displacements, MPI_INT,
                                             &created) == MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  MPI_Datatype make_indexed_block_pair() {
    static MPI_Datatype type = [] {
      const int displacements[2] = {0, 1};
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_indexed_block(2, 1, displacements, MPI_INT, &created) ==
                    MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  MPI_Datatype make_hindexed_block_pair() {
    static MPI_Datatype type = [] {
      const MPI_Aint displacements[2] = {0, static_cast<MPI_Aint>(sizeof(int))};
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_hindexed_block(2, 1, displacements, MPI_INT, &created) ==
                    MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  MPI_Datatype make_struct_pair() {
    static MPI_Datatype type = [] {
      const int blocklengths[2] = {1, 1};
      const MPI_Aint displacements[2] = {0, static_cast<MPI_Aint>(sizeof(int))};
      const MPI_Datatype types[2] = {MPI_INT, MPI_INT};
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_struct(2, blocklengths, displacements, types, &created) ==
                    MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  MPI_Datatype make_subarray_pair() {
    static MPI_Datatype type = [] {
      const int sizes[1] = {2};
      const int subsizes[1] = {2};
      const int starts[1] = {0};
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_subarray(1, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT,
                                             &created) == MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  MPI_Datatype make_resized_pair() {
    static MPI_Datatype type = [] {
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_resized(MPI_INT, 0,
                                            static_cast<MPI_Aint>(sizeof(resized_pair)),
                                            &created) == MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  // A struct with a positive int entry and a zero-blocklength double entry: the zero entry
  // contributes no basic components and must not make the datatype heterogeneous.
  MPI_Datatype make_zero_child_pair() {
    static MPI_Datatype type = [] {
      const int blocklengths[2] = {1, 0};
      const MPI_Aint displacements[2] = {0, 0};
      const MPI_Datatype types[2] = {MPI_INT, MPI_DOUBLE};
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_struct(2, blocklengths, displacements, types, &created) ==
                    MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Type_commit(&created) == MPI_SUCCESS);
      return resize_committed_and_own(created, sizeof(zero_child_pair));
    }();
    return type;
  }

  // A legal zero-size datatype: no positive-multiplicity basic component, so the annotation
  // must be rejected even though every listed child has the declared U.
  MPI_Datatype make_only_zero_pair() {
    static MPI_Datatype type = [] {
      const int blocklengths[1] = {0};
      const MPI_Aint displacements[1] = {0};
      const MPI_Datatype types[1] = {MPI_DOUBLE};
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_struct(1, blocklengths, displacements, types, &created) ==
                    MPI_SUCCESS);
      BOOST_REQUIRE(MPI_Type_commit(&created) == MPI_SUCCESS);
      return resize_committed_and_own(created, sizeof(only_zero_pair));
    }();
    return type;
  }

  MPI_Datatype make_gapped_vector() {
    static MPI_Datatype type = [] {
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_vector(2, 1, 2, MPI_INT, &created) == MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  MPI_Datatype make_cache_pair() {
    static MPI_Datatype type = [] {
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_contiguous(2, MPI_INT, &created) == MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  // Unsupported combiners: DARRAY, F90 and VALUE_INDEX.  The constructors are provider-aware;
  // MPI_Type_create_f90_real returns MPI_DATATYPE_NULL on a provider built without Fortran
  // support, which the NULL-handle guard must reject as well.
  MPI_Datatype make_darray_pair() {
    static MPI_Datatype type = [] {
      const int sizes[1] = {2};
      const int distribs[1] = {MPI_DISTRIBUTE_BLOCK};
      const int dargs[1] = {MPI_DISTRIBUTE_DFLT_DARG};
      const int psizes[1] = {1};
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_darray(1, 0, 1, sizes, distribs, dargs, psizes, MPI_ORDER_C,
                                           MPI_INT, &created) == MPI_SUCCESS);
      return commit_and_own(created);
    }();
    return type;
  }

  MPI_Datatype make_f90_real() {
    static MPI_Datatype type = [] {
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_create_f90_real(6, 37, &created) == MPI_SUCCESS);
      return created;  // unnamed predefined or MPI_DATATYPE_NULL; never freed
    }();
    return type;
  }

  MPI_Datatype make_value_index() {
    static MPI_Datatype type = [] {
      MPI_Datatype created = MPI_DATATYPE_NULL;
      BOOST_REQUIRE(MPI_Type_get_value_index(MPI_DOUBLE, MPI_INT, &created) == MPI_SUCCESS);
      return own_if_derived(created);
    }();
    return type;
  }

  // A named multi-component predefined handle: not the modeled basic component U.
  MPI_Datatype make_named_pair_handle() {
    return MPI_2INT;
  }

  // Opt-in custom annotations whose committed datatype is MPI_DATATYPE_NULL.
  struct null_custom {
    int value{0};
  };

  struct null_manual {
    int value{0};
  };

  // U itself maps to MPI_DATATYPE_NULL; only the validator helper is exercised, because a
  // predefined_op<U> cannot classify a null operand handle.
  struct null_operand {
    int value{0};
  };

  struct null_basic_committed {
    int value{0};
  };

  MPI_Datatype make_null_datatype() {
    return MPI_DATATYPE_NULL;
  }

  void release_custom_datatypes() {
    for (MPI_Datatype& type : custom_datatypes()) {
      if (type != MPI_DATATYPE_NULL) {
        (void)MPI_Type_free(&type);
        type = MPI_DATATYPE_NULL;
      }
    }
  }

}  // namespace

// The reflective macros deliberately open namespace mp1, so they must be used at global scope;
// the reflected types themselves live in the anonymous namespace above.
MP1_REFLECTION(refl_pair, a, b)
MP1_REFLECTION(nested_refl, inner, tail)
MP1_REFLECTION(array_member, values)
MP1_REFLECTION(padded_pair, a, b)
MP1_REFLECTION(mixed_value, i, d)
MP1_REFLECTION(outer_wrong, inner, tail)

// MP1_DATATYPE_TRAITS_PACKED + MP1_DATATYPE_BASIC: the supported opt-in for a custom
// non-predefined homogeneous derived datatype.
MP1_DATATYPE_TRAITS_PACKED(packed_pair, make_contiguous_two_ints())
MP1_DATATYPE_BASIC(packed_pair, int)

MP1_DATATYPE_TRAITS_PACKED(packed_pair_large, make_contiguous_two_ints_large())
MP1_DATATYPE_BASIC(packed_pair_large, int)

// The intentionally wrong annotation: rejected by the one-time committed-datatype validation
// before the first accumulating MPI call (including MPI_REPLACE, the MPICH abort trigger).
MP1_DATATYPE_TRAITS_PACKED(wrong_mixed, make_wrong_mixed_datatype())
MP1_DATATYPE_BASIC(wrong_mixed, int)

// A manual datatype_traits mapping without a byte_contiguous marker: the supported conservative
// spelling for a non-byte-contiguous derived handle (vector/indexed/subarray).  The RMA
// homogeneity contract is declared separately by MP1_DATATYPE_BASIC.
#define MP1_TEST_DERIVED_TRAITS(Type, Factory)                     \
  namespace mp1 {                                                  \
    template<>                                                     \
    struct datatype_traits<Type> {                                 \
      using data_type_category = detail::basic_or_fixed_size_type; \
      static MPI_Datatype get_datatype() {                         \
        return Factory;                                            \
      }                                                            \
    };                                                             \
  }

MP1_TEST_DERIVED_TRAITS(dup_pair, make_dup_pair())
MP1_DATATYPE_BASIC(dup_pair, int)
MP1_TEST_DERIVED_TRAITS(vector_pair, make_vector_pair())
MP1_DATATYPE_BASIC(vector_pair, int)
MP1_TEST_DERIVED_TRAITS(hvector_pair, make_hvector_pair())
MP1_DATATYPE_BASIC(hvector_pair, int)
MP1_TEST_DERIVED_TRAITS(indexed_pair, make_indexed_pair())
MP1_DATATYPE_BASIC(indexed_pair, int)
MP1_TEST_DERIVED_TRAITS(hindexed_pair, make_hindexed_pair())
MP1_DATATYPE_BASIC(hindexed_pair, int)
MP1_TEST_DERIVED_TRAITS(indexed_block_pair, make_indexed_block_pair())
MP1_DATATYPE_BASIC(indexed_block_pair, int)
MP1_TEST_DERIVED_TRAITS(hindexed_block_pair, make_hindexed_block_pair())
MP1_DATATYPE_BASIC(hindexed_block_pair, int)
MP1_TEST_DERIVED_TRAITS(struct_pair, make_struct_pair())
MP1_DATATYPE_BASIC(struct_pair, int)
MP1_TEST_DERIVED_TRAITS(subarray_pair, make_subarray_pair())
MP1_DATATYPE_BASIC(subarray_pair, int)
MP1_TEST_DERIVED_TRAITS(resized_pair, make_resized_pair())
MP1_DATATYPE_BASIC(resized_pair, int)
MP1_TEST_DERIVED_TRAITS(gapped_vector, make_gapped_vector())
MP1_DATATYPE_BASIC(gapped_vector, int)
MP1_TEST_DERIVED_TRAITS(zero_child_pair, make_zero_child_pair())
MP1_DATATYPE_BASIC(zero_child_pair, int)
MP1_TEST_DERIVED_TRAITS(only_zero_pair, make_only_zero_pair())
MP1_DATATYPE_BASIC(only_zero_pair, int)
MP1_TEST_DERIVED_TRAITS(cache_pair, make_cache_pair())
MP1_DATATYPE_BASIC(cache_pair, int)

// Fail-closed combiners: DARRAY/VALUE_INDEX are unsupported by the walk; F90 may be an unnamed
// predefined handle or MPI_DATATYPE_NULL (provider built without Fortran support); a named
// MPI_2* pair handle is not the modeled basic component.
struct f90_real_type {
  float value{0};
};

struct value_index_type {
  int value{0};
};

struct named_pair_type {
  int first{0};
  int second{0};
};

MP1_TEST_DERIVED_TRAITS(darray_pair, make_darray_pair())
MP1_DATATYPE_BASIC(darray_pair, int)
MP1_TEST_DERIVED_TRAITS(f90_real_type, make_f90_real())
MP1_DATATYPE_BASIC(f90_real_type, float)
MP1_TEST_DERIVED_TRAITS(value_index_type, make_value_index())
MP1_DATATYPE_BASIC(value_index_type, int)
MP1_TEST_DERIVED_TRAITS(named_pair_type, make_named_pair_handle())
MP1_DATATYPE_BASIC(named_pair_type, int)

// Committed handles that are themselves MPI_DATATYPE_NULL.
MP1_TEST_DERIVED_TRAITS(null_custom, make_null_datatype())
MP1_DATATYPE_BASIC(null_custom, int)
MP1_DATATYPE_TRAITS(null_operand, make_null_datatype())
MP1_TEST_DERIVED_TRAITS(null_basic_committed, make_null_datatype())
MP1_DATATYPE_BASIC(null_basic_committed, null_operand)

// A manual struct_builder exposing component_types but not the library-internal
// component_types_trusted marker: always untrusted and validated once at run time.
namespace mp1 {

  template<>
  class struct_builder<manual_pair_runtime> : public base_struct_builder<manual_pair_runtime> {
  public:
    using component_types = detail::type_list<int, int>;

    struct_builder() {
      manual_pair_runtime object{};
      this->register_object(object);
      this->register_element(object.first);
      this->register_element(object.second);
      this->define_struct();
    }
  };

  // A manual builder whose committed datatype is MPI_DATATYPE_NULL.  The component_types list
  // looks homogeneous, but the validator must reject the null committed handle before MPI.
  template<>
  class struct_builder<null_manual> : public base_struct_builder<null_manual> {
  public:
    using component_types = detail::type_list<int>;

    static MPI_Datatype get_datatype() {
      return MPI_DATATYPE_NULL;
    }
  };

}  // namespace mp1

namespace {

  struct homogeneous_fixture {
    homogeneous_fixture() {
      mp1::init();
      // The untrusted-annotation tests deliberately use invalid committed datatypes.  A
      // returning error handler turns any accidental provider diagnosis into an mpi_error
      // exception instead of a process abort; the local validation is still expected to throw
      // std::invalid_argument first.
      mp1::set_error_handler(mp1::comm_world(), mp1::error_handler::returns);
    }

    ~homogeneous_fixture() {
      try {
        release_custom_datatypes();
        mp1::finalize();
        // NOLINTNEXTLINE(bugprone-empty-catch): fixture cleanup must not throw
      } catch (...) {
        // The harness or another fixture may own MPI; never throw from a destructor.
      }
    }
  };

  template<class Function>
  bool throws_invalid_argument(Function&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  // ---- MPI call counters (the --wrap shim; absent on non-GNU/Clang/ELF builds) -------------
  bool wrap_supported() {
#if MP1_HOMOGENEOUS_WRAP
    return mpi_wrap::supported();
#else
    return false;
#endif
  }

  // The four accumulating data-movement calls.  A rejected untrusted annotation must not
  // increment any of them.
  std::size_t data_movement_calls() {
#if MP1_HOMOGENEOUS_WRAP
    return mpi_wrap::count("MPI_Accumulate_c") + mpi_wrap::count("MPI_Raccumulate_c") +
           mpi_wrap::count("MPI_Get_accumulate_c") + mpi_wrap::count("MPI_Rget_accumulate_c");
#else
    return 0;
#endif
  }

  // Envelope/contents queries made by the one-time untrusted-annotation walk.
  std::size_t walk_metadata_calls() {
#if MP1_HOMOGENEOUS_WRAP
    return mpi_wrap::count("MPI_Type_get_envelope_c") +
           mpi_wrap::count("MPI_Type_get_contents_c");
#else
    return 0;
#endif
  }

  std::size_t contents_calls() {
#if MP1_HOMOGENEOUS_WRAP
    return mpi_wrap::count("MPI_Type_get_contents_c");
#else
    return 0;
#endif
  }

  void reset_call_counts() {
#if MP1_HOMOGENEOUS_WRAP
    mpi_wrap::reset_counts();
#endif
  }

  // Boost.Test can print a bool but not std::pair; compare the components so a failure shows
  // two integers instead of failing to compile in the printer.
  void check_pair(const std::pair<int, int>& value, int first, int second) {
    BOOST_TEST(value.first == first);
    BOOST_TEST(value.second == second);
  }

  // Make all origin-side completion and target updates visible before the target rank reads
  // its own window memory (MPI 4.1 §12.7.4; the separate memory model needs MPI_Win_sync).
  void sync_after_passive(const mp1::mpi_communicator& c, const mp1::mpi_window& w) {
    mp1::barrier(c);
    mp1::lock_all(w);
    mp1::sync(w);
    mp1::unlock_all(w);
    mp1::barrier(c);
  }

  // Every accumulating call targets the calling rank itself, so the expected value is exact.
  template<class T>
  void accumulate_self(const mp1::mpi_communicator& c, const T& origin, const T& initial,
                       const T& expected) {
    mp1::window w = mp1::window::allocate(
        c, {.size = static_cast<MPI_Aint>(sizeof(T)), .disp_unit = sizeof(T)});
    T* target = static_cast<T*>(w.base());
    *target = initial;

    mp1::lock_all(w);
    mp1::accumulate(w, mp1::span_from(&origin, mp1::count_t{1}), c.rank(), 0, mp1::count_t{1},
                    MPI_SUM);
    mp1::flush_all(w);
    mp1::unlock_all(w);
    sync_after_passive(c, w);

    const bool matched = (*target == expected);
    BOOST_TEST(matched);
    w.free();
  }

  // Every accumulating family must reject an invalid untrusted annotation with
  // std::invalid_argument; the caller checks that no data-movement call was made.
  template<class T>
  void expect_custom_annotation_rejected(mp1::window& w, int rank) {
    T origin{};
    T result{};
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), rank, 0, 1, MPI_SUM); }));
    BOOST_TEST(throws_invalid_argument([&] {
      mp1::get_accumulate(w, mp1::span_from(&origin, 1), mp1::span_from(&result, 1), rank, 0, 1,
                          MPI_SUM);
    }));
    BOOST_TEST(throws_invalid_argument(
        [&] { (void)mp1::raccumulate(w, mp1::span_from(&origin, 1), rank, 0, 1, MPI_SUM); }));
    BOOST_TEST(throws_invalid_argument([&] {
      (void)mp1::rget_accumulate(w, mp1::span_from(&origin, 1), mp1::span_from(&result, 1),
                                 rank, 0, 1, MPI_REPLACE);
    }));
    // The validation runs before the MPI_PROC_NULL no-op branch.
    BOOST_TEST(throws_invalid_argument(
        [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), MPI_PROC_NULL, 0, 1, MPI_SUM); }));
  }

}  // namespace


BOOST_TEST_GLOBAL_FIXTURE(homogeneous_fixture);

BOOST_AUTO_TEST_SUITE(rma_homogeneous_contract)

BOOST_AUTO_TEST_CASE(accumulate_pair_componentwise_sum) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  using pair_type = std::pair<int, int>;
  constexpr int elements = 2;

  mp1::window w =
      mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(elements * sizeof(pair_type)),
                                .disp_unit = sizeof(pair_type)});
  pair_type* target = static_cast<pair_type*>(w.base());
  for (int i = 0; i < elements; ++i)
    target[i] = pair_type{0, 0};

  const mp1::layout<pair_type> target_layout = mp1::layout<pair_type>::contiguous(elements);
  const mp1::layout<pair_type> origin_layout = mp1::layout<pair_type>::contiguous(elements);
  pair_type origin[elements] = {{1, 2}, {3, 4}};

  mp1::lock_all(w);
  // Flat origin, layout target, raw predefined handle.
  mp1::accumulate(w, std::span{origin}, rank, 0, target_layout, MPI_SUM);
  // Layout origin, flat target, predefined_op<int> (the basic component, not the aggregate).
  mp1::accumulate(w, mp1::region{origin, origin_layout}, rank, 0, elements,
                  mp1::predefined_op<int>{MPI_SUM});
  // Layout origin, layout target, typed functor with a matching operand.
  mp1::accumulate(w, mp1::region{origin, origin_layout}, rank, 0, target_layout,
                  std::plus<int>{});
  // Flat origin, flat target, transparent standard functor.
  mp1::accumulate(w, mp1::span_from(origin, elements), rank, 0, elements, std::plus<>{});
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);

  // Four additions of {{1,2},{3,4}}.
  check_pair(target[0], 4, 8);
  check_pair(target[1], 12, 16);
  w.free();
}

BOOST_AUTO_TEST_CASE(get_accumulate_pair_old_values_and_replace) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  using pair_type = std::pair<int, int>;
  constexpr int elements = 2;

  mp1::window w =
      mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(elements * sizeof(pair_type)),
                                .disp_unit = sizeof(pair_type)});
  pair_type* target = static_cast<pair_type*>(w.base());
  target[0] = pair_type{10, 20};
  target[1] = pair_type{30, 40};

  const mp1::layout<pair_type> target_layout = mp1::layout<pair_type>::contiguous(elements);
  const mp1::layout<pair_type> origin_layout = mp1::layout<pair_type>::contiguous(elements);
  const mp1::layout<pair_type> result_layout = mp1::layout<pair_type>::contiguous(elements);
  pair_type origin[elements] = {{1, 2}, {3, 4}};
  pair_type result[elements] = {};

  mp1::lock_all(w);
  // 1. Flat origin, flat result, flat target: result is the old pair value.
  mp1::get_accumulate(w, mp1::span_from(origin, elements), mp1::span_from(result, elements),
                      rank, 0, elements, MPI_SUM);
  mp1::flush_all(w);
  check_pair(result[0], 10, 20);
  check_pair(result[1], 30, 40);

  // 2. Layout origin, layout result, layout target.
  result[0] = result[1] = pair_type{};
  mp1::get_accumulate(w, mp1::region{origin, origin_layout}, mp1::region{result, result_layout},
                      rank, 0, target_layout, mp1::predefined_op<int>{MPI_SUM});
  mp1::flush_all(w);
  check_pair(result[0], 11, 22);
  check_pair(result[1], 33, 44);

  // 3. MPI_NO_OP with an ignored origin: the target is read and not updated.  MPI ignores
  // the origin, but span_from rejects a null pointer with a positive count, so pass a real
  // dummy buffer.
  result[0] = result[1] = pair_type{};
  const pair_type ignored_origin[2] = {};
  mp1::get_accumulate(w, mp1::span_from(ignored_origin, 2), std::span{result}, rank, 0,
                      target_layout, mp1::predefined_op<int>{MPI_NO_OP});
  mp1::flush_all(w);
  check_pair(result[0], 12, 24);
  check_pair(result[1], 36, 48);

  // 4. MPI_REPLACE over the homogeneous derived datatype: the old value is returned and the
  // target is overwritten component-wise.
  result[0] = result[1] = pair_type{};
  mp1::get_accumulate(w, mp1::span_from(origin, elements), mp1::span_from(result, elements),
                      rank, 0, elements, mp1::predefined_op<int>{MPI_REPLACE});
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  check_pair(result[0], 12, 24);
  check_pair(result[1], 36, 48);
  check_pair(target[0], 1, 2);
  check_pair(target[1], 3, 4);
  w.free();
}

BOOST_AUTO_TEST_CASE(request_forms_accumulate_pair) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  using pair_type = std::pair<int, int>;

  mp1::window w = mp1::window::allocate(
      c, {.size = static_cast<MPI_Aint>(sizeof(pair_type)), .disp_unit = sizeof(pair_type)});
  pair_type* target = static_cast<pair_type*>(w.base());
  *target = pair_type{0, 0};
  const mp1::layout<pair_type> one = mp1::layout<pair_type>::contiguous(1);
  pair_type origin{1, 2};
  pair_type result{};

  mp1::lock_all(w);
  mp1::request accumulate_request = mp1::raccumulate(
      w, mp1::span_from(&origin, one.size()), rank, 0, one, mp1::predefined_op<int>{MPI_SUM});
  accumulate_request.wait();
  mp1::flush_all(w);

  mp1::request get_request =
      mp1::rget_accumulate(w, mp1::span_from(&origin, one.size()),
                           mp1::span_from(&result, one.size()), rank, 0, one, MPI_SUM);
  get_request.wait();
  mp1::flush_all(w);
  check_pair(result, 1, 2);

  // MPI_REPLACE in the request-based fetching family overwrites the target.
  pair_type replace_origin{7, 8};
  mp1::request replace_request = mp1::rget_accumulate(
      w, mp1::span_from(&replace_origin, one.size()), mp1::span_from(&result, one.size()), rank,
      0, one, mp1::predefined_op<int>{MPI_REPLACE});
  replace_request.wait();
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  check_pair(result, 2, 4);
  check_pair(*target, 7, 8);
  w.free();
}

BOOST_AUTO_TEST_CASE(tuple_array_nested_and_reflected_homogeneous) {
  const mp1::mpi_communicator c = mp1::comm_world();
  using tuple_type = std::tuple<int, int>;
  using array_type = std::array<int, 2>;
  using nested_pair = std::pair<std::pair<int, int>, int>;

  accumulate_self(c, tuple_type{1, 2}, tuple_type{10, 20}, tuple_type{11, 22});
  accumulate_self(c, array_type{1, 2}, array_type{10, 20}, array_type{11, 22});
  accumulate_self(c, nested_pair{{1, 2}, 3}, nested_pair{{10, 20}, 30},
                  nested_pair{{11, 22}, 33});
  accumulate_self(c, refl_pair{1, 2}, refl_pair{10, 20}, refl_pair{11, 22});
  accumulate_self(c, nested_refl{{1, 2}, 3}, nested_refl{{10, 20}, 30},
                  nested_refl{{11, 22}, 33});
  accumulate_self(c, array_member{{1, 2, 3}}, array_member{{10, 20, 30}},
                  array_member{{11, 22, 33}});
  accumulate_self(c, packed_pair{1, 2}, packed_pair{10, 20}, packed_pair{11, 22});
  accumulate_self(c, packed_pair_large{1, 2}, packed_pair_large{10, 20},
                  packed_pair_large{11, 22});
}

BOOST_AUTO_TEST_CASE(canonical_identity_enum_and_utf16) {
  const mp1::mpi_communicator c = mp1::comm_world();
  using enum_pair = std::pair<rma_enum, int>;
  using utf16_pair = std::pair<char16_t, std::uint16_t>;

  // pair<enum,int>: both members have the int canonical identity, U = int.
  accumulate_self(c, enum_pair{rma_enum::one, 2}, enum_pair{rma_enum::zero, 10},
                  enum_pair{rma_enum::one, 12});
  // pair<char16_t,uint16_t>: both map to the uint_least16_t C integer datatype.
  accumulate_self(c, utf16_pair{char16_t{2}, 2}, utf16_pair{char16_t{1}, 10},
                  utf16_pair{char16_t{3}, 12});
}

BOOST_AUTO_TEST_CASE(padding_and_untouched_bytes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  mp1::window w = mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(sizeof(padded_pair)),
                                            .disp_unit = sizeof(padded_pair)});
  auto* target = static_cast<unsigned char*>(w.base());
  // Fill every byte, then set the two mapped members: the tail keeps the 0xA5 pattern.
  std::memset(target, 0xA5, sizeof(padded_pair));
  padded_pair* value = static_cast<padded_pair*>(w.base());
  value->a = 10;
  value->b = 20;

  padded_pair origin{1, 2, {}};
  mp1::lock_all(w);
  mp1::accumulate(w, mp1::span_from(&origin, mp1::count_t{1}), rank, 0, mp1::count_t{1},
                  MPI_SUM);
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);

  BOOST_TEST(value->a == 11);
  BOOST_TEST(value->b == 22);
  for (std::size_t i = 8; i < sizeof(padded_pair); ++i)
    BOOST_TEST(target[i] == 0xA5);
  w.free();
}

BOOST_AUTO_TEST_CASE(replace_accepted_for_homogeneous_derived) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  using pair_type = std::pair<int, int>;

  // Existing source-compat anchor: layout<int> + MPI_REPLACE stays valid.
  {
    constexpr int elements = 2;
    mp1::window w = mp1::window::allocate(
        c, {.size = static_cast<MPI_Aint>(elements * sizeof(int)), .disp_unit = sizeof(int)});
    int* target = static_cast<int*>(w.base());
    target[0] = 5;
    target[1] = 6;
    int origin[elements] = {7, 8};
    const mp1::layout<int> layout = mp1::layout<int>::contiguous(elements);
    mp1::lock_all(w);
    mp1::accumulate(w, std::span{origin}, rank, 0, layout,
                    mp1::predefined_op<int>{MPI_REPLACE});
    mp1::flush_all(w);
    mp1::unlock_all(w);
    sync_after_passive(c, w);
    BOOST_TEST(target[0] == 7);
    BOOST_TEST(target[1] == 8);
    w.free();
  }

  // Permissive homogeneous-derived policy (MPI 4.1 §12.3.4): layout<pair<int,int>> +
  // MPI_REPLACE is accepted; the target is replaced component-wise.
  {
    constexpr int elements = 2;
    mp1::window w =
        mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(elements * sizeof(pair_type)),
                                  .disp_unit = sizeof(pair_type)});
    pair_type* target = static_cast<pair_type*>(w.base());
    target[0] = pair_type{1, 2};
    target[1] = pair_type{3, 4};
    pair_type origin[elements] = {{7, 8}, {9, 10}};
    const mp1::layout<pair_type> layout = mp1::layout<pair_type>::contiguous(elements);
    mp1::lock_all(w);
    mp1::accumulate(w, std::span{origin}, rank, 0, layout,
                    mp1::predefined_op<int>{MPI_REPLACE});
    mp1::flush_all(w);
    mp1::unlock_all(w);
    sync_after_passive(c, w);
    check_pair(target[0], 7, 8);
    check_pair(target[1], 9, 10);
    w.free();
  }
}

BOOST_AUTO_TEST_CASE(op_category_mismatch_rejected_before_mpi) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  using pair_type = std::pair<int, int>;

  mp1::window w = mp1::window::allocate(
      c, {.size = static_cast<MPI_Aint>(sizeof(pair_type)), .disp_unit = sizeof(pair_type)});
  pair_type* target = static_cast<pair_type*>(w.base());
  *target = pair_type{5, 6};

  pair_type int_pair{1, 2};
  std::pair<char, char> char_pair{'a', 'b'};
  std::pair<double, double> double_pair{1., 2.};
  std::pair<bool, bool> bool_pair{true, false};

  mp1::lock_all(w);
  // Character is predefined but belongs to no §6.9.2 reduction category (§6.9.3).
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::accumulate(w, mp1::span_from(&char_pair, 1), rank, 0, 1,
                    mp1::predefined_op<char>{MPI_SUM});
  }));
  // Floating point is not in the bitwise row.
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::accumulate(w, mp1::span_from(&double_pair, 1), rank, 0, 1,
                    mp1::predefined_op<double>{MPI_BAND});
  }));
  // Logical is not in the SUM row.
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::accumulate(w, mp1::span_from(&bool_pair, 1), rank, 0, 1,
                    mp1::predefined_op<bool>{MPI_SUM});
  }));
  // MAXLOC needs an MPI_2* pair datatype, which mp1 does not model.
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::accumulate(w, mp1::span_from(&int_pair, 1), rank, 0, 1,
                    mp1::predefined_op<int>{MPI_MAXLOC});
  }));
  // MPI_NO_OP is legal only in the fetching families, even for MPI_PROC_NULL.
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::accumulate(w, mp1::span_from(&int_pair, 1), rank, 0, 1,
                    mp1::predefined_op<int>{MPI_NO_OP});
  }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::accumulate(w, mp1::span_from(&int_pair, 1), MPI_PROC_NULL, 0, 1,
                    mp1::predefined_op<int>{MPI_NO_OP});
  }));

  // A user-created MPI_Op is rejected for RMA (MPI 4.1 §12.3.4 forbids user-defined ops).
  MPI_Op user_op = MPI_OP_NULL;
  BOOST_REQUIRE(MPI_Op_create([](void*, void*, int*, MPI_Datatype*) {}, 1, &user_op) ==
                MPI_SUCCESS);
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::accumulate(w, mp1::span_from(&int_pair, 1), rank, 0, 1,
                    mp1::predefined_op<int>{user_op});
  }));
  (void)MPI_Op_free(&user_op);

  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  check_pair(*target, 5, 6);  // no rejected call moved data
  w.free();
}

BOOST_AUTO_TEST_CASE(proc_null_and_zero_element_accumulate) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  using pair_type = std::pair<int, int>;

  mp1::window w = mp1::window::allocate(
      c,
      {.size = static_cast<MPI_Aint>(2 * sizeof(pair_type)), .disp_unit = sizeof(pair_type)});
  pair_type* target = static_cast<pair_type*>(w.base());
  target[0] = pair_type{1, 2};
  target[1] = pair_type{3, 4};
  pair_type origin[2] = {{9, 9}, {9, 9}};
  pair_type result{};
  const mp1::layout<pair_type> empty = mp1::layout<pair_type>::contiguous(0);

  mp1::lock_all(w);
  // A homogeneous pair with MPI_PROC_NULL is a normalized no-op.
  mp1::accumulate(w, mp1::span_from(origin, 1), MPI_PROC_NULL, 0, 1, MPI_SUM);
  // MPI_PROC_NULL normalizes the operation to a no-op; the origin is ignored, but
  // span_from rejects a null pointer with a positive count.
  mp1::get_accumulate(w, mp1::span_from(origin, 1), mp1::span_from(&result, 1), MPI_PROC_NULL,
                      0, 1, mp1::predefined_op<int>{MPI_NO_OP});
  mp1::request null_request = mp1::raccumulate(w, mp1::span_from(origin, 1), MPI_PROC_NULL, 0,
                                               1, mp1::predefined_op<int>{MPI_SUM});
  null_request.wait();
  // A zero-element layout<pair<int,int>> of a non-empty homogeneous T is accepted.
  mp1::accumulate(w, mp1::region{origin, empty}, rank, 0, empty, MPI_SUM);
  mp1::accumulate(w, mp1::span_from(origin, mp1::count_t{0}), rank, 0, mp1::count_t{0},
                  MPI_SUM);
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  check_pair(target[0], 1, 2);
  check_pair(target[1], 3, 4);
  w.free();
}

BOOST_AUTO_TEST_CASE(concurrent_accumulate_to_one_target_is_elementwise) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();
  const int size = c.size();
  using pair_type = std::pair<int, int>;

  mp1::window w = mp1::window::allocate(
      c, {.size = static_cast<MPI_Aint>(sizeof(pair_type)), .disp_unit = sizeof(pair_type)});
  pair_type* target = static_cast<pair_type*>(w.base());
  *target = pair_type{0, 0};
  mp1::barrier(c);

  pair_type origin{1, 2};
  mp1::lock_all(w);
  mp1::barrier(c);
  mp1::accumulate(w, mp1::span_from(&origin, 1), 0, 0, 1, MPI_SUM);
  mp1::flush_all(w);
  mp1::barrier(c);
  mp1::unlock_all(w);
  sync_after_passive(c, w);

  if (rank == 0)
    check_pair(*target, size, 2 * size);
  w.free();
}

BOOST_AUTO_TEST_CASE(custom_homogeneous_annotation_rejected_before_mpi) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  mp1::window w = mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(sizeof(wrong_mixed)),
                                            .disp_unit = sizeof(wrong_mixed)});
  mp1::set_error_handler(w, mp1::error_handler::returns);
  wrong_mixed* target = static_cast<wrong_mixed*>(w.base());
  target->i = 10;
  target->d = 20.;

  wrong_mixed origin{1, 2.};
  wrong_mixed result{};
  mp1::lock_all(w);
  // The committed datatype is heterogeneous, but the annotation claims int: the one-time
  // validation rejects every accumulating family before the MPI call.
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), rank, 0, 1, MPI_SUM); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), rank, 0, 1, MPI_REPLACE); }));
  BOOST_TEST(throws_invalid_argument([&] {
    mp1::get_accumulate(w, mp1::span_from(&origin, 1), mp1::span_from(&result, 1), rank, 0, 1,
                        MPI_SUM);
  }));
  BOOST_TEST(throws_invalid_argument(
      [&] { (void)mp1::raccumulate(w, mp1::span_from(&origin, 1), rank, 0, 1, MPI_SUM); }));
  BOOST_TEST(throws_invalid_argument([&] {
    (void)mp1::rget_accumulate(w, mp1::span_from(&origin, 1), mp1::span_from(&result, 1), rank,
                               0, 1, MPI_REPLACE);
  }));
  // The validation runs before the MPI_PROC_NULL no-op branch.
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), MPI_PROC_NULL, 0, 1, MPI_SUM); }));
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  BOOST_TEST(target->i == 10);
  BOOST_TEST(target->d == 20.);
  w.free();
}

BOOST_AUTO_TEST_CASE(nested_untrusted_member_annotation_rejected) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  mp1::window w = mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(sizeof(outer_wrong)),
                                            .disp_unit = sizeof(outer_wrong)});
  mp1::set_error_handler(w, mp1::error_handler::returns);
  outer_wrong* target = static_cast<outer_wrong*>(w.base());
  target->inner = wrong_mixed{10, 20.};
  target->tail = 30;

  outer_wrong origin{wrong_mixed{1, 2.}, 3};
  mp1::lock_all(w);
  // The outer member list is homogeneous over int, but it contains an untrusted member whose
  // committed datatype is heterogeneous; the resulting outer datatype is validated and
  // rejected.
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), rank, 0, 1, MPI_SUM); }));
  BOOST_TEST(throws_invalid_argument(
      [&] { mp1::accumulate(w, mp1::span_from(&origin, 1), rank, 0, 1, MPI_REPLACE); }));
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  BOOST_TEST(target->inner.i == 10);
  BOOST_TEST(target->inner.d == 20.);
  BOOST_TEST(target->tail == 30);
  w.free();
}

BOOST_AUTO_TEST_CASE(combiner_walk_accepts_homogeneous_constructors) {
  const mp1::mpi_communicator c = mp1::comm_world();

  // The manual struct_builder path is untrusted but its committed MPI_Type_create_struct_c +
  // RESIZED datatype is homogeneous; it must be accepted and accumulated end to end.
  static_assert(mp1::detail::component_info<manual_pair_runtime>::runtime_validate,
                "a manual component_types builder must stay runtime-validated");

  // One-time validation of every constructor the walk decodes.  These calls also populate the
  // per-T decision so the end-to-end calls below do not re-walk.
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<dup_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<vector_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<hvector_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<indexed_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<hindexed_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<indexed_block_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<hindexed_block_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<struct_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<subarray_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<resized_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<zero_child_pair>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<gapped_vector>());
  BOOST_TEST(mp1::detail::rma_committed_datatype_is_homogeneous<manual_pair_runtime>());

  // End-to-end checks with the committed map of each constructor.  DUP and RESIZED map one
  // int, so the second member keeps its initial value; the others map both.
  accumulate_self(c, dup_pair{1, 2}, dup_pair{10, 20}, dup_pair{11, 20});
  accumulate_self(c, vector_pair{1, 2}, vector_pair{10, 20}, vector_pair{11, 22});
  accumulate_self(c, hvector_pair{1, 2}, hvector_pair{10, 20}, hvector_pair{11, 22});
  accumulate_self(c, indexed_pair{1, 2}, indexed_pair{10, 20}, indexed_pair{11, 22});
  accumulate_self(c, hindexed_pair{1, 2}, hindexed_pair{10, 20}, hindexed_pair{11, 22});
  accumulate_self(c, indexed_block_pair{1, 2}, indexed_block_pair{10, 20},
                  indexed_block_pair{11, 22});
  accumulate_self(c, hindexed_block_pair{1, 2}, hindexed_block_pair{10, 20},
                  hindexed_block_pair{11, 22});
  accumulate_self(c, struct_pair{1, 2}, struct_pair{10, 20}, struct_pair{11, 22});
  accumulate_self(c, subarray_pair{1, 2}, subarray_pair{10, 20}, subarray_pair{11, 22});
  accumulate_self(c, resized_pair{1, 2}, resized_pair{10, 20}, resized_pair{11, 20});
  // zero_child_pair is accepted by the local validator (asserted above) but is not passed
  // through a data-movement call here: a legal homogeneous datatype that contains a
  // zero-blocklength entry may be rejected by the provider (see the COMPATIBILITY.md provider
  // caveat), while the validator must still accept it instead of rejecting a legal type.
  // A manual datatype_traits mapping without byte_contiguous: two ints at offsets 0 and 8,
  // middle int outside the type map.
  accumulate_self(c, gapped_vector{1, 99, 3}, gapped_vector{10, 20, 30},
                  gapped_vector{11, 20, 33});
  accumulate_self(c, manual_pair_runtime{1, 2}, manual_pair_runtime{10, 20},
                  manual_pair_runtime{11, 22});
}

BOOST_AUTO_TEST_CASE(combiner_walk_fails_closed_on_unmodeled_datatypes) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  // DARRAY and VALUE_INDEX are unmodeled combiners; a named MPI_2* handle is a named leaf that
  // is not the annotated basic component; the only-zero-multiplicity type has no component.
  BOOST_TEST(!mp1::detail::rma_committed_datatype_is_homogeneous<darray_pair>());
  BOOST_TEST(!mp1::detail::rma_committed_datatype_is_homogeneous<f90_real_type>());
  BOOST_TEST(!mp1::detail::rma_committed_datatype_is_homogeneous<value_index_type>());
  BOOST_TEST(!mp1::detail::rma_committed_datatype_is_homogeneous<named_pair_type>());
  BOOST_TEST(!mp1::detail::rma_committed_datatype_is_homogeneous<only_zero_pair>());

  mp1::window w = mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(8), .disp_unit = 1});
  mp1::set_error_handler(w, mp1::error_handler::returns);
  if (wrap_supported())
    reset_call_counts();

  mp1::lock_all(w);
  expect_custom_annotation_rejected<darray_pair>(w, rank);
  expect_custom_annotation_rejected<f90_real_type>(w, rank);
  expect_custom_annotation_rejected<value_index_type>(w, rank);
  expect_custom_annotation_rejected<named_pair_type>(w, rank);
  expect_custom_annotation_rejected<only_zero_pair>(w, rank);
  mp1::flush_all(w);
  mp1::unlock_all(w);
  w.free();

  if (wrap_supported()) {
    // The validator is the only path allowed to touch MPI for these types; no accumulating
    // data-movement call may be made.
    BOOST_TEST(data_movement_calls() == 0);
  }
}

BOOST_AUTO_TEST_CASE(null_handle_annotations_fail_closed) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  // Direct guards.  The providers used here reject constructing a datatype with an
  // MPI_DATATYPE_NULL child (MPI_ERR_TYPE), so the child guard is exercised through the walk
  // entry point directly: a null handle is rejected before any MPI metadata call and is never
  // freed.
  {
    bool any_component = false;
    bool unsupported = false;
    BOOST_TEST(!mp1::detail::rma_walk_basic_components(MPI_DATATYPE_NULL, MPI_INT,
                                                       any_component, unsupported, 0));
    BOOST_TEST(unsupported);
    BOOST_TEST(mp1::detail::rma_datatype_is_borrowed(MPI_DATATYPE_NULL));
    BOOST_TEST(throws_invalid_argument(
        [] { (void)mp1::detail::rma_datatype_envelope(MPI_DATATYPE_NULL); }));
    // native_datatype<U>() is null too: the validator returns false without touching MPI.
    BOOST_TEST(!mp1::detail::rma_committed_datatype_is_homogeneous<null_basic_committed>());
  }

  mp1::window w = mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(8), .disp_unit = 1});
  mp1::set_error_handler(w, mp1::error_handler::returns);
  auto* target = static_cast<int*>(w.base());
  *target = 42;
  if (wrap_supported())
    reset_call_counts();

  mp1::lock_all(w);
  // The opt-in annotation and the manual component_types annotation both claim int but commit
  // MPI_DATATYPE_NULL.  Every accumulating family must throw std::invalid_argument, including
  // the MPI_PROC_NULL path, and no provider call may abort the process.
  expect_custom_annotation_rejected<null_custom>(w, rank);
  expect_custom_annotation_rejected<null_manual>(w, rank);
  // U itself maps to MPI_DATATYPE_NULL: the predefined_op<U> classification fails closed in
  // datatype_info_of before any MPI query.
  expect_custom_annotation_rejected<null_basic_committed>(w, rank);
  mp1::flush_all(w);
  mp1::unlock_all(w);
  BOOST_TEST(*target == 42);
  w.free();

  if (wrap_supported()) {
    BOOST_TEST(data_movement_calls() == 0);
    // The null handle is rejected before the envelope/contents walk.
    BOOST_TEST(walk_metadata_calls() == 0);
  }
}

BOOST_AUTO_TEST_CASE(untrusted_validation_is_one_time) {
  if (not wrap_supported()) {
    BOOST_TEST_MESSAGE("MPI call counters unavailable; one-time cache not measurable");
    return;
  }
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  mp1::window w = mp1::window::allocate(
      c, {.size = static_cast<MPI_Aint>(sizeof(cache_pair)), .disp_unit = sizeof(cache_pair)});
  auto* target = static_cast<cache_pair*>(w.base());
  *target = cache_pair{0, 0};
  const cache_pair origin{1, 2};

  mp1::lock_all(w);
  reset_call_counts();
  mp1::accumulate(w, mp1::span_from(&origin, mp1::count_t{1}), rank, 0, mp1::count_t{1},
                  MPI_SUM);
  BOOST_TEST(contents_calls() == 1);
  BOOST_TEST(data_movement_calls() == 1);

  reset_call_counts();
  mp1::accumulate(w, mp1::span_from(&origin, mp1::count_t{1}), rank, 0, mp1::count_t{1},
                  MPI_SUM);
  BOOST_TEST(contents_calls() == 0);
  BOOST_TEST(walk_metadata_calls() == 0);
  BOOST_TEST(data_movement_calls() == 1);

  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  BOOST_TEST((*target == cache_pair{2, 4}));
  w.free();
}

BOOST_AUTO_TEST_CASE(put_get_keep_accepting_heterogeneous_types) {
  const mp1::mpi_communicator c = mp1::comm_world();
  const int rank = c.rank();

  mp1::window w = mp1::window::allocate(c, {.size = static_cast<MPI_Aint>(sizeof(mixed_value)),
                                            .disp_unit = sizeof(mixed_value)});
  auto* target = static_cast<mixed_value*>(w.base());
  target->i = 0;
  target->d = 0.;

  mixed_value origin{7, 2.5};
  mixed_value fetched{};
  const mp1::layout<mixed_value> layout = mp1::layout<mixed_value>::contiguous(1);
  mp1::lock_all(w);
  // put/get have no homogeneity rule: a mixed layout<T> remains legal.
  mp1::put(w, mp1::span_from(&origin, layout.size()), rank, 0, layout);
  mp1::flush_all(w);
  mp1::get(w, mp1::span_from(&fetched, layout.size()), rank, 0, layout);
  mp1::flush_all(w);
  mp1::unlock_all(w);
  sync_after_passive(c, w);
  BOOST_TEST(target->i == 7);
  BOOST_TEST(target->d == 2.5);

  // The put update is visible to the local get only after the epoch; fetched was read through
  // MPI from the local target, so it also carries the put value.
  BOOST_TEST(fetched.i == 7);
  BOOST_TEST(fetched.d == 2.5);
  w.free();
}

BOOST_AUTO_TEST_SUITE_END()
