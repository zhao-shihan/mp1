// Public-API compile check: datatypes, layouts, reflections.
//
// `layout<void>` and the public `layout<T>::from_range` are removed and are
// deliberately absent here; the typed layout factories remain.
//
// MPI 4.1 basis: Ch. 5 (datatypes) defines derived datatypes and their
// metadata inquiries; the wrapper factories are thin bindings of the
// MPI_Type_* constructors.

#include "mp1/mp1.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace {

  struct reflected_point {
    int x = 0;
    double y = 0;
  };

  template<class T>
  struct reflected_box {
    T value{};
  };

}  // namespace

MP1_REFLECTION(reflected_point, x, y)
MP1_REFLECTION_TEMPLATE((class T), (reflected_box<T>), value)

void api_datatypes() {
  const MPI_Datatype integer = mp1::native_datatype<int>();
  const MPI_Datatype point = mp1::native_datatype<reflected_point>();
  const MPI_Datatype boxed = mp1::native_datatype<reflected_box<int>>();
  (void)integer;
  (void)point;
  (void)boxed;
}

void api_layouts() {
  mp1::layout<int> value;
  const bool valid = value.is_valid();
  const bool dense = value.dense();
  const mp1::count_t size = value.size();
  const mp1::count_t bytes = value.byte_size();
  const mp1::count_t extent = value.byte_extent();
  const mp1::count_t lower = value.byte_lower_bound();
  const mp1::count_t upper = value.byte_upper_bound();
  const mp1::count_t true_lower = value.true_byte_lower_bound();
  const mp1::count_t true_extent = value.true_byte_extent();
  const mp1::count_t true_upper = value.true_byte_upper_bound();
  const MPI_Datatype native = value.native_handle();
  (void)valid;
  (void)dense;
  (void)size;
  (void)bytes;
  (void)extent;
  (void)lower;
  (void)upper;
  (void)true_lower;
  (void)true_extent;
  (void)true_upper;
  (void)native;

  value = mp1::layout<int>::contiguous(4);
  value = mp1::layout<int>::contiguous(0);
  mp1::layout<int> contiguous = mp1::layout<int>::contiguous(2);
  mp1::layout<int> repeated = mp1::layout<int>::contiguous(3, contiguous);
  mp1::layout<int> vector = mp1::layout<int>::vector(2, 3, 4);
  mp1::layout<int> vector_block = mp1::layout<int>::vector(2, 3, 4, contiguous);
  mp1::layout<int> strided = mp1::layout<int>::strided_vector(2, 3, 16);
  mp1::layout<int> strided_block = mp1::layout<int>::strided_vector(2, 3, 32, contiguous);
  (void)repeated;
  (void)vector;
  (void)vector_block;
  (void)strided;
  (void)strided_block;

  const std::array indexed_entries{std::pair{mp1::count_t{2}, mp1::count_t{0}},
                                   std::pair{mp1::count_t{1}, mp1::count_t{4}}};
  const mp1::indexed_parameter indexed_parameter{indexed_entries};
  mp1::layout<int> indexed = mp1::layout<int>::indexed(indexed_parameter);
  mp1::layout<int> indexed_block = mp1::layout<int>::indexed(indexed_parameter, contiguous);
  const std::array hindexed_entries{std::pair{mp1::count_t{2}, MPI_Aint{0}},
                                    std::pair{mp1::count_t{1}, MPI_Aint{16}}};
  const mp1::hindexed_parameter hindexed_parameter{hindexed_entries};
  mp1::layout<int> hindexed = mp1::layout<int>::hindexed(hindexed_parameter);
  mp1::layout<int> hindexed_block = mp1::layout<int>::hindexed(hindexed_parameter, contiguous);
  const std::array indexed_block_offsets{mp1::count_t{0}, mp1::count_t{3}};
  const mp1::indexed_block_parameter indexed_block_parameter{indexed_block_offsets};
  mp1::layout<int> block = mp1::layout<int>::indexed_block(2, indexed_block_parameter);
  const std::array indexed_block_with_type_offsets{mp1::count_t{0}, mp1::count_t{2}};
  const mp1::indexed_block_parameter indexed_block_with_type_parameter{
      indexed_block_with_type_offsets};
  mp1::layout<int> block_with_type =
      mp1::layout<int>::indexed_block(1, indexed_block_with_type_parameter, contiguous);
  const std::array hindexed_block_offsets{MPI_Aint{0}, MPI_Aint{16}};
  const mp1::hindexed_block_parameter hindexed_block_parameter{hindexed_block_offsets};
  mp1::layout<int> hblock = mp1::layout<int>::hindexed_block(2, hindexed_block_parameter);
  mp1::layout<int> hblock_with_type =
      mp1::layout<int>::hindexed_block(1, hindexed_block_parameter, contiguous);
  const std::array subarray_dimensions{
      std::array{mp1::count_t{4}, mp1::count_t{2}, mp1::count_t{1}},
      std::array{mp1::count_t{5}, mp1::count_t{3}, mp1::count_t{2}}};
  mp1::subarray_parameter subarray_parameter{subarray_dimensions};
  mp1::layout<int> subarray = mp1::layout<int>::subarray(subarray_parameter);
  mp1::layout<int> subarray_block = mp1::layout<int>::subarray(subarray_parameter, contiguous);
  subarray_parameter.order(mp1::array_order::column_major);
  const mp1::array_order order = subarray_parameter.order();
  subarray_parameter.add(4, 2, 0);
  (void)indexed;
  (void)indexed_block;
  (void)hindexed;
  (void)hindexed_block;
  (void)block;
  (void)block_with_type;
  (void)hblock;
  (void)hblock_with_type;
  (void)subarray;
  (void)subarray_block;
  (void)order;

  mp1::layout<int> copied{contiguous};
  mp1::layout<int> moved{std::move(copied)};
  copied = moved;
  moved = std::move(copied);
  contiguous.swap(moved);
  contiguous.byte_resize(0, 8);
  (void)contiguous;
}

int main() {
  return 0;
}
