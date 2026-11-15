#include <cstdlib>
#include <iostream>
#include <iterator>
#include <mp1/mp1.hpp>
#include <numeric>
#include <vector>


// some structures
struct structure {
  double d{0};
  int i[9]{0, 0, 0, 0, 0, 0, 0, 0, 0};
};


// print elements of structure
template<typename ch, typename tr>
std::basic_ostream<ch, tr>& operator<<(std::basic_ostream<ch, tr>& out, const structure& s) {
  out << '(' << s.d << ",[" << s.i[0];
  for (std::size_t i{1}; i < std::size(s.i); ++i)
    out << ',' << s.i[i];
  return out << "])";
}


struct structure_2 {
  double d{0};
  structure str;
};


// print elements of structure2
template<typename ch, typename tr>
std::basic_ostream<ch, tr>& operator<<(std::basic_ostream<ch, tr>& out, const structure_2& s) {
  return out << '(' << s.d << "," << s.str << ")";
}


// specialize the struct_builder hook for the structure defined above
namespace mp1 {

  template<>
  class struct_builder<structure> : public base_struct_builder<structure> {
  public:
    struct_builder() {
      structure object{};
      this->register_object(object);
      // register each element of struct structure
      this->register_element(object.d);
      this->register_element(object.i);
      // finalize
      this->define_struct();
    }

    static MPI_Datatype get_datatype() {
      detail::require_mpi("mp1: reflected datatype construction");
      static struct_builder instance;
      return instance.handle();
    }
  };

}  // namespace mp1


// MP1_REFLECTION is a convenient macro which creates the required
// specialization of the struct_builder template automatically.  Just
// pass the class name and the public members as arguments to the
// macro.  MP1_REFLECTION is limited to 119 class members.
MP1_REFLECTION(structure_2, d, str)


int main() {
  mp1::environment env;
  const auto comm_world{mp1::comm_world()};
  // run the program with two or more processes
  if (comm_world.size() < 2)
    comm_world.abort(EXIT_FAILURE);
  // send / receive a single structure
  structure str;
  if (comm_world.rank() == 0) {
    str.d = 1;
    std::iota(str.i, str.i + 9, 1);
    mp1::send(comm_world, str, 1);
  }
  if (comm_world.rank() == 1) {
    mp1::recv(comm_world, str, 0);
    std::cout << str << '\n';
  }
  // send / receive a single structure containing another structure
  structure_2 str2;
  if (comm_world.rank() == 0) {
    str2.d = 1;
    str2.str.d = 1;
    std::iota(str2.str.i, str2.str.i + 9, 1);
    mp1::send(comm_world, str2, 1);
  }
  if (comm_world.rank() == 1) {
    mp1::recv(comm_world, str2, 0);
    std::cout << str2 << '\n';
  }
  // send / receive a vector of structures
  const int field_size{8};
  std::vector<structure> str_field(field_size);
  const auto str_field_layout{mp1::layout<structure>::contiguous(field_size)};
  if (comm_world.rank() == 0) {
    // populate vector of structures
    for (int k{0}; k < field_size; ++k) {
      str_field[k].d = k + 1;
      std::iota(str_field[k].i, str_field[k].i + 9, 1 + k);
    }
    // send vector of structures
    mp1::send(comm_world, mp1::region{str_field.data(), str_field_layout}, 1);
  }
  if (comm_world.rank() == 1) {
    // receive vector of structures
    mp1::recv(comm_world, mp1::region{str_field.data(), str_field_layout}, 0);
    for (int k{0}; k < field_size; ++k)
      std::cout << str_field[k] << '\n';
  }

  return EXIT_SUCCESS;
}
