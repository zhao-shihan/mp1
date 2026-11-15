#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mp1/mp1.hpp>
#include <vector>


int main() {
  mp1::environment env;
  // get a reference to communicator "world"
  const auto comm_world{mp1::comm_world()};
  // generate some data
  using value_type = std::array<std::uint8_t, 2>;
  std::vector<value_type> vec;
  for (std::uint8_t i{0}; i < 16; ++i)
    vec.push_back(
        {static_cast<std::uint8_t>(i + 1), static_cast<std::uint8_t>(comm_world.rank() + 1)});
  // write data into file
  try {
    // wrap i/o operations in try-catch block, i/o operations may throw
    mp1::file file;
    // opening a file is collective over all processes within the employed communicator
    file.open(comm_world, "test.bin", mp1::access_mode::create | mp1::access_mode::read_write);
    // set file view; the element type of the view is the value type
    const auto view_type{mp1::layout<value_type>::contiguous(1)};
    file.set_view(0, view_type);
    // write data
    const auto write_layout{mp1::layout<value_type>::contiguous(vec.size())};
    const auto offset{static_cast<mp1::offset_t>(vec.size()) * comm_world.rank()};
    file.write_at_all(offset, mp1::region{vec.data(), write_layout});
    // close file
    file.close();
  } catch (const mp1::mpi_error& error) {
    std::cerr << error.what() << '\n';
  }
  return EXIT_SUCCESS;
}
