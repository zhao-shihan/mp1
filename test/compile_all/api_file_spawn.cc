// Public-API compile check: file I/O and dynamic process management.
//
// `file::iread_ordered` and `file::iwrite_ordered` were phantom names for
// ordered nonblocking collective procedures that MPI does not define; the
// surface removal deleted them, so there is nothing to instantiate here.  The
// blocking ordered forms are covered.
//
// MPI 4.1 basis: Ch. 14 (I/O) defines the file handle and the collective/
// nonblocking/ordered operations bound here; Ch. 11.7-11.8 defines the dynamic
// process model used by spawn/spawn_multiple.

#include "mp1/mp1.hpp"

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

void api_file(const mp1::mpi_communicator& c, const std::filesystem::path& path) {
  mp1::file f;
  f.open(c, path, mp1::access_mode::read_write | mp1::access_mode::create);
  const bool open = f.is_open();
  const MPI_File native = f.native_handle();
  const mp1::offset_t bytes = f.size();
  const mp1::access_mode access = f.mode();
  (void)open;
  (void)native;
  (void)bytes;
  (void)access;
  f.resize(static_cast<mp1::offset_t>(sizeof(int)));
  f.preallocate(static_cast<mp1::offset_t>(sizeof(int)));
  f.sync();
  f.set_view(0, "native");
  f.seek(0, mp1::whence_mode::set);
  f.seek_shared(0, mp1::whence_mode::set);
  const mp1::offset_t position = f.position();
  const mp1::offset_t shared_position = f.get_position_shared();
  const mp1::offset_t offset = f.byte_offset(0);
  (void)position;
  (void)shared_position;
  (void)offset;
  f.set_info(mp1::info{});
  const mp1::info table = f.get_info();
  const mp1::group group = f.get_group();
  (void)table;
  (void)group;
  f.set_atomicity(false);
  const bool atomic = f.get_atomicity();
  (void)atomic;
  const MPI_Aint scalar_extent = f.get_type_extent<int>();
  (void)scalar_extent;
  f.set_error_handler(mp1::error_handler::returns);
  const std::optional<mp1::error_handler> handler = f.get_error_handler();
  (void)handler;

  int value = 0;
  int carr[4] = {0, 0, 0, 0};
  std::array<int, 4> fixed = {0, 0, 0, 0};
  std::vector<int> container(4, 0);
  std::string text = "text";
  std::span<int> span{container};
  const mp1::layout<int> layout = mp1::layout<int>::contiguous(1);
  const mp1::layout<int> four = mp1::layout<int>::contiguous(4);
  f.set_view(0, four);
  const MPI_Aint layout_extent = f.get_type_extent(four);
  (void)layout_extent;
  mp1::status s{};
  mp1::request r{};
  (void)s;
  (void)r;
  s = f.read(value);
  s = f.read(container);
  s = f.read(mp1::region{carr, four});
  s = f.read_at(0, value);
  s = f.read_at(0, container);
  s = f.read_at(0, mp1::region{carr, four});
  s = f.read_shared(value);
  s = f.read_shared(container);
  s = f.read_shared(mp1::region{carr, four});
  s = f.read_all(value);
  s = f.read_all(container);
  s = f.read_all(mp1::region{carr, four});
  s = f.read_at_all(0, value);
  s = f.read_at_all(0, container);
  s = f.read_at_all(0, mp1::region{carr, four});
  s = f.read_ordered(value);
  s = f.read_ordered(container);
  s = f.read_ordered(mp1::region{carr, four});
  r = f.iread(value);
  r = f.iread(container);
  r = f.iread(mp1::region{carr, four});
  r = f.iread_at(0, value);
  r = f.iread_at(0, container);
  r = f.iread_at(0, mp1::region{carr, four});
  r = f.iread_shared(value);
  r = f.iread_shared(container);
  r = f.iread_shared(mp1::region{carr, four});
  r = f.iread_all(value);
  r = f.iread_all(container);
  r = f.iread_all(mp1::region{carr, four});
  r = f.iread_at_all(0, value);
  r = f.iread_at_all(0, container);
  r = f.iread_at_all(0, mp1::region{carr, four});
  f.read_at_all_begin(0, value);
  f.read_at_all_begin(0, container);
  f.read_at_all_begin(0, mp1::region{carr, four});
  f.read_all_begin(value);
  f.read_all_begin(container);
  f.read_all_begin(mp1::region{carr, four});
  f.read_ordered_begin(value);
  f.read_ordered_begin(container);
  f.read_ordered_begin(mp1::region{carr, four});
  s = f.write(value);
  s = f.write(container);
  s = f.write(mp1::region{carr, four});
  s = f.write_at(0, value);
  s = f.write_at(0, container);
  s = f.write_at(0, mp1::region{carr, four});
  s = f.write_shared(value);
  s = f.write_shared(container);
  s = f.write_shared(mp1::region{carr, four});
  s = f.write_all(value);
  s = f.write_all(container);
  s = f.write_all(mp1::region{carr, four});
  s = f.write_at_all(0, value);
  s = f.write_at_all(0, container);
  s = f.write_at_all(0, mp1::region{carr, four});
  s = f.write_ordered(value);
  s = f.write_ordered(container);
  s = f.write_ordered(mp1::region{carr, four});
  r = f.iwrite(value);
  r = f.iwrite(container);
  r = f.iwrite(mp1::region{carr, four});
  r = f.iwrite_at(0, value);
  r = f.iwrite_at(0, container);
  r = f.iwrite_at(0, mp1::region{carr, four});
  r = f.iwrite_shared(value);
  r = f.iwrite_shared(container);
  r = f.iwrite_shared(mp1::region{carr, four});
  r = f.iwrite_all(value);
  r = f.iwrite_all(container);
  r = f.iwrite_all(mp1::region{carr, four});
  r = f.iwrite_at_all(0, value);
  r = f.iwrite_at_all(0, container);
  r = f.iwrite_at_all(0, mp1::region{carr, four});
  f.write_at_all_begin(0, value);
  f.write_at_all_begin(0, container);
  f.write_at_all_begin(0, mp1::region{carr, four});
  f.write_all_begin(value);
  f.write_all_begin(container);
  f.write_all_begin(mp1::region{carr, four});
  f.write_ordered_begin(value);
  f.write_ordered_begin(container);
  f.write_ordered_begin(mp1::region{carr, four});
  s = f.read_at_all_end(value);
  s = f.read_at_all_end(container);
  s = f.read_all_end(value);
  s = f.read_all_end(container);
  s = f.read_ordered_end(value);
  s = f.read_ordered_end(container);
  s = f.write_at_all_end(value);
  s = f.write_at_all_end(container);
  s = f.write_all_end(value);
  s = f.write_all_end(container);
  s = f.write_ordered_end(value);
  s = f.write_ordered_end(
      container);  // Every contiguous buffer kind reaches the same constrained member.
  s = f.read_at(0, fixed);
  s = f.read_at(0, carr);
  s = f.read_at(0, text);
  s = f.read_at(0, span);
  s = f.write_at(0, fixed);
  s = f.write_at(0, carr);
  s = f.write_at(0, text);
  s = f.write_at(0, span);
  s = f.write_at(0, std::string_view{"text"});
  r = f.iread_at(0, std::span<int>{carr});
  r = f.iwrite_at(0, std::span<int>{carr});

  f.close();
}

void api_spawn(const mp1::mpi_communicator& c) {
  const mp1::command_line command{"mp1-child", "argument"};
  const mp1::command_lines commands{command, command};
  const std::vector<int> maxprocs{1, 1};
  const std::vector<mp1::info> infos{mp1::info{}, mp1::info{}};

  const mp1::spawn_options single{.maxprocs = 1, .command = command, .info = mp1::info{}};
  mp1::spawn_multiple_options multiple;
  multiple.add(commands[0], maxprocs[0], infos[0]).add(commands[1], maxprocs[1], infos[1]);
  mp1::communicator child = mp1::spawn(c, 0, single);
  mp1::communicator children = mp1::spawn_multiple(c, 0, multiple);
  const bool valid = child.is_valid();
  const int remote = child.remote_size();
  (void)valid;
  (void)remote;
  child.disconnect();
  children.disconnect();

  // The errcodes variants return the owning intercommunicator plus the OUT array.
  mp1::spawn_result with_codes = mp1::spawn_with_errcodes(c, 0, single);
  mp1::spawn_result multiple_with_codes = mp1::spawn_multiple_with_errcodes(c, 0, multiple);
  const std::size_t single_codes = with_codes.errcodes.size();
  const std::size_t multiple_codes = multiple_with_codes.errcodes.size();
  (void)single_codes;
  (void)multiple_codes;
  with_codes.intercomm.disconnect();
  multiple_with_codes.intercomm.disconnect();

  // release() + adopt_connected(): the raw-handle / DPM interop path.
  MPI_Comm raw = child.release();
  mp1::communicator connected = mp1::communicator::adopt_connected(raw);
  connected.disconnect();
}

int main() {
  return 0;
}
