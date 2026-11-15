// Positive control for the file dense nonblocking probes: every contiguous-array, vector,
// array, string and span spelling takes a named lvalue (or a borrowed span), whose address
// stays stable until the request completes.

#include "mp1/mp1.hpp"

#include <array>
#include <span>
#include <string>
#include <vector>

void dense_nonblocking_lvalues(mp1::file& f) {
  int buffer[4] = {0, 0, 0, 0};
  std::array<int, 4> fixed = {0, 0, 0, 0};
  std::vector<int> vector(4, 0);
  std::string text = "text";
  std::span<int> span{buffer};

  (void)f.iread_at(0, std::span<int>{buffer});
  (void)f.iread_at(0, fixed);
  (void)f.iread_at(0, vector);
  (void)f.iread_at(0, text);
  (void)f.iread_at(0, span);
  (void)f.iwrite_at(0, std::span<int>{buffer});
  (void)f.iwrite_at(0, fixed);
  (void)f.iwrite_at(0, vector);
  (void)f.iwrite_at(0, text);
  (void)f.iwrite_at(0, span);

  (void)f.iread(std::span<int>{buffer});
  (void)f.iread(fixed);
  (void)f.iread(vector);
  (void)f.iread(text);
  (void)f.iwrite(std::span<int>{buffer});
  (void)f.iwrite(fixed);
  (void)f.iwrite(vector);
  (void)f.iwrite(text);

  (void)f.iread_shared(std::span<int>{buffer});
  (void)f.iread_shared(fixed);
  (void)f.iread_shared(vector);
  (void)f.iread_shared(text);
  (void)f.iwrite_shared(std::span<int>{buffer});
  (void)f.iwrite_shared(fixed);
  (void)f.iwrite_shared(vector);
  (void)f.iwrite_shared(text);

  (void)f.iread_at_all(0, std::span<int>{buffer});
  (void)f.iread_at_all(0, fixed);
  (void)f.iread_at_all(0, vector);
  (void)f.iread_at_all(0, text);
  (void)f.iwrite_at_all(0, std::span<int>{buffer});
  (void)f.iwrite_at_all(0, fixed);
  (void)f.iwrite_at_all(0, vector);
  (void)f.iwrite_at_all(0, text);

  (void)f.iread_all(std::span<int>{buffer});
  (void)f.iread_all(fixed);
  (void)f.iread_all(vector);
  (void)f.iread_all(text);
  (void)f.iwrite_all(std::span<int>{buffer});
  (void)f.iwrite_all(fixed);
  (void)f.iwrite_all(vector);
  (void)f.iwrite_all(text);
}
