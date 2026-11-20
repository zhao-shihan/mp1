// The stringtag is a C-string vocabulary (std::string_view); mp1::tag is not accepted.

#include "mp1/mp1.hpp"


void create(const mp1::group& g) {
  (void)mp1::communicator::create_from_group(g, mp1::tag{0}, mp1::info{},
                                             mp1::error_handler::returns);
  // An int message tag is not a stringtag either.
  (void)mp1::communicator::create_from_group(g, 0, mp1::info{}, mp1::error_handler::returns);
}
