// From-group factories require the info argument; there is no default.

#include "mp1/mp1.hpp"


void create(const mp1::group& g) {
  (void)mp1::communicator::create_from_group(g, "mp1-probe", mp1::error_handler::returns);
}
