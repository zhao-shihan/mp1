// take_communicator(request&) mutates the request: it must not accept a const request.

#include "mp1/mp1.hpp"


void take_const(const mp1::request& request) {
  (void)mp1::take_communicator(request);
}
