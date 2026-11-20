// Compile-pass control: communicator factories added with the advanced creation surface.
// This is a compile-only TU; it does not run and must not call MPI.

#include "mp1/mp1.hpp"

#include <string_view>
#include <utility>


void communicator_advanced_forms(const mp1::mpi_communicator& comm, const mp1::group& local,
                                 const mp1::group& remote, const mp1::info& hints) {
  mp1::request copy = mp1::communicator::idup(comm);
  mp1::request copy_with_info = mp1::communicator::idup_with_info(comm, hints);
  (void)copy;
  (void)copy_with_info;

  mp1::communicator from_group = mp1::communicator::create_from_group(
      local, "mp1-control-tag", hints, mp1::error_handler::returns);
  mp1::communicator from_group2 = mp1::communicator::create_from_group(
      local, std::string_view{"mp1-control-tag"}, mp1::info{}, mp1::error_handler::are_fatal);
  mp1::communicator inter = mp1::communicator::intercomm_create_from_groups(
      local, 0, remote, 0, "mp1-control-tag", hints, mp1::error_handler::returns);
  mp1::communicator inter2 = mp1::communicator::intercomm_create_from_groups(
      local, 0, remote, 0, std::string_view{"mp1-control-tag"}, mp1::info{},
      mp1::error_handler::abort);
  (void)from_group;
  (void)from_group2;
  (void)inter;
  (void)inter2;

  // take_communicator() takes an lvalue request and returns an owning communicator.
  if (copy.has_result()) {
    mp1::communicator taken = mp1::take_communicator(copy);
    (void)taken;
  }
}
