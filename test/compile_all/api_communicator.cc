// Public-API compile check: communicator, request, p2p, buffer ownership.
//
// Compile-only.  Every call is compiled but never linked or executed; invalid
// runtime objects are acceptable here.  The removed topology communicators are
// not referenced; these families remain.

#include "mp1/mp1.hpp"

#include <array>
#include <cstddef>
#include <iterator>
#include <list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

  // The C++20 contiguity classifier must recognize every standard character string
  // specialization, and classify list iterators as staged.  The list<pair> lines are the
  // compile regression for the old hand-written classifier: it formed
  // std::basic_string<std::pair<int, int>> in decltype, which libstdc++ 15+ rejects in its
  // class body; std::contiguous_iterator never names basic_string.
  static_assert(std::contiguous_iterator<std::string::iterator>);
  static_assert(std::contiguous_iterator<std::string::const_iterator>);
  static_assert(std::contiguous_iterator<std::wstring::iterator>);
  static_assert(std::contiguous_iterator<std::u8string::iterator>);
  static_assert(std::contiguous_iterator<std::u16string::iterator>);
  static_assert(std::contiguous_iterator<std::u32string::iterator>);
  static_assert(!std::contiguous_iterator<std::list<int>::iterator>);
  static_assert(!std::contiguous_iterator<std::list<std::pair<int, int>>::iterator>);

}  // namespace

void api_communicator(const mp1::mpi_communicator& other) {
  mp1::mpi_communicator borrowed = mp1::mpi_communicator::borrow(MPI_COMM_WORLD);
  const bool valid = borrowed.is_valid();
  const MPI_Comm native = borrowed.native_handle();
  const int size = borrowed.size();
  const int rank = borrowed.rank();
  const bool inter = borrowed.is_inter();
  const int remote = borrowed.remote_size();
  const mp1::group remote_group = borrowed.remote_group();
  const mp1::equality_type equality = borrowed.compare(other);
  const mp1::info table = borrowed.get_info();
  borrowed.set_info(table);
  (void)valid;
  (void)native;
  (void)size;
  (void)rank;
  (void)inter;
  (void)remote;
  (void)remote_group;
  (void)equality;

  mp1::communicator owned = mp1::communicator::adopt(MPI_COMM_NULL);
  mp1::communicator duplicate = mp1::communicator::dup(other);
  mp1::communicator created = mp1::communicator::create(other, remote_group);
  mp1::communicator merged = mp1::communicator::merge(other, true);
  mp1::communicator split = mp1::communicator::split(other, 0);
  mp1::communicator split_type = mp1::communicator::split_type(other, mp1::comm_type_shared);
  (void)duplicate;
  (void)created;
  (void)merged;
  (void)split;
  (void)split_type;

  // Owner/view split and release policy.
  const mp1::mpi_communicator owner_view = duplicate.view();
  const mp1::mpi_communicator implicit_view = duplicate;  // implicit by-value view borrow
  const mp1::release_policy policy = duplicate.policy();
  (void)owner_view;
  (void)implicit_view;
  (void)policy;
  MPI_Comm detached = duplicate.release();
  mp1::communicator connected = mp1::communicator::adopt_connected(detached);
  connected.disconnect();

  owned.free();
  duplicate.free();
  created.free();
  merged.free();
  split.free();
  split_type.free();
}

void api_world_factories() {
  mp1::mpi_communicator world = mp1::comm_world();
  mp1::mpi_communicator self = mp1::comm_self();
  mp1::mpi_communicator null = mp1::comm_null();
  mp1::communicator parent = mp1::parent();
  (void)world;
  (void)self;
  (void)null;
  (void)parent;
}

void api_request() {
  mp1::request r;
  bool valid = r.is_valid();
  bool active = r.is_active();
  bool persistent = r.is_persistent();
  (void)valid;
  (void)active;
  (void)persistent;
  r.start();
  mp1::status waited = r.wait();
  const std::optional<mp1::status> tested = r.test();
  const std::optional<mp1::status> polled = r.get_status();
  (void)waited;
  (void)tested;
  (void)polled;
  r.cancel();
  r.release();

  mp1::request from_send = mp1::request::from_send(MPI_REQUEST_NULL);
  mp1::request from_recv = mp1::request::from_recv(MPI_REQUEST_NULL);
  mp1::request from_persistent_send = mp1::request::from_persistent_send(MPI_REQUEST_NULL);
  mp1::request from_persistent_recv = mp1::request::from_persistent_recv(MPI_REQUEST_NULL);
  mp1::request from_collective = mp1::request::from_collective(MPI_REQUEST_NULL);
  mp1::request from_persistent_collective =
      mp1::request::from_persistent_collective(MPI_REQUEST_NULL);
  mp1::request from_file = mp1::request::from_file(MPI_REQUEST_NULL);
  mp1::request from_matched_recv = mp1::request::from_matched_recv(MPI_REQUEST_NULL);
  mp1::request from_flush = mp1::request::from_flush(MPI_REQUEST_NULL);
  mp1::request from_rma = mp1::request::from_rma(MPI_REQUEST_NULL);
  (void)from_send;
  (void)from_recv;
  (void)from_persistent_send;
  (void)from_persistent_recv;
  (void)from_collective;
  (void)from_persistent_collective;
  (void)from_file;
  (void)from_matched_recv;
  (void)from_flush;
  (void)from_rma;

  std::vector<mp1::request> requests;
  std::vector<mp1::status> statuses = mp1::wait_all(requests);
  const std::optional<std::vector<mp1::status>> all = mp1::test_all(requests);
  const mp1::completion any = mp1::wait_any(requests);
  const std::optional<mp1::completion> test_any = mp1::test_any(requests);
  const std::vector<mp1::completion> some = mp1::wait_some(requests);
  const std::optional<std::vector<mp1::completion>> test_some = mp1::test_some(requests);
  const std::optional<std::vector<mp1::status>> status_all = mp1::get_status_all(requests);
  const std::optional<mp1::completion> status_any = mp1::get_status_any(requests);
  const std::optional<std::vector<mp1::completion>> status_some =
      mp1::get_status_some(requests);
  mp1::start_all(requests);
  (void)statuses;
  (void)all;
  (void)any;
  (void)test_any;
  (void)some;
  (void)test_some;
  (void)status_all;
  (void)status_any;
  (void)status_some;
}

void api_p2p(const mp1::mpi_communicator& c, int rank) {
  int scalar = 0;
  int array[4] = {0, 0, 0, 0};
  int other[4] = {0, 0, 0, 0};
  std::array<int, 4> fixed = {0, 0, 0, 0};
  std::vector<int> dynamic(4, 0);
  std::vector<int> received_dynamic(4, 0);
  const std::vector<int> const_dynamic(4, 0);
  std::list<int> sequence(4, 0);
  mp1::layout<int> layout = mp1::layout<int>::contiguous(4);

  // Blocking send: value / range (contiguous, non-contiguous, view) / region.  The empty
  // spelling is std::span<T>{}; there are no *_empty names.
  mp1::send(c, scalar, rank);
  mp1::send(c, array, rank);
  mp1::send(c, fixed, rank);
  mp1::send(c, dynamic, rank);
  mp1::send(c, const_dynamic, rank);
  mp1::send(c, std::span(array), rank);
  mp1::send(c, std::span<const int>(dynamic), rank);
  mp1::send(c, sequence, rank);
  mp1::send(c, std::string_view{"abc"}, rank);
  mp1::send(c, mp1::region{array, layout}, rank);
  mp1::send(c, std::span<int>{}, rank);
  mp1::bsend(c, scalar, rank);
  mp1::ssend(c, dynamic, rank);
  mp1::rsend(c, mp1::region{array, layout}, rank);

  // Nonblocking send: an lvalue value, a borrowable range, or a region (layout duplicated).
  mp1::request r = mp1::isend(c, scalar, rank);
  mp1::request rb = mp1::ibsend(c, scalar, rank);
  mp1::request rs = mp1::issend(c, scalar, rank);
  mp1::request rr = mp1::irsend(c, scalar, rank);
  mp1::request rp = mp1::isend(c, std::span(array), rank);
  mp1::request rl = mp1::isend(c, mp1::region{array, layout}, rank);
  mp1::request rc = mp1::isend(c, dynamic, rank);
  mp1::request ri = mp1::isend(c, sequence, rank);
  (void)r;
  (void)rb;
  (void)rs;
  (void)rr;
  (void)rp;
  (void)rl;
  (void)rc;
  (void)ri;

  // Receive: five blocking policies.
  mp1::status received = mp1::recv(c, scalar, rank);          // 1 value
  received = mp1::recv(c, std::span(array), rank);            // 2 fixed
  received = mp1::recv(c, array, rank);                       // 2 fixed C array
  received = mp1::recv(c, fixed, rank);                       // 2 fixed array
  received = mp1::recv(c, mp1::region{array, layout}, rank);  // 3 typed region
  received = mp1::recv(c, dynamic, rank);                     // 4 auto-size
  received = mp1::recv(c, dynamic, mp1::capacity{4}, rank);   // 5 bounded
  received = mp1::recv(c, std::span<int>{}, rank);            // empty
  (void)received;

  // Nonblocking receive: 1, 2, 3, 5 (no auto-size).
  mp1::request ir = mp1::irecv(c, scalar, rank);
  mp1::request irp = mp1::irecv(c, std::span(array), rank);
  mp1::request irl = mp1::irecv(c, mp1::region{array, layout}, rank);
  mp1::request irc = mp1::irecv(c, dynamic, mp1::capacity{4}, rank);
  mp1::request irf = mp1::irecv(c, fixed, rank);
  (void)ir;
  (void)irp;
  (void)irl;
  (void)irc;
  (void)irf;

  // Persistent send: the same three forms; a container lvalue is revalidated every start.
  mp1::request ps = mp1::send_init(c, scalar, rank);
  mp1::request pfr = mp1::recv_init(c, std::span(array), rank);
  mp1::request pr = mp1::recv_init(c, scalar, rank);
  mp1::request psc = mp1::send_init(c, dynamic, rank);
  mp1::request pbc = mp1::bsend_init(c, dynamic, rank);
  mp1::request pssc = mp1::ssend_init(c, dynamic, rank);
  mp1::request prc = mp1::rsend_init(c, mp1::region{array, layout}, rank);
  mp1::request pcrc = mp1::recv_init(c, dynamic, mp1::capacity{4}, rank);
  mp1::request prl = mp1::recv_init(c, mp1::region{array, layout}, rank);
  (void)ps;
  (void)pfr;
  (void)pr;
  (void)psc;
  (void)pbc;
  (void)pssc;
  (void)prc;
  (void)pcrc;
  (void)prl;

  // Combined: one declaration per family; the three forms are parameter alternatives.
  received = mp1::sendrecv(c, scalar, rank, mp1::tag{0}, scalar, rank, mp1::tag{1});
  received = mp1::sendrecv(c, std::span(array), rank, mp1::tag{0}, std::span(other), rank,
                           mp1::tag{1});
  received = mp1::sendrecv(c, dynamic, rank, mp1::tag{0}, received_dynamic, rank, mp1::tag{1});
  received = mp1::sendrecv(c, sequence, rank, mp1::tag{0}, received_dynamic, rank, mp1::tag{1});
  received = mp1::sendrecv(c, mp1::region{array, layout}, rank, mp1::tag{0},
                           mp1::region{other, layout}, rank, mp1::tag{1});
  received = mp1::sendrecv(c, std::span<int>{}, rank, mp1::tag{0}, std::span<int>{}, rank,
                           mp1::tag{1});
  mp1::status replaced = mp1::sendrecv_replace(c, scalar, rank, mp1::tag{0}, rank, mp1::tag{1});
  replaced = mp1::sendrecv_replace(c, std::span(array), rank, mp1::tag{0}, rank, mp1::tag{1});
  replaced = mp1::sendrecv_replace(c, received_dynamic, rank, mp1::tag{0}, rank, mp1::tag{1});
  replaced = mp1::sendrecv_replace(c, mp1::region{other, layout}, rank, mp1::tag{0}, rank,
                                   mp1::tag{1});
  (void)received;
  (void)replaced;

  // Combined nonblocking: value lvalue / borrowable range / region.
  mp1::request irs = mp1::isendrecv(c, scalar, rank, mp1::tag{0}, scalar, rank, mp1::tag{1});
  mp1::request isr_span = mp1::isendrecv(c, std::span(array), rank, mp1::tag{0},
                                         std::span(other), rank, mp1::tag{1});
  mp1::request isr_region = mp1::isendrecv(c, mp1::region{array, layout}, rank, mp1::tag{0},
                                           mp1::region{other, layout}, rank, mp1::tag{1});
  mp1::request ircv =
      mp1::isendrecv(c, dynamic, rank, mp1::tag{0}, received_dynamic, rank, mp1::tag{1});
  mp1::request irrs = mp1::isendrecv_replace(c, scalar, rank, mp1::tag{0}, rank, mp1::tag{1});
  mp1::request irrp =
      mp1::isendrecv_replace(c, std::span(array), rank, mp1::tag{0}, rank, mp1::tag{1});
  mp1::request irrl = mp1::isendrecv_replace(c, mp1::region{array, layout}, rank, mp1::tag{0},
                                             rank, mp1::tag{1});
  mp1::request irrc =
      mp1::isendrecv_replace(c, received_dynamic, rank, mp1::tag{0}, rank, mp1::tag{1});
  (void)irs;
  (void)isr_span;
  (void)isr_region;
  (void)ircv;
  (void)irrs;
  (void)irrp;
  (void)irrl;
  (void)irrc;

  // Matched receive: five policies for mrecv, four for imrecv, message& only.
  mp1::message value_message = mp1::mprobe(c, rank, mp1::tag{0});
  received = mp1::mrecv(scalar, value_message);
  mp1::message fixed_message = mp1::mprobe(c, rank, mp1::tag{0});
  received = mp1::mrecv(std::span(array), fixed_message);
  mp1::message region_message = mp1::mprobe(c, rank, mp1::tag{0});
  received = mp1::mrecv(mp1::region{array, layout}, region_message);
  mp1::message auto_message = mp1::mprobe(c, rank, mp1::tag{0});
  received = mp1::mrecv(dynamic, auto_message);
  mp1::message bounded_message = mp1::mprobe(c, rank, mp1::tag{0});
  received = mp1::mrecv(dynamic, mp1::capacity{4}, bounded_message);
  (void)received;
  mp1::message ivalue_message = mp1::mprobe(c, rank, mp1::tag{0});
  mp1::request imr = mp1::imrecv(scalar, ivalue_message);
  mp1::message ifixed_message = mp1::mprobe(c, rank, mp1::tag{0});
  mp1::request imf = mp1::imrecv(std::span(array), ifixed_message);
  mp1::message iregion_message = mp1::mprobe(c, rank, mp1::tag{0});
  mp1::request iml = mp1::imrecv(mp1::region{array, layout}, iregion_message);
  mp1::message ibounded_message = mp1::mprobe(c, rank, mp1::tag{0});
  mp1::request imb = mp1::imrecv(dynamic, mp1::capacity{4}, ibounded_message);
  (void)imr;
  (void)imf;
  (void)iml;
  (void)imb;

  const mp1::status probed = mp1::probe(c, rank, mp1::tag{0});
  const std::optional<mp1::status> iprobed = mp1::iprobe(c, rank, mp1::tag{0});
  const std::optional<mp1::message> improbed = mp1::improbe(c, rank, mp1::tag{0});
  (void)probed;
  (void)iprobed;
  (void)improbed;
}

int main() {
  return 0;
}
