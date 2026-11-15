#if !(defined MP1_IMPL_FWD_HPP)

#define MP1_IMPL_FWD_HPP

#include "mp1/impl/vocabulary.hpp"


namespace mp1 {

  // count_t, capacity, region<T> and blocks are defined in vocabulary.hpp (the vocabulary
  // split).  This forward header includes it so every current consumer of fwd.hpp keeps the
  // same names.  The transitional `counts`/`displacements`/`layouts<T>`/`layout_view`
  // spellings are not part of the API: the dense descriptor is `blocks` and the typed
  // descriptor is
  // `std::span<const layout<T>>` + `std::span<const count_t>`.

  // Opaque declaration only: environment.hpp defines the enumerators. The fixed underlying
  // type makes the forward-declared enum a complete type usable as a member before that.
  enum class threading_mode : int;

  // Owning, move-only handle of a message matched by mprobe/improbe.  Defined in
  // p2p.hpp, the only consumer.
  class message;

  template<class T>
  class layout;

  class group;
  class info;
  class file;
  class request;
  class mpi_communicator;
  class communicator;
  class mpi_window;
  class window;

}  // namespace mp1

#endif
