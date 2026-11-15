#if !(defined MP1_HPP)

#define MP1_HPP

// C++20 is a hard requirement.  The CMake target raises consumers through
// cxx_std_20; this guard catches a compiler invocation that overrides the
// standard flag (for example CMAKE_CXX_FLAGS=-std=c++17) with one clear error
// instead of a cascade of diagnostics from the implementation headers.
// MSVC reports the language mode in _MSVC_LANG unless /Zc:__cplusplus is set.
#if defined(_MSVC_LANG)
#if _MSVC_LANG < 202002L
#error "mp1 requires C++20"
#endif
#elif defined(__cplusplus) && __cplusplus < 202002L
#error "mp1 requires C++20"
#endif

// Public version macros.  They are generated from the top-level VERSION file
// into mp1/version.hpp by CMake; this header defines none of them itself and
// the generated header is included after the C++20 guard so a pre-C++20 compile
// still reports the guard diagnostic above.
#include "mp1/version.hpp"

// The public include order follows the dependency graph: forward
// declarations and errors first, then lifecycle/status/datatypes, the
// communicator layers, the operation families, and finally the file and spawn
// APIs.  Each impl header is self-contained;
// this order keeps the declarations readable and matches the released layout.
// clang-format off
#include "mp1/impl/fwd.hpp"
#include "mp1/impl/vocabulary.hpp"
#include "mp1/impl/error.hpp"
#include "mp1/impl/environment.hpp"
#include "mp1/impl/status.hpp"
#include "mp1/impl/datatype.hpp"
#include "mp1/impl/layout.hpp"
#include "mp1/impl/info.hpp"
#include "mp1/impl/group.hpp"
#include "mp1/impl/communicator.hpp"
#include "mp1/impl/window.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/buffer.hpp"
#include "mp1/impl/operator.hpp"
#include "mp1/impl/p2p.hpp"
#include "mp1/impl/collective.hpp"
#include "mp1/impl/rma.hpp"
#include "mp1/impl/file.hpp"
#include "mp1/impl/spawn.hpp"
// clang-format on

#endif
