// MPI identity/version gate (scenario, one rank).
//
// MPI 4.1 §9.1.1 (Version Inquiries): MPI_VERSION/MPI_SUBVERSION are the
// compile-time version of the standard and MPI_Get_version returns the pair at
// run time; both are always available, including before MPI_Init and after
// MPI_Finalize.  §11.4.1/Table 11.1 lists MPI_Get_version and
// MPI_Get_library_version among the functions that may be called at any time.
//
// The gate fails when the header macros, the runtime pair, the CMake-detected
// implementation and the provider family disagree.  This is the check that
// catches a stale MPI include path, a mismatched mpiexec, or a header/runtime
// version disagreement such as "Open MPI 5.0.7 advertises MPI 3.1".

#include "mp1/mp1.hpp"

#include <cstdio>
#include <cstring>
#include <mpi.h>
#include <string>

#if !(defined(MPI_VERSION)) || MPI_VERSION < 4 || \
    (MPI_VERSION == 4 && (!defined(MPI_SUBVERSION) || MPI_SUBVERSION < 1))
#error "mp1 requires MPI 4.1 or newer (MPI 4.1 §9.1.1)"
#endif

namespace {

  int failures = 0;

  void expect(bool condition, const char* what) {
    if (!condition) {
      std::fprintf(stderr, "mpi_identity: FAIL: %s\n", what);
      ++failures;
    }
  }

  std::string version_string(int major, int minor) {
    return std::to_string(major) + "." + std::to_string(minor);
  }

  const char* family_of(const char* library) {
    if (library == nullptr) {
      return "unknown";
    }
    if (std::strstr(library, "Open MPI") != nullptr) {
      return "Open MPI";
    }
    if (std::strstr(library, "MPICH") != nullptr || std::strstr(library, "Hydra") != nullptr ||
        std::strstr(library, "mpich") != nullptr) {
      return "MPICH";
    }
    return "unknown";
  }

}  // namespace

int main(int argc, char** argv) {
  static_cast<void>(argc);
  static_cast<void>(argv);

  // Before MPI_Init: version inquiries are always available (§9.1.1,
  // §11.4.1/Table 11.1).
  int runtime_major = -1;
  int runtime_minor = -1;
  expect(MPI_Get_version(&runtime_major, &runtime_minor) == MPI_SUCCESS,
         "MPI_Get_version before MPI_Init");
  expect(runtime_major == MPI_VERSION, "runtime MPI_VERSION matches the header macro");
  expect(runtime_minor == MPI_SUBVERSION, "runtime MPI_SUBVERSION matches the header macro");
  expect(runtime_major > 4 || (runtime_major == 4 && runtime_minor >= 1),
         "runtime implementation advertises MPI 4.1 or newer");

  char library[MPI_MAX_LIBRARY_VERSION_STRING + 1] = {};
  int library_length = 0;
  expect(MPI_Get_library_version(library, &library_length) == MPI_SUCCESS,
         "MPI_Get_library_version before MPI_Init");
  expect(library_length > 0 && library_length <= MPI_MAX_LIBRARY_VERSION_STRING,
         "MPI_Get_library_version length");
  const char* family = family_of(library);
  expect(std::strcmp(family, "unknown") != 0,
         "library version names Open MPI or MPICH");

  const std::pair<int, int> mp1_version = mp1::get_version();
  expect(mp1_version.first == runtime_major, "mp1::get_version major");
  expect(mp1_version.second == runtime_minor, "mp1::get_version minor");

#ifdef MP1_TEST_MPI_CMAKE_MAJOR_MINOR
  const std::string cmake_version = MP1_TEST_MPI_CMAKE_MAJOR_MINOR;
  expect(cmake_version == version_string(runtime_major, runtime_minor),
         "CMake-detected MPI major.minor matches MPI_Get_version");
#endif
#ifdef MP1_TEST_MPI_EXPECTED_FAMILY
  expect(std::strcmp(family, MP1_TEST_MPI_EXPECTED_FAMILY) == 0,
         "provider family matches the CMake-detected implementation");
#endif

  mp1::init();
  expect(mp1::initialized(), "mp1::initialized after init");
  expect(!mp1::finalized(), "not finalized after init");
  const std::pair<int, int> active_version = mp1::get_version();
  expect(active_version == mp1_version, "mp1::get_version stable across init");
  expect(mp1::get_library_version().find(library) != std::string::npos ||
             std::string{library}.find(mp1::get_library_version()) != std::string::npos,
         "mp1::get_library_version agrees with MPI_Get_library_version");
  mp1::finalize();
  expect(mp1::finalized(), "mp1::finalized after finalize");

  // After MPI_Finalize: the same always-available inquiries must still work
  // (§11.2.2: only §11.4.1 functions may be called after finalize).
  int after_major = -1;
  int after_minor = -1;
  expect(MPI_Get_version(&after_major, &after_minor) == MPI_SUCCESS,
         "MPI_Get_version after MPI_Finalize");
  expect(after_major == runtime_major && after_minor == runtime_minor,
         "runtime version stable across finalize");

  if (failures != 0) {
    std::fprintf(stderr, "mpi_identity: %d failure(s)\n", failures);
    return 1;
  }
  std::printf("mpi_identity: PASS %s %s (%s)\n", family,
              version_string(runtime_major, runtime_minor).c_str(), library);
  return 0;
}
