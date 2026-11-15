// Info test: null-safe state, key/value/size/key(n), copy/move, and the
// after-finalize destructor path.
//
// The shared test harness links the Boost.Test main; this file defines neither main nor
// the Boost.Test module macro. MPI is initialized by the global fixture below.

#include "mp1/mp1.hpp"

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <cstddef>
#include <exception>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>


namespace {

  template<class F>
  bool throws_invalid_argument(F&& function) {
    try {
      function();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  // An info object kept alive until after mp1::finalize(): MPI 4.1 Table 11.1 makes every
  // MPI_Info_* call always available, so the check verifies that accessors still work and
  // that the destructor frees the handle (it must not be counted as abandoned).  The check
  // object is declared before the storage, so it is destroyed after the global fixture
  // finalized MPI.
  alignas(mp1::info) unsigned char after_finalize_storage[sizeof(mp1::info)];
  bool after_finalize_constructed = false;

  struct info_after_finalize_check {
    ~info_after_finalize_check() {
      if (not after_finalize_constructed)
        return;
      if (not mp1::finalized())
        return;  // The fixture did not finalize MPI; leave the object alone.
      auto* table = std::launder(reinterpret_cast<mp1::info*>(after_finalize_storage));
      if (table->native_handle() == MPI_INFO_NULL)
        return;  // Never constructed a handle; nothing to free.

      // MPI 4.1 §11.4.1/Table 11.1: every MPI_Info_* routine is always available, so all
      // accessors and the destructor's MPI_Info_free work after MPI_FINALIZE.  A surviving
      // info is freed, not abandoned.
      try {
        (void)table->size();
        (void)table->value("after");
        (void)table->key(0);
        table->set("after-finalize", "works");
        if (not table->remove("after-finalize"))
          std::terminate();
        const std::size_t before = mp1::detail::abandoned_handle_count();
        table->~info();
        if (mp1::detail::abandoned_handle_count() != before)
          std::terminate();
      } catch (...) {
        std::terminate();
      }
      after_finalize_constructed = false;
    }
  };

  info_after_finalize_check after_finalize_check_instance;

}  // namespace


struct mp1_info_fixture {
  mp1_info_fixture() {
    mp1::init();
  }

  ~mp1_info_fixture() {
    try {
      mp1::finalize();
      // NOLINTNEXTLINE(bugprone-empty-catch): fixture destructor
    } catch (...) {
      // Another owner may finalize; a fixture destructor must not throw.
    }
  }
};

BOOST_TEST_GLOBAL_FIXTURE(mp1_info_fixture);

BOOST_AUTO_TEST_SUITE(info_test)

BOOST_AUTO_TEST_CASE(null_info_is_mpi_free) {
  mp1::info empty_info;
  BOOST_TEST(empty_info.native_handle() == MPI_INFO_NULL);
  BOOST_TEST(empty_info.size() == 0);
  BOOST_TEST(not empty_info.value("missing").has_value());
  BOOST_TEST(throws_invalid_argument([&empty_info] { (void)empty_info.key(0); }));
  BOOST_TEST(throws_invalid_argument([&empty_info] { (void)empty_info.key(-1); }));

  // remove() on a null info is a documented no-op; it must not create a handle.
  empty_info.remove("missing");
  BOOST_TEST(empty_info.native_handle() == MPI_INFO_NULL);
  BOOST_TEST(empty_info.size() == 0);

  // Copying a null info stays null without MPI_Info_dup(MPI_INFO_NULL).
  mp1::info copied{empty_info};
  BOOST_TEST(copied.native_handle() == MPI_INFO_NULL);
  BOOST_TEST(copied.size() == 0);

  mp1::info assigned;
  assigned.set("temporary", "value");
  BOOST_TEST(assigned.size() == 1);
  assigned = empty_info;
  BOOST_TEST(assigned.native_handle() == MPI_INFO_NULL);
  BOOST_TEST(assigned.size() == 0);

  mp1::info moved{std::move(empty_info)};
  BOOST_TEST(moved.native_handle() == MPI_INFO_NULL);
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(empty_info.native_handle() == MPI_INFO_NULL);
}

BOOST_AUTO_TEST_CASE(set_get_size_key_remove) {
  mp1::info table;
  table.set("Douglas Adams", "The Hitchhiker's Guide to the Galaxy");
  BOOST_TEST(table.native_handle() != MPI_INFO_NULL);
  BOOST_TEST(table.size() == 1);
  const auto hitchhiker = table.value("Douglas Adams");
  BOOST_TEST(hitchhiker.has_value());
  if (hitchhiker)
    BOOST_TEST(*hitchhiker == "The Hitchhiker's Guide to the Galaxy");

  // set() copies a non-null-terminated string_view; it must not read past the view.
  const std::string key_storage{"Isaac Asimov"};
  const std::string value_storage{"Nightfall"};
  table.set(std::string_view{key_storage.data(), 12},
            std::string_view{value_storage.data(), 9});
  BOOST_TEST(table.size() == 2);
  const auto nightfall = table.value("Isaac Asimov");
  BOOST_TEST(nightfall.has_value());
  if (nightfall)
    BOOST_TEST(*nightfall == "Nightfall");

  BOOST_TEST(not table.value("no such thing").has_value());

  // key(n) returns every key exactly once; MPI does not define key order.
  std::vector<std::string> keys;
  keys.reserve(static_cast<std::size_t>(table.size()));
  for (int i = 0; i < table.size(); ++i)
    keys.push_back(table.key(i));
  std::sort(keys.begin(), keys.end());
  const std::vector<std::string> expected{"Douglas Adams", "Isaac Asimov"};
  BOOST_TEST(keys == expected);
  BOOST_TEST(throws_invalid_argument([&table] { (void)table.key(-1); }));
  BOOST_TEST(throws_invalid_argument([&table] { (void)table.key(table.size()); }));

  table.remove("Douglas Adams");
  BOOST_TEST(table.size() == 1);
  BOOST_TEST(not table.value("Douglas Adams").has_value());

  // A missing key is specified as an mpi_error; the default MPI error handler is fatal in
  // the supported runtimes, so that path stays a compile/contract check rather than a runtime
  // probe that would abort every rank.

  table.remove("Isaac Asimov");
  BOOST_TEST(table.size() == 0);
  BOOST_TEST(table.native_handle() != MPI_INFO_NULL);
}

BOOST_AUTO_TEST_CASE(copy_move_assignment_are_independent) {
  mp1::info original;
  original.set("key", "original");

  mp1::info copy{original};
  BOOST_TEST(copy.native_handle() != MPI_INFO_NULL);
  BOOST_TEST(copy.native_handle() != original.native_handle());
  original.set("key", "changed");
  const auto copied_value = copy.value("key");
  const auto original_value = original.value("key");
  BOOST_TEST(copied_value.has_value());
  BOOST_TEST(original_value.has_value());
  if (copied_value)
    BOOST_TEST(*copied_value == "original");
  if (original_value)
    BOOST_TEST(*original_value == "changed");

  // Copy assignment is copy-and-swap: the target's old handle is released and the new table
  // is independent of the source.
  mp1::info assigned;
  assigned.set("stale", "value");
  assigned = original;
  BOOST_TEST(assigned.native_handle() != original.native_handle());
  BOOST_TEST(not assigned.value("stale").has_value());
  const auto assigned_value = assigned.value("key");
  BOOST_TEST(assigned_value.has_value());
  if (assigned_value)
    BOOST_TEST(*assigned_value == "changed");

  // Self-assignment keeps the handle and the value.
  mp1::info& alias = assigned;
  assigned = alias;
  BOOST_TEST(assigned.size() == 1);
  const auto aliased_value = assigned.value("key");
  BOOST_TEST(aliased_value.has_value());
  if (aliased_value)
    BOOST_TEST(*aliased_value == "changed");

  // Move steals the handle; the source becomes a null info.
  mp1::info moved{std::move(original)};
  // NOLINTNEXTLINE(bugprone-use-after-move): intentional moved-from state
  BOOST_TEST(original.native_handle() == MPI_INFO_NULL);
  BOOST_TEST(original.size() == 0);
  BOOST_TEST(moved.native_handle() != MPI_INFO_NULL);
  const auto moved_value = moved.value("key");
  BOOST_TEST(moved_value.has_value());
  if (moved_value)
    BOOST_TEST(*moved_value == "changed");
}

BOOST_AUTO_TEST_CASE(info_after_finalize) {
  // Placement-new one owning info; the post-finalize check above verifies that its accessors
  // still work after finalize, that set() lazily creates/uses the handle, and that the
  // destructor frees the handle instead of abandoning it (MPI 4.1 §11.4.1/Table 11.1).
  auto* table = new (after_finalize_storage) mp1::info{};
  table->set("after", "finalize");
  BOOST_TEST(table->native_handle() != MPI_INFO_NULL);
  BOOST_TEST(table->size() == 1);
  after_finalize_constructed = true;
}

BOOST_AUTO_TEST_CASE(create_create_env_and_validation) {
  // Always-available factories (MPI 4.1 §10 MPI_Info_create / §11.4.1 Table 11.1).
  mp1::info table = mp1::info::create();
  BOOST_TEST(table.is_valid());
  BOOST_TEST(table.empty());
  mp1::info environment_table = mp1::info::create_env();
  BOOST_TEST(environment_table.is_valid());

  int argc = 1;
  char program[] = "mp1-test";
  char* arguments[] = {program, nullptr};
  char** argv = arguments;
  mp1::info forwarded = mp1::info::create_env(argc, argv);
  BOOST_TEST(forwarded.is_valid());

  // Conservative local validation.  MPI 4.1 §10 MPI_Info_set raises MPI_ERR_INFO_KEY/VALUE
  // only when the key/value is *longer* than the maximum; mp1 rejects >= the maximum to stay
  // portable (the portable accepted maximum is MPI_MAX_INFO_KEY/VAL - 1).
  BOOST_TEST(throws_invalid_argument([] { mp1::info{}.set("", "value"); }));
  BOOST_TEST(throws_invalid_argument([] {
    mp1::info{}.set(std::string(static_cast<std::size_t>(MPI_MAX_INFO_KEY), 'k'), "v");
  }));
  BOOST_TEST(throws_invalid_argument([] {
    mp1::info{}.set("key", std::string(static_cast<std::size_t>(MPI_MAX_INFO_VAL), 'v'));
  }));
  const std::string embedded_nul{"a\0b", 3};
  BOOST_TEST(throws_invalid_argument([&embedded_nul] { mp1::info{}.set(embedded_nul, "v"); }));
  BOOST_TEST(
      throws_invalid_argument([&embedded_nul] { mp1::info{}.set("key", embedded_nul); }));

  const std::string maximum_key(static_cast<std::size_t>(MPI_MAX_INFO_KEY) - 1, 'k');
  const std::string maximum_value(static_cast<std::size_t>(MPI_MAX_INFO_VAL) - 1, 'v');
  table.set(maximum_key, maximum_value);
  BOOST_TEST(table.size() == 1);
  const auto stored = table.value(maximum_key);
  BOOST_TEST(stored.has_value());
  if (stored)
    BOOST_TEST(*stored == maximum_value);

  // The conservative policy is shared by every key-taking operation: the exact maximum is
  // rejected before the MPI call by remove() and value() as well as set(), so no provider
  // sees a boundary-length key.
  const std::string too_long_key(static_cast<std::size_t>(MPI_MAX_INFO_KEY), 'k');
  BOOST_TEST(throws_invalid_argument([&] { (void)table.remove(too_long_key); }));
  BOOST_TEST(throws_invalid_argument([&] { (void)table.value(too_long_key); }));
  BOOST_TEST(table.size() == 1);
}

BOOST_AUTO_TEST_CASE(remove_reports_presence) {
  mp1::info table = mp1::info::create();
  // A missing key must not reach MPI_Info_delete (MPI_ERR_INFO_NOKEY is fatal by default,
  // MPI 4.1 §10 MPI_Info_delete); remove reports absence instead.
  BOOST_TEST(not table.remove("missing"));
  table.set("key", "value");
  BOOST_TEST(table.remove("key"));
  BOOST_TEST(not table.remove("key"));
  BOOST_TEST(table.is_valid());
  BOOST_TEST(table.size() == 0);

  // An engaged empty value is distinguishable from an absent key (MPI 4.1 §10
  // MPI_Info_get_string flag semantics).
  table.set("empty", "");
  const auto empty_value = table.value("empty");
  BOOST_TEST(empty_value.has_value());
  if (empty_value)
    BOOST_TEST(empty_value->empty());
  BOOST_TEST(not table.value("absent").has_value());
}

BOOST_AUTO_TEST_SUITE_END()
