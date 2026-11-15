// Scenario test for the core: lifecycle, status, info and request behavior.
//
// This file intentionally uses only core types (no communicator/layout/file/group) and
// provides its own main so it can be registered as KIND scenario. The CTest registration runs
// it in several modes via MP1_TEST_SCENARIO:
//   owned    - mp1 owns MPI (environment ctor calls init)
//   external - the process calls MPI_Init_thread itself; mp1 adopts
//   none     - MPI is never initialized (rank 1)
//
// The MULTIPLE-only MPI_Init configuration (MPI 4.1 §11.2.1) needs a process-wide MPI_Init
// override, so it is a separate executable: test/test_environment_owned_multiple.cc.

#include "mp1/impl/buffer.hpp"
#include "mp1/impl/environment.hpp"
#include "mp1/impl/info.hpp"
#include "mp1/impl/request.hpp"
#include "mp1/impl/status.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(MP1_TEST_WRAP)
#include "support/mpi_wrap.hpp"
#define MP1_TEST_HAVE_MPI_WRAP 1
#else
#define MP1_TEST_HAVE_MPI_WRAP 0
#endif


namespace {

  // Namespace-scope core objects: default construction is MPI-free and happens before init;
  // their destructors run after main (possibly after the atexit finalizer) and must not call
  // MPI once MPI is inactive. static_info is made non-null while MPI is active so its
  // destructor exercises the abandoned-handle path.
  mp1::info static_info;
  mp1::request static_request;

  int failure_count = 0;

  void report_failure(const char* file, int line, const char* expression) {
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expression);
    ++failure_count;
  }

#define MP1_CHECK(condition)                          \
  do {                                                \
    if (not(condition))                               \
      report_failure(__FILE__, __LINE__, #condition); \
  } while (false)

  template<class F>
  bool throws_runtime_error(F&& f) {
    try {
      f();
    } catch (const std::runtime_error&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class F>
  bool throws_invalid_argument(F&& f) {
    try {
      f();
    } catch (const std::invalid_argument&) {
      return true;
    } catch (...) {
      return false;
    }
    return false;
  }

  template<class F>
  bool throws_nothing(F&& f) {
    try {
      f();
    } catch (...) {
      return false;
    }
    return true;
  }

  bool is_empty_status(const mp1::status& s) {
    return s.source() == MPI_ANY_SOURCE and s.tag() == mp1::any_tag and s.error() == MPI_SUCCESS;
  }

  // After an explicit finalize, no wrapped MPI entry point may be reached.  The wrap shim
  // counts (and refuses) such calls; without it the check is skipped.
  void check_no_mpi_after_finalize() {
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      MP1_CHECK(mpi_wrap::after_finalize_calls() == 0);
#endif
  }

  std::string scenario_from_environment() {
    const char* value = std::getenv("MP1_TEST_SCENARIO");
    if (value == nullptr or *value == '\0')
      return "owned";
    return value;
  }

  // ---- status (empty status paths work before init and after finalize) -------------------
  void check_status() {
    // The provenance byte is allowed to grow the wrapper beyond MPI_Status (the wrapper adds
    // a one-byte provenance tag); it is not a stable-layout type.
    static_assert(sizeof(mp1::status) <= sizeof(MPI_Status) + alignof(MPI_Status),
                  "status must wrap MPI_Status without excessive extra data");
    const mp1::status s;
    MP1_CHECK(s.kind() == mp1::status_kind::normalized);
    MP1_CHECK(s.source() == MPI_ANY_SOURCE);
    MP1_CHECK(s.tag() == mp1::any_tag);
    MP1_CHECK(s.error() == MPI_SUCCESS);
    MP1_CHECK(not s.is_cancelled());  // empty status: no MPI call
    const MPI_Status& native = s.native_handle();
    MP1_CHECK(native.MPI_SOURCE == MPI_ANY_SOURCE);
    MP1_CHECK(native.MPI_TAG == MPI_ANY_TAG);
    MP1_CHECK(native.MPI_ERROR == MPI_SUCCESS);
    mp1::status copy{s};
    MP1_CHECK(copy.source() == MPI_ANY_SOURCE);
    copy = s;
    MP1_CHECK(copy.tag() == mp1::any_tag);
  }

  // ---- info (always available: null paths and real tables work before init/after finalize)
  void check_info_null() {
    mp1::info i;
    MP1_CHECK(i.native_handle() == MPI_INFO_NULL);
    MP1_CHECK(i.size() == 0);
    MP1_CHECK(not i.value("key").has_value());
    MP1_CHECK(throws_invalid_argument([&i] { (void)i.key(0); }));
    MP1_CHECK(throws_invalid_argument([&i] { (void)i.key(-1); }));
    MP1_CHECK(not i.remove("key"));  // null remove reports absent, no MPI call
    MP1_CHECK(throws_invalid_argument([&i] { (void)i.remove(""); }));
    const mp1::info copy{i};  // null copy: no MPI_Info_dup
    MP1_CHECK(copy.native_handle() == MPI_INFO_NULL);
    mp1::info assigned;
    assigned = copy;  // null assignment: no MPI call
    MP1_CHECK(assigned.native_handle() == MPI_INFO_NULL);
    mp1::info moved{std::move(assigned)};
    MP1_CHECK(moved.native_handle() == MPI_INFO_NULL);
    mp1::infos list;
    list.push_back(mp1::info{});
    MP1_CHECK(list.size() == 1);
    MP1_CHECK(list[0].native_handle() == MPI_INFO_NULL);

    // MPI 4.1 Table 11.1: all MPI_Info_* calls are always available, so a real table works
    // with MPI inactive (before init and after finalize).
    mp1::info table = mp1::info::create();
    MP1_CHECK(table.is_valid());
    MP1_CHECK(table.empty());
    table.set("mp1-always-key", "mp1-always-value");
    MP1_CHECK(table.size() == 1);
    MP1_CHECK(not table.empty());
    MP1_CHECK(table.value("mp1-always-key").has_value());
    MP1_CHECK(*table.value("mp1-always-key") == "mp1-always-value");
    MP1_CHECK(table.key(0) == "mp1-always-key");
    MP1_CHECK(table.remove("mp1-always-key"));
    MP1_CHECK(not table.remove("mp1-always-key"));
    mp1::info env_table = mp1::info::create_env();
    MP1_CHECK(env_table.is_valid());
    const mp1::info table_copy{table};
    MP1_CHECK(table_copy.is_valid());
    MP1_CHECK(throws_invalid_argument([] { mp1::info{}.set("", "value"); }));
    MP1_CHECK(throws_invalid_argument(
        [] { mp1::info{}.set(std::string(MPI_MAX_INFO_KEY, 'k'), "v"); }));
    MP1_CHECK(throws_invalid_argument(
        [] { mp1::info{}.set("k", std::string(MPI_MAX_INFO_VAL, 'v')); }));
  }

  // ---- request (null/inactive paths are MPI-free) ---------------------------------------
  void check_request_null() {
    mp1::request r;
    MP1_CHECK(not r.is_valid());
    MP1_CHECK(not r.is_active());
    MP1_CHECK(not r.is_persistent());
    MP1_CHECK(not static_cast<bool>(r));

    const mp1::status waited = r.wait();
    MP1_CHECK(waited.source() == MPI_ANY_SOURCE);
    const std::optional<mp1::status> tested = r.test();
    MP1_CHECK(tested.has_value());
    MP1_CHECK(tested->source() == MPI_ANY_SOURCE);
    const std::optional<mp1::status> polled = r.get_status();
    MP1_CHECK(polled.has_value());
    MP1_CHECK(polled->tag() == mp1::any_tag);

    MP1_CHECK(throws_nothing([&r] { r.cancel(); }));  // invalid: documented no-op
    MP1_CHECK(throws_invalid_argument([&r] { r.start(); }));
    MP1_CHECK(throws_nothing([&r] { r.release(); }));
    MP1_CHECK(not r.is_valid());

    MP1_CHECK(not mp1::request::from_send(MPI_REQUEST_NULL).is_valid());
    MP1_CHECK(not mp1::request::from_recv(MPI_REQUEST_NULL).is_active());
    MP1_CHECK(not mp1::request::from_persistent_send(MPI_REQUEST_NULL).is_active());
    MP1_CHECK(not mp1::request::from_persistent_recv(MPI_REQUEST_NULL).is_active());
    MP1_CHECK(not mp1::request::from_collective(MPI_REQUEST_NULL).is_valid());
    MP1_CHECK(not mp1::request::from_file(MPI_REQUEST_NULL).is_valid());
    MP1_CHECK(not mp1::request::from_matched_recv(MPI_REQUEST_NULL).is_valid());

    std::vector<mp1::request> empty;
    MP1_CHECK(mp1::wait_all(empty).empty());
    const std::optional<std::vector<mp1::status>> all = mp1::test_all(empty);
    MP1_CHECK(all.has_value() and all->empty());
    const mp1::completion any = mp1::wait_any(empty);
    MP1_CHECK(any.first == MPI_UNDEFINED);
    const std::optional<mp1::completion> test_any = mp1::test_any(empty);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): MP1_CHECK adds negation
    MP1_CHECK(test_any.has_value() and test_any->first == MPI_UNDEFINED);
    MP1_CHECK(mp1::wait_some(empty).empty());
    MP1_CHECK(not mp1::test_some(empty).has_value());
    MP1_CHECK(throws_nothing([&empty] { mp1::start_all(empty); }));

    // MPI 4.1 non-destructive queries: empty input is MPI-free even when MPI is inactive.
    const std::optional<std::vector<mp1::status>> status_all = mp1::get_status_all(empty);
    MP1_CHECK(status_all.has_value() and status_all->empty());
    const std::optional<mp1::completion> status_any = mp1::get_status_any(empty);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): MP1_CHECK adds negation
    MP1_CHECK(status_any.has_value() and status_any->first == MPI_UNDEFINED);
    MP1_CHECK(is_empty_status(status_any->second));
    MP1_CHECK(not mp1::get_status_some(empty).has_value());

    // A non-empty query over only-inactive slots is MPI-free: the per-slot loop skips every
    // inactive/null handle and reports the empty status (MPI 4.1 3.7.6, 05 I15).
    std::vector<mp1::request> inactive(2);
    const std::optional<std::vector<mp1::status>> inactive_all = mp1::get_status_all(inactive);
    MP1_CHECK(inactive_all.has_value() and inactive_all->size() == 2);
    MP1_CHECK(is_empty_status((*inactive_all)[0]) and is_empty_status((*inactive_all)[1]));
    const std::optional<mp1::completion> inactive_any = mp1::get_status_any(inactive);
    MP1_CHECK(inactive_any.has_value() and inactive_any->first == MPI_UNDEFINED);
    MP1_CHECK(is_empty_status(inactive_any->second));
    MP1_CHECK(not mp1::get_status_some(inactive).has_value());
  }

  // ---- checks valid before init ---------------------------------------------------------
  void check_process_before_init() {
    MP1_CHECK(not mp1::initialized());
    MP1_CHECK(not mp1::finalized());
    MP1_CHECK(not mp1::available());
    MP1_CHECK(mp1::provided() == mp1::threading_mode::single);

    const std::pair<int, int> version = mp1::get_version();
    MP1_CHECK(version.first >= 4);
    MP1_CHECK(not mp1::get_library_version().empty());

    MP1_CHECK(throws_runtime_error([] { (void)mp1::query_thread(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::is_thread_main(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::processor_name(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::wtime(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::wtick(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::wtime_is_global(); }));
    MP1_CHECK(throws_runtime_error([] { mp1::buffer_attach(nullptr, 0); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::buffer_detach(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::get_hw_resource_info(); }));
    MP1_CHECK(throws_runtime_error([] { mp1::buffer_flush(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::buffer_iflush(); }));
    MP1_CHECK(throws_runtime_error([] { mp1::finalize(); }));

    check_status();
    check_info_null();
    check_request_null();
  }

  // ---- checks valid after finalize ------------------------------------------------------
  void check_process_after_finalize() {
    MP1_CHECK(mp1::initialized());
    MP1_CHECK(mp1::finalized());
    MP1_CHECK(not mp1::available());
    MP1_CHECK(throws_runtime_error([] { mp1::init(); }));
    MP1_CHECK(throws_runtime_error([] { mp1::finalize(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::query_thread(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::is_thread_main(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::processor_name(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::wtime(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::wtick(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::wtime_is_global(); }));
    MP1_CHECK(throws_runtime_error([] { mp1::buffer_attach(nullptr, 0); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::buffer_detach(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::get_hw_resource_info(); }));
    MP1_CHECK(throws_runtime_error([] { mp1::buffer_flush(); }));
    MP1_CHECK(throws_runtime_error([] { (void)mp1::buffer_iflush(); }));
    MP1_CHECK(not mp1::get_library_version().empty());
    check_status();
    check_info_null();
    check_request_null();
  }

  // ---- owned scenario -------------------------------------------------------------------
  void run_owned(int& argc, char**& argv) {
    check_process_before_init();

    std::optional<mp1::info> live_info;
    std::optional<mp1::info> copied_info;
    std::optional<mp1::info> assigned_info;
    std::optional<mp1::request> live_persistent;

    {
      // The environment object owns MPI here and initializes it through the argc/argv
      // overload; mp1::init() below must then be the idempotent owner path.
      mp1::environment environment{argc, argv};
      MP1_CHECK(mp1::initialized());
      MP1_CHECK(not mp1::finalized());
      MP1_CHECK(mp1::available());
      MP1_CHECK(mp1::provided() == mp1::query_thread());
      // The zero-argument init/environment use MPI_Init; MPI 4.1 §11.2.1 permits the
      // provider to install a higher level than SINGLE, so only the truthfulness of
      // provided() is asserted here, not a SINGLE assumption.
#if MP1_TEST_HAVE_MPI_WRAP
      if (mpi_wrap::supported()) {
        MP1_CHECK(mpi_wrap::count("MPI_Init") == 1);
        MP1_CHECK(mpi_wrap::count("MPI_Init_thread") == 0);
      }
#endif
      MP1_CHECK(throws_nothing([] { (void)mp1::is_thread_main(); }));
      MP1_CHECK(throws_nothing([] { (void)mp1::processor_name(); }));
      MP1_CHECK(mp1::wtick() > 0.0);
      MP1_CHECK(mp1::wtime() > 0.0);
      MP1_CHECK(throws_nothing([] { (void)mp1::wtime_is_global(); }));

      mp1::init();  // idempotent second init by the owner
      MP1_CHECK(mp1::provided() == mp1::query_thread());

      {
        mp1::environment adopted;  // adopts; does not own and must not finalize
        MP1_CHECK(mp1::available());
        MP1_CHECK(mp1::provided() == mp1::query_thread());
      }
      MP1_CHECK(mp1::available());

      // Namespace-scope object made non-null while MPI is active; destroyed after main.
      static_info.set("mp1-static-key", "mp1-static-value");
      MP1_CHECK(static_info.size() == 1);

      // Non-null info paths while MPI is active.
      live_info.emplace();
      live_info->set("mp1-test-key", "mp1-test-value");
      MP1_CHECK(live_info->size() == 1);
      MP1_CHECK(live_info->value("mp1-test-key").has_value());
      MP1_CHECK(*live_info->value("mp1-test-key") == "mp1-test-value");
      MP1_CHECK(not live_info->value("mp1-missing-key").has_value());
      MP1_CHECK(live_info->key(0) == "mp1-test-key");
      copied_info.emplace(*live_info);
      MP1_CHECK(copied_info->size() == 1);
      assigned_info.emplace();
      *assigned_info = *live_info;
      MP1_CHECK(assigned_info->size() == 1);
      MP1_CHECK(throws_nothing([&live_info] { live_info->remove("mp1-test-key"); }));
      MP1_CHECK(live_info->size() == 0);

      // A valid but inactive persistent request: its destructor after finalize must abandon
      // the handle without an MPI call.
      int send_buffer = 0;
      MPI_Request raw_request = MPI_REQUEST_NULL;
      MP1_CHECK(MPI_Send_init(&send_buffer, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &raw_request) ==
                MPI_SUCCESS);
      live_persistent.emplace(mp1::request::from_persistent_send(raw_request));
      MP1_CHECK(live_persistent->is_valid());
      MP1_CHECK(not live_persistent->is_active());
      MP1_CHECK(live_persistent->is_persistent());

      mp1::finalize();
      MP1_CHECK(mp1::finalized());
      MP1_CHECK(not mp1::available());
      check_process_after_finalize();

      // Info is always available (MPI 4.1 Table 11.1): every operation, including copy
      // (MPI_Info_dup) and the destructor's MPI_Info_free, works after finalize.
      MP1_CHECK(throws_nothing([&live_info] { (void)live_info->size(); }));
      MP1_CHECK(not live_info->remove("mp1-absent-key"));  // absent is false, not an exception
      MP1_CHECK(throws_nothing([&live_info] { live_info->set("mp1-after-key", "v"); }));
      MP1_CHECK(live_info->value("mp1-after-key").has_value());
      MP1_CHECK(*live_info->value("mp1-after-key") == "v");
      MP1_CHECK(throws_nothing([&live_info] { (void)live_info->key(0); }));
      MP1_CHECK(throws_nothing([&live_info] { (void)live_info->remove("mp1-after-key"); }));
      MP1_CHECK(throws_nothing([&copied_info] { (void)copied_info->size(); }));
      MP1_CHECK(throws_nothing([&live_info] {
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization): copy under test
        const mp1::info copy{*live_info};
        (void)copy;
      }));

      // A surviving info is freed, not abandoned.
      const std::size_t before = mp1::detail::abandoned_handle_count();
      live_info.reset();
      MP1_CHECK(mp1::detail::abandoned_handle_count() == before);

      const std::size_t before_request = mp1::detail::abandoned_handle_count();
      live_persistent.reset();  // valid persistent request after finalize: abandon, no MPI
      MP1_CHECK(mp1::detail::abandoned_handle_count() == before_request + 1);
    }

    // The environment destructor ran after finalize: it must be a no-op (owned but inactive).
    MP1_CHECK(mp1::finalized());
    check_no_mpi_after_finalize();
  }

  // ---- external scenario ----------------------------------------------------------------
  void run_external(int& argc, char**& argv) {
    check_process_before_init();

    int provided_level = MPI_THREAD_SINGLE;
    MP1_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_level) ==
              MPI_SUCCESS);
    // NOLINTNEXTLINE(readability-simplify-boolean-expr): MP1_CHECK adds negation
    MP1_CHECK(provided_level >= MPI_THREAD_SINGLE and provided_level <= MPI_THREAD_MULTIPLE);
    MP1_CHECK(mp1::initialized());
    MP1_CHECK(mp1::available());

    const mp1::threading_mode live_level = mp1::query_thread();
    mp1::init();  // adopts the external initialization; must not call MPI_Init_thread
    MP1_CHECK(mp1::provided() == live_level);
    MP1_CHECK(throws_runtime_error([] { mp1::finalize(); }));  // external: not owned
    MP1_CHECK(mp1::available());

    std::optional<mp1::info> live_info;
    {
      mp1::environment adopted;  // adopts; its destructor must not finalize external MPI
      MP1_CHECK(mp1::available());
      MP1_CHECK(mp1::provided() == mp1::query_thread());

      static_info.set("mp1-static-key", "mp1-static-value");
      live_info.emplace();
      live_info->set("mp1-external-key", "value");
      MP1_CHECK(live_info->size() == 1);

      // External owner finalizes MPI itself while the mp1 environment object is still alive.
      MP1_CHECK(MPI_Finalize() == MPI_SUCCESS);
      MP1_CHECK(mp1::finalized());
      MP1_CHECK(not mp1::available());
      MP1_CHECK(throws_runtime_error([] { mp1::init(); }));
      MP1_CHECK(throws_runtime_error([] { mp1::finalize(); }));

      const std::size_t before = mp1::detail::abandoned_handle_count();
      live_info.reset();  // info dtor after finalize: MPI_Info_free, not an abandon
      MP1_CHECK(mp1::detail::abandoned_handle_count() == before);
    }
    MP1_CHECK(mp1::finalized());  // adopted environment dtor did not finalize anything
    check_no_mpi_after_finalize();
  }


  // ---- external single-level adoption ---------------------------------------------------
  //
  // An application that initialized MPI itself at MPI_THREAD_SINGLE keeps ownership.  The
  // explicit threading-mode form must reject a higher requirement without finalizing MPI; the
  // default form accepts the provided level.
  void run_external_single(int& argc, char**& argv) {
    check_process_before_init();

    int provided_level = MPI_THREAD_SINGLE;
    MP1_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided_level) == MPI_SUCCESS);
    MP1_CHECK(mp1::available());

    MP1_CHECK(throws_runtime_error([] { mp1::init(mp1::threading_mode::multiple); }));
    MP1_CHECK(mp1::available());  // the failed requirement did not roll back MPI
    MP1_CHECK(mp1::provided() == mp1::threading_mode::single);
    MP1_CHECK(mp1::provided() == mp1::query_thread());

    mp1::init();  // accept-any-level form adopts the existing environment
    MP1_CHECK(mp1::provided() == mp1::threading_mode::single);
    MP1_CHECK(throws_runtime_error([] { mp1::finalize(); }));  // external owner finalizes
    MP1_CHECK(mp1::available());

    MP1_CHECK(MPI_Finalize() == MPI_SUCCESS);
    MP1_CHECK(mp1::finalized());
    MP1_CHECK(not mp1::available());
    check_no_mpi_after_finalize();
  }

  // ---- concurrent init ------------------------------------------------------------------
  //
  // Eight threads released together all call mp1::init(); every call must return and exactly
  // one MPI_Init may be issued.  This is the concurrent-init scenario.
  void run_concurrent_init() {
    check_process_before_init();
    constexpr int thread_count = 8;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::future<int>> callers;
    callers.reserve(thread_count);
    for (int i = 0; i < thread_count; ++i) {
      callers.push_back(std::async(std::launch::async, [&ready, &go]() -> int {
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (not go.load(std::memory_order_acquire)) {
          std::atomic_signal_fence(std::memory_order_seq_cst);
        }
        try {
          mp1::init();
          return 0;
        } catch (const std::exception& error) {
          std::fprintf(stderr, "concurrent init caller threw: %s\n", error.what());
          return 1;
        } catch (...) {
          return 2;
        }
      }));
    }
    while (ready.load(std::memory_order_acquire) != thread_count) {
      std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    go.store(true, std::memory_order_release);
    int failures = 0;
    for (auto& caller : callers)
      failures += caller.get();
    MP1_CHECK(failures == 0);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported()) {
      MP1_CHECK(mpi_wrap::count("MPI_Init") == 1);
      MP1_CHECK(mpi_wrap::count("MPI_Init_thread") == 0);
    }
#endif
    MP1_CHECK(mp1::available());
    // MPI_Init is equivalent to MPI_Init_thread(SINGLE), but a MULTIPLE-only launch may
    // install a higher level; the recorded level must match the live one, not SINGLE.
    MP1_CHECK(mp1::provided() == mp1::query_thread());
    mp1::finalize();
    MP1_CHECK(mp1::finalized());
    check_no_mpi_after_finalize();
  }

  // ---- concurrent finalize --------------------------------------------------------------
  //
  // Four threads released together all call mp1::finalize(); exactly one returns and the
  // others get std::runtime_error.  Exactly one MPI_Finalize may be issued and the process
  // must exit normally.  This is the concurrent-finalize scenario.
  void run_concurrent_finalize() {
    check_process_before_init();
    mp1::init();
    MP1_CHECK(mp1::available());
    constexpr int thread_count = 4;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::future<int>> callers;
    callers.reserve(thread_count);
    for (int i = 0; i < thread_count; ++i) {
      callers.push_back(std::async(std::launch::async, [&ready, &go]() -> int {
        ready.fetch_add(1, std::memory_order_acq_rel);
        while (not go.load(std::memory_order_acquire)) {
          std::atomic_signal_fence(std::memory_order_seq_cst);
        }
        try {
          mp1::finalize();
          return 0;  // the one winner
        } catch (const std::runtime_error&) {
          return 10;  // expected for every loser
        } catch (...) {
          return 11;
        }
      }));
    }
    while (ready.load(std::memory_order_acquire) != thread_count) {
      std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    go.store(true, std::memory_order_release);
    int succeeded = 0;
    int rejected = 0;
    int unexpected = 0;
    for (auto& caller : callers) {
      switch (caller.get()) {
        case 0:
          ++succeeded;
          break;
        case 10:
          ++rejected;
          break;
        default:
          ++unexpected;
          break;
      }
    }
    MP1_CHECK(succeeded == 1);
    MP1_CHECK(rejected == thread_count - 1);
    MP1_CHECK(unexpected == 0);
#if MP1_TEST_HAVE_MPI_WRAP
    if (mpi_wrap::supported())
      MP1_CHECK(mpi_wrap::count("MPI_Finalize") == 1);
#endif
    MP1_CHECK(mp1::finalized());
    check_no_mpi_after_finalize();
  }

  // ---- none scenario --------------------------------------------------------------------
  void run_none() {
    check_process_before_init();
    MP1_CHECK(not mp1::initialized());
    MP1_CHECK(not mp1::finalized());
    MP1_CHECK(mp1::provided() == mp1::threading_mode::single);
    // No MPI_Init call in this process: do not construct mp1::environment here.
  }

}  // namespace

int main(int argc, char** argv) {
  const std::string scenario = scenario_from_environment();
  if (scenario == "owned") {
    run_owned(argc, argv);
  } else if (scenario == "external") {
    run_external(argc, argv);
  } else if (scenario == "external_single") {
    run_external_single(argc, argv);
  } else if (scenario == "concurrent_init") {
    run_concurrent_init();
  } else if (scenario == "concurrent_finalize") {
    run_concurrent_finalize();
  } else if (scenario == "none") {
    run_none();
  } else {
    std::fprintf(stderr, "unknown MP1_TEST_SCENARIO=%s\n", scenario.c_str());
    return 2;
  }

  if (failure_count != 0) {
    std::fprintf(stderr, "test_environment: %d failure(s)\n", failure_count);
    return 1;
  }
  return 0;
}
