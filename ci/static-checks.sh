#!/usr/bin/env bash
# Static identity and hygiene gate for mp1.
#
# Usage: ci/static-checks.sh [--shadow]
#
# Default mode scans every tracked file.  --shadow additionally skips test/ and
# examples/ so the gate can run on a reduced tree that omits them; CI uses the
# default mode.
#
# Exclusions:
#   build-scan/**  untracked scratch reports
#   test/compile_fail/no_inter_communicator.cc
#                  negative probe that must name the removed identifiers
#   MIGRATION.md   only its marked historical reference block may use the
#                  mpl 0.5.0 names; mplr/MPLR/0.25/0.26 are forbidden everywhere

set -euo pipefail

mode=final
case "${1:-}" in
  "")
    ;;
  --shadow)
    mode=shadow
    ;;
  -h | --help)
    printf 'usage: %s [--shadow]\n' "${0##*/}"
    exit 0
    ;;
  *)
    printf 'usage: %s [--shadow]\n' "${0##*/}" >&2
    exit 2
    ;;
esac

if ! root=$(git rev-parse --show-toplevel 2>/dev/null); then
  printf 'static-checks: not inside a git working tree\n' >&2
  exit 2
fi
cd "${root}"

if ! command -v python3 >/dev/null 2>&1; then
  printf 'static-checks: python3 is required for the markdown link check\n' >&2
  exit 2
fi

fail=0
report() {
  printf 'static-checks: FAIL: %s\n' "$1" >&2
  fail=1
}

tmp=$(mktemp -d)
trap 'rm -rf "${tmp}"' EXIT

word_start='(^|[^A-Za-z0-9_])'
word_end='([^A-Za-z0-9_]|$)'
old_identity_re="mplr|MPLR|mpl::|mplConfig|mplTargets|find_package[[:space:]]*\\([[:space:]]*mpl([^A-Za-z0-9_]|$)|add_mpl_executable|compile-mplr|_mpl([^A-Za-z0-9_]|$)"
# The three removed layout classes are templates.  Requiring the following '<'
# keeps a local variable named empty_layout in a contract test out of the gate.
old_api_re="${word_start}(tag_t|duty_ratio|test_result|completed_status|no_index|inter_communicator|flat_memory|stopwatch|thread_stopwatch)${word_end}|${word_start}(null_layout|empty_layout|contiguous_layouts)[[:space:]]*<|<thread>|${word_start}MPI_Grequest${word_end}"
forbidden_re="(${old_identity_re})|(${old_api_re})"

exclude=(
  ':(exclude)build-scan/**'
  ':(exclude)test/compile_fail/no_inter_communicator.cc'
  ':(exclude)test/CMakeLists.txt' # the probe-registration row names the negative probe
  ':(exclude)test/spawn_child.cc' # threaded parent() race probe; thread creation is the test
  ':(exclude)MIGRATION.md'
  ':(exclude)ci/static-checks.sh'
)

if [[ "${mode}" == "shadow" ]]; then
  exclude+=(':(exclude)test/**' ':(exclude)examples/**')
fi

# Run a grep command and classify its exit status: 0 = match (report),
# 1 = no match (pass), >1 = error (report and fail).  A plain
# `if hits=$(git grep ...)` treats exit 2 as "no match" and is fail-open.
# The command is passed as arguments so patterns and exclude
# pathspecs keep their quoting.
check_grep() {
  local what=$1
  shift
  local output status=0
  output=$("$@") || status=$?
  if ((status > 1)); then
    report "grep command failed (exit ${status}) while checking: ${what}"
    return 0
  fi
  if ((status == 0)); then
    report "${what}"
    printf '%s\n' "${output}" >&2
  fi
}

# ---------------------------------------------------------------------------
# 1. Old identity tokens and removed API names in tracked files.
# ---------------------------------------------------------------------------
check_grep 'old identity tokens or removed API names in tracked files' \
  git grep -n -I -E "${forbidden_re}" -- . "${exclude[@]}"

# The historical block is exempt from the mpl::/MPL_* spellings, but the later
# mplr identity and its 0.25/0.26 version strings are forbidden in every
# tracked file, including that block.
check_grep 'removed previous-library identity (mplr/MPLR/0.25/0.26) in tracked files' \
  git grep -n -I -E "${word_start}(mplr|MPLR|0\\.25|0\\.26)${word_end}" -- . \
  ':(exclude)build-scan/**' ':(exclude)ci/static-checks.sh'

# MIGRATION.md is the single historical reference.  Old names are allowed only
# between the two markers; blank out that block and check the rest.
if [[ -f MIGRATION.md ]]; then
  awk '
    /<!-- BEGIN HISTORICAL MIGRATION REFERENCE/ { skip = 1; print ""; next }
    /<!-- END HISTORICAL MIGRATION REFERENCE/ { skip = 0; print ""; next }
    { if (skip) print ""; else print }
  ' MIGRATION.md >"${tmp}/migration.md"
  migration_hits=""
  migration_status=0
  migration_hits=$(grep -n -E "${forbidden_re}" "${tmp}/migration.md") || \
    migration_status=$?
  if ((migration_status > 1)); then
    report 'grep failed while checking MIGRATION.md outside the historical block'
  elif ((migration_status == 0)); then
    report 'old identity tokens or removed API names outside the historical block in MIGRATION.md'
    printf '%s\n' "${migration_hits}" | sed 's/^/MIGRATION.md:/' >&2
  fi
fi

# ---------------------------------------------------------------------------
# 2. Structural rules.
# ---------------------------------------------------------------------------
# Variable collectives must not use the Alltoallw engine.
check_grep 'MPI_Alltoallw/MPI_Ialltoallw in the variable-collective implementation' \
  git grep -n -I -E '(^|[^A-Za-z0-9_])MPI_I?Alltoallw([^A-Za-z0-9_]|$)' -- mp1/impl

check_grep 'reinterpret_cast to MPI_Datatype is forbidden' \
  git grep -n -I -E 'reinterpret_cast[[:space:]]*<[^>]*MPI_Datatype' -- . "${exclude[@]}"

sizeof_lines=""
sizeof_status=0
sizeof_lines=$(git grep -n -I -F 'sizeof(T)' -- mp1/impl) || sizeof_status=$?
if ((sizeof_status > 1)); then
  report 'grep failed while checking sizeof(T) displacements'
elif [[ -n "${sizeof_lines}" ]]; then
  hits=""
  hits=$(printf '%s\n' "${sizeof_lines}" | grep -E 'displ|offset|stride') || true
  if [[ -n "${hits}" ]]; then
    report 'sizeof(T) used as a displacement or stride'
    printf '%s\n' "${hits}" >&2
  fi
fi

# The managed-reduction callback image is typed over native_datatype<T>(); a
# full-object MPI_BYTE image (contiguous sizeof(T) bytes) must never return.
check_grep 'MPI_BYTE full-object reduction image' \
  git grep -n -I -E 'MPI_Type_contiguous(_c)?\([^)]*MPI_BYTE' -- mp1/impl/operator.hpp
if ! git grep -q -I -F 'MPI_Type_create_resized' -- mp1/impl/operator.hpp; then
  report 'operator.hpp does not build the reduction image with MPI_Type_create_resized'
fi
# The built-in reduction functors must not regain the blanket commutative_op
# base; their commutativity is the explicit op_traits truth table.
check_grep 'blanket commutative_op inheritance on a built-in reduction functor' \
  git grep -n -I -E 'struct[[:space:]]+(max|min|plus|multiplies|logical_and|logical_or|logical_xor|bit_and|bit_or|bit_xor)([[:space:]]*<[^>]*>)?[[:space:]]*:[[:space:]]*commutative_op' -- mp1/impl/operator.hpp

# ---------------------------------------------------------------------------
# 3. C++20 baseline.
# ---------------------------------------------------------------------------
# Every build and CI entry point must require C++20.  This catches a
# reintroduced -std=c++17 (a target-level standard can otherwise win over the
# library's cxx_std_20 feature in some configurations) and a dropped feature
# requirement or umbrella guard.
check_grep 'build or CI file forces C++17' \
  git grep -n -I -E '(-std=c\+\+17|cxx_std_17|CXX_STANDARD[[:space:]]+17)' -- \
  '*CMakeLists.txt' 'cmake/*.cmake' 'ci/*.sh' 'ci/*.yml' '.github/workflows/*.yml' \
  ':(exclude)ci/static-checks.sh' ':(exclude)ci/check-cxx20-guard.sh'

if ! grep -qF 'target_compile_features(mp1 INTERFACE cxx_std_20)' CMakeLists.txt; then
  report 'CMakeLists.txt does not require cxx_std_20 on the mp1 target'
fi
if ! grep -qF '#error "mp1 requires C++20"' mp1/mp1.hpp; then
  report 'mp1/mp1.hpp does not fail closed below C++20'
fi

# ---------------------------------------------------------------------------
# 4. NOLINT policy.
# ---------------------------------------------------------------------------
# Every suppression must name the check and explain why.  A bare NOLINT or
# NOLINTNEXTLINE disables every check on the line and is not accepted.
check_grep 'bare NOLINT/NOLINTNEXTLINE; use NOLINTNEXTLINE(check-name): reason' \
  git grep -n -I -E 'NOLINT(NEXTLINE)?([[:space:]]|$)' -- \
  '*.hpp' '*.cc' '*.c' ':(exclude)build-scan/**'

check_grep 'NOLINT without a reason' \
  git grep -n -I -E 'NOLINT(NEXTLINE)?\([^)]*\)[[:space:]]*(\*/)?[[:space:]]*\\?$' -- \
  '*.hpp' '*.cc' '*.c' ':(exclude)build-scan/**'

# ---------------------------------------------------------------------------
# 5. Documentation rules.
# ---------------------------------------------------------------------------
check_grep 'tracked markdown links or mentions a build-scan path' \
  git grep -n -I -F 'build-scan' -- '*.md'

check_grep 'MPL lineage in tracked documentation' \
  git grep -n -I -E "${word_start}MPL${word_end}" -- README.md MIGRATION.md CHANGELOG.md

if ! python3 - "${root}" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
documents = [root / name for name in ("README.md", "MIGRATION.md", "CHANGELOG.md")]
link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
errors = []

for document in documents:
    if not document.is_file():
        continue
    text = document.read_text(encoding="utf-8")
    visible = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            visible.append(line)
    for match in link_pattern.finditer("\n".join(visible)):
        target = match.group(1).strip()
        if target.startswith("<") and ">" in target:
            target = target[1:target.index(">")]
        else:
            target = target.split()[0]
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = target.split("#", 1)[0]
        if not target:
            continue
        resolved = (document.parent / target).resolve()
        if not resolved.exists():
            errors.append(
                f"{document.relative_to(root)}: missing link target {target}"
            )

if errors:
    print("\n".join(errors), file=sys.stderr)
    sys.exit(1)
PY
then
  report 'broken relative link in README.md, MIGRATION.md or CHANGELOG.md'
fi

# ---------------------------------------------------------------------------
# 6. Version single source.
# ---------------------------------------------------------------------------
# VERSION is the only build version literal.  CMake generates the public
# mp1/version.hpp from mp1/version.hpp.in; neither the generated header nor a
# second macro definition may be committed, and the CHANGELOG release record
# must name the version that VERSION declares.
if [[ ! -f VERSION ]]; then
  report 'VERSION file is missing'
else
  version=$(tr -d '[:space:]' <VERSION)
  if [[ ! ${version} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    report "VERSION is not major.minor.patch: '${version}'"
  fi
  if git ls-files --error-unmatch mp1/version.hpp >/dev/null 2>&1; then
    report 'mp1/version.hpp is tracked; it must be generated from mp1/version.hpp.in'
  fi
  if ! grep -qF '#include "mp1/version.hpp"' mp1/mp1.hpp; then
    report 'mp1/mp1.hpp does not include the generated mp1/version.hpp'
  fi
  if grep -qE '^[[:space:]]*mp1/version\.hpp[[:space:]]*$' mp1/headers.txt; then
    report 'mp1/headers.txt must not list the generated mp1/version.hpp'
  fi
  first_release_heading=$(grep -m1 '^## ' CHANGELOG.md || true)
  if [[ -z ${first_release_heading} ]]; then
    report 'CHANGELOG.md has no release heading'
  elif [[ ${first_release_heading} != "## ${version}"* ]]; then
    report "CHANGELOG.md first release heading '${first_release_heading}' does not begin with VERSION '${version}'"
  fi
fi

check_grep 'version macro defined outside mp1/version.hpp.in' \
  git grep -n -I -E '#define[[:space:]]+MP1_VERSION_(MAJOR|MINOR|PATCH|STRING)([[:space:]]|$)' -- . \
  ':(exclude)build-scan/**' ':(exclude)mp1/version.hpp.in' ':(exclude)ci/static-checks.sh'

check_grep 'hard-coded release version outside VERSION and the docs' \
  git grep -n -I -F '1.0.0' -- CMakeLists.txt cmake mp1 test ci .github/workflows examples \
  ':(exclude)ci/static-checks.sh'

# 7. MPI address arithmetic.
#
# MPI 4.1 5.1.5: "arithmetic on MPI addresses must be performed using the
# MPI_AINT_ADD and MPI_AINT_DIFF procedures."  Intrinsic subtraction is not
# portable, and 5.1.12 forbids a displacement that is the difference of
# addresses from different allocations.  The only permitted spellings are the
# checked helpers in detail/mpi_call.hpp and the direct MPI_Aint_add run probe
# in detail/buffer.hpp.
# ---------------------------------------------------------------------------
check_grep 'intrinsic MPI_Aint subtraction in mp1/impl' \
  git grep -n -E '\b(member_address|base_address|address|run_start|base_)[[:space:]]*-[[:space:]]*[A-Za-z_(]' -- mp1/impl

for required_procedure in MPI_Aint_diff MPI_Aint_add; do
  if ! git grep -q -F "${required_procedure}" -- mp1/impl/detail/mpi_call.hpp; then
    report "MPI address arithmetic: ${required_procedure} is missing from detail/mpi_call.hpp"
  fi
done
if ! git grep -q -F 'MPI_Aint_add' -- mp1/impl/detail/buffer.hpp; then
  report 'MPI address arithmetic: MPI_Aint_add is missing from detail/buffer.hpp'
fi

# The former relative iterator map subtracted two addresses from
# different allocations.  Reintroducing it requires the cross-allocation
# rejection documented in CONTRIBUTING.md.
check_grep 'relative iterator hindexed map reintroduced in layout.hpp' \
  git grep -n -E 'address[[:space:]]*-[[:space:]]*base_address|from_iterator_range' -- mp1/impl/layout.hpp


# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------
if ((fail != 0)); then
  printf 'static-checks: FAILED\n' >&2
  exit 1
fi

printf 'static-checks: PASS (%s mode)\n' "${mode}"
