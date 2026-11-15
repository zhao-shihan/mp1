#!/usr/bin/env python3
"""Validate ci/mpi-matrix.json and emit the GitHub Actions matrix.

The JSON file is the single source of truth for the MPI pins.  This script is
the only place that maps a provider pin to the workflow matrix fields; it
fails closed on a missing checksum, a second floor entry, a floor that does not
advertise exactly MPI 4.1, a profile without a floor, or a profile without a
newer-than-floor provider.

Usage:
    ci/matrix-plan.py --profile pr|nightly [--format github|json|text]
    ci/matrix-plan.py --check-pin --family mpich --version 4.3.2 [--sha HEX]
    ci/matrix-plan.py --list

`--format github` prints a JSON object with an `include` array suitable for
`fromJSON(needs.plan.outputs.matrix)`.
"""

import argparse
import json
import pathlib
import re
import sys

MATRIX = pathlib.Path(__file__).with_name("mpi-matrix.json")
FLOOR = "4.1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TIERS = {"floor", "stable", "provisional", "rc", "dev"}
FAMILIES = {"mpich", "openmpi"}
REQUIRED_PROVIDER = {
    "id",
    "family",
    "version",
    "tier",
    "advertised_mpi_version",
    "url",
    "sha256",
}
REQUIRED_PROFILE = {"provider", "name", "build_type"}


def fail(message: str) -> None:
    print(f"matrix-plan: {message}", file=sys.stderr)
    raise SystemExit(2)


def load() -> dict:
    try:
        data = json.loads(MATRIX.read_text(encoding="utf-8"))
    except OSError as error:
        fail(f"cannot read {MATRIX}: {error}")
    except json.JSONDecodeError as error:
        fail(f"invalid JSON in {MATRIX}: {error}")

    if data.get("schema") != 1:
        fail(f"{MATRIX}: unsupported schema {data.get('schema')!r}")
    if data.get("mpi_floor") != FLOOR:
        fail(f"{MATRIX}: mpi_floor must be {FLOOR!r}")

    providers = data.get("providers")
    if not isinstance(providers, list) or not providers:
        fail(f"{MATRIX}: providers must be a non-empty list")
    ids = set()
    for provider in providers:
        missing = REQUIRED_PROVIDER - provider.keys()
        if missing:
            fail(f"provider {provider.get('id')!r} is missing {sorted(missing)}")
        if provider["id"] in ids:
            fail(f"duplicate provider id {provider['id']!r}")
        ids.add(provider["id"])
        if provider["family"] not in FAMILIES:
            fail(f"{provider['id']}: unknown family {provider['family']!r}")
        if provider["tier"] not in TIERS:
            fail(f"{provider['id']}: unknown tier {provider['tier']!r}")
        if not provider["url"].startswith("https://"):
            fail(f"{provider['id']}: url must be https")
        if not HEX64.match(provider["sha256"]):
            fail(f"{provider['id']}: sha256 must be 64 lowercase hex digits")
        if not re.match(r"^[0-9]+(\.[0-9]+)+", provider["advertised_mpi_version"]):
            fail(f"{provider['id']}: advertised_mpi_version is not a version")

    floors = [provider for provider in providers if provider["tier"] == "floor"]
    if len(floors) != 1:
        fail(f"exactly one floor provider is required, found {len(floors)}")
    if floors[0]["advertised_mpi_version"] != FLOOR:
        fail(
            "the floor provider must advertise exactly MPI "
            f"{FLOOR} (it is the oldest supported implementation)"
        )

    # A family that is present must have a blocking release pin: the floor is
    # blocking by definition, as are stable/provisional entries.  rc/dev-only
    # families are rejected because an RC never replaces a release.
    for family in FAMILIES:
        family_entries = [entry for entry in providers if entry["family"] == family]
        if not family_entries:
            continue
        blocking = [
            entry
            for entry in family_entries
            if entry["tier"] in {"floor", "stable", "provisional"}
        ]
        if not blocking:
            fail(
                f"family {family!r} has no blocking provider "
                "(floor/stable/provisional)"
            )

    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        fail(f"{MATRIX}: profiles must be a non-empty object")
    by_id = {provider["id"]: provider for provider in providers}
    for name, rows in profiles.items():
        if not isinstance(rows, list) or not rows:
            fail(f"profile {name!r} must be a non-empty list")
        profile_ids = set()
        floor_seen = False
        newer_seen = False
        for row in rows:
            missing = REQUIRED_PROFILE - row.keys()
            if missing:
                fail(f"profile {name!r} row is missing {sorted(missing)}")
            features = row.get("require_features", [])
            if not isinstance(features, list) or not all(
                isinstance(feature, str) and feature for feature in features
            ):
                fail(
                    f"profile {name!r} row {row.get('name')!r}: "
                    "require_features must be a list of non-empty strings"
                )
            provider_id = row["provider"]
            if provider_id not in by_id:
                fail(f"profile {name!r} references unknown provider {provider_id!r}")
            if provider_id in profile_ids:
                fail(f"profile {name!r} repeats provider {provider_id!r}")
            profile_ids.add(provider_id)
            provider = by_id[provider_id]
            if provider["tier"] == "floor":
                floor_seen = True
            elif provider["tier"] != "rc":
                newer_seen = True
        if not floor_seen:
            fail(f"profile {name!r} does not include the floor provider")
        if not newer_seen:
            fail(f"profile {name!r} has no newer-than-floor provider")

    return data


def launch_flags(provider: dict) -> str:
    if provider["family"] == "openmpi":
        return "--oversubscribe"
    return ""


def matrix_rows(data: dict, profile: str) -> list:
    rows = []
    by_id = {provider["id"]: provider for provider in data["providers"]}
    for entry in data["profiles"][profile]:
        provider = by_id[entry["provider"]]
        row = {
            "name": entry["name"],
            "mpi": provider["family"],
            "mpi_version": provider["version"],
            "mpi_sha256": provider["sha256"],
            "mpi_url": provider["url"],
            "tier": provider["tier"],
            "expected_mpi_version": provider["advertised_mpi_version"],
            "build_type": entry["build_type"],
            "cxx_flags": entry.get("cxx_flags", ""),
            "mpiexec_preflags": entry.get(
                "mpiexec_preflags", launch_flags(provider)
            ),
            "compile_fail": bool(entry.get("compile_fail", False)),
            "example_smoke": bool(entry.get("example_smoke", False)),
            "extra_ranks": bool(entry.get("extra_ranks", False)),
            "require_features": entry.get("require_features", []),
        }
        rows.append(row)
    return rows


def print_text(data: dict, profile: str, rows: list) -> None:
    print(f"profile: {profile}")
    print(f"floor:   MPI {data['mpi_floor']} (must be advertised exactly)")
    for row in rows:
        checks = []
        if row["compile_fail"]:
            checks.append("compile-fail")
        if row["example_smoke"]:
            checks.append("examples")
        if row["extra_ranks"]:
            checks.append("extra-ranks")
        flags = " ".join(checks) if checks else "-"
        print(
            f"  {row['name']:<38} {row['mpi']:<8} {row['mpi_version']:<10} "
            f"{row['tier']:<11} {row['build_type']:<8} {flags}"
        )


def check_pin(data: dict, family: str, version: str, sha: str | None) -> None:
    matches = [
        provider
        for provider in data["providers"]
        if provider["family"] == family and provider["version"] == version
    ]
    if not matches:
        fail(
            f"pin {family} {version!r} is not in {MATRIX}; update the matrix "
            "instead of adding an unmanaged provider"
        )
    if sha is not None and matches[0]["sha256"] != sha:
        fail(
            f"pin {family} {version} has sha256 {matches[0]['sha256']}, "
            f"not the supplied {sha}"
        )
    print(f"pin ok: {matches[0]['id']} ({matches[0]['tier']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["pr", "nightly"])
    parser.add_argument(
        "--format", choices=["github", "json", "text"], default="text"
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check-pin", action="store_true")
    parser.add_argument("--family", choices=sorted(FAMILIES))
    parser.add_argument("--version")
    parser.add_argument("--sha")
    parser.add_argument("--provider")
    parser.add_argument(
        "--field",
        choices=[
            "id",
            "family",
            "version",
            "sha256",
            "tier",
            "advertised_mpi_version",
        ],
    )
    args = parser.parse_args()

    data = load()
    if args.provider:
        if not args.field:
            fail("--provider requires --field")
        matches = [
            provider
            for provider in data["providers"]
            if provider["id"] == args.provider
        ]
        if not matches:
            fail(f"unknown provider id {args.provider!r}")
        print(matches[0][args.field])
        return 0
    if args.list:
        for provider in data["providers"]:
            print(
                f"{provider['id']:<24} {provider['family']:<8} "
                f"{provider['version']:<10} {provider['tier']:<11} "
                f"advertises MPI {provider['advertised_mpi_version']}"
            )
        return 0
    if args.check_pin:
        if not args.family or not args.version:
            fail("--check-pin requires --family and --version")
        check_pin(data, args.family, args.version, args.sha)
        return 0
    if not args.profile:
        fail("--profile is required unless --list/--check-pin is used")

    rows = matrix_rows(data, args.profile)
    if args.format == "text":
        print_text(data, args.profile, rows)
        return 0
    document = {"include": rows}
    print(json.dumps(document, indent=2 if args.format == "json" else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
