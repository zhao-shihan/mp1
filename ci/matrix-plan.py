#!/usr/bin/env python3
"""Validate ci/mpi-matrix.json and emit the GitHub Actions matrix.

The JSON file is the single source of truth for the MPI pins.  This script is
the only place that maps a provider pin to the workflow matrix fields; it
fails closed on a missing checksum, an unknown family/install kind, or a
provider that does not advertise MPI 4.1 or newer.

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
MINIMUM = "4.1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
FAMILIES = {"mpich", "openmpi", "intel"}
INSTALL_KINDS = {"tarball", "oneapi-apt"}
REQUIRED_PROVIDER = {
    "id",
    "family",
    "version",
    "advertised_mpi_version",
    "install",
}
REQUIRED_PROFILE = {"provider", "name", "build_type"}


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"[0-9]+", value))


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

    if data.get("schema") != 3:
        fail(f"{MATRIX}: unsupported schema {data.get('schema')!r}")
    if data.get("mpi_minimum") != MINIMUM:
        fail(f"{MATRIX}: mpi_minimum must be {MINIMUM!r}")

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
        install = provider["install"]
        if install not in INSTALL_KINDS:
            fail(f"{provider['id']}: unknown install kind {install!r}")
        if install == "tarball":
            for field in ("url", "sha256"):
                if field not in provider:
                    fail(f"{provider['id']}: tarball provider is missing {field}")
            if not provider["url"].startswith("https://"):
                fail(f"{provider['id']}: url must be https")
            if not HEX64.match(provider["sha256"]):
                fail(f"{provider['id']}: sha256 must be 64 lowercase hex digits")
        else:
            packages = provider.get("apt_packages")
            if (
                not isinstance(packages, list)
                or not packages
                or not all(isinstance(package, str) and package for package in packages)
            ):
                fail(
                    f"{provider['id']}: oneapi-apt provider needs a non-empty "
                    "apt_packages list"
                )
            if "url" in provider or "sha256" in provider:
                fail(
                    f"{provider['id']}: oneapi-apt provider must not carry "
                    "url/sha256 (it is not a tarball)"
                )
        advertised = provider["advertised_mpi_version"]
        if not re.match(r"^[0-9]+(\.[0-9]+)+", advertised):
            fail(f"{provider['id']}: advertised_mpi_version is not a version")
        if version_tuple(advertised) < version_tuple(MINIMUM):
            fail(
                f"{provider['id']}: advertised MPI {advertised} is below "
                f"the minimum {MINIMUM}"
            )

    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        fail(f"{MATRIX}: profiles must be a non-empty object")
    by_id = {provider["id"]: provider for provider in providers}
    for name, rows in profiles.items():
        if not isinstance(rows, list) or not rows:
            fail(f"profile {name!r} must be a non-empty list")
        profile_ids = set()
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

        if not any(
            version_tuple(by_id[provider_id]["advertised_mpi_version"])
            == version_tuple(MINIMUM)
            for provider_id in profile_ids
        ):
            fail(f"profile {name!r} does not exercise an MPI {MINIMUM} provider")

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
            "install": provider["install"],
            "apt_packages": provider.get("apt_packages", []),
            "mpi_sha256": provider.get("sha256", ""),
            "mpi_url": provider.get("url", ""),
            "runs_on": entry.get("runs_on", provider.get("runs_on", "ubuntu-24.04")),
            "build_type": entry["build_type"],
            "cxx_flags": entry.get("cxx_flags", ""),
            "mpiexec_preflags": entry.get(
                "mpiexec_preflags", launch_flags(provider)
            ),
            "compile_fail": bool(entry.get("compile_fail", False)),
            "example_smoke": bool(entry.get("example_smoke", False)),
            "require_features": entry.get("require_features", []),
        }
        rows.append(row)
    return rows


def print_text(data: dict, profile: str, rows: list) -> None:
    print(f"profile: {profile}")
    for row in rows:
        checks = []
        if row["compile_fail"]:
            checks.append("compile-fail")
        if row["example_smoke"]:
            checks.append("examples")
        flags = " ".join(checks) if checks else "-"
        print(
            f"  {row['name']:<38} {row['mpi']:<8} {row['mpi_version']:<10} "
            f"{row['install']:<12} {row['build_type']:<8} {flags}"
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
    if sha is not None and matches[0].get("sha256") != sha:
        fail(
            f"pin {family} {version} has sha256 {matches[0].get('sha256')}, "
            f"not the supplied {sha}"
        )
    print(f"pin ok: {matches[0]['id']}")


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
            "install",
            "apt_packages",
            "sha256",
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
        if args.field not in matches[0]:
            fail(f"provider {args.provider!r} has no field {args.field!r}")
        value = matches[0][args.field]
        if isinstance(value, list):
            value = " ".join(value)
        print(value)
        return 0
    if args.list:
        for provider in data["providers"]:
            print(
                f"{provider['id']:<24} {provider['family']:<8} "
                f"{provider['version']:<10} advertises MPI {provider['advertised_mpi_version']}"
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
