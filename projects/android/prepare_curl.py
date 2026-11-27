#!/usr/bin/env python3
"""Prepare a pinned static curl binary for the Android container.

`sandbox-cli` and the vendor installers it launches need curl, and the
container intentionally ships no system CA certificates. Use a pinned,
statically linked musl curl with an embedded CA bundle instead of installing
packages on the phone: HTTPS works without apt and without PRoot's /etc state.
Every archive is verified by SHA-256 and exact byte count, and the extracted
binary is checked against its own SHA-256 pin.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

from prepare_proot import download, write_executable

ANDROID_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ANDROID_ROOT / "app/src/main/assets/container/catalog.json"
DEFAULT_OUTPUT = ANDROID_ROOT / "build/curl-bin"
DEFAULT_CACHE = ANDROID_ROOT / "build/curl-packages"


def prepare(asset: dict, output: Path, cache: Path) -> None:
    filename = asset["url"].rsplit("/", 1)[-1]
    archive_blob = download(asset["url"], cache / filename, asset["sha256"])
    if len(archive_blob) != asset["bytes"]:
        raise ValueError(
            f"size mismatch for {filename}: expected {asset['bytes']}, got {len(archive_blob)}"
        )

    with tarfile.open(fileobj=io.BytesIO(archive_blob), mode="r:xz") as archive:
        member = archive.getmember("curl")
        if not member.isfile():
            raise ValueError(f"{filename} has no regular curl member")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"cannot read curl from {filename}")
        binary = handle.read()

    actual = hashlib.sha256(binary).hexdigest()
    if actual != asset["binary_sha256"]:
        raise ValueError(
            f"curl checksum mismatch: expected {asset['binary_sha256']}, got {actual}"
        )
    write_executable(output / "curl", binary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--abi", default="arm64-v8a")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    packages = catalog["runtime"]["curl"]["packages"]
    if args.abi not in packages:
        raise ValueError(f"catalog has no curl package for {args.abi}")
    prepare(packages[args.abi], args.output / args.abi, args.cache / args.abi)
    print(f"prepared pinned curl {catalog['runtime']['curl']['version']} for {args.abi}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(f"prepare_curl.py: {error}", file=sys.stderr)
        raise SystemExit(1)
