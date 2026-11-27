#!/usr/bin/env python3
"""Prepare PRoot native libraries for the Android container build.

Android blocks execve() of binaries in app-writable storage.  PRoot and its
static ELF loader therefore have to live in the APK's native library
directory, which is installed read-only.  This script downloads the pinned
Termux .deb packages, verifies their SHA-256, and extracts the two binaries
into app/src/main/jniLibs/<abi>/ as libproot.so and libproot_loader.so.

The shared libraries PRoot itself needs (libtalloc and libandroid-shmem) are
still extracted at runtime into app-writable storage; dlopen-class loading is
allowed there and they are data, not executed entry points.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

ANDROID_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ANDROID_ROOT / "app/src/main/assets/container/catalog.json"
DEFAULT_JNI_LIBS = ANDROID_ROOT / "app/src/main/jniLibs"
DEFAULT_CACHE = ANDROID_ROOT / "build/proot-packages"

# Destination name -> path inside the Termux data.tar archive.
MEMBERS = {
    "libproot.so": "bin/proot",
    "libproot_loader.so": "libexec/proot/loader",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> bytes:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        data = destination.read_bytes()
        if sha256(data) == expected_sha256:
            return data
        destination.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "g4f-android-build/2.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    actual = sha256(data)
    if actual != expected_sha256:
        raise ValueError(f"checksum mismatch for {url}: expected {expected_sha256}, got {actual}")
    destination.write_bytes(data)
    return data


def ar_members(data: bytes):
    """Yield (name, payload) from a Debian ar archive (short names only)."""
    if not data.startswith(b"!<arch>\n"):
        raise ValueError("not an ar archive")
    offset = 8
    while offset + 60 <= len(data):
        header = data[offset:offset + 60].decode("ascii", "replace")
        name = header[0:16].strip().rstrip("/")
        size_text = header[48:58].strip()
        if not size_text.isdigit():
            raise ValueError(f"invalid ar entry size for {name!r}")
        size = int(size_text)
        start = offset + 60
        end = start + size
        if end > len(data):
            raise ValueError(f"truncated ar entry {name!r}")
        yield name, data[start:end]
        offset = end + (size % 2)


def data_tar(data: bytes) -> bytes:
    for name, payload in ar_members(data):
        if name.startswith("data.tar"):
            return payload
    raise ValueError("no data.tar member in .deb")


def normalize_member(name: str) -> str:
    if name.startswith("./"):
        name = name[2:]
    prefix = "data/data/com.termux/files/usr/"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def write_executable(destination: Path, data: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(data)
    temporary.chmod(0o755)
    os.replace(temporary, destination)


def prepare(abi: str, asset: dict, jni_libs: Path, cache: Path) -> None:
    filename = asset["url"].rsplit("/", 1)[-1]
    package = download(asset["url"], cache / abi / filename, asset["sha256"])
    with tarfile.open(fileobj=io.BytesIO(data_tar(package)), mode="r:*") as archive:
        members = {normalize_member(member.name): member for member in archive.getmembers()
                   if member.isfile()}
        for output, member_name in MEMBERS.items():
            member = members.get(member_name)
            if member is None:
                raise ValueError(f"{filename} does not contain {member_name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read {member_name} from {filename}")
            write_executable(jni_libs / abi / output, handle.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--jni-libs", type=Path, default=DEFAULT_JNI_LIBS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    packages = catalog["runtime"]["proot"]["packages"]
    for abi in ("arm64-v8a", "x86_64"):
        if abi not in packages:
            raise ValueError(f"catalog has no PRoot package for {abi}")
        prepare(abi, packages[abi], args.jni_libs, args.cache)
        print(f"prepared PRoot libraries for {abi}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"prepare_proot.py: {error}", file=sys.stderr)
        sys.exit(1)
