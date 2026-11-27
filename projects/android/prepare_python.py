#!/usr/bin/env python3
"""Prepare a minimal Debian CPython runtime for the Android container.

The pinned prebuilt g4f overlay contains /opt/g4f-venv but not the system
interpreter it points to.  Extracting these pinned Debian packages into the
assets tree gives the rootfs /usr/bin/python3 and its stdlib without running
apt on the phone.  Every archive is verified by SHA-256 and exact byte count.
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from prepare_proot import data_tar, download, normalize_member

ANDROID_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ANDROID_ROOT / "app/src/main/assets/container/catalog.json"
DEFAULT_ASSETS = ANDROID_ROOT / "app/src/main/assets/container/python"
DEFAULT_CACHE = ANDROID_ROOT / "build/python-packages"

# Which archive members are actually needed.  Everything else (docs, manpages,
# lintian metadata, Debian helper scripts) is left out of the APK.
INCLUDES = {
    "libpython3.11-minimal": ("usr/lib/python3.11/", "etc/python3.11/"),
    "python3.11-minimal": ("usr/bin/python3.11",),
    "libpython3.11-stdlib": ("usr/lib/python3.11/",),
    "python3-minimal": ("usr/bin/python3",),
    "python3.11": (),
    "python3": (),
    "libexpat1": ("usr/lib/*/libexpat.so.1*",),
    "libssl3": ("usr/lib/*/libssl.so.3", "usr/lib/*/libcrypto.so.3"),
    "libsqlite3-0": ("usr/lib/*/libsqlite3.so.0*",),
    "libreadline8": ("usr/lib/*/libreadline.so.8*",),
    "libncursesw6": ("usr/lib/*/libncursesw.so.6*",),
    "libtinfo6": ("usr/lib/*/libtinfo.so.6*",),
    "libnsl2": ("usr/lib/*/libnsl.so.2*",),
    "libtirpc3": ("usr/lib/*/libtirpc.so.3*",),
}


def matches(name: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/"):
            if name.startswith(pattern) or name == pattern.rstrip("/"):
                return True
        elif fnmatch.fnmatch(name, pattern):
            return True
    return False


def output_name(name: str) -> str:
    # Debian bookworm's /lib is a symlink to /usr/lib in the pinned rootfs.
    return "usr/" + name if name.startswith("lib/") else name


def resolve(member: tarfile.TarInfo, members: dict[str, tarfile.TarInfo],
            archive: tarfile.TarFile, assets: Path,
            seen: frozenset[str] = frozenset()) -> bytes | None:
    if not (member.issym() or member.islnk()):
        handle = archive.extractfile(member)
        return handle.read() if handle else None
    target = member.linkname
    if member.issym() and not target.startswith("/"):
        target = str(Path(member.name).parent / target)
    target = normalize_member(target.lstrip("/"))
    if target in seen:
        return None
    if target not in members:
        # /usr/bin/python3 is a symlink owned by a different package.
        fallback = assets / output_name(target)
        return fallback.read_bytes() if fallback.is_file() else None
    return resolve(members[target], members, archive, assets, seen | {target})


def prepare(package: dict, cache: Path, assets: Path) -> int:
    name = package["name"]
    filename = package["url"].rsplit("/", 1)[-1]
    blob = download(package["url"], cache / filename, package["sha256"])
    if len(blob) != package["bytes"]:
        raise ValueError(f"size mismatch for {filename}: expected {package['bytes']}, got {len(blob)}")

    written = 0
    with tarfile.open(fileobj=io.BytesIO(data_tar(blob)), mode="r:*") as archive:
        members = {normalize_member(m.name): m for m in archive.getmembers()
                   if not m.isdir()}
        patterns = INCLUDES.get(name)
        if patterns is None:
            raise ValueError(f"no extraction rules for package {name!r}")
        for member_name, member in members.items():
            normalized = output_name(member_name)
            if not matches(normalized, patterns):
                continue
            data = resolve(member, members, archive, assets)
            if data is None:
                continue
            destination = assets / normalized
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            destination.chmod(0o755 if member.mode & 0o111 else 0o644)
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--abi", default="arm64-v8a",
                        help="single ABI to prepare; assets cannot hold two ABIs at once")
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    packages = catalog["runtime"]["python"]["packages"]
    if args.abi not in packages:
        raise ValueError(f"catalog has no Python packages for {args.abi}")
    shutil.rmtree(args.assets, ignore_errors=True)
    args.assets.mkdir(parents=True, exist_ok=True)
    count = 0
    for package in packages[args.abi]:
        count += prepare(package, args.cache / args.abi, args.assets)
    print(f"prepared {count} Python runtime files for {args.abi}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, tarfile.TarError, KeyError) as error:
        print(f"prepare_python.py: {error}", file=__import__("sys").stderr)
        raise SystemExit(1)
