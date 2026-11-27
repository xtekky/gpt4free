#!/usr/bin/env python3
"""Build a branded noVNC client for the Android container desktop.

The app serves noVNC from the g4f API at /desktop/. Debian's ``novnc`` package
ships the stock client, so this helper extracts the pinned upstream release,
layers the sandbox-desktop landing page/touch controls on top, and runs the
same ``novnc-brand.sh`` used by the Docker image. The result replaces
/usr/share/novnc on the phone, giving the app the same g4f logo, colors,
clipboard/key bridge and fullscreen control as the sandbox desktop.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from prepare_proot import download

ANDROID_ROOT = Path(__file__).resolve().parent
REPO = ANDROID_ROOT.parents[1]
DEFAULT_CATALOG = ANDROID_ROOT / "app/src/main/assets/container/catalog.json"
DEFAULT_OUTPUT = ANDROID_ROOT / "build/novnc-client"
DEFAULT_CACHE = ANDROID_ROOT / "build/novnc-packages"

UI_FILES = {
    "docker/novnc-index.html": "index.html",
    "docker/novnc-controls.js": "g4f-controls.js",
    "docker/novnc-controls.css": "g4f-controls.css",
    "docker/novnc-manifest.webmanifest": "manifest.webmanifest",
    "docker/novnc-sw.js": "sw.js",
    "docker/g4f.svg": "g4f.svg",
}


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive.getmembers():
        raw = member.name
        parts = Path(raw).parts
        if not parts or parts[0] in {".", ".."} or raw.startswith("/"):
            raise ValueError(f"unsafe noVNC archive entry: {raw!r}")
        # GitHub tag tarballs wrap everything in noVNC-<version>/.
        target = destination.joinpath(*parts[1:])
        resolved = target.resolve()
        if not resolved.is_relative_to(destination.resolve()):
            raise ValueError(f"noVNC archive entry escapes output: {raw!r}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read noVNC archive entry: {raw!r}")
            target.write_bytes(handle.read())
        else:
            raise ValueError(f"unsupported noVNC archive entry: {raw!r}")


def prepare(asset: dict, output: Path, cache: Path) -> None:
    filename = asset["url"].rsplit("/", 1)[-1]
    archive_blob = download(asset["url"], cache / filename, asset["sha256"])
    if len(archive_blob) != asset["bytes"]:
        raise ValueError(
            f"size mismatch for {filename}: expected {asset['bytes']}, got {len(archive_blob)}"
        )

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive_blob), mode="r:gz") as archive:
        safe_extract(archive, output)

    for source, name in UI_FILES.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / source, target)

    subprocess.run([str(REPO / "bin" / "novnc-brand.sh"), str(output)], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    asset = catalog["runtime"]["novnc"]
    prepare(asset, args.output, args.cache)
    print(f"prepared branded noVNC {catalog['runtime']['novnc']['version']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError,
            tarfile.TarError) as error:
        print(f"prepare_novnc.py: {error}", file=sys.stderr)
        raise SystemExit(1)
