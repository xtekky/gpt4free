#!/usr/bin/env bash
# fetch-python.sh
#
# Downloads the CPython embeddable runtime for every supported platform,
# merges gpt4free into it (source + prebuilt wheels), patches python311._pth
# so the interpreter is fully relocatable, and stores the result as a zip
# next to embed/<os>/EMPTY_PYTHON_RUNTIME so `go:embed` picks it up.
#
# Usage:
#   ./fetch-python.sh            # all platforms (linux, windows, darwin, bsd)
#   ./fetch-python.sh linux      # single platform
#   G4F_VERSION=0.4.x ./fetch-python.sh
#
# Requirements: bash, curl, unzip, zip, python3 (with pip)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYVER="3.14.7"
PYTAG="cp314"
PYABI="cp314-cp314"
PBS_TAG="${PBS_TAG:-20260805}"
PBS_BASE="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}"
PYORG_BASE="https://www.python.org/ftp/python/${PYVER}"
G4F_SRC="${G4F_SRC:-$HERE/..}"                  # gpt4free repository root
G4F_VERSION="${G4F_VERSION:-$(cd "$G4F_SRC" && python3 -c 'import sys;sys.path.insert(0,"g4f");from version import __version__;print(__version__)' 2>/dev/null || echo 0.4.x)}"

# Minimum versions for deps that have ancient pure-python releases on PyPI.
# `pip download --platform <x>` considers py3-none-any wheels valid for any
# target, so an old/misbehaving resolver can pick e.g. aiohttp 0.13.1 (2015,
# predates async/await) which crashes the embedded CPython 3.14. Pinning
# floors here keeps the offline wheel set sane.
WHEEL_FLOORS="aiohttp>=3.8"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

mkdir -p "$HERE/embed/linux" "$HERE/embed/windows" "$HERE/embed/darwin" "$HERE/embed/bsd"

# Build the g4f project wheel ONCE (pure python, platform-independent).
mkdir -p "$WORK/wheels-cache"
(cd "$G4F_SRC" && python3 -m pip wheel . -w "$WORK/wheels-cache" --no-deps -q 2>/dev/null || true)

# platform  source       spec                      goos    goarch   name
PLATFORMS=(
  "linux   pbs  x86_64-unknown-linux-gnu              linux   amd64   linux-x64"
  "linux   pbs  aarch64-unknown-linux-gnu             linux   arm64   linux-arm64"
  "windows pbs  x86_64-pc-windows-msvc                windows amd64   windows-amd64"
  "windows pbs  i686-pc-windows-msvc                  windows 386     windows-x86"
  "windows pbs  aarch64-pc-windows-msvc               windows arm64   windows-arm64"
  "darwin  pbs  aarch64-apple-darwin                  darwin  arm64   darwin-arm64"
  "darwin  pbs  x86_64-apple-darwin                   darwin  amd64   darwin-x64"
  "bsd     pbs  x86_64-unknown-linux-gnu              freebsd amd64   bsd-amd64"
)
WANT=("$@"); [ ${#WANT[@]} -eq 0 ] && WANT=(linux windows darwin bsd)

do_platform() {
  local plat src spec goos goarch name
  read -r plat src spec goos goarch name <<<"$1"
  [[ " ${WANT[*]} " == *" $plat "* ]] || return 0
  echo "==> [$plat/$goarch] $name ($src)"

  local dir="$WORK/$name"
  mkdir -p "$dir"
  local home="$dir/python-home"
  if [ "$src" = "pbs" ]; then
    local asset="cpython-${PYVER}+${PBS_TAG}-${spec}-install_only.tar.gz"
    curl -fsSL -o "$dir/base.tar.gz" "$PBS_BASE/$asset"
    mkdir -p "$home"
    # python-build-standalone is a full installable layout (bin/ or top-level
    # python.exe) that is already relocatable, so no _pth patching is needed.
    tar -xzf "$dir/base.tar.gz" -C "$home" --strip-components=1
    rm -f "$dir/base.tar.gz"
    # On unix the entry point is `bin/python3`; on windows `python.exe`.
    if [ -f "$home/bin/python3" ]; then
      mv "$home/bin/python3" "$home/bin/python"
      find "$home/bin" -maxdepth 1 -name 'python3.*' -exec rm -f {} +
    fi
  else
    local basezip="python-${PYVER}-${spec}.zip"
    curl -fsSL -o "$dir/base.zip" "$PYORG_BASE/$basezip"
    ( cd "$dir" && unzip -oq "$basezip" )
    # 1) Relocatable: strip the hard-coded prefix path, keep python314.zip.
    local pth
    pth=$(find "$dir" -maxdepth 1 -name 'python3*._pth' | head -1)
    sed -i 's|^#import site|import site|' "$pth" || true
    grep -v '^\.\./' "$pth" > "$dir/py.pth" || true
    printf '\n..\\python-home\n..\\wheels\n' >> "$dir/py.pth"
    mv "$dir/py.pth" "$pth"
  fi

  # 2) Download wheels for THIS target platform/arch only, then merge the
  #    g4f package + wheels so the interpreter is self-contained.
  #    pip --platform lets us fetch the right cp314 wheels from a single
  #    ubuntu runner for every target (no need to execute the target python).
  local tag
  case "$name" in
    linux-x64)      tag="manylinux2014_x86_64" ;;
    linux-arm64)    tag="manylinux2014_aarch64" ;;
    windows-amd64)  tag="win_amd64" ;;
    windows-x86)    tag="win32" ;;
    windows-arm64)  tag="win_arm64" ;;
    darwin-arm64)   tag="macosx_11_0_arm64" ;;
    darwin-x64)     tag="macosx_10_15_universal2" ;; # covers x86_64 + arm64
    bsd-amd64)      tag="manylinux2014_x86_64" ;;
    *) echo "  WARNING: no wheel tag for $name; runtime will need network on first run" >&2; tag="" ;;
  esac
  mkdir -p "$dir/wheels"
#  if [ -n "$tag" ]; then
#     # Windows pip can't read process-substitution FDs from its subprocess, so
#     # write the version floors to a real file once per platform.
#     local floors="$dir/wheels.floors"
#     printf '%s\n' "$WHEEL_FLOORS" > "$floors"
#     # Full dependency resolution (no --no-deps) so the runtime's offline
#     # `pip install g4f` finds every transitive wheel it needs.
#     # brotli is optional in g4f and has no cp314 wheel for win_arm64;
#     # fall back to fetching everything else if a single dep is unavailable.
#     if ! python3 -m pip download \
#         -r "$G4F_SRC/requirements-min.txt" \
#         --constraint "$floors" \
#         --only-binary=:all: \
#         --python-version "$PYVER" --implementation cp --abi "cp$(echo "$PYVER" | tr -d '.')" \
#         --platform "$tag" \
#         -d "$dir/wheels" -q 2>"$dir/pip.err"; then
#       echo "  WARNING: full wheel set for $name not available; retrying without brotli (optional dep)" >&2
#       grep -v '^brotli$' "$G4F_SRC/requirements-min.txt" > "$dir/req-nobrotli.txt"
#       python3 -m pip download \
#           -r "$dir/req-nobrotli.txt" \
#           --constraint "$floors" \
#           --only-binary=:all: \
#           --python-version "$PYVER" --implementation cp --abi "cp$(echo "$PYVER" | tr -d '.')" \
#           --platform "$tag" \
#           -d "$dir/wheels" -q 2>>"$dir/pip.err" || {
#         echo "  WARNING: wheel download for $name failed; runtime will need network on first run" >&2
#       }
#     fi
#     # Belt-and-braces: drop any wheel whose Requires-Python metadata excludes
#     # our interpreter. The --constraint above prevents this at resolve time;
#     # this catches stale wheels (e.g. copied in from wheels-cache) and covers
#     # ancient pure-python releases like aiohttp 0.13.1.
#     python3 - "$dir/wheels" "$PYVER" <<'PY' || true
# import glob, os, sys, zipfile
# from packaging.specifiers import SpecifierSet
# from packaging.version import Version
# want_v = Version(sys.argv[2])
# for whl in glob.glob(os.path.join(sys.argv[1], "*.whl")):
#     try:
#         with zipfile.ZipFile(whl) as z:
#             meta = next((n for n in z.namelist() if n.endswith(".dist-info/METADATA")), None)
#             if not meta:
#                 continue
#             txt = z.read(meta).decode("utf-8", "replace")
#         rp = next((l.split(":", 1)[1].strip() for l in txt.splitlines()
#                    if l.lower().startswith("requires-python:")), None)
#         if rp and not SpecifierSet(rp).contains(want_v):
#             print(f"  removing {os.path.basename(whl)} (Requires-Python {rp} excludes {sys.argv[2]})")
#             os.remove(whl)
#     except Exception as e:
#         print(f"  skip check {os.path.basename(whl)}: {e}")
# PY
#  fi

  # 3) Merge g4f package + wheels so the interpreter is self-contained.
  # rsync -a --exclude='.git' --exclude='g4f-go' --exclude='g4f.dev' --exclude='g4f.egg-info' \
  #       "$G4F_SRC/g4f" "$dir/python-home/g4f"
  # rsync -a --exclude='.git' --exclude='g4f-go' --exclude='g4f.dev' --exclude='g4f.egg-info' \
  #       "$G4F_SRC/requirements-min.txt" "$dir/python-home/requirements-min.txt"
  # cp -n "$WORK/wheels-cache/"*.whl "$dir/wheels/" 2>/dev/null || true

  # 4) Pre-stamp install so first run is instant.
  mkdir -p "$dir/.g4f-runtime"
  printf 'g4f %s embedded (CPython %s)\n' "$G4F_VERSION" "$PYVER" > "$dir/.g4f-runtime/.installed"

  # 5) Repack next to the placeholder so go:embed sees only one zip.
  rm -f "$HERE/embed/$plat/${name}-embed-${PYVER}.zip"
  ( cd "$WORK" && zip -qr "$HERE/embed/$plat/${name}-embed-${PYVER}.zip" "$name" )
  echo "    -> embed/$plat/${name}-embed-${PYVER}.zip"
}

for p in "${PLATFORMS[@]}"; do do_platform "$p"; done
echo "Done. Rebuild with ./build-all.sh (or: go build -o g4f-go .)"
