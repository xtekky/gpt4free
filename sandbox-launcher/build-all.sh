#!/usr/bin/env bash
# build-all.sh
#
# Cross-compiles the standalone g4f-sandbox launcher for every supported
# OS/arch and packs one zip per platform. The image is NOT embedded: the
# launcher pulls the multi-arch sandbox image (or the smaller API-only image)
# on first run.
#
# Usage:
#   ./build-all.sh                         # build every target
#   ./build-all.sh linux                   # build only one OS
#   ./build-all.sh android                 # Termux (static linux/arm64)
#   G4F_VERSION=0.1.0 ./build-all.sh       # custom version
#
# The android target is built as a static linux/arm64 binary: Termux executes
# Linux ELF binaries directly, and the static build has no glibc dependency.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VERSION="${G4F_VERSION:-0.1.0}"
OUT="${OUT:-$HERE/dist}"
mkdir -p "$OUT"

WANT_OS="${1:-${OS_ONLY:-}}"
case "$WANT_OS" in
    "linux"|"windows"|"darwin"|"android") echo "==> Building only: $WANT_OS" ;;
    "") echo "==> Building all targets" ;;
    *) echo "Unknown OS filter: $WANT_OS (expected linux|windows|darwin|android)" >&2; exit 1 ;;
esac

# os      arch   binary name
TARGETS=(
    "linux   amd64  g4f-sandbox"
    "linux   arm64  g4f-sandbox"
    "windows amd64  g4f-sandbox.exe"
    "windows arm64  g4f-sandbox.exe"
    "darwin  amd64  g4f-sandbox"
    "darwin  arm64  g4f-sandbox"
    "android arm64  g4f-sandbox"
)

for t in "${TARGETS[@]}"; do
    read -r GOOS GOARCH BIN <<<"$t"
    case "$WANT_OS" in
        "linux")   [[ "$GOOS" == "linux" ]] || continue ;;
        "windows") [[ "$GOOS" == "windows" ]] || continue ;;
        "darwin")  [[ "$GOOS" == "darwin" ]] || continue ;;
        "android") [[ "$GOOS" == "android" ]] || continue ;;
        "") ;;
    esac

    build_goos="$GOOS"
    if [ "$GOOS" = "android" ]; then
        build_goos="linux"
    fi

    dir="$OUT/$GOOS-$GOARCH"
    mkdir -p "$dir"
    echo "==> $GOOS/$GOARCH"
    CGO_ENABLED=0 GOOS="$build_goos" GOARCH="$GOARCH" \
        go build -trimpath -ldflags "-s -w -X main.Version=$VERSION" -o "$dir/$BIN" .
    ( cd "$dir" && zip -qr "../g4f-sandbox-$VERSION-$GOOS-$GOARCH.zip" . )
done

echo "Built releases in $OUT:"
ls -lh "$OUT"/*.zip
