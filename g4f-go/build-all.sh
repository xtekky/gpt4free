#!/usr/bin/env bash
# build-all.sh
#
# Cross-compiles the g4f-go launcher for every supported OS/arch and packs a
# release zip per platform. The embedded Python archive is expected to have
# been prepared by ./fetch-python.sh first (the launcher still builds without
# it, but then refuses to run).
#
# Usage:
#   ./build-all.sh                       # build every target
#   ./build-all.sh linux                 # build only one OS (linux|windows|darwin|bsd)
#   G4F_VERSION=0.1.0 ./build-all.sh     # custom version

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
VERSION="${G4F_VERSION:-0.1.0}"
OUT="${OUT:-$HERE/dist}"
mkdir -p "$OUT"

# Optional OS filter: maps the fetch-python.sh platform names (linux,
# windows, darwin, bsd) onto GOOS. Empty = build everything.
WANT_OS="${1:-${OS_ONLY:-}}"
case "$WANT_OS" in
  "linux"|"windows"|"darwin"|"bsd") echo "==> Building only: $WANT_OS" ;;
  "") echo "==> Building all targets" ;;
  *) echo "Unknown OS filter: $WANT_OS (expected linux|windows|darwin|bsd)" >&2; exit 1 ;;
esac

# os      arch   ext  name
TARGETS=(
  "linux   amd64  g4f-go"
  "linux   arm64  g4f-go"
  "windows amd64  g4f-go.exe"
  "windows 386    g4f-go.exe"
  "darwin  arm64  g4f-go"
  "darwin  amd64  g4f-go"
  "freebsd amd64  g4f-go"
)

for t in "${TARGETS[@]}"; do
  read -r GOOS GOARCH BIN <<<"$t"
  case "$WANT_OS" in
    "linux")   [[ "$GOOS" == "linux" ]] || continue ;;
    "windows") [[ "$GOOS" == "windows" ]] || continue ;;
    "darwin")  [[ "$GOOS" == "darwin" ]] || continue ;;
    "bsd")     [[ "$GOOS" == "freebsd" ]] || continue ;;
    "") ;;
  esac
  echo "==> $GOOS/$GOARCH"
  CGO_ENABLED=0 GOOS="$GOOS" GOARCH="$GOARCH" \
    go build -trimpath -ldflags "-s -w -X main.Version=$VERSION" -o "$OUT/$GOOS-$GOARCH/$BIN" .
  ( cd "$OUT/$GOOS-$GOARCH" && zip -qr "../g4f-go-$VERSION-$GOOS-$GOARCH.zip" . )
done

echo "Built releases in $OUT:"
ls -lh "$OUT"/*.zip
