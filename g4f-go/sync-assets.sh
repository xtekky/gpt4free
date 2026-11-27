#!/usr/bin/env bash
# sync-assets.sh
#
# Regenerate the files embedded by g4f-go/polli from the single source of truth
# in bin/. They are build artifacts, not checked-in command copies. Builds and
# tests run this first so `go:embed` always sees the current bin/ commands.
#
# Usage:
#   ./sync-assets.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DEST="$HERE/polli/assets"

ASSETS=(
    sandbox-cli
    polli-sandbox-bootstrap.sh
    install-desktop.sh
    patch-xfce-menu.py
    sandbox-discover
    sandbox-docs
    start-browser
)

rm -rf "$DEST"
mkdir -p "$DEST"
for name in "${ASSETS[@]}"; do
    cp "$REPO/bin/$name" "$DEST/$name"
    chmod 0644 "$DEST/$name"
done

echo "Synced ${#ASSETS[@]} embedded assets into $DEST"
