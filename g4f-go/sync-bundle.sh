#!/usr/bin/env bash
# sync-bundle.sh
#
# Copies the Python packages that g4f-go ships inside its binary from their
# source projects into g4f-go/bundle/. The bundle is committed, so a plain
# `go build` works without running this; run it after changing a bundled
# project and before building a release.
#
# Usage:
#   ./sync-bundle.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DEST="$HERE/bundle"

rm -rf "$DEST"
mkdir -p "$DEST"

# remote-desktop: the remote_desktop package plus the web assets it serves.
# app.py resolves its assets as <package>/../web, so `web` is a sibling of the
# package inside the bundle (and therefore inside the extracted bundle dir).
cp -r "$REPO/projects/remote-desktop/remote_desktop" "$DEST/remote_desktop"
cp -r "$REPO/projects/remote-desktop/web" "$DEST/web"

# Never ship bytecode caches.
find "$DEST" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$DEST" -name '*.pyc' -delete

echo "Synced bundle into $DEST"
