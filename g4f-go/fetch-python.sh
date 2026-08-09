#!/usr/bin/env bash
# fetch-python.sh
#
# Downloads the CPython runtimes for every supported platform and pins their
# size + sha256 into runtime.json. The g4f-go binary itself does NOT embed
# the runtimes anymore: it downloads the matching archive on first run (with
# live progress feedback) and verifies it against this manifest.
#
# Usage:
#   ./fetch-python.sh            # download all platforms
#   ./fetch-python.sh linux      # single platform group (linux|windows|darwin|android)
#
# Requirements: bash, curl, python3 (for sha256/size pinning)
#
# After updating URLs in runtime.json, re-run this to refresh the pins.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
MANIFEST="runtime.json"

pick_urls() {
  python3 - "$MANIFEST" "$WANT" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
want = sys.argv[2]
for name, spec in m["platforms"].items():
    if want == "all" or name.startswith(want):
        print(name, spec["url"])
PY
}

WANT="${1:-all}"
case "$WANT" in
  linux|windows|darwin|android|all) ;;
  *) echo "Unknown filter: $WANT (expected linux|windows|darwin|android|all)" >&2; exit 1 ;;
esac

if ! command -v python3 >/dev/null; then
  echo "python3 required for manifest pinning" >&2; exit 1
fi

pin() {
  # pin <name> <url>: download, compute sha256+size, update runtime.json
  local name url file sha size
  name="$1"; url="$2"
  file="$(mktemp)"
  echo "==> [$name] downloading $url"
  curl -fsSL -o "$file" "$url"
  sha="$(sha256sum "$file" | cut -d' ' -f1)"
  size="$(stat -c%s "$file")"
  rm -f "$file"
  echo "    sha256=$sha"
  echo "    size=$size"
  python3 - "$MANIFEST" "$name" "$sha" "$size" <<'PY'
import json, sys
path, name, sha, size = sys.argv[1:5]
m = json.load(open(path))
m["platforms"][name]["sha256"] = sha
m["platforms"][name]["size"] = int(size)
json.dump(m, open(path, "w"), indent=2, sort_keys=True)
print("    -> runtime.json updated for", name)
PY
}

pick_urls | while read -r name url; do
  pin "$name" "$url"
done

echo "Done. Build with ./build-all.sh (or: go build -o g4f-go .)"
