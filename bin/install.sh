#!/usr/bin/env bash
#
# install.sh — install the g4f-go launcher for this machine.
#
# Downloads the prebuilt g4f-go release zip that matches the host OS/arch from
# https://github.com/xtekky/gpt4free/releases, verifies its SHA-256 against the
# digest published by GitHub, and installs the binary into a bin directory.
#
# Usage:
#   ./install.sh                    install the latest release
#   ./install.sh --version v8.7.7   install a specific release
#   ./install.sh --prefix /usr/local/bin
#   ./install.sh --uninstall        remove an installed g4f-go
#
# Environment:
#   G4F_GO_VERSION   same as --version
#   G4F_GO_PREFIX    same as --prefix

set -euo pipefail

REPO="xtekky/gpt4free"
BINARY="g4f-go"
API_BASE="https://api.github.com/repos/${REPO}"

VERSION="${G4F_GO_VERSION:-}"
PREFIX="${G4F_GO_PREFIX:-}"
VERIFY=1
UNINSTALL=0

log()  { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Install the g4f-go launcher (${REPO}).

Options:
  -v, --version VERSION   Release to install (default: latest; accepts 8.7.7 or v8.7.7)
  -p, --prefix DIR        Install directory (default: /usr/local/bin when root,
                             otherwise ~/.local/bin)
      --no-verify         Skip the SHA-256 check of the downloaded archive
      --uninstall         Remove the installed g4f-go binary
  -h, --help              Show this help

After installing, the first \`g4f-go\` run downloads a CPython runtime into
~/.g4f/ (this needs network access, but no system Python).
EOF
}

# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        -v|--version) [ $# -ge 2 ] || die "$1 requires an argument"; VERSION="$2"; shift 2 ;;
        --version=*)  VERSION="${1#*=}"; shift ;;
        -p|--prefix)  [ $# -ge 2 ] || die "$1 requires an argument"; PREFIX="$2"; shift 2 ;;
        --prefix=*)   PREFIX="${1#*=}"; shift 2 ;;
        --no-verify)  VERIFY=0; shift ;;
        --uninstall)  UNINSTALL=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            usage >&2; die "unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
# http_get URL — print a remote document to stdout
http_get() {
    local url="$1"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --retry-delay 1 "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O - "$url"
    else
        die "need curl or wget to download files"
    fi
}

# download URL DEST
download() {
    local url="$1" dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --retry-delay 1 -o "$dest" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$dest" "$url"
    else
        die "need curl or wget to download files"
    fi
}

# json_field JSON FIELD  (first "field": "value" occurrence on one long line)
json_field() {
    printf '%s' "$1" | tr -d '\n' | sed -n 's/.*"'"$2"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        echo ""
    fi
}

# ---------------------------------------------------------------------------
# platform detection
# ---------------------------------------------------------------------------
detect_os() {
    local u
    u="$(uname -s)"
    case "$u" in
        Linux)                 echo linux ;;
        Darwin)                echo darwin ;;
        FreeBSD)               echo freebsd ;;
        CYGWIN*|MINGW*|MSYS*)  echo windows ;;
        *)
            if [ "$(uname -o 2>/dev/null)" = "Android" ]; then
                die "Android/Termux is supported by building from source — see g4f-go/README.md"
            fi
            die "unsupported OS: $u (supported: linux, darwin, freebsd, windows)"
            ;;
    esac
}

detect_arch() {
    local m
    m="$(uname -m)"
    case "$m" in
        x86_64|amd64)  echo amd64 ;;
        aarch64|arm64) echo arm64 ;;
        *) die "unsupported architecture: $m (supported: amd64, arm64)" ;;
    esac
}

# ---------------------------------------------------------------------------
# uninstall
# ---------------------------------------------------------------------------
if [ "$UNINSTALL" -eq 1 ]; then
    if [ -z "$PREFIX" ]; then
        found="$(command -v "$BINARY" 2>/dev/null || true)"
        [ -n "$found" ] || die "g4f-go not found on PATH"
        PREFIX="$(dirname "$found")"
    fi
    target="$PREFIX/$BINARY"
    if [ ! -e "$target" ] && [ ! -e "$target.exe" ]; then
        die "nothing to remove at $target"
    fi
    rm -f "$target" "$target.exe"
    log "removed $target"
    log "user data in ~/.g4f was left untouched"
    exit 0
fi

# ---------------------------------------------------------------------------
# resolve platform + release
# ---------------------------------------------------------------------------
if [ -z "$PREFIX" ]; then
    if [ "$(id -u)" -eq 0 ]; then
        PREFIX="/usr/local/bin"
    else
        PREFIX="$HOME/.local/bin"
    fi
fi

OS="$(detect_os)"
ARCH="$(detect_arch)"
BIN_NAME="$BINARY"
[ "$OS" = windows ] && BIN_NAME="${BINARY}.exe"

# Release metadata doubles as the source of the SHA-256 digest; parsing it
# needs python3, so without it we skip verification (with a warning).
SKIPPED_REASON=""
if [ "$VERIFY" -eq 1 ] && ! command -v python3 >/dev/null 2>&1; then
    VERIFY=0
    SKIPPED_REASON="needs python3 to parse release metadata"
fi

RELEASE_JSON=""
if [ -z "$VERSION" ]; then
    log "resolving latest release of ${REPO}"
    RELEASE_JSON="$(http_get "${API_BASE}/releases/latest" 2>/dev/null || true)"
    if [ -n "$RELEASE_JSON" ]; then
        VERSION="$(json_field "$RELEASE_JSON" tag_name)"
    fi
    if [ -z "$VERSION" ] && command -v curl >/dev/null 2>&1; then
        # Fall back to the /releases/latest redirect (no API rate limit).
        redirect="$(curl -fsSL -o /dev/null -w '%{redirect_url}' \
            "https://github.com/${REPO}/releases/latest" 2>/dev/null || true)"
        VERSION="$(printf '%s' "$redirect" | sed -n 's#.*/tag/\(v[^/]*\)$#\1#p')"
    fi
    [ -n "$VERSION" ] || die "could not determine the latest release tag (pass --version)"
else
    # Normalise "8.7.7" -> "v8.7.7".
    case "$VERSION" in
        v*) ;;
        *)  VERSION="v$VERSION" ;;
    esac
    if [ "$VERIFY" -eq 1 ]; then
        RELEASE_JSON="$(http_get "${API_BASE}/releases/tags/${VERSION}" 2>/dev/null || true)"
    fi
fi

ASSET="${BINARY}-${VERSION}-${OS}-${ARCH}.zip"
URL="https://github.com/${REPO}/releases/download/${VERSION}/${ASSET}"

log "installing ${BINARY} ${VERSION} for ${OS}/${ARCH}"
log "downloading ${URL}"

TMPDIR_INSTALL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_INSTALL"' EXIT

ARCHIVE="$TMPDIR_INSTALL/$ASSET"
if ! download "$URL" "$ARCHIVE"; then
    die "download failed — ${ASSET} is not published for ${OS}/${ARCH} (see https://github.com/${REPO}/releases)"
fi

# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
if [ "$VERIFY" -eq 1 ] && [ -n "$RELEASE_JSON" ]; then
    expected="$(printf '%s' "$RELEASE_JSON" | python3 -c '
import json, sys
asset = sys.argv[1]
try:
    data = json.load(sys.stdin)
    for entry in data.get("assets", []):
        if entry.get("name") == asset:
            digest = entry.get("digest") or ""
            if digest.startswith("sha256:"):
                print(digest[len("sha256:"):])
            break
except Exception:
    pass
' "$ASSET")"
    if [ -n "$expected" ]; then
        actual="$(sha256_file "$ARCHIVE")"
        if [ -z "$actual" ]; then
            warn "no sha256 tool available (sha256sum/shasum/openssl) — skipping verification"
        elif [ "$actual" = "$expected" ]; then
            log "sha256 verified"
        else
            die "sha256 mismatch (expected ${expected}, got ${actual})"
        fi
    else
        warn "no published digest for ${ASSET} — skipping verification"
    fi
elif [ -n "$SKIPPED_REASON" ]; then
    warn "verification skipped (${SKIPPED_REASON})"
elif [ "$VERIFY" -eq 1 ]; then
    warn "no release metadata available — skipping verification"
fi

# ---------------------------------------------------------------------------
# extract + install
# ---------------------------------------------------------------------------
EXTRACT="$TMPDIR_INSTALL/x"
mkdir -p "$EXTRACT"
if command -v unzip >/dev/null 2>&1; then
    unzip -qo "$ARCHIVE" -d "$EXTRACT"
elif command -v python3 >/dev/null 2>&1; then
    python3 -m zipfile -e "$ARCHIVE" "$EXTRACT"
else
    die "unzip (or python3) is required to extract the release archive"
fi

SRC="$(find "$EXTRACT" -type f -name "$BIN_NAME" | head -n1)"
[ -n "$SRC" ] || die "archive did not contain ${BIN_NAME}"

mkdir -p "$PREFIX"
if command -v install >/dev/null 2>&1; then
    install -m 0755 "$SRC" "$PREFIX/$BIN_NAME"
else
    cp "$SRC" "$PREFIX/$BIN_NAME"
    chmod 0755 "$PREFIX/$BIN_NAME"
fi
log "installed ${PREFIX}/${BIN_NAME}"

case ":${PATH}:" in
    *":${PREFIX}:"*) ;;
    *) warn "${PREFIX} is not in PATH — add it with:  export PATH=\"${PREFIX}:\$PATH\"" ;;
esac

log "done. Try: ${BINARY} help"
log "the first run downloads a CPython runtime into ~/.g4f/ (network required)"
