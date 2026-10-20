#!/usr/bin/env bash
#
# sandbox-cli — the installer behind the sandbox desktop's "Install & Open ..."
# menu entries (docker/sandbox-menu/).
#
#   sandbox-cli <tool>          install <tool> if missing, then run it here
#   sandbox-cli install <tool>  install only, don't run it
#   sandbox-cli settings        choose the g4f model, HOME, provider and proxy
#   sandbox-cli reset           clear the saved settings, back to the defaults
#   sandbox-cli list            list the tools the desktop menu knows about
#   sandbox-cli urls            print the public URLs served by the g4f API
#
# Each menu item runs:
#   xfce4-terminal --command="/usr/local/bin/sandbox-cli <tool>"
# The install happens on FIRST CLICK (vendor install script, npm package,
# `pip install --user`, or Google's Chrome .deb), so the image stays small;
# afterwards the CLI starts instantly in the same terminal window (Chrome
# opens on the desktop instead). nodejs/npm are already in the image for the
# npm-based tools, everything else is one click. Runs as `pn`.

set -u

# pip --user and the vendor installers write to ~/.local/bin, ~/.opencode/bin,
# ~/.openclaw/bin, ~/.kimi-code/bin, ~/.qoder/bin or ~/.lmstudio/bin, which are
# not on PATH inside a bare terminal.
if [ -n "${HOME:-}" ]; then
    PATH="$HOME/.local/bin:$HOME/.opencode/bin:$HOME/.openclaw/bin:$HOME/.kimi-code/bin:$HOME/.qoder/bin:$HOME/.lmstudio/bin:$PATH"
fi
export PATH

ANTIGRAVITY_INSTALLER="https://antigravity.google/cli/install.sh"
OPENCODE_INSTALLER="https://opencode.ai/v2/install"
GOOSE_INSTALLER="https://github.com/block/goose/releases/download/stable/download_cli.sh"
CURSOR_INSTALLER="https://cursor.com/install"
OPENCLAW_INSTALLER="https://openclaw.ai/install.sh"
# Google publishes per-architecture .deb files; append the dpkg arch (amd64,
# arm64) to this prefix. The .deb pulls its own Chrome runtime libraries.
CHROME_DEB_BASE="https://dl.google.com/linux/direct/google-chrome-stable_current"
# Vendor install scripts whose binaries land on PATH (~/.local/bin or
# /usr/local/bin, both searched above).
GROK_INSTALLER="https://x.ai/cli/install.sh"
DROID_INSTALLER="https://app.factory.ai/cli"
OPENHANDS_INSTALLER="https://install.openhands.dev/install.sh"
PLANDEX_INSTALLER="https://plandex.ai/install.sh"
FABRIC_INSTALLER="https://raw.githubusercontent.com/danielmiessler/fabric/main/scripts/installer/install.sh"
HERMES_INSTALLER="https://hermes-agent.nousresearch.com/install.sh"
KIRO_INSTALLER="https://cli.kiro.dev/install"
KIMI_INSTALLER="https://code.kimi.com/kimi-code/install.sh"
# Roomote is a self-hosted service, not a CLI: its installer needs root, brings
# up Docker plus a Compose stack under /opt/roomote and drops the `roomote`
# management CLI in /usr/local/bin (see ensure_roomote).
ROOMOTE_INSTALLER="https://get.roomote.dev"
QODER_INSTALLER="https://qoder.com/install"
VIBE_INSTALLER="https://mistral.ai/vibe/install.sh"
UV_INSTALLER="https://astral.sh/uv/install.sh"
# assistant-ui's setup-agent drives a setup that is shared live between the
# browser that opened this URL and the agent installing the products. The URL
# is the browser's link to its agent, not one setup: a finished setup stays on
# it and the browser's next one replaces it.
ASSISTANT_UI_SETUP_URL="https://setup.assistant-ui.com/Q2Exyx6AidfK"
# Local model runtimes and their web UIs.
OLLAMA_INSTALLER="https://ollama.com/install.sh"
LMSTUDIO_INSTALLER="https://lmstudio.ai/install.sh"
# llama.cpp publishes per-build tarballs; the release tag is resolved at install
# time (see ensure_llamacpp) and the arch suffix appended to this prefix.
LLAMACPP_RELEASE_BASE="https://github.com/ggml-org/llama.cpp/releases/download"
# Jan ships a Linux AppImage; the release tag is resolved at install time.
JAN_RELEASE_BASE="https://github.com/janhq/jan/releases/download"
# Chat UIs. LobeChat publishes a Linux AppImage per release (tag resolved at
# install time, x64 only); Msty Studio serves a stable "latest" AppImage URL
# (x64 only). LibreChat is not packaged: it is cloned and built from source
# (see ensure_librechat) and needs MongoDB, which the image does not ship.
LOBECHAT_RELEASE_BASE="https://github.com/lobehub/lobehub/releases/download"
MSTY_APPIMAGE_URL="https://next-assets.msty.studio/app/latest/linux/MstyStudio_x86_64.AppImage"
LIBRECHAT_REPO="https://github.com/danny-avila/LibreChat.git"
# MongoDB publishes no trixie build of mongodb-org-server, so the server comes
# from the bookworm repository (the .deb is self-contained) or, on arm64, from
# the ubuntu2404 tarball -- fastdl serves no aarch64 debian12 tarball.
MONGODB_APT_BASE="https://repo.mongodb.org/apt/debian"
MONGODB_APT_DIST="bookworm/mongodb-org/8.0"
MONGODB_GPG_URL="https://pgp.mongodb.com/server-8.0.asc"
MONGODB_TARBALL_BASE="https://fastdl.mongodb.org/linux"
MONGODB_VERSION="8.0.34"
# SillyTavern is a Node web app without an upstream package: it is cloned and
# served with npm (see ensure_sillytavern). The port is moved off SillyTavern's
# default 8000 so a running Open WebUI Computer (cptr, also 8000) does not
# collide with it.
SILLYTAVERN_REPO="https://github.com/SillyTavern/SillyTavern.git"
SILLYTAVERN_DIR_NAME=".sillytavern"
SILLYTAVERN_PORT="8601"
# Open WebUI Computer (cptr) is pip-installable: one local server that serves
# files, terminal, git, editor and AI to any browser, on its documented
# default port 8000.
CPTR_PORT="8000"
# code-server (VS Code in the browser) ships a vendor install script; the
# standalone method keeps the install under ~/.local. Its default port 8080 is
# the g4f API's, so it listens on 8602 instead (see the code-server case).
CODE_SERVER_INSTALLER="https://code-server.dev/install.sh"
CODE_SERVER_PORT="8602"
# The tunnel entry ("Install & Run Tunnel Client") is backed by Cloudflare's
# cloudflared: `cloudflared tunnel --url <target>` opens a public HTTPS URL
# (trycloudflare.com) for a local port with no account and no config file. The
# per-architecture static binary is fetched from the latest release over HTTPS;
# the arch suffix is appended at install time (see ensure_cloudflared).
CLOUDFLARED_RELEASE_BASE="https://github.com/cloudflare/cloudflared/releases/latest/download"

INSTALL_ONLY=0

have() { command -v "$1" >/dev/null 2>&1; }
log()  { printf '==> %s\n' "$*"; }

# True when something accepts connections on 127.0.0.1:<port>.
port_open() {
    if have curl; then
        curl -s --max-time 1 "http://127.0.0.1:$1/json/version" >/dev/null 2>&1
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
    fi
}

# Menu entries open a fresh terminal that closes on exit; hold it so the
# output of a one-shot command stays readable.
hold_open() {
    [ -t 0 ] || return 0
    printf '\nPress Enter to close...'
    read -r _ || true
}

die() {
    printf '\nsandbox-cli: error: %s\n' "$*" >&2
    hold_open
    exit 1
}

# Menu installs run as `pn`, which has passwordless sudo in the image.
as_root() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo -n "$@" || die "need root privileges for: $*"
    fi
}

# True when something accepts a TCP connection on 127.0.0.1:<port>. Unlike
# port_open this makes no HTTP request, so it also works for non-HTTP services
# such as the bundled MongoDB.
tcp_open() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1
}

# Set KEY=value in an env-style file: replace the line when the key is already
# there (commented out or not), append it otherwise.
set_env_var() { # <file> <key> <value>
    local file="$1" key="$2" value="$3" tmp
    tmp="$(mktemp)" || die "could not create a temporary file"
    if grep -qE "^[[:space:]]*#?[[:space:]]*${key}=" "$file"; then
        sed -E "0,/^[[:space:]]*#?[[:space:]]*${key}=/s|^[[:space:]]*#?[[:space:]]*${key}=.*|${key}=${value}|" \
            "$file" >"$tmp" || die "could not update $key in $file"
    else
        cat "$file" >"$tmp"
        printf '%s=%s\n' "$key" "$value" >>"$tmp"
    fi
    mv "$tmp" "$file" || die "could not write $file"
}

# ---------------------------------------------------------------------------
# Sandbox settings (g4f model, g4f provider + HOME)
#
# A desktop menu entry cannot export environment variables, so the launchers
# read their overrides from a small settings file that the "Sandbox Settings"
# menu entry writes. The file deliberately lives outside $HOME: changing HOME
# must not move the file that records it.
#
# Precedence, highest first:
#   G4F_SANDBOX_MODEL / G4F_SANDBOX_HOME / G4F_SANDBOX_PROVIDER /
#   G4F_SANDBOX_PROXY                      explicit environment override
#   the settings file                      what the menu entry saved
#   G4F_MODEL / $HOME / G4F_PROVIDER / G4F_PROXY
#                                          ambient environment, then defaults
#
# HOME applies to every menu entry (see apply_sandbox_home): agents keep their
# config, credentials and caches under $HOME, so pointing it at a scratch
# directory gives a throwaway profile.
#
# PROXY applies to every menu entry too (see apply_sandbox_proxy): it is
# exported as G4F_PROXY (read by g4f itself) and as the conventional
# HTTP_PROXY/HTTPS_PROXY/ALL_PROXY variables (read by the agents and the tools
# they shell out to). It typically points at the local "Install & Run Proxy
# Server" entry (mitmproxy on 127.0.0.1:8081); loopback stays direct. The
# proxy's CA is trusted at the same time (see trust_proxy_ca), and OpenCode's
# background server is told about both (see configure_opencode_service).
#
# A model only reaches the g4f-aware launchers when one was actually chosen —
# G4F_SANDBOX_MODEL, or G4F_MODEL in the settings file. With no model set, the
# launchers leave the tool's own provider/model configuration alone instead of
# forcing it onto the g4f API. The provider is optional: with none set, g4f
# picks from its own rotation. When one is set, the launchers pin it by
# prefixing the model name ("<provider>:<model>"), which is how g4f resolves a
# provider for a request; G4F_PROVIDER itself is read only by the g4f server.
# ---------------------------------------------------------------------------
real_home() {
    local home
    home="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
    [ -n "$home" ] || home="${HOME:-/home/pn}"
    printf '%s' "$home"
}

SANDBOX_ENV_FILE="${G4F_SANDBOX_ENV:-$(real_home)/.config/g4f-sandbox/env}"
# mitmproxy signs the HTTPS traffic it intercepts with its own CA. The CA lives
# in a fixed confdir under the real home (not $HOME): the per-entry HOME
# setting must not move it, or a tool launched with another HOME would not find
# the certificate the proxy is using. The path is also exported to the tools
# (see apply_sandbox_proxy), so a custom location works the same way.
PROXY_CA_DIR="${G4F_SANDBOX_MITM_DIR:-$(real_home)/.config/g4f-sandbox/mitmproxy}"
PROXY_CA_CERT="$PROXY_CA_DIR/mitmproxy-ca-cert.pem"
PROXY_CA_BUNDLE="$PROXY_CA_DIR/ca-bundle.pem"
# The system trust store, used as the base of the bundle the tools are pointed
# at: mitmproxy's CA is appended to it rather than replacing it, so direct
# (NO_PROXY) connections keep working.
SYSTEM_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"
SANDBOX_MODEL=""
SANDBOX_HOME=""
SANDBOX_PROVIDER=""
SANDBOX_PROXY=""
# 1 once a model was chosen (G4F_SANDBOX_MODEL, or the settings file), else 0.
SANDBOX_MODEL_SET=0

load_sandbox_settings() {
    SANDBOX_MODEL=""
    SANDBOX_HOME=""
    SANDBOX_PROVIDER=""
    SANDBOX_PROXY=""
    SANDBOX_MODEL_SET=0
    [ -f "$SANDBOX_ENV_FILE" ] || return 0
    local line key value
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*) continue ;; esac
        key="${line%%=*}"
        value="${line#*=}"
        case "$key" in
            G4F_MODEL)
                SANDBOX_MODEL="$value"
                [ -n "$value" ] && SANDBOX_MODEL_SET=1 ;;
            SANDBOX_HOME) SANDBOX_HOME="$value" ;;
            G4F_PROVIDER) SANDBOX_PROVIDER="$value" ;;
            SANDBOX_PROXY) SANDBOX_PROXY="$value" ;;
        esac
    done < "$SANDBOX_ENV_FILE"
}

save_sandbox_settings() { # <model> <home> <provider> <proxy>
    local dir
    dir="$(dirname "$SANDBOX_ENV_FILE")"
    mkdir -p "$dir" || die "cannot create $dir"
    {
        printf '# g4f sandbox settings — written by: sandbox-cli settings\n'
        printf '# Read by the menu launchers (HOME, proxy) and the g4f-aware ones\n'
        printf '# (OpenCode, Copilot, Hermes).\n'
        printf 'G4F_MODEL=%s\n' "$1"
        printf 'SANDBOX_HOME=%s\n' "$2"
        printf 'G4F_PROVIDER=%s\n' "$3"
        printf 'SANDBOX_PROXY=%s\n' "$4"
    } > "$SANDBOX_ENV_FILE" || die "cannot write $SANDBOX_ENV_FILE"
}

show_sandbox_settings() {
    printf 'settings file: %s\n' "$SANDBOX_ENV_FILE"
    if [ -f "$SANDBOX_ENV_FILE" ]; then
        printf 'G4F_MODEL:     %s\n' "${SANDBOX_MODEL:-(unset)}"
        printf 'HOME:          %s\n' "${SANDBOX_HOME:-(unset)}"
        printf 'G4F_PROVIDER:  %s\n' "${SANDBOX_PROVIDER:-(unset)}"
        printf 'PROXY:         %s\n' "${SANDBOX_PROXY:-(unset)}"
    else
        printf 'G4F_MODEL:     %s (default)\n' "${G4F_MODEL:-(unset)}"
        printf 'HOME:          %s (default)\n' "$(real_home)"
        printf 'G4F_PROVIDER:  %s (default)\n' "${G4F_PROVIDER:-(unset)}"
        printf 'PROXY:         %s (default)\n' "${G4F_PROXY:-(unset)}"
    fi
}

validate_sandbox_home() { # <dir>
    case "$1" in
        /*) ;;
        *) die "HOME must be an absolute path: $1" ;;
    esac
    mkdir -p "$1" 2>/dev/null || true
    [ -w "$1" ] && return 0
    # A path outside the user's own tree (e.g. /opt/agent-home) needs root.
    # Create it as root, then hand it to the invoking user so the agents can
    # actually write their config there.
    log "creating $1 as root (needs sudo)..."
    as_root mkdir -p "$1" || die "cannot create HOME directory: $1"
    as_root chown "$(id -u):$(id -g)" "$1" \
        || die "cannot take ownership of HOME directory: $1"
    [ -w "$1" ] || die "HOME directory is not writable: $1"
}

prompt_sandbox_settings() {
    local model home provider proxy answer
    model="${SANDBOX_MODEL:-${G4F_MODEL:-}}"
    home="${SANDBOX_HOME:-$(real_home)}"
    provider="${SANDBOX_PROVIDER:-${G4F_PROVIDER:-}}"
    proxy="${SANDBOX_PROXY:-${G4F_PROXY:-}}"
    printf 'g4f sandbox settings — press Enter to keep the value in brackets.\n\n'
    printf 'Model passed to the g4f API (G4F_MODEL).\n'
    read -r -p "  model [$model]: " answer || true
    [ -n "$answer" ] && model="$answer"
    printf '\nHOME used by every menu entry (agents keep their config here).\n'
    read -r -p "  HOME [$home]: " answer || true
    [ -n "$answer" ] && home="$answer"
    printf '\ng4f provider the g4f-aware launchers pin (G4F_PROVIDER).\n'
    printf 'It is sent as "<provider>:<model>"; Enter keeps it, clear it with:\n'
    printf '  sandbox-cli settings --provider ""\n'
    read -r -p "  provider [${provider:-none}]: " answer || true
    [ -n "$answer" ] && provider="$answer"
    printf '\nProxy used by every menu entry (HTTP_PROXY/HTTPS_PROXY and G4F_PROXY).\n'
    printf 'Point it at the Proxy Server entry, e.g. http://127.0.0.1:8081.\n'
    printf 'Enter keeps it, clear it with:\n'
    printf '  sandbox-cli settings --proxy ""\n'
    read -r -p "  proxy [${proxy:-none}]: " answer || true
    [ -n "$answer" ] && proxy="$answer"
    printf '\n'
    validate_sandbox_home "$home"
    save_sandbox_settings "$model" "$home" "$provider" "$proxy"
    load_sandbox_settings
    show_sandbox_settings
}

settings_usage() {
    cat <<'EOF'
usage: sandbox-cli settings                    ask for model, HOME, provider and proxy
       sandbox-cli settings --show             print the saved settings
       sandbox-cli settings --model <id>       set the model (keeps the rest)
       sandbox-cli settings --home <dir>       set HOME (keeps the rest)
       sandbox-cli settings --provider <name>  set the g4f provider
       sandbox-cli settings --proxy <url>      set the HTTP(S) proxy
       sandbox-cli settings --unset            drop the file and use the defaults

HOME and PROXY apply to every menu entry. The g4f-aware launchers (OpenCode,
Copilot, Hermes) additionally read the model and provider; the environment
variables G4F_SANDBOX_MODEL, G4F_SANDBOX_HOME, G4F_SANDBOX_PROVIDER and
G4F_SANDBOX_PROXY override them. PROXY is exported as G4F_PROXY (for g4f) and
as HTTP_PROXY/HTTPS_PROXY/ALL_PROXY (for the agents), with loopback kept direct.
Those launchers only repoint the tool at the g4f provider once a model is set,
so with no model saved they keep the tool's own provider and model.
EOF
}

run_settings() {
    local model="" home="" provider="" proxy=""
    local set_model=0 set_home=0 set_provider=0 set_proxy=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --show|show)
                show_sandbox_settings
                return 0 ;;
            --unset|unset)
                rm -f "$SANDBOX_ENV_FILE" || die "cannot remove $SANDBOX_ENV_FILE"
                log "cleared $SANDBOX_ENV_FILE"
                return 0 ;;
            --model)
                [ $# -ge 2 ] || die "--model needs a value"
                model="$2"; set_model=1; shift 2 ;;
            --home)
                [ $# -ge 2 ] || die "--home needs a value"
                home="$2"; set_home=1; shift 2 ;;
            --provider)
                [ $# -ge 2 ] || die "--provider needs a value"
                provider="$2"; set_provider=1; shift 2 ;;
            --proxy)
                [ $# -ge 2 ] || die "--proxy needs a value"
                proxy="$2"; set_proxy=1; shift 2 ;;
            -h|--help|help)
                settings_usage
                return 0 ;;
            *)
                die "unknown settings option \"$1\" — run: sandbox-cli settings --help" ;;
        esac
    done
    if [ "$set_model" = 1 ] || [ "$set_home" = 1 ] || [ "$set_provider" = 1 ] \
        || [ "$set_proxy" = 1 ]; then
        [ "$set_model" = 1 ] || model="${SANDBOX_MODEL:-${G4F_MODEL:-}}"
        [ "$set_home" = 1 ] || home="${SANDBOX_HOME:-$(real_home)}"
        [ "$set_provider" = 1 ] || provider="${SANDBOX_PROVIDER:-${G4F_PROVIDER:-}}"
        [ "$set_proxy" = 1 ] || proxy="${SANDBOX_PROXY:-${G4F_PROXY:-}}"
        validate_sandbox_home "$home"
        save_sandbox_settings "$model" "$home" "$provider" "$proxy"
        load_sandbox_settings
        show_sandbox_settings
        return 0
    fi
    [ -t 0 ] || die "sandbox-cli settings needs a terminal (or use --model/--home/--provider/--proxy/--show)"
    prompt_sandbox_settings
}

reset_usage() {
    cat <<'EOF'
usage: sandbox-cli reset            clear the saved settings (asks first)
       sandbox-cli reset --yes      clear them without asking
       sandbox-cli reset --show     print what would be cleared

Drops the settings file written by "sandbox-cli settings", so the launchers
fall back to the ambient environment and the built-in defaults (G4F_MODEL,
$HOME, G4F_PROVIDER). Only that file is removed — installed tools, agent
configs and the g4f runtime are left alone.
EOF
}

run_reset() {
    local assume_yes=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --yes|-y|yes)
                assume_yes=1; shift ;;
            --show|show)
                show_sandbox_settings
                return 0 ;;
            -h|--help|help)
                reset_usage
                return 0 ;;
            *)
                die "unknown reset option \"$1\" — run: sandbox-cli reset --help" ;;
        esac
    done
    if [ ! -f "$SANDBOX_ENV_FILE" ]; then
        log "no settings file at $SANDBOX_ENV_FILE — nothing to reset"
        hold_open
        return 0
    fi
    if [ "$assume_yes" = 0 ]; then
        [ -t 0 ] || die "sandbox-cli reset needs a terminal (or use --yes)"
        printf 'This removes %s and restores the default model, HOME and provider.\n' \
            "$SANDBOX_ENV_FILE"
        local answer
        read -r -p "Reset the sandbox settings? [y/N]: " answer || true
        case "$answer" in
            y|Y|yes|YES) ;;
            *) log "kept $SANDBOX_ENV_FILE"; hold_open; return 0 ;;
        esac
    fi
    rm -f "$SANDBOX_ENV_FILE" || die "cannot remove $SANDBOX_ENV_FILE"
    load_sandbox_settings
    log "cleared $SANDBOX_ENV_FILE"
    show_sandbox_settings
    hold_open
}

load_sandbox_settings

# HOME is a per-menu-entry setting, so it is applied for every tool, not just
# the g4f-aware ones: agents keep their config, credentials and caches under
# $HOME, and a scratch HOME gives a throwaway profile. The PATH entries added
# at the top of this script were derived from the original HOME, so they are
# re-prepended here.
apply_sandbox_home() {
    local home="${G4F_SANDBOX_HOME:-${SANDBOX_HOME:-}}"
    [ -n "$home" ] || return 0
    # A HOME saved before it existed (or created by root) would make every
    # agent fail on its first write, so make sure it is there and ours.
    if [ ! -d "$home" ] || [ ! -w "$home" ]; then
        validate_sandbox_home "$home"
    fi
    export HOME="$home"
    PATH="$HOME/.local/bin:$HOME/.opencode/bin:$HOME/.openclaw/bin:$HOME/.kimi-code/bin:$HOME/.qoder/bin:$PATH"
    export PATH
}

# mitmproxy signs the HTTPS traffic it intercepts with its own CA, which no
# tool trusts by default: without it every intercepted HTTPS request fails with
# "unable to verify the first certificate" (OpenCode logs exactly that for its
# update check and for models.opencode.ai). Point the tools at a bundle that
# holds the system CAs plus mitmproxy's, and install the CA into the system
# store so anything that reads it directly (curl, Python, Chrome) works too.
#
# The bundle is rebuilt on every launch: mitmproxy regenerates its CA when the
# confdir is recreated, and a stale bundle would silently break TLS again.
# Every failure here is non-fatal — a proxy that has not run yet simply has no
# CA to trust, and the launcher must still start the tool. as_root is
# deliberately not used: it dies (and waits for Enter) on failure, which would
# turn a missing CA into a launcher that never starts.
trust_proxy_ca() {
    [ -r "$PROXY_CA_CERT" ] || return 0
    local bundle="$PROXY_CA_BUNDLE"
    if ! { cat "$SYSTEM_CA_BUNDLE" 2>/dev/null || :; cat "$PROXY_CA_CERT"; } > "$bundle" 2>/dev/null; then
        log "could not write the proxy CA bundle at $bundle; HTTPS through the proxy may fail"
        return 0
    fi
    export NODE_EXTRA_CA_CERTS="$bundle"
    export SSL_CERT_FILE="$bundle" REQUESTS_CA_BUNDLE="$bundle" CURL_CA_BUNDLE="$bundle"
    export GIT_SSL_CAINFO="$bundle" AWS_CA_BUNDLE="$bundle"
    # The system store is what Chrome and the other GUI apps read. Installing
    # the CA there is idempotent: the anchor is overwritten in place and
    # update-ca-certificates rebuilds the store from the anchors, so repeated
    # launches do not pile up copies. update-ca-certificates lives in /usr/sbin,
    # which is not on the PATH the menu entries run with.
    local anchor="/usr/local/share/ca-certificates/g4f-sandbox-proxy.crt"
    [ -f "$anchor" ] && cmp -s "$PROXY_CA_CERT" "$anchor" && return 0
    if [ "$(id -u)" -eq 0 ]; then
        install -m 0644 "$PROXY_CA_CERT" "$anchor" 2>/dev/null || return 0
        /usr/sbin/update-ca-certificates >/dev/null 2>&1 || :
    elif sudo -n install -m 0644 "$PROXY_CA_CERT" "$anchor" 2>/dev/null; then
        sudo -n /usr/sbin/update-ca-certificates >/dev/null 2>&1 || :
    else
        log "could not install the proxy CA into the system store; GUI apps may not trust the proxy"
    fi
}

# PROXY is the other per-menu-entry setting that applies to every tool: it is
# exported both as G4F_PROXY, which g4f itself reads for its own requests, and
# as the conventional HTTP_PROXY/HTTPS_PROXY/ALL_PROXY variables, which the
# agents and the tools they shell out to honour. The typical value is the local
# "Install & Run Proxy Server" entry (mitmproxy on 127.0.0.1:8081). Loopback is
# added to NO_PROXY so the launchers can still reach the g4f API and the other
# local servers directly instead of looping back through the proxy.
apply_sandbox_proxy() {
    local proxy="${G4F_SANDBOX_PROXY:-${SANDBOX_PROXY:-${G4F_PROXY:-}}}"
    [ -n "$proxy" ] || return 0
    export G4F_PROXY="$proxy"
    export HTTP_PROXY="$proxy" HTTPS_PROXY="$proxy" ALL_PROXY="$proxy"
    export http_proxy="$proxy" https_proxy="$proxy" all_proxy="$proxy"
    local no_proxy="${NO_PROXY:-${no_proxy:-127.0.0.1,localhost,::1}}"
    export NO_PROXY="$no_proxy" no_proxy="$no_proxy"
    trust_proxy_ca
}

ensure_node() {
    have npm && return 0
    log "installing Node.js + npm (first run)..."
    as_root apt-get update -qq || die "apt-get update failed"
    as_root apt-get install -y -qq nodejs npm || die "could not install nodejs/npm"
    have npm || die "npm not found after installing nodejs/npm"
}

# Some vendor installers (Kiro CLI) unpack a .zip; the image ships zip but not
# unzip, so pull it in on demand.
ensure_unzip() {
    have unzip && return 0
    log "installing unzip (needed by the installer)..."
    as_root apt-get update -qq || die "apt-get update failed"
    as_root apt-get install -y -qq unzip || die "could not install unzip"
    have unzip || die "unzip not found after installing it"
}

# Hermes clones its own checkout, so it needs git; the image ships none.
ensure_git() {
    have git && return 0
    log "installing git (needed by the installer)..."
    as_root apt-get update -qq || die "apt-get update failed"
    as_root apt-get install -y -qq git || die "could not install git"
    have git || die "git not found after installing it"
}

ensure_npm() { # <npm package> <binary>
    have "$2" && return 0
    ensure_node
    log "installing $1 (npm install -g)..."
    as_root npm install -g --no-fund --no-audit "$1" || die "npm install $1 failed"
    have "$2" || die "$2 not on PATH after installing $1"
}

ensure_pip() { # <pypi package> <binary>
    have "$2" && return 0
    log "installing $1 (pip install --user)..."
    python3 -m pip install --user --quiet "$1" || die "pip install $1 failed"
    have "$2" || die "$2 not on PATH after installing $1"
}

# uv is the only Python toolchain here that can pick an interpreter version:
# the image's embedded CPython is 3.15, which is too new for SWE-agent, whose
# pinned dependencies ship no wheels for it. uv downloads a matching CPython
# on demand.
ensure_uv() {
    have uv && return 0
    log "installing uv (needed to pick a Python version)..."
    curl -fsSL "$UV_INSTALLER" | sh || die "uv install failed"
    have uv || die "uv not on PATH after installing it"
}

# SWE-agent ships its config/, tools/ and trajectories/ directories next to the
# package and asserts they exist at import time, but its wheel does not carry
# them -- a plain `uv tool install` of the git checkout installs a package that
# crashes on start. An editable install of a real clone keeps the repo layout,
# so the assertions hold.
ensure_sweagent() {
    have sweagent && return 0
    ensure_uv
    ensure_git
    local repo="$HOME/.swe-agent"
    if [ ! -d "$repo/.git" ]; then
        log "cloning SWE-agent into $repo..."
        rm -rf "$repo"
        git clone --depth 1 https://github.com/SWE-agent/SWE-agent.git "$repo" ||
            die "could not clone SWE-agent"
    fi
    log "installing SWE-agent (uv tool install --editable)..."
    uv tool install --force --editable --python 3.13 "$repo" ||
        die "uv tool install SWE-agent failed"
    have sweagent || die "sweagent not on PATH after installing it"
}

# The real Chrome binary, ignoring the ~/.local/bin wrapper installed below
# (which would otherwise make `command -v google-chrome` succeed on its own).
chrome_real_bin() {
    local p
    for p in /usr/bin/google-chrome-stable /usr/bin/google-chrome \
             /opt/google/chrome/google-chrome; do
        [ -x "$p" ] && { printf '%s\n' "$p"; return 0; }
    done
    return 1
}

# Chrome refuses to start its sandbox inside the container: Docker's default
# seccomp profile blocks the user namespaces its zygote needs, so it dies with
# "Failed to move to new namespace ... Operation not permitted" even as the
# unprivileged `pn` user. Put a wrapper named `google-chrome` in the user's bin
# that always passes --no-sandbox. ~/.local/bin comes first on PATH, so the
# shell, start-browser and g4f's own browser lookup all pick it up, while the
# real binary stays untouched.
link_chrome_no_sandbox() {
    local real bin
    real="$(chrome_real_bin)" || die "google-chrome not found after install"
    bin="$HOME/.local/bin/google-chrome"
    mkdir -p "$HOME/.local/bin" || die "cannot create $HOME/.local/bin"
    cat > "$bin" <<EOF
#!/bin/sh
# Generated by sandbox-cli: Chrome's sandbox cannot run in this container.
exec "$real" --no-sandbox "\$@"
EOF
    chmod 0755 "$bin" || die "cannot make $bin executable"
    ln -sf google-chrome "$HOME/.local/bin/google-chrome-stable"
    log "linked $bin -> $real (--no-sandbox)"
}

ensure_chrome() {
    if ! chrome_real_bin >/dev/null; then
        install_chrome_deb
    fi
    link_chrome_no_sandbox
}

install_chrome_deb() {
    local arch deb_url
    arch="$(dpkg --print-architecture || true)"
    case "$arch" in
        amd64) deb_url="${CHROME_DEB_BASE}_amd64.deb" ;;
        arm64) deb_url="${CHROME_DEB_BASE}_arm64.deb" ;;
        aarch64) deb_url="${CHROME_DEB_BASE}_arm64.deb" ;;
        *) deb_url="${CHROME_DEB_BASE}_amd64.deb" ;;
    esac
    log "installing Google Chrome (apt install of ${deb_url##*/}) on first run..."
    local tmp
    tmp="$(mktemp --suffix=.deb)" || die "cannot create temp file for Chrome .deb"
    curl -fsSL -o "$tmp" "$deb_url" || die "failed to download Google Chrome .deb"
    as_root apt-get -qq update || die "apt-get update failed"
    as_root apt-get install -y -qq "$tmp" || die "failed to install Google Chrome .deb"
    rm -f "$tmp"
    chrome_real_bin >/dev/null || die "google-chrome not found after installing the .deb"
}

ensure_lightpanda() {
    have g4f-go || die "g4f-go is not installed"
    g4f-go browser path >/dev/null 2>&1 && return 0
    log "installing Lightpanda (g4f-go browser install)..."
    g4f-go browser install || die "Lightpanda install failed"
    g4f-go browser path >/dev/null 2>&1 || die "Lightpanda not installed after g4f-go browser install"
}

# Remote Desktop is a bundled g4f-go module (`g4f-go -m remote_desktop`):
# g4f-go extracts it into its own runtime and pip-installs the extra packages
# it needs there on first use -- qrcode (the pairing QR SVG) and pynput
# (pointer and keyboard injection). Installing them here up front makes a
# failure visible in this terminal instead of leaving a server that silently
# has no QR code and no input control. `python3` is the embedded interpreter,
# so the probe looks at the same site-packages g4f-go does (PYTHONNOUSERSITE
# mirrors its runtime env).
#
# pynput's wheel metadata pulls in `evdev` on Linux, a C extension that needs
# kernel headers and a compiler the slim image does not carry, so a plain
# `pip install pynput` fails to build it. The sandbox session is X11, where
# pynput talks to the server through python-xlib instead of evdev, so pynput is
# installed without its dependencies and python-xlib is pulled explicitly.
ensure_remote_desktop() {
    have g4f-go || die "g4f-go is not installed"
    if have python3 && PYTHONNOUSERSITE=1 python3 -c \
        'import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in ("qrcode", "pynput", "Xlib")) else 1)' \
        2>/dev/null; then
        return 0
    fi
    log "installing Remote Desktop dependencies (qrcode, python-xlib, pynput)..."
    g4f-go install qrcode python-xlib || die "could not install qrcode/python-xlib"
    g4f-go install --no-deps pynput || die "could not install pynput"
}

ensure_script() { # <install script url> <binary> <label>
    have "$2" && return 0
    log "installing $3..."
    curl -fsSL "$1" | bash || die "$3 install script failed"
    have "$2" || die "$2 not on PATH after the $3 install"
}

# Roomote's installer is a host installer: it runs as root, installs Docker if
# it is missing, writes /opt/roomote and starts a Compose stack. The sandbox
# image has no init system, so the systemd unit it writes is inert here and the
# stack is started directly instead. `--tls-mode internal` keeps Caddy on a
# self-signed certificate, so the install needs no public DNS or ACME round
# trip; the printed /setup link is reachable through the sandbox's own URLs.
ensure_roomote() {
    have roomote && return 0
    log "installing Roomote (self-hosted coding agent)..."
    curl -fsSL "$ROOMOTE_INSTALLER" | as_root bash -s -- --tls-mode internal \
        || die "Roomote install script failed"
    have roomote || die "roomote not on PATH after the Roomote install"
}

# Hermes installs into ~/.hermes/hermes-agent/.hermes/bin and only adds
# ~/.local/bin to the shell rc, so link the binary there for the next run.
ensure_hermes() {
    have hermes && return 0
    ensure_git
    log "installing Hermes Agent (Nous Research)..."
    curl -fsSL "$HERMES_INSTALLER" | bash -s -- --non-interactive \
        || die "Hermes Agent install script failed"
    local bin="$HOME/.hermes/hermes-agent/.hermes/bin/hermes"
    if [ -x "$bin" ]; then
        mkdir -p "$HOME/.local/bin"
        ln -sf "$bin" "$HOME/.local/bin/hermes"
    fi
    have hermes || die "hermes not on PATH after the Hermes Agent install"
}

# Ollama's installer needs a real init system or falls back to a manual start;
# it also wants `zstd` to unpack its payload, which the slim image lacks.
ensure_ollama() {
    have ollama && return 0
    have zstd || {
        log "installing zstd (needed by the Ollama installer)..."
        as_root apt-get update -qq || die "apt-get update failed"
        as_root apt-get install -y -qq zstd || die "could not install zstd"
    }
    log "installing Ollama..."
    curl -fsSL "$OLLAMA_INSTALLER" | sh || die "Ollama install script failed"
    have ollama || die "ollama not on PATH after the install"
}

# LM Studio's installer drops the `lms` CLI into ~/.lmstudio/bin (already on
# PATH above) and, on a machine without a desktop session, only the daemon.
ensure_lmstudio() {
    have lms && return 0
    log "installing LM Studio (lms CLI + daemon)..."
    curl -fsSL "$LMSTUDIO_INSTALLER" | sh || die "LM Studio install script failed"
    have lms || die "lms not on PATH after the LM Studio install"
}

# llama.cpp ships prebuilt tarballs per release build; the tag is resolved from
# the GitHub API so the entry does not pin a build that later disappears.
ensure_llamacpp() {
    have llama-server && return 0
    local arch tag url dir
    case "$(dpkg --print-architecture 2>/dev/null || echo amd64)" in
        arm64|aarch64) arch="arm64" ;;
        *) arch="x64" ;;
    esac
    tag="$(curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1" \
        | grep -m1 '"tag_name"' | cut -d'"' -f4)"
    [ -n "$tag" ] || die "could not resolve the latest llama.cpp release"
    url="$LLAMACPP_RELEASE_BASE/$tag/llama-$tag-bin-ubuntu-$arch.tar.gz"
    log "installing llama.cpp ($tag, $arch)..."
    dir="$HOME/.llama.cpp"
    rm -rf "$dir"
    mkdir -p "$dir"
    curl -fsSL "$url" | tar -xz -C "$dir" --strip-components=1 \
        || die "could not download llama.cpp ($url)"
    mkdir -p "$HOME/.local/bin"
    ln -sf "$dir/llama-server" "$HOME/.local/bin/llama-server"
    ln -sf "$dir/llama-cli" "$HOME/.local/bin/llama-cli"
    have llama-server || die "llama-server not on PATH after the install"
}

# Jan is an Electron desktop app shipped as an AppImage; the tag is resolved
# from the GitHub API and the AppImage unpacked into ~/.local (no FUSE in the
# container, so it is extracted rather than mounted).
ensure_jan() {
    have jan && return 0
    local tag url dir
    tag="$(curl -fsSL "https://api.github.com/repos/janhq/jan/releases/latest" \
        | grep -m1 '"tag_name"' | cut -d'"' -f4)"
    [ -n "$tag" ] || die "could not resolve the latest Jan release"
    url="$JAN_RELEASE_BASE/$tag/Jan_${tag#v}_amd64.AppImage"
    log "installing Jan ($tag)..."
    dir="$HOME/.local/opt/jan"
    rm -rf "$dir"
    mkdir -p "$dir"
    curl -fsSL -o "$dir/jan.AppImage" "$url" || die "could not download Jan ($url)"
    chmod +x "$dir/jan.AppImage"
    ( cd "$dir" && ./jan.AppImage --appimage-extract >/dev/null 2>&1 ) \
        || die "could not unpack the Jan AppImage"
    mkdir -p "$HOME/.local/bin"
    ln -sf "$dir/squashfs-root/AppRun" "$HOME/.local/bin/jan"
    have jan || die "jan not on PATH after the install"
}

# Electron desktop apps (LobeChat, Msty Studio) link against NSS/NSPR and
# ALSA, which the slim image does not carry. Chrome's .deb pulls the same
# libraries in, but these entries never install Chrome, so they are installed
# here on first click.
ensure_electron_libs() {
    have ldconfig || return 0
    if ldconfig -p 2>/dev/null | grep -q "libnss3.so" \
        && ldconfig -p 2>/dev/null | grep -q "libasound.so.2"; then
        return 0
    fi
    log "installing the Electron runtime libraries (NSS, ALSA)..."
    as_root apt-get update -qq || die "apt-get update failed"
    as_root apt-get install -y -qq libnss3 libnspr4 libasound2t64 \
        || die "could not install the Electron runtime libraries"
}

# LobeChat (lobehub) ships an Electron desktop app as an AppImage. Like Jan it
# is unpacked rather than mounted (no FUSE in the container) and the AppRun is
# symlinked onto PATH. The release only carries an x64 AppImage.
ensure_lobechat() {
    have lobechat && return 0
    local tag url dir
    case "$(dpkg --print-architecture 2>/dev/null || echo amd64)" in
        amd64|x86_64) ;;
        *) die "LobeChat publishes no Linux AppImage for $(dpkg --print-architecture); only x64 is available" ;;
    esac
    ensure_electron_libs
    tag="$(curl -fsSL "https://api.github.com/repos/lobehub/lobehub/releases/latest" \
        | grep -m1 '"tag_name"' | cut -d'"' -f4)"
    [ -n "$tag" ] || die "could not resolve the latest LobeChat release"
    url="$LOBECHAT_RELEASE_BASE/$tag/LobeHub-${tag#v}.AppImage"
    log "installing LobeChat ($tag)..."
    dir="$HOME/.local/opt/lobechat"
    rm -rf "$dir"
    mkdir -p "$dir"
    curl -fsSL -o "$dir/lobechat.AppImage" "$url" \
        || die "could not download LobeChat ($url)"
    chmod +x "$dir/lobechat.AppImage"
    ( cd "$dir" && ./lobechat.AppImage --appimage-extract >/dev/null 2>&1 ) \
        || die "could not unpack the LobeChat AppImage"
    mkdir -p "$HOME/.local/bin"
    ln -sf "$dir/squashfs-root/AppRun" "$HOME/.local/bin/lobechat"
    have lobechat || die "lobechat not on PATH after the install"
}

# Msty Studio is the current Msty desktop app (the old msty.app download URLs
# are dead). It serves a stable "latest" AppImage URL, so no release lookup is
# needed; x64 only, like LobeChat.
ensure_msty() {
    have msty && return 0
    local dir
    case "$(dpkg --print-architecture 2>/dev/null || echo amd64)" in
        amd64|x86_64) ;;
        *) die "Msty Studio publishes no Linux AppImage for $(dpkg --print-architecture); only x64 is available" ;;
    esac
    ensure_electron_libs
    log "installing Msty Studio..."
    dir="$HOME/.local/opt/msty"
    rm -rf "$dir"
    mkdir -p "$dir"
    curl -fsSL -o "$dir/msty.AppImage" "$MSTY_APPIMAGE_URL" \
        || die "could not download Msty Studio ($MSTY_APPIMAGE_URL)"
    chmod +x "$dir/msty.AppImage"
    ( cd "$dir" && ./msty.AppImage --appimage-extract >/dev/null 2>&1 ) \
        || die "could not unpack the Msty Studio AppImage"
    mkdir -p "$HOME/.local/bin"
    # Msty's AppRun cannot be used directly: its APPDIR detection walks up the
    # tree looking for a file named after the first argument, so passing any
    # argument resolves APPDIR to an empty string and it then tries to exec
    # "/mstystudio". It also lacks the --no-sandbox fail-safe that LobeChat's
    # AppRun has, which is needed where unprivileged user namespaces are
    # unavailable (containers, hardened hosts).
    cat > "$HOME/.local/bin/msty" <<'MSTY_WRAPPER'
#!/bin/bash
appdir="$HOME/.local/opt/msty/squashfs-root"
export APPDIR="$appdir"
no_sandbox=()
if ! unshare -Ur true 2>/dev/null; then
    no_sandbox=(--no-sandbox)
fi
exec "$appdir/AppRun" "${no_sandbox[@]}" "$@"
MSTY_WRAPPER
    chmod +x "$HOME/.local/bin/msty"
    have msty || die "msty not on PATH after the install"
}

# LibreChat is not packaged: it is a Node monorepo that needs MongoDB and a
# full frontend build. The checkout lives in ~/.librechat, the database in
# ~/.librechat-data (started by ensure_librechat, kept running by run_tool).
LIBRECHAT_DIR_NAME=".librechat"
LIBRECHAT_DATA_DIR_NAME=".librechat-data"
LIBRECHAT_PORT="3080"

# MongoDB publishes no trixie build of mongodb-org-server, so the server is
# taken from the bookworm repository (the .deb is self-contained and runs on
# trixie) or, on arm64, from the ubuntu2404 tarball. The mongosh shell comes
# from the same repository and is only used to verify the server is up.
ensure_mongodb() {
    have mongod && return 0
    local arch dist
    case "$(dpkg --print-architecture 2>/dev/null || echo amd64)" in
        arm64|aarch64) arch="arm64" ;;
        *) arch="amd64" ;;
    esac
    if [ "$arch" = "amd64" ]; then
        log "installing MongoDB ($MONGODB_VERSION, apt)..."
        as_root mkdir -p /usr/share/keyrings
        # apt (>= 2.4, trixie ships 3.x) reads ASCII-armored keys directly, so
        # the .asc is used as-is; the image carries no gpg to dearmor it.
        curl -fsSL -o /tmp/mongodb-server-8.0.asc "$MONGODB_GPG_URL" \
            || die "could not download the MongoDB signing key"
        as_root install -m 0644 /tmp/mongodb-server-8.0.asc \
            /usr/share/keyrings/mongodb-server-8.0.asc \
            || die "could not install the MongoDB signing key"
        printf 'deb [ signed-by=/usr/share/keyrings/mongodb-server-8.0.asc ] %s %s main\n' \
            "$MONGODB_APT_BASE" "$MONGODB_APT_DIST" \
            | as_root tee /etc/apt/sources.list.d/mongodb-org-8.0.list >/dev/null
        as_root apt-get update -qq || die "apt-get update failed"
        as_root apt-get install -y -qq mongodb-org-server mongodb-mongosh \
            || die "could not install mongodb-org-server"
    else
        # fastdl serves no aarch64 debian12 tarball, so arm64 uses the
        # ubuntu2404 build; it is a static server binary and runs on trixie.
        log "installing MongoDB ($MONGODB_VERSION, tarball)..."
        dist="$HOME/.local/opt/mongodb"
        rm -rf "$dist"
        mkdir -p "$dist"
        curl -fsSL "$MONGODB_TARBALL_BASE/mongodb-linux-$arch-ubuntu2404-$MONGODB_VERSION.tgz" \
            | tar -xz -C "$dist" --strip-components=1 \
            || die "could not download the MongoDB tarball"
        mkdir -p "$HOME/.local/bin"
        ln -sf "$dist/bin/mongod" "$HOME/.local/bin/mongod"
    fi
    have mongod || die "mongod not on PATH after the install"
}

# Start the bundled MongoDB on 127.0.0.1:27017 if nothing answers there yet.
# The data directory lives outside the checkout so a re-clone keeps the data.
start_mongodb() {
    local data="$HOME/$LIBRECHAT_DATA_DIR_NAME"
    if tcp_open 27017; then
        log "MongoDB is already running on 127.0.0.1:27017"
        return 0
    fi
    mkdir -p "$data"
    log "starting MongoDB on 127.0.0.1:27017 (data in $data)"
    nohup setsid mongod --dbpath "$data" --bind_ip 127.0.0.1 --port 27017 \
        >/tmp/mongod.log 2>&1 &
    local i
    for i in $(seq 1 30); do
        tcp_open 27017 && return 0
        sleep 1
    done
    die "MongoDB did not start; see /tmp/mongod.log"
}

# LibreChat's .env: keep the upstream defaults and only pin what the sandbox
# needs (the local MongoDB, the port, and the g4f endpoint when a model was
# chosen). CREDS_KEY/CREDS_IV are left blank on purpose -- LibreChat generates
# them into .env.temp on first start.
write_librechat_env() {
    local env="$HOME/$LIBRECHAT_DIR_NAME/.env"
    [ -f "$env" ] || cp "$HOME/$LIBRECHAT_DIR_NAME/.env.example" "$env" \
        || die "could not create the LibreChat .env"
    set_env_var "$env" MONGO_URI "mongodb://127.0.0.1:27017/LibreChat"
    set_env_var "$env" PORT "$LIBRECHAT_PORT"
    set_env_var "$env" HOST "127.0.0.1"
    # The sandbox runs exactly one LibreChat process, so the scheduler's
    # replica check can be asserted instead of leaving its write routes off.
    set_env_var "$env" SCHEDULES_SINGLE_PROCESS "true"
    if [ "$SANDBOX_MODEL_SET" = 1 ]; then
        set_env_var "$env" G4F_API_BASE_URL "$G4F_API_BASE_URL"
        set_env_var "$env" G4F_API_KEY "$G4F_API_KEY"
        set_env_var "$env" G4F_MODEL "$G4F_PROVIDER_MODEL"
    fi
}

# LibreChat reads custom OpenAI-compatible endpoints from librechat.yaml. The
# g4f endpoint is written there (and pointed at the sandbox's g4f API) only
# when a model was chosen in Sandbox Settings; otherwise the upstream example
# is copied so the file LibreChat expects exists, and the user's own
# configuration is left alone.
write_librechat_yaml() {
    local dir yaml
    dir="$HOME/$LIBRECHAT_DIR_NAME"
    yaml="$dir/librechat.yaml"
    if [ "$SANDBOX_MODEL_SET" != 1 ]; then
        [ -f "$yaml" ] || cp "$dir/librechat.example.yaml" "$yaml" \
            || die "could not create the LibreChat librechat.yaml"
        return 0
    fi
    cat >"$yaml" <<EOF
# Written by sandbox-cli from the Sandbox Settings model/provider.
version: 1.2.8
cache: true
endpoints:
  custom:
    - name: "g4f"
      apiKey: "${G4F_API_KEY}"
      baseURL: "${G4F_API_BASE_URL}"
      models:
        default: ["${G4F_PROVIDER_MODEL}"]
        fetch: false
      titleConvo: true
      titleModel: "${G4F_PROVIDER_MODEL}"
      modelDisplayLabel: "g4f"
EOF
}

ensure_librechat() {
    ensure_node
    ensure_git
    ensure_mongodb
    local dir="$HOME/$LIBRECHAT_DIR_NAME"
    if [ ! -d "$dir/.git" ]; then
        log "cloning LibreChat into $dir..."
        rm -rf "$dir"
        git clone --depth 1 "$LIBRECHAT_REPO" "$dir" \
            || die "could not clone LibreChat"
    fi
    if [ ! -d "$dir/node_modules" ] || [ ! -d "$dir/client/dist" ]; then
        # `npm run reinstall` builds the workspace packages and the frontend;
        # it is the documented first-run step and needs a large heap.
        log "building LibreChat (npm run reinstall, this takes a while)..."
        ( cd "$dir" && NODE_OPTIONS="--max-old-space-size=8192" npm run reinstall ) \
            || die "the LibreChat build failed"
    fi
    have mongod || die "mongod not on PATH after the install"
}

# SillyTavern is a Node web app with no upstream package: the checkout lives
# in ~/.sillytavern, `npm install` fetches the dependencies. If no config.yaml
# exists yet, the checkout's default/config.yaml is copied to the repo root --
# the path server.js reads (./config.yaml; the config/ directory also seen in
# older releases is not used) -- and the port there is moved off SillyTavern's
# default 8000 so a running Open WebUI Computer (also 8000) does not collide
# with it, and the startup browser launch is turned off: the launcher terminal
# is the console and nothing guarantees a desktop browser is installed.
ensure_sillytavern() {
    ensure_node
    ensure_git
    local dir="$HOME/$SILLYTAVERN_DIR_NAME" config
    if [ ! -d "$dir/.git" ]; then
        log "cloning SillyTavern into $dir..."
        rm -rf "$dir"
        git clone --depth 1 "$SILLYTAVERN_REPO" "$dir" \
            || die "could not clone SillyTavern"
    fi
    if [ ! -d "$dir/node_modules" ]; then
        log "installing SillyTavern's dependencies (npm install)..."
        ( cd "$dir" && npm install --no-fund --no-audit ) \
            || die "the SillyTavern npm install failed"
    fi
    config="$dir/config.yaml"
    if [ ! -f "$config" ]; then
        cp "$dir/default/config.yaml" "$config" \
            || die "could not create the SillyTavern config.yaml"
    fi
    if grep -q '^port: 8000$' "$config"; then
        sed -i "s/^port: 8000$/port: $SILLYTAVERN_PORT/" "$config" \
            || die "could not set the SillyTavern port"
    fi
    sed -i '/^browserLaunch:/,/^[a-zA-Z]/{s/^  enabled: true/  enabled: false/}' \
        "$config" || die "could not disable SillyTavern's browser launch"
}

# Open WebUI Computer ("cptr") is pip-installable: one local server that
# serves files, terminal, git, editor and AI to any browser on port 8000 --
# the machine it runs on, not a copy of it.
ensure_cptr() {
    ensure_pip cptr cptr
}

# code-server (VS Code in the browser) ships its own vendor installer. Explicit
# --method standalone skips the deb/package-manager path (and its systemctl
# step): the release tarball is unarchived into $prefix/lib/code-server-* and
# symlinked into $prefix/bin, which ~/.local/bin already covers on PATH; the
# fallback link below catches an installer revision without the symlink.
ensure_code_server() {
    have code-server && return 0
    log "installing code-server (vendor install script, ~/.local)..."
    curl -fsSL "$CODE_SERVER_INSTALLER" | \
        bash -s -- --method standalone --prefix "$HOME/.local" \
        || die "the code-server install script failed"
    if ! have code-server; then
        local candidate found=""
        for candidate in "$HOME"/.local/lib/code-server-*/bin/code-server; do
            [ -x "$candidate" ] && found="$candidate" && break
        done
        [ -n "$found" ] || die "code-server not on PATH after the install"
        mkdir -p "$HOME/.local/bin"
        ln -sf "$found" "$HOME/.local/bin/code-server" \
            || die "cannot link code-server"
    fi
    have code-server || die "code-server not on PATH after the install"
}

# mitmproxy is the "Install & Run Proxy Server" entry: a debugging HTTP(S)
# proxy installed with pip --user. The console binary is `mitmproxy` (the other
# two, mitmdump/mitmweb, come with the same package).
ensure_mitmproxy() {
    have mitmproxy && return 0
    ensure_pip mitmproxy mitmproxy
}

# The "Install & Run Tunnel Client" entry is backed by Cloudflare's
# cloudflared. The static binary is architecture-specific, so append the dpkg
# arch (amd64/arm64) to the release prefix and drop it in ~/.local/bin, which
# is already first on PATH. `cloudflared tunnel --url <target>` then opens a
# public https://<random>.trycloudflare.com URL with no account or config.
ensure_cloudflared() {
    have cloudflared && return 0
    local arch
    case "$(dpkg --print-architecture 2>/dev/null || echo amd64)" in
        arm64|aarch64) arch="arm64" ;;
        *) arch="amd64" ;;
    esac
    local url="$CLOUDFLARED_RELEASE_BASE/cloudflared-linux-$arch"
    log "installing cloudflared ($arch, the tunnel client)..."
    mkdir -p "$HOME/.local/bin"
    curl -fsSL -o "$HOME/.local/bin/cloudflared" "$url" \
        || die "could not download cloudflared ($url)"
    chmod 0755 "$HOME/.local/bin/cloudflared" \
        || die "cannot make cloudflared executable"
    have cloudflared || die "cloudflared not on PATH after the install"
}

configure_g4f() {
    local base_url api_key port
    api_key="${G4F_API_KEY:-}"
    base_url="${G4F_API_BASE_URL:-}"
    if [ -z "$base_url" ]; then
        for port in 8080 9090; do
            if curl -fsS --connect-timeout 1 --max-time 2 \
                -H "Authorization: Bearer $api_key" \
                "http://127.0.0.1:$port/v1" >/dev/null 2>&1; then
                base_url="http://127.0.0.1:$port/v1"
                break
            fi
        done
        if [ -z "$base_url" ]; then
            base_url="http://127.0.0.1:8080/v1"
            log "g4f API was not detected; using $base_url (set G4F_API_BASE_URL to override)"
        fi
    fi
    base_url="${base_url%/}"
    export G4F_API_BASE_URL="$base_url"
    export G4F_API_KEY="$api_key"
    # Model and provider come from the settings file (menu → Sandbox Settings);
    # G4F_SANDBOX_MODEL / G4F_SANDBOX_PROVIDER override it. Only a model that
    # was actually chosen is exported: with none set, G4F_MODEL is left as the
    # caller had it and the launchers do not touch the tool's own provider.
    if [ -n "${G4F_SANDBOX_MODEL:-}" ]; then
        SANDBOX_MODEL_SET=1
    fi
    if [ "$SANDBOX_MODEL_SET" = 1 ]; then
        export G4F_MODEL="${G4F_SANDBOX_MODEL:-${SANDBOX_MODEL:-${G4F_MODEL:-}}}"
    fi
    # g4f resolves the provider from the model name itself ("Provider:model"),
    # so every launcher carries it in the model id. G4F_PROVIDER is only read by
    # the g4f server process, so exporting it here would not select anything.
    # The provider is optional: with none set, g4f picks from its own rotation.
    local provider="${G4F_SANDBOX_PROVIDER:-${SANDBOX_PROVIDER:-${G4F_PROVIDER:-}}}"
    G4F_PROVIDER_MODEL="${G4F_MODEL:-}"
    if [ -n "$provider" ] && [ "$SANDBOX_MODEL_SET" = 1 ]; then
        case "$G4F_PROVIDER_MODEL" in
            "$provider":*) ;;
            *) G4F_PROVIDER_MODEL="$provider:$G4F_PROVIDER_MODEL" ;;
        esac
    fi
    export G4F_PROVIDER_MODEL
}

configure_opencode_g4f() {
    # Only repoint OpenCode at the g4f provider when a model was chosen in the
    # menu. Otherwise leave the user's own OpenCode config/provider intact.
    [ "$SANDBOX_MODEL_SET" = 1 ] || return 0
    local previous config
    previous="${OPENCODE_CONFIG_CONTENT:-}"
    [ -n "$previous" ] || previous='{}'
    if ! config=$(jq -cn \
        --argjson previous "$previous" \
        --arg base_url "$G4F_API_BASE_URL" \
        --arg api_key "$G4F_API_KEY" \
        --arg model "$G4F_PROVIDER_MODEL" '
        def merge($left; $right):
            if ($left | type) == "object" and ($right | type) == "object" then
                reduce ($right | keys_unsorted[]) as $key ($left;
                    .[$key] = if has($key) then merge(.[$key]; $right[$key]) else $right[$key] end)
            else
                $right
            end;
        merge($previous; {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "g4f": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "g4f",
                    "options": {
                        "baseURL": $base_url,
                        "apiKey": $api_key
                    },
                    "models": {
                        ($model): {"name": $model}
                    }
                }
            },
            "model": ("g4f/" + $model)
        })' 2>/dev/null); then
        die "OPENCODE_CONFIG_CONTENT must contain valid JSON to enable the g4f provider"
    fi
    export OPENCODE_CONFIG_CONTENT="$config"
}

# OpenCode does its network traffic in a persistent background server
# (`opencode serve --service`) that the CLI only talks to over loopback. The
# server is started once and reused by every later invocation, and it does not
# inherit the CLI's environment, so the proxy variables exported for the CLI
# never reach the process that actually makes the requests: the CLI looks
# proxied while the server keeps connecting directly. OpenCode stores the
# server's environment in its own service config, so the proxy and CA variables
# are written there and the server is restarted when they change. With no proxy
# configured the variables are removed again, so turning the setting off takes
# effect as well.
configure_opencode_service() {
    have opencode || return 0
    local before after name value
    before=$(opencode service get 2>/dev/null) || return 0
    local -a vars=(
        HTTP_PROXY HTTPS_PROXY NO_PROXY
        NODE_EXTRA_CA_CERTS SSL_CERT_FILE REQUESTS_CA_BUNDLE CURL_CA_BUNDLE
        GIT_SSL_CAINFO AWS_CA_BUNDLE
    )
    if [ -n "${G4F_PROXY:-}" ]; then
        for name in "${vars[@]}"; do
            value=$(printenv "$name" 2>/dev/null) || continue
            opencode service set env "$name" "$value" >/dev/null 2>&1 || true
        done
    else
        for name in "${vars[@]}"; do
            opencode service unset env "$name" >/dev/null 2>&1 || true
        done
    fi
    after=$(opencode service get 2>/dev/null) || return 0
    [ "$before" = "$after" ] && return 0
    log "restarting the OpenCode background server to apply the proxy settings"
    opencode service restart >/dev/null 2>&1 \
        || log "could not restart the OpenCode background server; it may still use the previous proxy settings"
}

configure_copilot_g4f() {
    # As with OpenCode: only BYOK the g4f provider when a model was chosen.
    [ "$SANDBOX_MODEL_SET" = 1 ] || return 0
    export COPILOT_PROVIDER_TYPE=openai
    export COPILOT_PROVIDER_BASE_URL="$G4F_API_BASE_URL"
    export COPILOT_PROVIDER_API_KEY="$G4F_API_KEY"
    export COPILOT_PROVIDER_MODEL_ID="$G4F_PROVIDER_MODEL"
    export COPILOT_MODEL="$G4F_PROVIDER_MODEL"
}

# Hermes has no environment-variable provider override, so its config file is
# written through `hermes config set` (the supported path; the file is YAML and
# the image ships no YAML parser). `model.provider custom` + `model.base_url`
# point Hermes at the g4f API and `model.key_env` names the variable holding
# the bearer token. The provider travels in the model name ("Provider:model"),
# which g4f resolves itself; a request body `extra_body.provider` is ignored by
# the server, and the base URL stays at /v1 so model listing keeps working.
configure_hermes_g4f() {
    [ "$SANDBOX_MODEL_SET" = 1 ] || return 0
    log "pointing Hermes at the g4f API ($G4F_API_BASE_URL)"
    hermes config set model.provider custom >/dev/null \
        || die "could not configure Hermes (model.provider)"
    hermes config set model.base_url "$G4F_API_BASE_URL" >/dev/null \
        || die "could not configure Hermes (model.base_url)"
    hermes config set model.default "$G4F_PROVIDER_MODEL" >/dev/null \
        || die "could not configure Hermes (model.default)"
    hermes config set model.key_env G4F_API_KEY >/dev/null \
        || die "could not configure Hermes (model.key_env)"
    # Drop a provider left behind by an earlier version of this launcher, which
    # wrote it to extra_body where g4f never read it.
    hermes config unset model.extra_body.provider >/dev/null 2>&1 || true
}

# The addresses the sandbox serves through the g4f API: the two gateway
# proxies (Ollama, OmniRoute) and the Remote Desktop page. Printed by
# "sandbox-cli urls" and echoed by every install & start entry (see launch),
# so they are visible without digging through the docs.
public_urls() {
    printf '%s\n' "Public Urls:"
    printf '  %s\n' \
        "http://localhost:8080/ollama/v1/models" \
        "http://localhost:8080/omniroute/v1/models" \
        "http://localhost:8080/omniroute" \
        "http://localhost:8080/remote_desktop"
}

launch() { # <label> <command...>
    if [ "$INSTALL_ONLY" = 1 ]; then
        log "$1 is installed and ready"
        exit 0
    fi
    public_urls
    shift
    exec "$@"
}

run_tool() {
    # The saved HOME and PROXY apply to every entry, g4f-aware or not.
    apply_sandbox_home
    apply_sandbox_proxy
    case "$1" in
        remote-desktop)
            ensure_remote_desktop
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "Remote Desktop is installed and ready"
                exit 0
            fi
            # `g4f-go -m remote_desktop` runs the bundled server from its own
            # runtime and brings up the embedded STUN/TURN relay. The terminal
            # stays open as its console (Ctrl-C stops it); the banner prints
            # the host page and the phone URL. Open the host page on this
            # desktop, press Share, then scan the QR code from the phone.
            log "starting the Remote Desktop server (Ctrl-C to stop)"
            launch "Remote Desktop" g4f-go -m remote_desktop "${@:2}" ;;
        opencode)
            ensure_script "$OPENCODE_INSTALLER" opencode "OpenCode"
            if [ "$INSTALL_ONLY" = 0 ]; then
                configure_g4f
                configure_opencode_g4f
                configure_opencode_service
            fi
            launch OpenCode opencode "${@:2}" ;;
        kilocode)
            ensure_npm @kilocode/cli kilo
            launch "Kilo Code" kilo "${@:2}" ;;
        antigravity)
            ensure_script "$ANTIGRAVITY_INSTALLER" agy "Antigravity CLI"
            launch "Antigravity CLI" agy "${@:2}" ;;
        gemini-cli)
            ensure_npm @google/gemini-cli gemini
            launch "Gemini CLI" gemini "${@:2}" ;;
        qwencode)
            ensure_npm @qwen-code/qwen-code qwen
            launch "Qwen Code" qwen "${@:2}" ;;
        claude-code)
            ensure_npm @anthropic-ai/claude-code claude
            launch "Claude Code" claude "${@:2}" ;;
        codex)
            ensure_npm @openai/codex codex
            launch Codex codex "${@:2}" ;;
        copilot)
            ensure_npm @github/copilot copilot
            if [ "$INSTALL_ONLY" = 0 ]; then
                configure_g4f
                configure_copilot_g4f
            fi
            launch "GitHub Copilot" copilot "${@:2}" ;;
        g4f-go)
            have g4f-go || die "g4f-go is not installed"
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "g4f-go is installed and ready"
                exit 0
            fi
            if [ $# -gt 1 ]; then
                launch "g4f-go" g4f-go "${@:2}"
            fi
            log "g4f-go is preinstalled. Run 'g4f-go help' to see available commands."
            exec bash -i ;;
        cline)
            ensure_npm cline cline
            launch Cline cline "${@:2}" ;;
        amp)
            ensure_npm @ampcode/cli amp
            launch Amp amp "${@:2}" ;;
        goose)
            ensure_script "$GOOSE_INSTALLER" goose "Goose"
            launch Goose goose "${@:2}" ;;
        aider)
            ensure_pip aider-chat aider
            launch Aider aider "${@:2}" ;;
        cursor)
            ensure_script "$CURSOR_INSTALLER" cursor-agent "Cursor CLI"
            launch "Cursor CLI" cursor-agent "${@:2}" ;;
        chrome)
            ensure_chrome
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "Google Chrome is installed and ready"
                exit 0
            fi
            # Chrome is a GUI app: open it on the desktop via start-browser
            # (debug port + port forward, --once = no relaunch loop), detached,
            # so the launcher terminal can close while the browser keeps running.
            log "opening Google Chrome on the desktop"
            nohup setsid /usr/local/bin/start-browser chrome --once \
                >/tmp/chrome.log 2>&1 &
            # start-browser exits as soon as Chrome does, so a crash would
            # otherwise close this terminal before the reason is readable.
            # Wait for the CDP port (or the process) and report the log.
            local waited=0
            while [ "$waited" -lt 20 ]; do
                port_open 57011 && break
                pgrep -f "remote-debugging-port=57011" >/dev/null 2>&1 || break
                sleep 1
                waited=$((waited + 1))
            done
            if port_open 57011; then
                log "Chrome is running on the desktop (CDP on 127.0.0.1:57011)"
                exit 0
            fi
            printf '\nChrome did not start. Last lines of /tmp/chrome.log:\n\n' >&2
            tail -n 20 /tmp/chrome.log >&2 2>/dev/null || true
            die "Chrome failed to start (see /tmp/chrome.log)" ;;
        lightpanda)
            ensure_lightpanda
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "Lightpanda is installed and ready"
                exit 0
            fi
            # Not baked into the image: installed on first click, then served
            # here in the foreground. g4f-go browser serve prints the CDP
            # endpoint and keeps running, so the terminal stays open as the
            # server console (Ctrl-C stops it). g4f picks the endpoint up
            # automatically when it runs headless.
            log "starting the Lightpanda CDP server (Ctrl-C to stop)"
            launch Lightpanda g4f-go browser serve "${@:2}" ;;
        omniroute)
            ensure_npm omniroute omniroute
            launch OmniRoute omniroute "${@:2}" ;;
        proxy)
            ensure_mitmproxy
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "mitmproxy is installed and ready"
                exit 0
            fi
            # mitmproxy's console UI is the proxy itself; the terminal stays
            # open as its console (Ctrl-C stops it). Listen on 8081, not the
            # default 8080, so it does not collide with the g4f API. Point the
            # Sandbox Settings PROXY at http://127.0.0.1:8081 to route the menu
            # entries through it.
            #
            # confdir is pinned to PROXY_CA_DIR so the CA lands where
            # trust_proxy_ca looks for it; mitmproxy's default (~/.mitmproxy)
            # would leave the tools trusting a certificate that does not exist.
            log "starting mitmproxy on 127.0.0.1:8081 (Ctrl-C to stop)"
            launch "Proxy Server" mitmproxy \
                --set confdir="$PROXY_CA_DIR" \
                --listen-host 127.0.0.1 --listen-port 8081 "${@:2}" ;;
        tunnel)
            ensure_cloudflared
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "the tunnel client (cloudflared) is installed and ready"
                exit 0
            fi
            # One-click tunnel: expose a local port on a public HTTPS URL. The
            # target defaults to the g4f API (the docs' phone/GUI guide uses
            # it); override it with TUNNEL_TARGET or extra arguments, e.g.
            #   sandbox-cli tunnel --url http://127.0.0.1:3000
            local target="${TUNNEL_TARGET:-http://127.0.0.1:8080}"
            log "opening a public tunnel to $target (Ctrl-C to stop)"
            launch "Tunnel Client" cloudflared tunnel --url "$target" \
                --no-autoupdate "${@:2}" ;;
        ollama)
            ensure_ollama
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "Ollama is installed and ready"
                exit 0
            fi
            # `ollama serve` is the daemon; the terminal stays open as its
            # console (Ctrl-C stops it). The API is on 127.0.0.1:11434 and g4f
            # reaches it through the /ollama proxy route.
            log "starting the Ollama server (Ctrl-C to stop)"
            launch Ollama ollama serve "${@:2}" ;;
        llamacpp)
            ensure_llamacpp
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "llama.cpp is installed and ready"
                exit 0
            fi
            # llama-server exposes an OpenAI-compatible API on 8080 by default;
            # move it to 11435 so it does not collide with the g4f API.
            log "starting llama-server on http://127.0.0.1:11435 (Ctrl-C to stop)"
            launch "llama.cpp" llama-server --port 11435 "${@:2}" ;;
        lmstudio)
            ensure_lmstudio
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "LM Studio is installed and ready"
                exit 0
            fi
            log "starting the LM Studio daemon (Ctrl-C to stop)"
            launch "LM Studio" lms server start "${@:2}" ;;
        jan)
            ensure_jan
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "Jan is installed and ready"
                exit 0
            fi
            # Jan is a GUI app: open it on the desktop, detached, so the
            # launcher terminal can close while the app keeps running.
            log "opening Jan on the desktop"
            nohup setsid jan >/tmp/jan.log 2>&1 &
            exit 0 ;;
        librechat)
            ensure_librechat
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "LibreChat is installed and ready"
                exit 0
            fi
            # LibreChat is a web app: MongoDB first, then the Node server in
            # the foreground so the terminal stays open as its console.
            configure_g4f
            write_librechat_env
            write_librechat_yaml
            start_mongodb
            log "starting LibreChat on http://127.0.0.1:$LIBRECHAT_PORT (Ctrl-C to stop)"
            cd "$HOME/$LIBRECHAT_DIR_NAME" || die "LibreChat checkout is missing"
            launch LibreChat npm run backend "${@:2}" ;;
        lobechat)
            ensure_lobechat
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "LobeChat is installed and ready"
                exit 0
            fi
            # Electron GUI app: open it on the desktop, detached, so the
            # launcher terminal can close while the app keeps running.
            log "opening LobeChat on the desktop"
            nohup setsid lobechat >/tmp/lobechat.log 2>&1 &
            exit 0 ;;
        msty)
            ensure_msty
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "Msty Studio is installed and ready"
                exit 0
            fi
            log "opening Msty Studio on the desktop"
            nohup setsid msty >/tmp/msty.log 2>&1 &
            exit 0 ;;
        sillytavern)
            ensure_sillytavern
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "SillyTavern is installed and ready"
                exit 0
            fi
            # SillyTavern is a web app: the Node server runs in the foreground
            # so the terminal stays open as its console (Ctrl-C stops it).
            log "starting SillyTavern on http://127.0.0.1:$SILLYTAVERN_PORT (Ctrl-C to stop)"
            cd "$HOME/$SILLYTAVERN_DIR_NAME" || die "SillyTavern checkout is missing"
            launch SillyTavern node server.js "${@:2}" ;;
        cptr)
            ensure_cptr
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "Open WebUI Computer is installed and ready"
                exit 0
            fi
            # cptr serves the machine it runs on — files, terminal, git,
            # editor and AI — to any browser; the terminal stays open as its
            # console (Ctrl-C stops it).
            log "starting Open WebUI Computer on http://127.0.0.1:$CPTR_PORT (Ctrl-C to stop)"
            launch "Open WebUI Computer" cptr run "${@:2}" ;;
        code-server)
            ensure_code_server
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "code-server is installed and ready"
                exit 0
            fi
            # code-server serves VS Code Web; the terminal stays open as its
            # console (Ctrl-C stops it). Listen on 8602, not the default
            # 8080, so it does not collide with the g4f API. --auth none keeps
            # the local loopback session free of a login prompt.
            log "starting code-server on http://127.0.0.1:$CODE_SERVER_PORT (Ctrl-C to stop)"
            launch "VS Code Web" code-server --auth none \
                --bind-addr "127.0.0.1:$CODE_SERVER_PORT" "${@:2}" ;;
        openclaw)
            ensure_script "$OPENCLAW_INSTALLER" openclaw "OpenClaw"
            launch OpenClaw openclaw "${@:2}" ;;
        crush)
            ensure_npm @charmland/crush crush
            launch Crush crush "${@:2}" ;;
        continue-cli)
            ensure_npm @continuedev/cli cn
            launch "Continue CLI" cn "${@:2}" ;;
        grok)
            ensure_script "$GROK_INSTALLER" grok "Grok Build CLI"
            launch "Grok Build CLI" grok "${@:2}" ;;
        droid)
            ensure_script "$DROID_INSTALLER" droid "Factory Droid"
            launch "Factory Droid" droid "${@:2}" ;;
        openhands)
            ensure_script "$OPENHANDS_INSTALLER" openhands "OpenHands CLI"
            launch "OpenHands CLI" openhands "${@:2}" ;;
        gptme)
            ensure_pip gptme gptme
            launch gptme gptme "${@:2}" ;;
        plandex)
            ensure_script "$PLANDEX_INSTALLER" plandex "Plandex"
            launch Plandex plandex "${@:2}" ;;
        hermes)
            ensure_hermes
            if [ "$INSTALL_ONLY" = 0 ]; then
                configure_g4f
                configure_hermes_g4f
            fi
            launch "Hermes Agent" hermes "${@:2}" ;;
        kiro)
            ensure_unzip
            ensure_script "$KIRO_INSTALLER" kiro-cli "Kiro CLI"
            launch "Kiro CLI" kiro-cli "${@:2}" ;;
        kimi)
            ensure_script "$KIMI_INSTALLER" kimi "Kimi Code"
            launch "Kimi Code" kimi "${@:2}" ;;
        roomote)
            ensure_roomote
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "Roomote is installed and ready"
                exit 0
            fi
            # Roomote is a web app: the installer already started the Compose
            # stack, so print the first-admin setup link and keep the terminal
            # open as its console (`roomote logs` follows the services).
            log "Roomote setup link:"
            as_root roomote setup-url || true
            log "following the Roomote logs (Ctrl-C to stop)"
            as_root roomote logs "${@:2}" ;;
        auggie)
            ensure_npm @augmentcode/auggie auggie
            launch Auggie auggie "${@:2}" ;;
        qoder)
            ensure_script "$QODER_INSTALLER" qodercli "Qoder CLI"
            launch "Qoder CLI" qodercli "${@:2}" ;;
        iflow)
            ensure_npm @iflow-ai/iflow-cli iflow
            launch "iFlow CLI" iflow "${@:2}" ;;
        vibe)
            ensure_script "$VIBE_INSTALLER" vibe "Mistral Vibe"
            launch "Mistral Vibe" vibe "${@:2}" ;;
        sweagent)
            ensure_sweagent
            launch "SWE-agent" sweagent "${@:2}" ;;
        fabric)
            ensure_script "$FABRIC_INSTALLER" fabric "Fabric"
            launch Fabric fabric "${@:2}" ;;
        litellm)
            ensure_pip 'litellm[proxy]' litellm
            launch "LiteLLM gateway" litellm "${@:2}" ;;
        sgpt)
            ensure_pip shell-gpt sgpt
            launch ShellGPT sgpt "${@:2}" ;;
        llm)
            ensure_pip llm llm
            launch "LLM CLI" llm "${@:2}" ;;
        assistant-ui)
            ensure_node
            have npx || die "npx is not on PATH"
            if [ "$INSTALL_ONLY" = 1 ]; then
                log "assistant-ui's setup-agent is ready (runs through npx)"
                exit 0
            fi
            # setup-agent is a thin Statewire client, not an installer: it
            # prints the setup's instructions and current state, then the
            # agent drives it from this shell (stream, wait, plan, add-step,
            # step, done). It is meant for a coding agent, so with none
            # detected it offers to start one -- which is why this entry opens
            # a terminal instead of installing anything itself.
            log "starting the assistant-ui setup (Ctrl-C to stop)"
            launch "assistant-ui" npx --yes setup-agent \
                "$ASSISTANT_UI_SETUP_URL" "${@:2}" ;;
        *)
            die "unknown tool \"$1\" — run: sandbox-cli list" ;;
    esac
}

usage() {
    cat <<'EOF'
usage: sandbox-cli <tool> [args...]   install (first click) and run a desktop tool
       sandbox-cli install <tool>     install only, don't run it
       sandbox-cli settings           choose the g4f model, HOME, provider and proxy
       sandbox-cli reset              clear the saved settings, back to the defaults
       sandbox-cli list               list the tools behind the desktop menu
       sandbox-cli urls               print the public URLs served by the g4f API
       sandbox-cli help               this help

Installs go to /usr/local (npm -g via sudo), ~/.local (pip --user and the
vendor install scripts) or wherever the installer puts them; the CLI then
runs in this terminal. The HOME and PROXY saved in the sandbox settings apply
to every entry.
EOF
}

list_tools() {
    # Grouped like the desktop menu (see docker/sandbox-menu/README.md).
    printf '%s\n' "g4f"
    printf '  %-15s %s\n' \
        g4f-go          "g4f Go launcher CLI (preinstalled)" \
        remote-desktop  "share this sandbox screen with a phone (WebRTC)" \
        settings        "g4f model, HOME and provider for the launchers" \
        reset           "clear the saved settings, back to the defaults" \
        urls            "public URLs served by the g4f API"
    printf '%s\n' "Browsers"
    printf '  %-15s %s\n' \
        chrome      "Google Chrome browser" \
        lightpanda  "Lightpanda headless browser (CDP server)"
    printf '%s\n' "Coding Agents"
    printf '  %-15s %s\n' \
        opencode    "OpenCode coding agent" \
        kilocode    "Kilo Code terminal agent" \
        antigravity "Antigravity CLI by Google" \
        gemini-cli  "Google Gemini CLI" \
        qwencode    "Qwen Code CLI" \
        claude-code "Anthropic Claude Code" \
        codex       "OpenAI Codex CLI" \
        copilot     "GitHub Copilot CLI" \
        cline       "Cline terminal agent" \
        amp         "Amp by Sourcegraph" \
        goose       "Goose by Block" \
        aider       "Aider pair-programming CLI" \
        cursor      "Cursor CLI" \
        crush       "Crush agent by Charm" \
        continue-cli "Continue CLI agent" \
        grok        "xAI Grok Build CLI" \
        droid       "Factory Droid agent" \
        openhands   "OpenHands coding agent" \
        plandex     "Plandex coding agent" \
        hermes      "Hermes Agent by Nous Research" \
        kiro        "Kiro CLI by AWS" \
        kimi        "Kimi Code by Moonshot AI" \
        roomote     "Roomote self-hosted coding agent" \
        auggie      "Auggie by Augment Code" \
        qoder       "Qoder CLI" \
        iflow       "iFlow CLI" \
        vibe        "Mistral Vibe" \
        sweagent    "SWE-agent by Princeton"
    printf '%s\n' "Gateways"
    printf '  %-15s %s\n' \
        omniroute   "OmniRoute AI router/gateway" \
        litellm     "LiteLLM AI gateway" \
        proxy       "mitmproxy HTTP(S) proxy server" \
        tunnel      "Tunnel client (public HTTPS URL via cloudflared)"
    printf '%s\n' "Local AI"
    printf '  %-15s %s\n' \
        ollama      "Ollama local model server" \
        llamacpp    "llama.cpp llama-server" \
        lmstudio    "LM Studio (lms CLI + daemon)" \
        jan         "Jan desktop app"
    printf '%s\n' "Chat UIs"
    printf '  %-15s %s\n' \
        librechat   "LibreChat web UI (Node + MongoDB)" \
        lobechat    "LobeChat desktop app" \
        msty        "Msty Studio desktop app" \
        sillytavern "SillyTavern web UI for role-play chats" \
        cptr        "Open WebUI Computer (files + terminal + AI)"
    printf '%s\n' "Editors"
    printf '  %-15s %s\n' \
        code-server "VS Code Web (code-server)"
    printf '%s\n' "Assistants"
    printf '  %-15s %s\n' \
        openclaw    "OpenClaw CLI" \
        fabric      "Fabric AI patterns CLI" \
        sgpt        "ShellGPT shell assistant" \
        gptme       "gptme terminal agent" \
        llm         "LLM CLI by Simon Willison" \
        assistant-ui "assistant-ui setup (setup-agent)"
}

case "${1:-}" in
    ""|help|-h|--help)
        usage ;;
    list)
        list_tools ;;
    urls)
        public_urls
        hold_open ;;
    settings)
        shift
        run_settings "$@" ;;
    reset)
        shift
        run_reset "$@" ;;
    install)
        [ $# -ge 2 ] || die "usage: sandbox-cli install <tool>"
        INSTALL_ONLY=1
        run_tool "$2" ;;
    *)
        tool="$1"
        shift
        run_tool "$tool" "$@" ;;
esac
