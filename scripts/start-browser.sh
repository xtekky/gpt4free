#!/bin/bash

# Browser selection: chrome (default), chromium, brave, msedge
BROWSER="${1:-chrome}"

BROWSER_FLAGS=(
    --remote-allow-origins=*
    --no-first-run
    --no-service-autorun
    --no-default-browser-check
    --homepage=about:blank
    --no-pings
    --password-store=basic
    --disable-infobars
    --disable-breakpad
    --disable-dev-shm-usage
    --disable-session-crashed-bubble
    --disable-search-engine-choice-screen
    --user-data-dir="$HOME/.config/g4f-nodriver"
    --disable-features=IsolateOrigins,site-per-process
    --remote-debugging-host=127.0.0.1
    --remote-debugging-port=57011
)

# Resolve a Windows path to the correct format for the current shell environment.
# Git Bash uses /c/... whereas WSL uses /mnt/c/...
win_path() {
    local p="$1"
    # Git Bash / MSYS2: OSTYPE contains "msys" or "cygwin"
    if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* ]]; then
        # Convert "C:\foo\bar" or "/mnt/c/foo/bar" -> "/c/foo/bar"
        p="${p//\\//}"                     # backslash -> slash
        p="${p/\/mnt\/c\//\/c\/}"          # /mnt/c/ -> /c/
        echo "$p"
    else
        # WSL / Linux: convert /c/ -> /mnt/c/
        p="${p//\\//}"
        p="${p/\/c\//\/mnt\/c\/}"
        echo "$p"
    fi
}

# Returns the first existing Windows path from a list of candidate paths.
find_win_bin() {
    for raw in "$@"; do
        local p
        p="$(win_path "$raw")"
        [ -f "$p" ] && echo "$p" && return
    done
}

case "$BROWSER" in
    chromium)
        for bin in chromium chromium-browser chromium-freeworld; do
            if command -v "$bin" &>/dev/null; then
                BROWSER_BIN="$bin"; break
            fi
        done
        if [ -z "$BROWSER_BIN" ]; then
            BROWSER_BIN="$(find_win_bin \
                "/c/Program Files/Chromium/Application/chrome.exe" \
                "/mnt/c/Program Files/Chromium/Application/chrome.exe")"
        fi
        ;;
    brave)
        for bin in brave-browser brave brave-browser-stable; do
            if command -v "$bin" &>/dev/null; then
                BROWSER_BIN="$bin"; break
            fi
        done
        if [ -z "$BROWSER_BIN" ]; then
            BROWSER_BIN="$(find_win_bin \
                "/c/Program Files (x86)/BraveSoftware/Brave-Browser/Application/brave.exe" \
                "/mnt/c/Program Files (x86)/BraveSoftware/Brave-Browser
                "/c/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe" \
                "/mnt/c/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe" \
                "$LOCALAPPDATA/BraveSoftware/Brave-Browser/Application/brave.exe")"
        fi
        ;;
    msedge)
        for bin in microsoft-edge msedge microsoft-edge-stable; do
            if command -v "$bin" &>/dev/null; then
                BROWSER_BIN="$bin"; break
            fi
        done
        if [ -z "$BROWSER_BIN" ]; then
            BROWSER_BIN="$(find_win_bin \
                "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
                "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
                "/c/Program Files/Microsoft/Edge/Application/msedge.exe" \
                "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe" \
                "$LOCALAPPDATA/Microsoft/Edge/Application/msedge.exe")"
        fi
        # Last resort: ask Windows itself
        if [ -z "$BROWSER_BIN" ] && command -v powershell.exe &>/dev/null; then
            BROWSER_BIN="$(powershell.exe -NoProfile -Command \
                "(Get-Command msedge -ErrorAction SilentlyContinue).Source" 2>/dev/null | tr -d '\r')"
        fi
        ;;
    chrome|*)
        for bin in google-chrome google-chrome-stable google-chrome-unstable; do
            if command -v "$bin" &>/dev/null; then
                BROWSER_BIN="$bin"; break
            fi
        done
        if [ -z "$BROWSER_BIN" ]; then
            BROWSER_BIN="$(find_win_bin \
                "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
                "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
                "/c/Program Files/Google/Chrome/Application/chrome.exe" \
                "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe" \
                "$LOCALAPPDATA/Google/Chrome/Application/chrome.exe")"
        fi
        ;;
esac

if [ -z "$BROWSER_BIN" ]; then
    echo "Error: No browser binary found for '$BROWSER'" >&2
    exit 1
fi

PROC_NAME="$(basename "$BROWSER_BIN")"
echo "Starting browser: $BROWSER_BIN"

if [[ "$BROWSER_BIN" == *.exe ]]; then
    # Windows: launch once and exit — the browser manages its own lifecycle
    rm -f ~/.g4f/cookies/.browser_is_open
    "$BROWSER_BIN" "${BROWSER_FLAGS[@]}"
else
    # Linux: loop and relaunch if the browser exits
    while true; do
        rm -f ~/.g4f/cookies/.browser_is_open
        "$BROWSER_BIN" "${BROWSER_FLAGS[@]}"
        sleep 5
    done
fi