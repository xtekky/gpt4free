#!/bin/bash

# Browser selection: chrome (default), chromium, brave, msedge
BROWSER="chrome"
DEBUG_PORT=57011
HEADLESS=0
for arg in "$@"; do
    case "$arg" in
        --headless) HEADLESS=1 ;;
        -h|--help)
            echo "Usage: $0 [chrome|chromium|brave|msedge] [--headless]"
            exit 0
            ;;
        -*) echo "Unknown argument: $arg" >&2; exit 1 ;;
        *) BROWSER="$arg" ;;
    esac
done

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
    --remote-debugging-port=$DEBUG_PORT
)

if [ "$HEADLESS" -eq 1 ]; then
    BROWSER_FLAGS+=(--headless=new)
fi

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
                "/mnt/c/Program Files/Chromium/Application/chrome.exe" \
                "$LOCALAPPDATA/Microsoft/WindowsApps/chromium.exe")"
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
                "/mnt/c/Program Files (x86)/BraveSoftware/Brave-Browser/Application/brave.exe" \
                "/c/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe" \
                "/mnt/c/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe" \
                "$LOCALAPPDATA/BraveSoftware/Brave-Browser/Application/brave.exe" \
                "$LOCALAPPDATA/Microsoft/WindowsApps/brave.exe")"
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
                "$LOCALAPPDATA/Microsoft/Edge/Application/msedge.exe" \
                "$LOCALAPPDATA/Microsoft/WindowsApps/msedge.exe")"
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
                "$LOCALAPPDATA/Google/Chrome/Application/chrome.exe" \
                "$LOCALAPPDATA/Microsoft/WindowsApps/chrome.exe")"
        fi
        ;;
esac

if [ -z "$BROWSER_BIN" ]; then
    echo "Error: No browser binary found for '$BROWSER'" >&2
    exit 1
fi

PROC_NAME="$(basename "$BROWSER_BIN")"
echo "Starting browser: $BROWSER_BIN"

# Check if the remote-debugging port is accepting connections.
port_open() {
    if command -v curl &>/dev/null; then
        curl -s --max-time 2 "http://127.0.0.1:$DEBUG_PORT/json/version" &>/dev/null
    elif command -v powershell.exe &>/dev/null; then
        powershell.exe -NoProfile -Command \
            "try { (New-Object Net.Sockets.TcpClient('127.0.0.1', $DEBUG_PORT)).Close(); exit 0 } catch { exit 1 }" &>/dev/null
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$DEBUG_PORT") &>/dev/null
    fi
}

# Expose the debug port on the network: 0.0.0.0:9223 -> 127.0.0.1:$DEBUG_PORT
start_port_forward() {
    if command -v socat &>/dev/null; then
        socat tcp-listen:9223,reuseaddr,bind=0.0.0.0,fork tcp:127.0.0.1:$DEBUG_PORT &
        echo "Forwarding 0.0.0.0:9223 -> 127.0.0.1:$DEBUG_PORT (socat)"
    elif command -v python &>/dev/null || command -v python3 &>/dev/null; then
        PYTHON_BIN="$(command -v python3 || command -v python)"
        "$PYTHON_BIN" - "$DEBUG_PORT" <<'EOF' &
import socket, sys, threading

DEBUG_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 57011
LISTEN_PORT = 9223

def pipe(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.close()
            except OSError:
                pass

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", LISTEN_PORT))
server.listen()
print(f"Forwarding 0.0.0.0:{LISTEN_PORT} -> 127.0.0.1:{DEBUG_PORT} (python)")
while True:
    client, _ = server.accept()
    remote = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        remote.connect(("127.0.0.1", DEBUG_PORT))
    except OSError:
        client.close()
        continue
    threading.Thread(target=pipe, args=(client, remote), daemon=True).start()
    threading.Thread(target=pipe, args=(remote, client), daemon=True).start()
EOF
    else
        echo "Warning: install socat or python to expose port 9223 on the network" >&2
        return
    fi
    echo "Note: allow inbound port 9223 in the Windows Firewall for external access."
}

# Clean up background forwarders when the script exits.
trap 'kill $(jobs -p) 2>/dev/null' EXIT

start_port_forward

if [[ "$BROWSER_BIN" == *.exe ]]; then
    # Windows: relaunch the browser whenever it is closed.
    while true; do
        rm -f ~/.g4f/cookies/.browser_is_open
        "$BROWSER_BIN" "${BROWSER_FLAGS[@]}" &
        BROWSER_PID=$!
        # Wait until the debugging port is reachable (browser fully started), max 30s.
        waited=0
        until port_open; do
            sleep 1
            waited=$((waited + 1))
            if [ "$waited" -ge 30 ]; then
                echo "Browser did not open the debug port, retrying..." >&2
                break
            fi
        done
        echo "Browser running (pid $BROWSER_PID). Close it to restart..."
        # Wait until the port closes again (browser shut down).
        while port_open; do sleep 2; done
        echo "Browser closed, restarting in 3 seconds... (Ctrl+C to stop)"
        sleep 3
    done
else
    # Linux: loop and relaunch if the browser exits
    while true; do
        rm -f ~/.g4f/cookies/.browser_is_open
        "$BROWSER_BIN" "${BROWSER_FLAGS[@]}"
        sleep 5
    done
fi