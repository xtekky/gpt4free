#!/bin/sh
# Provision the Debian rootfs for gpt4free.
# Runs inside PRoot as fake root. The app binds the bundled g4f source at
# /g4f-src and these scripts at /g4f-scripts. Python itself is not installed
# here: g4f-go downloads its own CPython runtime and installs the bundled
# g4f source into it.
set -eu
export DEBIAN_FRONTEND=noninteractive
export PATH=/g4f-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Installs upgraded from the old venv image keep /etc/profile.d/g4f.sh, which
# exported G4F_PYTHON=/opt/g4f-venv/bin/python. g4f-go must manage its own
# runtime here, otherwise pip installs into a venv that may not exist anymore.
unset G4F_PYTHON VIRTUAL_ENV PYTHONPATH PYTHONHOME
# g4f-go provisions its embedded CPython runtime under G4F_CONFIG_DIR (or
# $HOME/.g4f when unset). Pin it to /root/.g4f so the python/pip/python3
# symlinks below point at a stable location inside the container.
export G4F_CONFIG_DIR=/root/.g4f

log() { echo "[g4f-provision] $*"; }
started=$(date +%s)
log "started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# dpkg's base-files conffile can restore the image's resolver while packages
# are being configured. Write it again before every network step so a brief
# network/DNS interruption does not turn into a fatal apt error (exit 100).
write_resolv() {
    rm -f /etc/resolv.conf
    cat > /etc/resolv.conf <<'EOF'
nameserver 1.1.1.1
nameserver 8.8.8.8
options timeout:5 attempts:3
EOF
}

# A previous session may have been killed while dpkg was running. Android can
# suspend or kill long-running child processes even with a foreground service,
# so always clear stale lock files and repair both dpkg and apt state before
# installing anything. The app prevents two provisioning sessions at once.
clear_apt_locks() {
    rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/cache/apt/archives/lock
}

log "repairing dpkg state"
clear_apt_locks
write_resolv
dpkg --configure -a || true
write_resolv
apt-get -f install -y || true
write_resolv

log "updating apt indexes"
updated=0
for attempt in 1 2 3; do
    write_resolv
    if apt-get update -qq; then
        updated=1
        break
    fi
    log "apt update failed (attempt $attempt/3), retrying"
    sleep 5
done
if [ "$updated" != 1 ]; then
    log "apt update failed after 3 attempts; check the phone's connection"
    exit 1
fi

log "installing base packages"
installed=0
for attempt in 1 2; do
    write_resolv
    # g4f-go downloads CPython and pip packages over HTTPS; it needs the
    # system CA bundle. sudo and git are part of the interactive container.
    # No Python, pip or venv is installed in the rootfs.
    if apt-get install -y --no-install-recommends ca-certificates sudo git; then
        installed=1
        break
    fi
    log "package install failed (attempt $attempt/2), repairing and retrying"
    dpkg --configure -a || true
    write_resolv
    apt-get update -qq || true
    sleep 5
done
if [ "$installed" != 1 ]; then
    log "base package installation failed after 2 attempts"
    exit 1
fi
write_resolv

log "installing gpt4free through g4f-go"
log "the first run downloads a ~60 MB CPython runtime; this can take a few minutes"
# g4f-go downloads the pinned CPython runtime, then installs the app's
# bundled source as an editable package. The editable install makes g4f-go
# treat it as builtin, so later `api`/`gui` runs never replace it with PyPI's
# latest release. Only the API extras plus the small Flask GUI dependencies
# are installed; the full "all" extra pulls desktop/printing packages that do
# not work under PRoot.
installed_g4f=0
for attempt in 1 2; do
    if /g4f-bin/g4f-go install -e "/g4f-src[api]" flask werkzeug asgiref pillow; then
        installed_g4f=1
        break
    fi
    log "g4f-go install failed (attempt $attempt/2); retrying"
    sleep 10
done
if [ "$installed_g4f" != 1 ]; then
    log "g4f-go could not install gpt4free; check the phone's connection"
    log "a partial runtime download resumes automatically on the next attempt"
    log "retry later with: /g4f-bin/g4f-go install -e '/g4f-src[api]' flask werkzeug asgiref pillow"
    exit 1
fi
/g4f-bin/g4f-go status || true

# Link g4f-go's embedded CPython as python, pip and python3 so the container
# has a working python interpreter without a system python or venv. Use
# G4F_PYTHON_ONLY to discover g4f-go's actual interpreter path (it downloads
# the runtime on first use if needed).
log "linking g4f-go python as python/pip/python3"
G4F_PYTHON_BIN="$(G4F_PYTHON_ONLY=1 /g4f-bin/g4f-go __g4f_python_path__ 2>/dev/null || true)"
if [ -z "$G4F_PYTHON_BIN" ] || [ ! -x "$G4F_PYTHON_BIN" ]; then
    log "g4f-go did not report a python binary; using default runtime path"
    G4F_PYTHON_BIN="${G4F_CONFIG_DIR}/python-embed/python-home/bin/python3.15"
fi
G4F_PYTHON_DIR="$(dirname "$G4F_PYTHON_BIN")"
ln -sf "$G4F_PYTHON_BIN" /usr/local/bin/python3
ln -sf "$G4F_PYTHON_BIN" /usr/local/bin/python
if [ -x "$G4F_PYTHON_DIR/pip3" ]; then
    ln -sf "$G4F_PYTHON_DIR/pip3" /usr/local/bin/pip3
    ln -sf "$G4F_PYTHON_DIR/pip3" /usr/local/bin/pip
elif [ -x "$G4F_PYTHON_DIR/pip" ]; then
    ln -sf "$G4F_PYTHON_DIR/pip" /usr/local/bin/pip3
    ln -sf "$G4F_PYTHON_DIR/pip" /usr/local/bin/pip
fi
log "python: $G4F_PYTHON_BIN"

log "writing launchers"
mkdir -p /root/.g4f /usr/local/bin
cat > /usr/local/bin/g4f-api <<'EOF'
#!/bin/sh
# A stale PYTHONHOME makes CPython fail before it can import encodings.
unset PYTHONHOME
export G4F_VNC_PORT=5901
export G4F_NOVNC_DIR=/usr/share/novnc
exec /g4f-bin/g4f-go api --bind 127.0.0.1:1337 --debug --no-reload
EOF
chmod 755 /usr/local/bin/g4f-api

cat > /g4f-scripts/.provisioned-base <<'EOF'
base provisioning complete
EOF
log "base provisioning complete in $(( $(date +%s) - started ))s"
