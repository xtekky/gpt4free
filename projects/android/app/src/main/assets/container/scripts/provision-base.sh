#!/bin/sh
# Provision the Debian rootfs with a Python environment for gpt4free.
# Runs inside PRoot as fake root. The app binds the bundled g4f source at
# /g4f-src and these scripts at /g4f-scripts.
set -eu
export DEBIAN_FRONTEND=noninteractive
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export G4F_PYTHON=/opt/g4f-venv/bin/python

log() { echo "[g4f-provision] $*"; }

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
    if apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ca-certificates; then
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

log "creating /opt/g4f-venv"
python3 -m venv /opt/g4f-venv
/opt/g4f-venv/bin/pip install --no-cache-dir --retries 5 --timeout 30 \
    --upgrade pip setuptools wheel

log "installing gpt4free from bundled source"
# Editable install keeps the app's bundled source authoritative. Only the
# API extras plus the small Flask GUI dependencies are installed; the full
# "all" extra pulls desktop/printing packages that do not work under PRoot.
/opt/g4f-venv/bin/pip install --no-cache-dir --retries 5 --timeout 30 \
    -e "/g4f-src[api]" flask werkzeug asgiref pillow

log "writing launchers"
mkdir -p /root/.g4f /usr/local/bin
cat > /usr/local/bin/g4f-api <<'EOF'
#!/bin/sh
# A stale PYTHONHOME makes CPython fail before it can import encodings.
unset PYTHONHOME
export G4F_VNC_PORT=5901
export G4F_NOVNC_DIR=/usr/share/novnc
exec /opt/g4f-venv/bin/python -m g4f api --bind 127.0.0.1:1337 --debug --no-reload
EOF
chmod 755 /usr/local/bin/g4f-api

cat > /g4f-scripts/.provisioned-base <<'EOF'
base provisioning complete
EOF
log "base provisioning complete"
