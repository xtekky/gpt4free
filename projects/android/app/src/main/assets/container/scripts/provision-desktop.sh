#!/bin/sh
# Optional XFCE desktop installation. Only executed when the user opts in.
set -eu
export DEBIAN_FRONTEND=noninteractive
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export G4F_PYTHON=/opt/g4f-venv/bin/python

log() { echo "[g4f-desktop] $*"; }

log "installing XFCE, TigerVNC and noVNC"
# One installer is shared with the Polli sandbox bootstrap. The Android run
# needs dpkg/apt repair and resolver rewrites, then overlays the branded noVNC
# UI that ships with the app.
G4F_DESKTOP_REPAIR=1 \
G4F_DESKTOP_PACKAGES='xfce4 tigervnc-standalone-server novnc dbus-x11 procps' \
G4F_DESKTOP_NOVNC_ASSETS=/g4f-scripts/novnc \
    /g4f-bin/install-desktop.sh

log "writing desktop launcher"
# One launcher is shipped as an app asset and copied by writeLaunchers() too.
# It owns Xvnc and XFCE, and gives XFCE a D-Bus session bus; the g4f API serves
# noVNC at /desktop/, so no second HTTP server is needed.
cp /g4f-scripts/g4f-desktop /usr/local/bin/g4f-desktop
chmod 755 /usr/local/bin/g4f-desktop

cat > /g4f-scripts/.provisioned-desktop <<'EOF'
desktop provisioning complete
EOF
log "desktop provisioning complete"
