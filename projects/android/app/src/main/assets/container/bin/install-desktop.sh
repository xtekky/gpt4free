#!/bin/sh
# install-desktop.sh — install the Xfce/TigerVNC/noVNC packages used by g4f
# sandboxes. Shared by Polli's bootstrap and the Android PRoot container.
#
# Usage:
#   G4F_DESKTOP_PACKAGES="xfce4 tigervnc-standalone-server novnc dbus-x11 procps" \
#   G4F_DESKTOP_REPAIR=1 \
#   G4F_DESKTOP_NOVNC_ASSETS=/g4f-scripts/novnc \
#       install-desktop.sh
#
# Environment:
#   G4F_DESKTOP_PACKAGES       Package list (default: Polli's Xfce stack)
#   G4F_DESKTOP_REPAIR=1       Clear apt/dpkg locks, repair dpkg state and
#                              rewrite /etc/resolv.conf before each apt step
#   G4F_DESKTOP_NOVNC_ASSETS   Optional directory copied over /usr/share/novnc
#   G4F_DESKTOP_UPDATE_ATTEMPTS  apt-get update retries (default: 3)
#   G4F_DESKTOP_INSTALL_ATTEMPTS apt-get install retries (default: 2)
#   G4F_DESKTOP_LOG_PREFIX     Log prefix (default: [g4f-desktop])
set -eu

DEFAULT_PACKAGES='xfce4-panel xfwm4 xfdesktop4 xfce4-settings xfce4-terminal dbus-x11 x11-xserver-utils x11-xkb-utils tigervnc-standalone-server fonts-dejavu-core fonts-noto-core fonts-noto-color-emoji librsvg2-common adwaita-icon-theme gtk-update-icon-cache procps'
PACKAGES="${G4F_DESKTOP_PACKAGES:-$DEFAULT_PACKAGES}"
REPAIR="${G4F_DESKTOP_REPAIR:-0}"
NOVNC_ASSETS="${G4F_DESKTOP_NOVNC_ASSETS:-}"
UPDATE_ATTEMPTS="${G4F_DESKTOP_UPDATE_ATTEMPTS:-3}"
INSTALL_ATTEMPTS="${G4F_DESKTOP_INSTALL_ATTEMPTS:-2}"
LOG_PREFIX="${G4F_DESKTOP_LOG_PREFIX:-[g4f-desktop]}"

export DEBIAN_FRONTEND=noninteractive

log() { printf '%s %s\n' "$LOG_PREFIX" "$*"; }

write_resolv() {
    rm -f /etc/resolv.conf
    cat > /etc/resolv.conf <<'EOF'
nameserver 1.1.1.1
nameserver 8.8.8.8
options timeout:5 attempts:3
EOF
}

clear_apt_locks() {
    rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/cache/apt/archives/lock
}

repair_state() {
    clear_apt_locks
    write_resolv
    dpkg --configure -a || true
    apt-get -f install -y || true
}

prepare_resolv() {
    if [ "$REPAIR" = 1 ]; then
        write_resolv
    fi
}

if [ "$REPAIR" = 1 ]; then
    log "repairing dpkg state"
    repair_state
fi

log "updating apt package lists"
updated=0
attempt=1
while [ "$attempt" -le "$UPDATE_ATTEMPTS" ]; do
    prepare_resolv
    if apt-get update -qq; then
        updated=1
        break
    fi
    log "apt update failed ($attempt/$UPDATE_ATTEMPTS), retrying"
    attempt=$((attempt + 1))
    sleep 5
done
if [ "$updated" != 1 ]; then
    log "apt update failed after $UPDATE_ATTEMPTS attempts; check the connection"
    exit 1
fi

log "installing desktop packages"
installed=0
attempt=1
while [ "$attempt" -le "$INSTALL_ATTEMPTS" ]; do
    prepare_resolv
    set -f
    # Word splitting is intentional: G4F_DESKTOP_PACKAGES is a package list.
    # shellcheck disable=SC2086
    if apt-get install -y --no-install-recommends $PACKAGES; then
        set +f
        installed=1
        break
    fi
    set +f
    log "desktop package install failed ($attempt/$INSTALL_ATTEMPTS), retrying"
    if [ "$REPAIR" = 1 ]; then
        dpkg --configure -a || true
        prepare_resolv
        apt-get update -qq || true
    fi
    attempt=$((attempt + 1))
    sleep 5
done
if [ "$installed" != 1 ]; then
    log "desktop package installation failed after $INSTALL_ATTEMPTS attempts"
    exit 1
fi
prepare_resolv

if [ -n "$NOVNC_ASSETS" ] && [ -d "$NOVNC_ASSETS" ]; then
    log "applying g4f noVNC UI"
    # Android ships a complete pinned/branded noVNC tree. Replace the stock
    # package directory so no old logos or client files survive the copy; a
    # partial overlay without vnc.html is still copied on top as before.
    if [ -f "$NOVNC_ASSETS/vnc.html" ]; then
        rm -rf /usr/share/novnc
    fi
    mkdir -p /usr/share/novnc
    cp -R "$NOVNC_ASSETS"/. /usr/share/novnc/
    chmod -R a+rX /usr/share/novnc
fi

log "desktop packages installed"
