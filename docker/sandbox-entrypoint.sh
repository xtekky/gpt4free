#!/bin/bash
# Entrypoint for the sandbox desktop image: fixes up the container basics
# Chrome needs (non-empty /etc/machine-id, system bus), starts the desktop
# stack (Xvnc, Xfce, noVNC) as the unprivileged `pn` user, then execs the
# container command with DISPLAY pointing at the sandbox display — so a
# headed Chrome launched by the g4f API shows up on the noVNC desktop.
#
# Env knobs:
#   SANDBOX_DESKTOP  set to 0 to skip the desktop stack (default: 1)
#   VNC_DISPLAY      X display for Xvnc   (default: :0)
#   VNC_GEOMETRY     screen size          (default: 1920x1080)
#   VNC_RFBPORT      VNC port             (default: 5900)
#   NOVNC_PORT       noVNC web port       (default: 6080)

set -u

# Non-root callers (e.g. `user: pn`) get no desktop setup, just the command.
if [ "$(id -u)" != "0" ]; then
    exec "$@"
fi

# Chrome reads /etc/machine-id (browser_dm_token_storage) and the slim base
# ships an empty file; a system bus silences its dbus connection errors.
if [ ! -s /etc/machine-id ]; then
    dbus-uuidgen > /etc/machine-id 2>/dev/null || \
        tr -d '-' < /proc/sys/kernel/random/uuid > /etc/machine-id
fi
dbus-daemon --system --fork 2>/dev/null || true

if [ "${SANDBOX_DESKTOP:-1}" = "1" ]; then
    export VNC_DISPLAY="${VNC_DISPLAY:-:0}"
    export VNC_GEOMETRY="${VNC_GEOMETRY:-1920x1080}"
    export VNC_RFBPORT="${VNC_RFBPORT:-5900}"
    export NOVNC_PORT="${NOVNC_PORT:-6080}"
    # Desktop processes run as `pn` (uid 1000): Xvnc and Xfce want a real
    # user session, and Chrome refuses to run as root without --no-sandbox.
    runuser -u pn -- bash -c '
        export HOME=/home/pn DISPLAY="$VNC_DISPLAY"
        nohup Xvnc "$VNC_DISPLAY" -rfbport "$VNC_RFBPORT" -localhost no \
            -SecurityTypes None -geometry "$VNC_GEOMETRY" -depth 24 \
            -nolisten tcp >/tmp/vnc.log 2>&1 &
        num="${VNC_DISPLAY#:}"
        for _ in $(seq 1 50); do
            [ -S "/tmp/.X11-unix/X$num" ] && break
            sleep 0.2
        done
        nohup xfsettingsd >/tmp/xfsettingsd.log 2>&1 &
        nohup xfwm4 >/tmp/xfwm4.log 2>&1 &
        nohup xfce4-panel >/tmp/xfce4-panel.log 2>&1 &
        nohup xfdesktop4 >/tmp/xfdesktop4.log 2>&1 &
        nohup websockify --web /usr/share/novnc "$NOVNC_PORT" localhost:"$VNC_RFBPORT" >/tmp/novnc.log 2>&1 &
    '
    # Headed Chrome spawned by the container command lands on the display.
    export DISPLAY="$VNC_DISPLAY"
fi

exec "$@"
