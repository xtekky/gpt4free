#!/bin/bash
# Entrypoint for the sandbox desktop image: fixes up the container basics
# Chrome needs (non-empty /etc/machine-id, system bus), starts the desktop
# stack (Xvnc, Xfce) as the unprivileged `pn` user, then execs the container
# command with DISPLAY pointing at the sandbox display — so a headed browser
# launched on the desktop (menu → Install & Open Chrome) or by the g4f API
# shows up there. The noVNC web client is served by the g4f API itself (see
# g4f/api/vnc_relay) on its own port, so no websockify process runs here.
#
# Env knobs:
#   SANDBOX_DESKTOP  set to 0 to skip the desktop stack (default: 1)
#   G4F_WALLPAPER    set to 0 to skip the animated g4f wallpaper (default: 1)
#   VNC_DISPLAY      X display for Xvnc   (default: :0)
#   VNC_GEOMETRY     screen size          (default: 1920x1080)
#   VNC_RFBPORT      VNC port             (default: 5900)

set -u

# Root-only container basics. Chrome reads /etc/machine-id
# (browser_dm_token_storage) and the slim base ships an empty file; a system
# bus silences its dbus connection errors. The image runs as `pn` (USER), who
# has passwordless sudo, so the same setup happens either way.
as_root() {
    if [ "$(id -u)" = "0" ]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo -n "$@"
    else
        return 1
    fi
}

if [ ! -s /etc/machine-id ]; then
    as_root sh -c 'dbus-uuidgen > /etc/machine-id 2>/dev/null ||
        tr -d "-" < /proc/sys/kernel/random/uuid > /etc/machine-id' || true
fi
as_root dbus-daemon --system --fork 2>/dev/null || true

# Host-based routing: <app>.g4f.local and <port>.g4f.local reach the sandbox
# apps through the g4f API on port 8080 (see g4f/api/domain_router.py). The
# entries are derived from the router's own registry so they cannot drift.
# Hosts outside the container need the same names (see the sandbox README).
if [ "${G4F_DOMAIN_ROUTER:-1}" != "0" ]; then
    suffix=$(python3 - <<'PY' 2>/dev/null || true
from g4f.api.domain_router import domain_suffix, hosts_entries

with open("/tmp/g4f-domain-hosts", "w") as fh:
    fh.write("# g4f sandbox domain routing\n")
    fh.write("\n".join(hosts_entries()) + "\n")
print(domain_suffix())
PY
)
    if [ -n "$suffix" ] && [ -s /tmp/g4f-domain-hosts ]; then
        # Drop a previous block (the container may be restarted in place).
        as_root sh -c "grep -v -e '# g4f sandbox domain routing' -e '\\.${suffix}\$' /etc/hosts > /tmp/g4f-hosts.keep && cat /tmp/g4f-hosts.keep > /etc/hosts && rm -f /tmp/g4f-hosts.keep" || true
        as_root sh -c 'cat /tmp/g4f-domain-hosts >> /etc/hosts' || true
    fi
fi

if [ "${SANDBOX_DESKTOP:-1}" = "1" ]; then
    export VNC_DISPLAY="${VNC_DISPLAY:-:0}"
    export VNC_GEOMETRY="${VNC_GEOMETRY:-1920x1080}"
    export VNC_RFBPORT="${VNC_RFBPORT:-5900}"
    # Desktop processes run as `pn` (the unprivileged uid set by G4F_USER_ID):
    # Xvnc and Xfce want a real user session, and Chrome refuses to run as root
    # without --no-sandbox. When the container already runs as `pn` (the image
    # default) no switch is needed; as root, drop privileges with runuser.
    desktop_launcher='
        export HOME=/home/pn DISPLAY="$VNC_DISPLAY"
        num="${VNC_DISPLAY#:}"
        # A restarted container keeps /tmp, so the previous run leaves the X lock
        # and socket behind. Xvnc then refuses to start ("Server is already active
        # for display 0"), the Xfce clients fail with "cannot open display", and
        # the desktop is gone even though the entrypoint did run. Drop the
        # leftovers unless a live Xvnc still owns them.
        lock="/tmp/.X${num}-lock"
        if [ -e "$lock" ]; then
            # Xvnc writes the pid space-padded ("%10d"), so strip the blanks.
            pid=$(tr -d " \n" < "$lock" 2>/dev/null || true)
            if [ -z "$pid" ] || ! tr "\0" " " < "/proc/$pid/cmdline" 2>/dev/null | grep -q Xvnc; then
                rm -f "$lock" "/tmp/.X11-unix/X${num}"
            fi
        fi
        nohup Xvnc "$VNC_DISPLAY" -rfbport "$VNC_RFBPORT" -localhost no \
            -SecurityTypes None -geometry "$VNC_GEOMETRY" -depth 24 \
            -nolisten tcp >/tmp/vnc.log 2>&1 &
        for _ in $(seq 1 50); do
            [ -S "/tmp/.X11-unix/X$num" ] && break
            sleep 0.2
        done
        nohup xfsettingsd >/tmp/xfsettingsd.log 2>&1 &
        nohup xfwm4 >/tmp/xfwm4.log 2>&1 &
        nohup xfce4-panel >/tmp/xfce4-panel.log 2>&1 &
        nohup xfdesktop >/tmp/xfdesktop.log 2>&1 &
        if [ "${G4F_WALLPAPER:-1}" = "1" ]; then
            nohup xfce4-terminal --disable-server --maximize \
                --hide-menubar --hide-borders --hide-scrollbar \
                --title="g4f live wallpaper" \
                --icon=g4f-sandbox-applications \
                --command=/usr/local/bin/g4f-livewall \
                >/tmp/g4f-livewall.log 2>&1 &
        fi
    '
    if [ "$(id -u)" = "0" ]; then
        runuser -u pn -- bash -c "$desktop_launcher"
    else
        bash -c "$desktop_launcher"
    fi
    # Headed Chrome spawned by the container command lands on the display.
    export DISPLAY="$VNC_DISPLAY"
fi

exec "$@"
