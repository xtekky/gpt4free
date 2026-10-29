#!/usr/bin/env bash
#
# novnc-brand — put the g4f mark and colors on the stock noVNC client the
# sandbox desktop serves (from the upstream noVNC release tarball, unpacked
# into /usr/share/novnc).
#
#   novnc-brand.sh [novnc-dir]   default: /usr/share/novnc
#
# Expects g4f.svg inside novnc-dir (the Dockerfile copies it there) and only
# touches what the branding needs:
#
#   vnc.html                    the two text ".noVNC_logo" wordmarks become
#                               <img src="g4f.svg"> (control bar + connect
#                               dialog); the favicon link points at g4f.svg,
#                               the browser tab's first g4f pixel
#   app/styles/constants.css    noVNC's blue accent palette -> g4f purple/blue
#   app/styles/base.css         connect dialog stacks its logo above the
#                               Connect button (flex column) and centers it
#   app/images/icons/novnc*.svg the noVNC icon marks -> g4f.svg (used by
#                               vnc_lite.html and as apple-touch/PWA icon)
#   defaults.json               resize=remote, so the client asks the VNC
#                               server to resize the desktop to the browser
#                               window (server-side scaling) instead of
#                               noVNC's stock resize=off
#
# Written against noVNC 1.6.0 (the upstream release tarball the sandbox image
# unpacks). Every substitution is verified below, so a changed noVNC layout
# fails the image build instead of silently shipping stock logos. Runs at image
# build: see docker/Dockerfile-sandbox-desktop.

set -euo pipefail

NOVNC_DIR="${1:-/usr/share/novnc}"
G4F_SVG="$NOVNC_DIR/g4f.svg"

fail() { echo "novnc-brand: $*" >&2; exit 1; }

[ -f "$NOVNC_DIR/vnc.html" ] || fail "no vnc.html under $NOVNC_DIR"
[ -f "$G4F_SVG" ] || fail "$G4F_SVG missing (copy docker/g4f.svg into the noVNC dir first)"

# The control-bar (<h1>) and connect-dialog (<p>) wordmarks are single-line
# text elements; swap each for the g4f mark, sized to its slot (the surrounding
# layout CSS — padding, centering, the + hr divider — keeps applying via the
# class). The favicon <link> swaps to the same SVG.
sed -i -E \
    -e 's#<h1 class="noVNC_logo".*</h1>#<img class="noVNC_logo" src="g4f.svg" alt="g4f" style="width:32px;height:32px">#' \
    -e 's#<p class="noVNC_logo".*</p>#<img class="noVNC_logo" src="g4f.svg" alt="g4f" style="width:120px;height:120px">#' \
    -e 's#<link rel="icon" type="image/x-icon" href="app/images/icons/novnc.ico">#<link rel="icon" type="image/svg+xml" href="g4f.svg">#' \
    "$NOVNC_DIR/vnc.html"

# Accent palette: noVNC's greys stay, its blues become the g4f gradient ends
# (#6e48aa / #4776E6) and a matching dark shade for pressed panel states.
sed -i -E \
    -e 's/--novnc-blue: rgb\(110, 132, 163\)/--novnc-blue: rgb(110, 72, 170)/' \
    -e 's/--novnc-lightblue: rgb\(74, 144, 217\)/--novnc-lightblue: rgb(71, 118, 230)/' \
    -e 's/--novnc-darkblue: rgb\(83, 99, 122\)/--novnc-darkblue: rgb(60, 45, 100)/' \
    "$NOVNC_DIR/app/styles/constants.css"

# Connect dialog: stack the logo above the Connect button instead of letting
# the two inline-level children sit on one line, and center the logo over the
# button (the logo is an <img> now, so noVNC's text-align: center no longer
# reaches it; align-self centers just the mark and leaves the button row's
# full width alone). Appended (not substituted) so it wins on source order over
# noVNC's own #noVNC_connect_dlg rules.
if ! grep -q 'g4f: stack the connect dialog' "$NOVNC_DIR/app/styles/base.css"; then
    cat >> "$NOVNC_DIR/app/styles/base.css" <<'CSS'

/* g4f: stack the connect dialog's logo above the Connect button */
#noVNC_connect_dlg {
    display: flex;
    flex-direction: column;
}
#noVNC_connect_dlg .noVNC_logo {
    align-self: center;
}
CSS
fi

# SVG icon marks (vnc_lite.html, apple-touch/PWA icons). The PNG/ICO variants
# are left alone — regenerating raster icons needs a converter the image
# doesn't carry, and every desktop-browser surface uses the SVG paths above.
for icon in novnc-icon.svg novnc-icon-sm.svg novnc-ios-icon.svg; do
    icon_path="$NOVNC_DIR/app/images/icons/$icon"
    if [ -f "$icon_path" ]; then
        cp "$G4F_SVG" "$icon_path"
    fi
done

# Server-side scaling by default: noVNC's stock default is resize=off (native
# framebuffer, scrollbars). Writing resize=remote into defaults.json makes the
# client ask the VNC server to resize the desktop to the browser window, so
# direct visits to vnc.html get server-side scaling too (the /desktop/ wrapper
# page also passes resize=remote in its query string). The value stays a
# default, not a lock: the settings panel can still change it.
cat > "$NOVNC_DIR/defaults.json" <<'JSON'
{
    "resize": "remote"
}
JSON

# Fail the build rather than serve stock branding if a patch did not land.
grep -q 'src="g4f\.svg"' "$NOVNC_DIR/vnc.html" \
    || fail "logo substitution did not apply to $NOVNC_DIR/vnc.html"
grep -q -- '--novnc-blue: rgb(110, 72, 170)' "$NOVNC_DIR/app/styles/constants.css" \
    || fail "color substitution did not apply to $NOVNC_DIR/app/styles/constants.css"
grep -q 'flex-direction: column' "$NOVNC_DIR/app/styles/base.css" \
    || fail "connect dialog rule did not apply to $NOVNC_DIR/app/styles/base.css"
grep -q 'align-self: center' "$NOVNC_DIR/app/styles/base.css" \
    || fail "connect dialog logo centering did not apply to $NOVNC_DIR/app/styles/base.css"
grep -q '"resize": "remote"' "$NOVNC_DIR/defaults.json" \
    || fail "resize default did not apply to $NOVNC_DIR/defaults.json"

echo "novnc-brand: g4f logo and colors applied to $NOVNC_DIR"
