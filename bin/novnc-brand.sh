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
#   app/ui.js                   clipboard bridge: remote clipboard events are
#                               mirrored into the browser clipboard, and a
#                               paste in the noVNC window is sent to the VM;
#                               also exposes the UI on window.g4fUI with
#                               helpers for text, key combos, modifiers,
#                               view-only, scaling and quality, used by the
#                               /desktop touch controls (g4f-controls.js)
#   defaults.json               resize=remote, so the VNC server resizes the
#                               desktop to the viewer's window; the Fit button
#                               on the touch overlay switches to local scaling
#                               for viewers that prefer the fixed geometry
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

# Clipboard bridge. noVNC's stock clipboard panel copies between its textarea
# and the remote machine; the bridge below makes the browser clipboard part of
# that loop too, so Ctrl+V in the noVNC window pastes into the VM and text
# copied in the VM is mirrored into the browser clipboard (the Async Clipboard
# API only works in a secure context, so http:// LAN URLs keep the panel as the
# fallback). Every edit is verified below; a changed noVNC layout fails the
# build instead of silently shipping stock clipboard behaviour.
if ! grep -q 'g4f: clipboard bridge' "$NOVNC_DIR/app/ui.js"; then
    python3 - "$NOVNC_DIR/app/ui.js" <<'PY'
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    src = handle.read()

receive_old = """    clipboardReceive(e) {
        Log.Debug(">> UI.clipboardReceive: " + e.detail.text.substr(0, 40) + "...");
        document.getElementById('noVNC_clipboard_text').value = e.detail.text;
        Log.Debug("<< UI.clipboardReceive");
    },"""
receive_new = """    clipboardReceive(e) {
        Log.Debug(">> UI.clipboardReceive: " + e.detail.text.substr(0, 40) + "...");
        document.getElementById('noVNC_clipboard_text').value = e.detail.text;
        // g4f: clipboard bridge - mirror the remote clipboard into the browser.
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(e.detail.text).catch(() => {});
        }
        Log.Debug("<< UI.clipboardReceive");
    },"""

handlers_old = """        document.getElementById("noVNC_clipboard_text")
            .addEventListener('change', UI.clipboardSend);
    },"""
handlers_new = """        document.getElementById("noVNC_clipboard_text")
            .addEventListener('change', UI.clipboardSend);
        // g4f: clipboard bridge - a paste in the noVNC window goes to the VM.
        document.addEventListener('paste', (event) => {
            if (!UI.rfb || !event.clipboardData) {
                return;
            }
            const text = event.clipboardData.getData('text/plain');
            if (text) {
                UI.rfb.clipboardPasteFrom(text);
            }
        });
    },"""

for old, new in ((receive_old, receive_new), (handlers_old, handlers_new)):
    if old not in src:
        raise SystemExit(f"novnc-brand: clipboard patch anchor not found in {path}")
    src = src.replace(old, new, 1)

with open(path, "w", encoding="utf-8") as handle:
    handle.write(src)
PY
fi

# Touch controls bridge. The /desktop landing page (novnc-index.html) overlays
# the Remote Desktop viewer's text input, key buttons and one-tap shortcuts on
# the noVNC iframe. Give that page a stable handle to the UI plus helpers that
# use noVNC's own keysym table, so it never has to reimplement RFB internals.
#
# The patch is versioned and replaces any earlier g4f bridge (including the
# unversioned one written by older images), so rebuilding over an already
# branded directory still ships the current helpers instead of silently
# keeping stale event handlers.
if ! grep -q 'g4f: /desktop key bridge v3' "$NOVNC_DIR/app/ui.js"; then
    python3 - "$NOVNC_DIR/app/ui.js" <<'PY'
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as handle:
    src = handle.read()

anchor = "export default UI;"
if anchor not in src:
    raise SystemExit(f"novnc-brand: UI export anchor not found in {path}")

# Drop a bridge installed by an older run before inserting this one. Legacy
# bridges had no end marker and ran up to the export statement.
legacy_start = src.find("/* g4f: /desktop key bridge")
end_marker = "/* g4f: end /desktop key bridge */"
if legacy_start != -1:
    marker_end = src.find(end_marker, legacy_start)
    if marker_end != -1:
        remove_end = src.index("\n", marker_end) + 1
    else:
        remove_end = src.index(anchor, legacy_start)
    src = src[:legacy_start] + src[remove_end:].lstrip("\n")

bridge = r"""/* g4f: /desktop key bridge v3
 *
 * The outer page (index.html) layers the Remote Desktop viewer's touch
 * controls over this client. These helpers keep the keysym mapping, sticky
 * modifiers and settings handling inside noVNC, where KeyTable, keysyms and
 * RFB are already in scope; the outer page only supplies key names such as
 * "Control", "ArrowLeft" or "F5".
 */
const G4F_KEY_TABLE = {
    Alt: [KeyTable.XK_Alt_L, "AltLeft"],
    Shift: [KeyTable.XK_Shift_L, "ShiftLeft"],
    Control: [KeyTable.XK_Control_L, "ControlLeft"],
    Meta: [KeyTable.XK_Super_L, "MetaLeft"],
    Enter: [KeyTable.XK_Return, "Enter"],
    Return: [KeyTable.XK_Return, "Enter"],
    Backspace: [KeyTable.XK_BackSpace, "Backspace"],
    Tab: [KeyTable.XK_Tab, "Tab"],
    Escape: [KeyTable.XK_Escape, "Escape"],
    Esc: [KeyTable.XK_Escape, "Escape"],
    Delete: [KeyTable.XK_Delete, "Delete"],
    Insert: [KeyTable.XK_Insert, "Insert"],
    Home: [KeyTable.XK_Home, "Home"],
    End: [KeyTable.XK_End, "End"],
    PageUp: [KeyTable.XK_Page_Up, "PageUp"],
    PageDown: [KeyTable.XK_Page_Down, "PageDown"],
    ArrowUp: [KeyTable.XK_Up, "ArrowUp"],
    ArrowDown: [KeyTable.XK_Down, "ArrowDown"],
    ArrowLeft: [KeyTable.XK_Left, "ArrowLeft"],
    ArrowRight: [KeyTable.XK_Right, "ArrowRight"],
    Up: [KeyTable.XK_Up, "ArrowUp"],
    Down: [KeyTable.XK_Down, "ArrowDown"],
    Left: [KeyTable.XK_Left, "ArrowLeft"],
    Right: [KeyTable.XK_Right, "ArrowRight"],
    Space: [KeyTable.XK_space, "Space"],
    PrintScreen: [KeyTable.XK_Print, "PrintScreen"],
    F1: [KeyTable.XK_F1, "F1"],
    F2: [KeyTable.XK_F2, "F2"],
    F3: [KeyTable.XK_F3, "F3"],
    F4: [KeyTable.XK_F4, "F4"],
    F5: [KeyTable.XK_F5, "F5"],
    F6: [KeyTable.XK_F6, "F6"],
    F7: [KeyTable.XK_F7, "F7"],
    F8: [KeyTable.XK_F8, "F8"],
    F9: [KeyTable.XK_F9, "F9"],
    F10: [KeyTable.XK_F10, "F10"],
    F11: [KeyTable.XK_F11, "F11"],
    F12: [KeyTable.XK_F12, "F12"],
};

UI.g4fDownKeys = new Set();

UI.g4fKey = function (name) {
    if (Object.prototype.hasOwnProperty.call(G4F_KEY_TABLE, name)) {
        return G4F_KEY_TABLE[name];
    }
    if (typeof name === "string" && Array.from(name).length === 1) {
        return [keysyms.lookup(name.codePointAt(0)), ""];
    }
    return null;
};

UI.g4fCanSend = function () {
    return Boolean(UI.rfb && UI.connected && !UI.rfb.viewOnly);
};

UI.g4fSendKey = function (name) {
    if (!UI.g4fCanSend()) return false;
    const key = UI.g4fKey(name);
    if (!key) return false;
    try {
        UI.rfb.sendKey(key[0], key[1]);
    } catch (error) {
        Log.Error("g4f: could not send key " + name + ": " + error);
        return false;
    }
    return true;
};

UI.g4fToggleModifier = function (name) {
    if (!UI.g4fCanSend()) return null;
    const key = UI.g4fKey(name);
    if (!key) return null;
    if (UI.g4fDownKeys.has(name)) {
        UI.rfb.sendKey(key[0], key[1], false);
        UI.g4fDownKeys.delete(name);
    } else {
        UI.rfb.sendKey(key[0], key[1], true);
        UI.g4fDownKeys.add(name);
    }
    return UI.g4fDownKeys.has(name);
};

UI.g4fReleaseModifiers = function () {
    const held = Array.from(UI.g4fDownKeys);
    UI.g4fDownKeys.clear();
    if (!UI.rfb) return;
    for (const name of held) {
        const key = UI.g4fKey(name);
        if (!key) continue;
        try {
            UI.rfb.sendKey(key[0], key[1], false);
        } catch (error) {
            Log.Error("g4f: could not release key " + name + ": " + error);
        }
    }
};

UI.g4fDisconnect = function () {
    // Leave must work in every connection state, not only while a session is
    // live: a failed or unreachable desktop sits in noVNC's reconnect loop,
    // where UI.rfb is either still connecting or undefined. Cancel the pending
    // reconnect, stop any live session, and return to the connect panel.
    UI.g4fReleaseModifiers();
    UI.inhibitReconnect = true;
    if (UI.reconnectCallback) {
        clearTimeout(UI.reconnectCallback);
        UI.reconnectCallback = null;
    }
    if (UI.rfb) {
        try {
            UI.disconnect();
        } catch (error) {
            Log.Error("g4f: could not disconnect: " + error);
            UI.rfb = undefined;
            UI.connected = false;
            UI.updateVisualState("disconnected");
            UI.openControlbar();
            UI.openConnectPanel();
        }
    } else {
        UI.connected = false;
        UI.updateVisualState("disconnected");
        UI.showStatus(_("Disconnected"), "normal");
        document.title = PAGE_TITLE;
        UI.openControlbar();
        UI.openConnectPanel();
    }
    return true;
};

UI.g4fSendCombo = function (names) {
    if (!UI.g4fCanSend() || !Array.isArray(names) || names.length === 0) {
        return false;
    }
    UI.g4fReleaseModifiers();
    const keys = [];
    for (const name of names) {
        const key = UI.g4fKey(name);
        if (!key) return false;
        keys.push(key);
    }
    // Press in order (modifiers first), release in reverse. The finally block
    // guarantees a thrown send never leaves a modifier stuck down remotely.
    const pressed = [];
    try {
        for (const key of keys) {
            UI.rfb.sendKey(key[0], key[1], true);
            pressed.push(key);
        }
        return true;
    } catch (error) {
        Log.Error("g4f: could not send combo: " + error);
        return false;
    } finally {
        for (const key of pressed.reverse()) {
            try {
                UI.rfb.sendKey(key[0], key[1], false);
            } catch (error) {
                Log.Error("g4f: could not release combo key: " + error);
            }
        }
    }
};

UI.g4fSendText = function (text) {
    if (!UI.g4fCanSend() || !text) return false;
    UI.g4fReleaseModifiers();
    // Sending "Enter" twice for CRLF would submit dialogs twice; normalise the
    // text first, then treat every remaining line break as one Return.
    const value = String(text).replace(/\r\n?/g, "\n");
    let sent = 0;
    for (const char of value) {
        const code = char.codePointAt(0);
        try {
            if (char === "\n") {
                UI.rfb.sendKey(KeyTable.XK_Return, "Enter");
            } else if (char === "\t") {
                UI.rfb.sendKey(KeyTable.XK_Tab, "Tab");
            } else if (code === 8) {
                UI.rfb.sendKey(KeyTable.XK_BackSpace, "Backspace");
            } else if (code >= 0x20) {
                UI.rfb.sendKey(keysyms.lookup(code));
            } else {
                continue;
            }
        } catch (error) {
            Log.Error("g4f: could not send text: " + error);
            return false;
        }
        sent += 1;
    }
    return sent > 0;
};

UI.g4fSetViewOnly = function (viewOnly) {
    viewOnly = Boolean(viewOnly);
    if (viewOnly) UI.g4fReleaseModifiers();
    WebUtil.writeSetting("view_only", viewOnly);
    if (UI.rfb) UI.rfb.viewOnly = viewOnly;
    UI.updateViewOnly();
    UI.updateSetting("view_only");
    return viewOnly;
};

UI.g4fSetResize = function (mode) {
    if (mode !== "off" && mode !== "scale") mode = "remote";
    WebUtil.writeSetting("resize", mode);
    UI.updateSetting("resize");
    UI.applyResizeMode();
    UI.updateViewClip();
    return mode;
};

UI.g4fToggleFit = function () {
    if (!UI.rfb) return false;
    // Drive noVNC's own resize setting instead of only poking scaleViewport:
    // the stock settings panel, reconnect and the Fit button then agree.
    // The default is remote (server-side resize), so Fit toggles local scaling
    // and returns to remote geometry on the next tap.
    const current = String(UI.getSetting("resize") || "remote");
    const next = current === "scale" ? "remote" : "scale";
    return UI.g4fSetResize(next);
};

UI.g4fSetQuality = function (level) {
    level = Math.max(0, Math.min(9, parseInt(level, 10) || 0));
    WebUtil.writeSetting("quality", level);
    if (UI.rfb) UI.rfb.qualityLevel = level;
    UI.updateSetting("quality");
    return level;
};

UI.g4fState = function () {
    return {
        connected: Boolean(UI.rfb && UI.connected),
        // A connection attempt, or the delay before noVNC retries, is not a
        // dead desktop: the controls use these flags for a useful status.
        connecting: Boolean(UI.rfb && !UI.connected),
        reconnecting: Boolean(UI.reconnectCallback),
        viewOnly: Boolean(UI.rfb && UI.rfb.viewOnly),
        scale: Boolean(UI.rfb && UI.rfb.scaleViewport),
        resize: String(UI.getSetting("resize") || "remote"),
        quality: UI.rfb ? UI.rfb.qualityLevel : (parseInt(UI.getSetting("quality"), 10) || 6),
        modifiers: Array.from(UI.g4fDownKeys),
    };
};

window.g4fUI = UI;

/* g4f: end /desktop key bridge */
"""

src = src.replace(anchor, bridge + anchor, 1)
with open(path, "w", encoding="utf-8") as handle:
    handle.write(src)
PY
fi

# Remote resize by default: noVNC asks the VNC server to resize the desktop to
# the browser window, so the session fills a phone, laptop or tablet without a
# manual zoom. The touch overlay's Fit button switches a viewer to local
# scaling (and back to remote) without touching the server-side default.
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
grep -q 'g4f: clipboard bridge' "$NOVNC_DIR/app/ui.js" \
    || fail "clipboard bridge did not apply to $NOVNC_DIR/app/ui.js"
grep -q 'g4f: /desktop key bridge v3' "$NOVNC_DIR/app/ui.js" \
    || fail "desktop key bridge did not apply to $NOVNC_DIR/app/ui.js"
grep -q 'g4f: end /desktop key bridge' "$NOVNC_DIR/app/ui.js" \
    || fail "desktop key bridge is missing its end marker in $NOVNC_DIR/app/ui.js"
grep -q 'window.g4fUI = UI' "$NOVNC_DIR/app/ui.js" \
    || fail "desktop key bridge was not exposed on window.g4fUI"
grep -q 'UI.g4fSendCombo' "$NOVNC_DIR/app/ui.js" \
    || fail "desktop key bridge is missing the combo helper"
grep -q 'UI.g4fSendText' "$NOVNC_DIR/app/ui.js" \
    || fail "desktop key bridge is missing the text helper"
grep -q 'UI.g4fSetResize' "$NOVNC_DIR/app/ui.js" \
    || fail "desktop key bridge is missing the resize helper"
grep -q '"resize": "remote"' "$NOVNC_DIR/defaults.json" \
    || fail "resize default did not apply to $NOVNC_DIR/defaults.json"

echo "novnc-brand: g4f logo, colors, clipboard bridge, touch controls and remote resize applied to $NOVNC_DIR"
