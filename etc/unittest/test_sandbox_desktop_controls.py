from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "docker"
BRAND = ROOT / "bin" / "novnc-brand.sh"


class TestSandboxDesktopControls(unittest.TestCase):
    def test_novnc_is_extracted_into_the_served_directory(self):
        text = (DOCKER / "Dockerfile-sandbox-desktop").read_text(encoding="utf-8")
        self.assertIn("mkdir -p /usr/share/novnc", text)
        self.assertIn("-C /usr/share/novnc --strip-components=1", text)
        self.assertNotIn("mkdir -p novnc", text)
        self.assertNotIn("-C novnc ", text)

    def test_desktop_controls_are_installed_with_the_client(self):
        text = (DOCKER / "Dockerfile-sandbox-desktop").read_text(encoding="utf-8")
        self.assertIn(
            "COPY docker/novnc-controls.js /usr/share/novnc/g4f-controls.js", text
        )
        self.assertIn(
            "COPY docker/novnc-controls.css /usr/share/novnc/g4f-controls.css", text
        )

    def test_landing_page_loads_the_controls_overlay(self):
        text = (DOCKER / "novnc-index.html").read_text(encoding="utf-8")
        self.assertIn('href="g4f-controls.css"', text)
        self.assertIn('src="g4f-controls.js"', text)
        for element_id in ("vnc", "toolbar", "keyboard", "shortcuts", "toast"):
            self.assertIn(f'id="{element_id}"', text)
        # Remote resize is the default: the VNC server follows the viewer's
        # window instead of forcing the session's original geometry.
        self.assertIn("resize=remote", text)
        self.assertNotIn("resize=off", text)

    def test_desktop_ships_an_installable_manifest(self):
        manifest = json.loads(
            (DOCKER / "novnc-manifest.webmanifest").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["start_url"], "./")
        self.assertEqual(manifest["scope"], "./")
        self.assertIn(manifest["display"], ("standalone", "fullscreen"))
        self.assertEqual(manifest["display_override"][-1], "standalone")
        self.assertEqual(
            manifest["icons"][0],
            {
                "src": "g4f.svg",
                "sizes": "any",
                "type": "image/svg+xml",
                "purpose": "any",
            },
        )
        self.assertEqual(len(manifest["icons"]), 1)

    def test_pwa_assets_are_installed_and_linked(self):
        dockerfile = (DOCKER / "Dockerfile-sandbox-desktop").read_text(
            encoding="utf-8"
        )
        for source, target in (
            ("docker/novnc-manifest.webmanifest", "manifest.webmanifest"),
            ("docker/novnc-sw.js", "sw.js"),
            ("docker/g4f.svg", "g4f.svg"),
        ):
            self.assertIn(f"COPY {source} /usr/share/novnc/{target}", dockerfile)

        landing = (DOCKER / "novnc-index.html").read_text(encoding="utf-8")
        self.assertIn('rel="manifest" href="manifest.webmanifest"', landing)
        self.assertIn('rel="icon" href="g4f.svg" type="image/svg+xml"', landing)
        self.assertIn('id="btn-install"', landing)

        controls = (DOCKER / "novnc-controls.js").read_text(encoding="utf-8")
        self.assertIn("beforeinstallprompt", controls)
        self.assertIn("appinstalled", controls)
        self.assertIn("serviceWorker.register", controls)
        self.assertIn('new URL("sw.js", document.baseURI)', controls)

    def test_service_worker_keeps_the_live_stream_out_of_the_cache(self):
        worker = (DOCKER / "novnc-sw.js").read_text(encoding="utf-8")
        self.assertIn("g4f-desktop-", worker)
        self.assertIn('url.pathname.endsWith("/websockify")', worker)
        self.assertIn("self.registration.scope", worker)
        self.assertIn("cache.match", worker)
        self.assertIn('request.mode === "navigate"', worker)

    def test_vnc_relay_serves_the_manifest_and_worker_types(self):
        relay = (ROOT / "g4f" / "api" / "vnc_relay.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"/desktop/manifest.webmanifest"', relay)
        self.assertIn('"/desktop/sw.js"', relay)
        self.assertIn('media_type="application/manifest+json"', relay)
        self.assertIn('media_type="application/javascript; charset=utf-8"', relay)
        self.assertIn('headers={"Cache-Control": "no-cache"}', relay)
        # The explicit files must be registered before the /desktop mount.
        self.assertLess(
            relay.index('"/desktop/manifest.webmanifest"'), relay.index("app.mount(")
        )
        self.assertLess(relay.index('"/desktop/sw.js"'), relay.index("app.mount("))

    def test_control_buttons_and_bridge_are_wired_up(self):
        controls = (DOCKER / "novnc-controls.js").read_text(encoding="utf-8")
        for method in (
            "g4fSendText",
            "g4fSendKey",
            "g4fSendCombo",
            "g4fToggleModifier",
            "g4fSetViewOnly",
            "g4fToggleFit",
            "g4fSetQuality",
        ):
            self.assertIn(method, controls)
        bridge = BRAND.read_text(encoding="utf-8")
        self.assertIn("window.g4fUI = UI", bridge)
        self.assertIn("g4f: /desktop key bridge v3", bridge)
        self.assertIn("g4f: end /desktop key bridge", bridge)
        self.assertIn("g4fSetResize", bridge)
        self.assertIn('"resize": "remote"', bridge)
        self.assertNotIn('"resize": "off"', bridge)

    def test_leave_disconnects_even_without_a_live_connection(self):
        """A failed desktop retries forever; Leave must break that loop."""
        controls = (DOCKER / "novnc-controls.js").read_text(encoding="utf-8")
        self.assertIn("g4fDisconnect", controls)
        self.assertNotIn("api.rfb && api.connected", controls)
        bridge = BRAND.read_text(encoding="utf-8")
        self.assertIn("UI.g4fDisconnect", bridge)
        self.assertIn("UI.inhibitReconnect = true", bridge)
        self.assertIn("clearTimeout(UI.reconnectCallback)", bridge)
        self.assertIn("UI.openConnectPanel()", bridge)
        # The status line distinguishes a retrying client from a dead one.
        self.assertIn("reconnecting: Boolean(UI.reconnectCallback)", bridge)
        self.assertIn("function statusMessage()", controls)

    def test_every_control_handler_has_a_matching_element(self):
        """A missing id used to abort boot and kill every later handler."""
        landing = (DOCKER / "novnc-index.html").read_text(encoding="utf-8")
        ids = set(re.findall(r'id="([^"]+)"', landing))
        controls = (DOCKER / "novnc-controls.js").read_text(encoding="utf-8")
        referenced = set(re.findall(r'\$\("([a-z][a-z0-9-]*)"\)', controls))
        self.assertTrue(referenced)
        missing = sorted(referenced - ids)
        self.assertEqual(missing, [], f"controls reference missing ids: {missing}")

    def test_every_shortcut_key_is_known_to_the_bridge(self):
        controls = (DOCKER / "novnc-controls.js").read_text(encoding="utf-8")
        shortcut_keys = set()
        for group in re.findall(r"keys:\s*\[([^\]]+)\]", controls):
            shortcut_keys.update(re.findall(r'"([^"]+)"', group))
        self.assertIn("Control", shortcut_keys)
        self.assertIn("ArrowLeft", shortcut_keys)

        bridge = BRAND.read_text(encoding="utf-8")
        table = bridge[bridge.index("const G4F_KEY_TABLE"):bridge.index("UI.g4fDownKeys")]
        named = set(re.findall(r"^\s{4}([A-Za-z0-9]+):\s*\[", table, re.MULTILINE))
        missing = sorted(
            key
            for key in shortcut_keys
            if key not in named and len(key) != 1
        )
        self.assertEqual(missing, [], f"shortcuts use unknown key names: {missing}")

    def test_touch_panels_track_the_on_screen_keyboard(self):
        controls = (DOCKER / "novnc-controls.js").read_text(encoding="utf-8")
        self.assertIn("visualViewport", controls)
        self.assertIn("--g4f-keyboard-inset", controls)
        self.assertIn("input.blur()", controls)
        css = (DOCKER / "novnc-controls.css").read_text(encoding="utf-8")
        self.assertIn("--g4f-keyboard-inset", css)
        self.assertIn("touch-action: manipulation", css)


if __name__ == "__main__":
    unittest.main()
