from __future__ import annotations

import json
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
        self.assertIn("test -f /usr/share/novnc/vnc.html", text)
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
        # The overlay switches Fit itself, so the page must not force the
        # server-side resize that would fight the fixed-geometry default.
        self.assertNotIn("resize=remote", text)
        self.assertIn("resize=off", text)

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
        self.assertIn("g4f: /desktop key bridge", bridge)
        self.assertIn('"resize": "off"', bridge)


if __name__ == "__main__":
    unittest.main()
