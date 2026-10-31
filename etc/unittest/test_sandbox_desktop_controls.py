from __future__ import annotations

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
