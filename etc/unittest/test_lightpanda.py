"""
Tests for the Lightpanda browser installed by ``g4f-go browser install``.

Verifies that the installed binary is detected in the shared config
directory and preferred over Chrome whenever headless mode is on, and that
the auto-started process is shut down again (Lightpanda does not implement
the CDP ``Browser.close`` command).
"""

import os
import unittest
import unittest.mock
import urllib.request

import g4f.requests.cdp as cdp_module
from g4f.requests.cdp import (
    _close_browser_via_cdp,
    _start_lightpanda,
    _terminate_shared_browser,
    find_lightpanda_path,
    get_lightpanda_dir,
    get_shared_browser,
    lightpanda_auto_start_enabled,
)

LIGHTPANDA_PATH = find_lightpanda_path()


def _reset_shared_browser():
    """Terminate any shared browser and clear the recorded state."""
    _terminate_shared_browser()
    cdp_module._shared_browser_port = None
    cdp_module._last_shared_browser_port = None
    cdp_module._shared_browser_adopted = False


def _cdp_version(host: str, port: int) -> dict:
    import json

    with urllib.request.urlopen(
        f"http://{host}:{port}/json/version", timeout=2
    ) as response:
        return json.loads(response.read().decode("utf-8"))


class TestLightpandaDetection(unittest.TestCase):
    """Detection helpers — no browser process required."""

    def test_lightpanda_dir_is_browser_subdir_of_config_dir(self):
        from g4f.config import get_config_dir

        self.assertEqual(
            get_lightpanda_dir(), os.path.join(str(get_config_dir()), "browser")
        )

    def test_path_override_wins(self):
        with unittest.mock.patch.dict(
            os.environ, {"G4F_BROWSER_LIGHTPANDA_PATH": __file__}
        ):
            self.assertEqual(find_lightpanda_path(), __file__)

    def test_missing_override_falls_back(self):
        with unittest.mock.patch.dict(
            os.environ, {"G4F_BROWSER_LIGHTPANDA_PATH": os.path.join(__file__, "nope")}
        ):
            self.assertNotEqual(find_lightpanda_path(), os.path.join(__file__, "nope"))

    def test_auto_start_requires_headless(self):
        self.assertFalse(lightpanda_auto_start_enabled(False))

    def test_auto_start_skipped_when_port_configured(self):
        with unittest.mock.patch.dict(os.environ, {"G4F_BROWSER_PORT": "9222"}):
            self.assertFalse(lightpanda_auto_start_enabled(True))

    def test_auto_start_skipped_for_non_cdp_mode(self):
        with unittest.mock.patch.dict(os.environ, {"G4F_BROWSER_MODE": "extension"}):
            self.assertFalse(lightpanda_auto_start_enabled(True))

    def test_auto_start_allowed_for_cdp_mode(self):
        with unittest.mock.patch.dict(os.environ, {"G4F_BROWSER_MODE": "cdp"}):
            self.assertEqual(
                lightpanda_auto_start_enabled(True), LIGHTPANDA_PATH is not None
            )

    def test_auto_start_requires_installed_binary(self):
        with unittest.mock.patch.object(cdp_module, "find_lightpanda_path", lambda: None):
            self.assertFalse(lightpanda_auto_start_enabled(True))


@unittest.skipUnless(LIGHTPANDA_PATH, "Lightpanda is not installed")
class TestLightpandaAutoStart(unittest.TestCase):
    """Integration tests against the installed Lightpanda binary."""

    def setUp(self):
        _reset_shared_browser()

    def tearDown(self):
        _reset_shared_browser()

    def test_start_lightpanda_serves_cdp(self):
        port = _start_lightpanda("127.0.0.1")
        self.assertIsNotNone(port)
        self.assertIsNotNone(cdp_module._shared_browser_process)
        self.assertFalse(cdp_module._shared_browser_adopted)
        version = _cdp_version("127.0.0.1", port)
        self.assertIn("Lightpanda", version.get("Browser", ""))

    def test_terminate_kills_lightpanda(self):
        port = _start_lightpanda("127.0.0.1")
        self.assertIsNotNone(port)
        proc = cdp_module._shared_browser_process
        _terminate_shared_browser()
        self.assertIsNone(cdp_module._shared_browser_process)
        self.assertIsNotNone(proc.poll())

    def test_browser_close_is_rejected(self):
        """Lightpanda answers -32601, so the caller must fall back to a kill."""
        port = _start_lightpanda("127.0.0.1")
        self.assertIsNotNone(port)
        self.assertFalse(_close_browser_via_cdp("127.0.0.1", port))
        # Still running — the process handle is the only way to stop it.
        self.assertIsNone(cdp_module._shared_browser_process.poll())

    def test_shared_browser_prefers_lightpanda_when_headless(self):
        port = get_shared_browser("127.0.0.1", None, headless=True)
        self.assertIsNotNone(port)
        self.assertIsNotNone(cdp_module._shared_browser_process)
        self.assertIn(
            "Lightpanda", _cdp_version("127.0.0.1", port).get("Browser", "")
        )

    def test_shared_browser_reuses_lightpanda(self):
        port = get_shared_browser("127.0.0.1", None, headless=True)
        self.assertEqual(get_shared_browser("127.0.0.1", None, headless=True), port)


if __name__ == "__main__":
    unittest.main()
