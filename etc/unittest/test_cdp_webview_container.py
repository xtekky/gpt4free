"""CDP-over-socket detection for the Android app and its PRoot container.

The container runs a normal Linux Python, so it cannot derive the app process'
``@webview_devtools_remote_<pid>`` socket from its own pid. These tests cover
the environment hand-off that makes provider automation work there too.
"""

import os
import unittest
from unittest.mock import patch

from g4f.requests import cdp


class WebViewContainerSocketTests(unittest.TestCase):
    def test_container_is_android(self):
        with patch.dict(os.environ, {"G4F_CONTAINER": "1"}, clear=True):
            self.assertTrue(cdp._is_android())

    def test_socket_env_marks_android(self):
        with patch.dict(
            os.environ,
            {"G4F_WEBVIEW_DEVTOOLS_SOCKET": "webview_devtools_remote_123"},
            clear=True,
        ):
            self.assertTrue(cdp._is_android())

    def test_not_android_without_signals(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "os.path.exists", return_value=False
        ), patch("builtins.__import__", side_effect=ImportError):
            self.assertFalse(cdp._is_android())

    def test_socket_override_wins(self):
        with patch.dict(
            os.environ,
            {"G4F_WEBVIEW_DEVTOOLS_SOCKET": "@webview_devtools_remote_4321"},
            clear=True,
        ):
            self.assertEqual(
                cdp._find_webview_devtools_socket(),
                "webview_devtools_remote_4321",
            )

    def test_app_pid_override(self):
        with patch.dict(os.environ, {"G4F_WEBVIEW_PID": "9876"}, clear=True):
            self.assertEqual(
                cdp._find_webview_devtools_socket(),
                "webview_devtools_remote_9876",
            )

    def test_container_auto_detect_requires_reachable_socket(self):
        with patch.dict(os.environ, {"G4F_CONTAINER": "1"}, clear=True), patch.object(
            cdp, "_find_webview_devtools_socket_exists", return_value=False
        ):
            self.assertFalse(cdp._webview_auto_detect())
        with patch.dict(os.environ, {"G4F_CONTAINER": "1"}, clear=True), patch.object(
            cdp, "_find_webview_devtools_socket_exists", return_value=True
        ):
            self.assertTrue(cdp._webview_auto_detect())

    def test_app_mode_auto_detects_without_container(self):
        with patch.dict(
            os.environ,
            {"G4F_WEBVIEW_DEVTOOLS_SOCKET": "webview_devtools_remote_1"},
            clear=True,
        ):
            self.assertTrue(cdp._webview_auto_detect())


if __name__ == "__main__":
    unittest.main()
