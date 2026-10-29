from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, filename: str):
    # The CLI scripts (sandbox-settings, sandbox-tray) ship without a .py
    # extension, so spec_from_file_location cannot infer a source loader for
    # them. Build the spec from an explicit SourceFileLoader instead.
    loader = SourceFileLoader(name, str(ROOT / "bin" / filename))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before execution so dataclass and other stdlib helpers that look
    # the module up by name in sys.modules can find it.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


settings = _load("sandbox_settings_under_test", "sandbox-settings")


class TestStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "env"
        self.store = settings.SettingsStore(self.path)

    def test_save_creates_private_file(self):
        self.store.save(
            {
                "model": "gpt-4o",
                "api_key": "sk-test",
                "home": "/home/pn/work",
                "provider": "PollinationsAI",
                "proxy": "http://127.0.0.1:8081",
            }
        )
        mode = stat.S_IMODE(self.path.stat().st_mode)
        self.assertEqual(mode, 0o600)
        values = self.store.load()
        self.assertEqual(values["G4F_MODEL"], "gpt-4o")
        self.assertEqual(values["G4F_API_KEY"], "sk-test")
        self.assertEqual(values["SANDBOX_HOME"], "/home/pn/work")
        self.assertEqual(values["G4F_PROVIDER"], "PollinationsAI")
        self.assertEqual(values["SANDBOX_PROXY"], "http://127.0.0.1:8081")

    def test_unknown_keys_are_preserved(self):
        self.path.write_text("CUSTOM=keep\nG4F_MODEL=old\n", encoding="utf-8")
        values = self.store.load()
        values.update(
            {
                "model": "new",
                "api_key": "",
                "home": "",
                "provider": "",
                "proxy": "",
            }
        )
        self.store.save(values)
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("CUSTOM=keep", text)
        self.assertIn("G4F_MODEL=new", text)

    def test_delete_reports_whether_a_file_was_removed(self):
        self.assertFalse(self.store.delete())
        self.store.save({})
        self.assertTrue(self.store.delete())
        self.assertFalse(self.path.exists())


class TestEffectiveValues(unittest.TestCase):
    def setUp(self):
        self.saved = {}

    def test_saved_key_wins_over_ambient(self):
        self.saved = {"G4F_API_KEY": "saved"}
        with patch.dict(os.environ, {"G4F_API_KEY": "ambient"}, clear=False):
            value, source = settings._effective(
                settings.FIELD_BY_ATTR["api_key"], self.saved
            )
        self.assertEqual(value, "saved")
        self.assertEqual(source, "settings file")

    def test_empty_saved_key_disables_ambient(self):
        self.saved = {"G4F_API_KEY": ""}
        with patch.dict(os.environ, {"G4F_API_KEY": "ambient"}, clear=False):
            value, source = settings._effective(
                settings.FIELD_BY_ATTR["api_key"], self.saved
            )
        self.assertEqual(value, "")
        self.assertIn("off", source)

    def test_empty_override_key_disables_everything(self):
        self.saved = {"G4F_API_KEY": "saved"}
        with patch.dict(
            os.environ,
            {"G4F_SANDBOX_API_KEY": "", "G4F_API_KEY": "ambient"},
            clear=False,
        ):
            value, source = settings._effective(
                settings.FIELD_BY_ATTR["api_key"], self.saved
            )
        self.assertEqual(value, "")
        self.assertEqual(source, "environment")

    def test_empty_saved_model_falls_through_to_ambient(self):
        self.saved = {"G4F_MODEL": ""}
        with patch.dict(os.environ, {"G4F_MODEL": "ambient-model"}, clear=False):
            value, source = settings._effective(
                settings.FIELD_BY_ATTR["model"], self.saved
            )
        self.assertEqual(value, "ambient-model")
        self.assertEqual(source, "environment")

    def test_override_model_wins_over_file(self):
        self.saved = {"G4F_MODEL": "saved-model"}
        with patch.dict(
            os.environ, {"G4F_SANDBOX_MODEL": "override-model"}, clear=False
        ):
            value, source = settings._effective(
                settings.FIELD_BY_ATTR["model"], self.saved
            )
        self.assertEqual(value, "override-model")
        self.assertEqual(source, "environment")


class TestValidation(unittest.TestCase):
    def test_valid_values(self):
        values = {
            "model": "gpt-4o",
            "api_key": "sk-test",
            "home": "/home/pn/work",
            "provider": "PollinationsAI",
            "proxy": "socks5h://127.0.0.1:1080",
        }
        self.assertEqual(settings.validate_values(values), {})

    def test_empty_values_are_valid(self):
        self.assertEqual(
            settings.validate_values({field.attr: "" for field in settings.FIELDS}),
            {},
        )

    def test_relative_home_is_rejected(self):
        error = settings.validate_field(
            settings.FIELD_BY_ATTR["home"], "relative/path"
        )
        self.assertIn("absolute", error)

    def test_proxy_requires_a_supported_scheme(self):
        error = settings.validate_field(
            settings.FIELD_BY_ATTR["proxy"], "127.0.0.1:8081"
        )
        self.assertIn("http://", error)

    def test_provider_rejects_model_prefix(self):
        error = settings.validate_field(
            settings.FIELD_BY_ATTR["provider"], "PollinationsAI:gpt-4o"
        )
        self.assertIn("name only", error)

    def test_key_rejects_whitespace(self):
        error = settings.validate_field(
            settings.FIELD_BY_ATTR["api_key"], "sk test"
        )
        self.assertIn("whitespace", error)

    def test_check_settings_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "env"
            path.write_text("SANDBOX_HOME=relative\n", encoding="utf-8")
            store = settings.SettingsStore(path)
            with patch("sys.stderr"):
                self.assertEqual(settings.check_settings(store), 1)
            path.write_text("SANDBOX_HOME=/home/pn/work\n", encoding="utf-8")
            with patch("sys.stdout"):
                self.assertEqual(settings.check_settings(store), 0)


class TestDraftPrefill(unittest.TestCase):
    def test_ambient_key_is_kept_when_saving_a_new_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = settings.SettingsStore(Path(tmp) / "env")
            with patch.dict(os.environ, {"G4F_API_KEY": "ambient-key"}, clear=False):
                app = settings.SettingsApp(store)
            self.assertEqual(app.draft["api_key"], "ambient-key")

    def test_explicitly_empty_saved_key_stays_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "env"
            path.write_text("G4F_API_KEY=\n", encoding="utf-8")
            store = settings.SettingsStore(path)
            with patch.dict(os.environ, {"G4F_API_KEY": "ambient-key"}, clear=False):
                app = settings.SettingsApp(store)
            self.assertEqual(app.draft["api_key"], "")

    def test_generated_keys_are_random_and_url_safe(self):
        first = settings.generate_key()
        second = settings.generate_key()
        self.assertGreaterEqual(len(first), 40)
        self.assertNotEqual(first, second)
        self.assertTrue(first.replace("-", "").replace("_", "").isalnum())


class _ModelsHandler(BaseHTTPRequestHandler):
    expected_key = "sk-ok"

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path != "/v1/models":
            self.send_response(404)
            self.end_headers()
            return
        if self.headers.get("Authorization") != "Bearer " + self.expected_key:
            self.send_response(401)
            self.end_headers()
            return
        body = json.dumps({"data": [{"id": "auto"}, {"id": "gpt-4o"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class TestProbeServer(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.base = "http://127.0.0.1:%s/v1" % self.server.server_port

    def test_successful_probe_returns_model_count(self):
        base, count, error = settings.probe_server("sk-ok", self.base)
        self.assertEqual(base, self.base)
        self.assertEqual(count, 2)
        self.assertIsNone(error)

    def test_rejected_key_is_reported(self):
        base, count, error = settings.probe_server("wrong", self.base)
        self.assertEqual(base, self.base)
        self.assertIsNone(count)
        self.assertIn("rejected", error)

    def test_unreachable_server_is_reported(self):
        base, count, error = settings.probe_server("sk-ok", "http://127.0.0.1:1/v1")
        self.assertIsNone(base)
        self.assertIsNone(count)
        self.assertTrue(error)


class TestTraySettings(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "env"

    def _module(self):
        return _load("sandbox_tray_under_test", "sandbox-tray")

    def test_server_env_uses_saved_key_model_provider_and_proxy(self):
        self.path.write_text(
            "G4F_API_KEY=sk-saved\n"
            "G4F_MODEL=gpt-4o\n"
            "G4F_PROVIDER=PollinationsAI\n"
            "SANDBOX_PROXY=http://127.0.0.1:8081\n",
            encoding="utf-8",
        )
        module = self._module()
        with patch.dict(
            os.environ,
            {"G4F_SANDBOX_ENV": str(self.path)},
            clear=False,
        ):
            for key in (
                "G4F_SANDBOX_API_KEY",
                "G4F_SANDBOX_MODEL",
                "G4F_SANDBOX_PROVIDER",
                "G4F_SANDBOX_PROXY",
                "G4F_API_KEY",
                "G4F_MODEL",
                "G4F_PROVIDER",
                "G4F_PROXY",
            ):
                os.environ.pop(key, None)
            env = module._server_env()
        self.assertEqual(env["G4F_API_KEY"], "sk-saved")
        self.assertEqual(env["G4F_MODEL"], "gpt-4o")
        self.assertEqual(env["G4F_PROVIDER"], "PollinationsAI")
        self.assertEqual(env["G4F_PROXY"], "http://127.0.0.1:8081")

    def test_empty_saved_key_suppresses_an_ambient_key(self):
        self.path.write_text("G4F_API_KEY=\n", encoding="utf-8")
        module = self._module()
        with patch.dict(
            os.environ,
            {"G4F_SANDBOX_ENV": str(self.path), "G4F_API_KEY": "ambient"},
            clear=False,
        ):
            os.environ.pop("G4F_SANDBOX_API_KEY", None)
            self.assertEqual(module._api_key(), "")

    def test_settings_token_changes_when_the_file_changes(self):
        module = self._module()
        with patch.dict(os.environ, {"G4F_SANDBOX_ENV": str(self.path)}):
            before = module._settings_token()
            self.path.write_text("G4F_MODEL=a\n", encoding="utf-8")
            after = module._settings_token()
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
