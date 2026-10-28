from __future__ import annotations

import os
import socket
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from g4f.api import sandbox_apps
from g4f.api.local_gateway_proxy import LocalGatewayProxy
from g4f.api.sandbox_apps import (
    autostart_enabled,
    cli_path,
    ensure_app_running,
    is_running,
    is_starting,
    start_app,
    starting_response,
    tool_for,
)

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

class _FakeProcess:
    """Stand-in for subprocess.Popen in the autostart bookkeeping."""

    def __init__(self, pid: int = 4242, running: bool = True) -> None:
        self.pid = pid
        self._running = running

    def poll(self):
        return None if self._running else 0

class TestAutostartConfig(unittest.TestCase):
    def test_enabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("G4F_DOMAIN_AUTOSTART", None)
            self.assertTrue(autostart_enabled())

    def test_disabled_values(self):
        for value in ("0", "false", "no", "off", "OFF"):
            with patch.dict(os.environ, {"G4F_DOMAIN_AUTOSTART": value}):
                self.assertFalse(autostart_enabled(), value)

    def test_enabled_values(self):
        for value in ("1", "true", "yes", "on"):
            with patch.dict(os.environ, {"G4F_DOMAIN_AUTOSTART": value}):
                self.assertTrue(autostart_enabled(), value)

    def test_tool_for_known_apps(self):
        for name in ("omniroute", "ollama", "litellm", "remote-desktop"):
            self.assertEqual(tool_for(name), name)

    def test_tool_for_apps_without_a_tool(self):
        # g4f is the API itself and desktop is the noVNC client in the image.
        for name in ("g4f", "desktop", "evil", ""):
            self.assertIsNone(tool_for(name))

    def test_cli_path_prefers_the_image_location(self):
        with patch("g4f.api.sandbox_apps.os.path.isfile", return_value=True), patch(
            "g4f.api.sandbox_apps.os.access", return_value=True
        ):
            self.assertEqual(cli_path(), sandbox_apps.CLI_PATH)

    def test_cli_path_falls_back_to_path_lookup(self):
        with patch("g4f.api.sandbox_apps.os.path.isfile", return_value=False), patch(
            "g4f.api.sandbox_apps.shutil.which", return_value="/opt/bin/sandbox-cli"
        ):
            self.assertEqual(cli_path(), "/opt/bin/sandbox-cli")

    def test_cli_path_is_none_outside_the_sandbox(self):
        with patch("g4f.api.sandbox_apps.os.path.isfile", return_value=False), patch(
            "g4f.api.sandbox_apps.shutil.which", return_value=None
        ):
            self.assertIsNone(cli_path())

class TestReachability(unittest.TestCase):
    def test_is_running_for_a_listening_port(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            self.assertTrue(is_running(f"http://127.0.0.1:{port}"))

    def test_is_running_for_a_closed_port(self):
        self.assertFalse(is_running(f"http://127.0.0.1:{_free_port()}"))

    def test_is_running_without_a_host(self):
        self.assertFalse(is_running("not-a-url"))

    def test_is_starting_tracks_the_process(self):
        sandbox_apps._processes.clear()
        self.addCleanup(sandbox_apps._processes.clear)
        sandbox_apps._processes["litellm"] = _FakeProcess()
        self.assertTrue(is_starting("litellm"))
        sandbox_apps._processes["litellm"] = _FakeProcess(running=False)
        self.assertFalse(is_starting("litellm"))
        # A finished process is forgotten, so the next request can start it.
        self.assertNotIn("litellm", sandbox_apps._processes)

class TestTerminalArgv(unittest.TestCase):
    def test_no_terminal_without_a_display(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPLAY", None)
            self.assertIsNone(
                sandbox_apps.terminal_argv(
                    "/usr/local/bin/sandbox-cli", "litellm", "LiteLLM"
                )
            )

    def test_no_terminal_when_xfce4_terminal_is_missing(self):
        with patch.dict(os.environ, {"DISPLAY": ":0"}), patch(
            "g4f.api.sandbox_apps.shutil.which", return_value=None
        ):
            self.assertIsNone(
                sandbox_apps.terminal_argv(
                    "/usr/local/bin/sandbox-cli", "litellm", "LiteLLM"
                )
            )

    def test_terminal_runs_the_cli_in_a_visible_window(self):
        with patch.dict(os.environ, {"DISPLAY": ":0"}), patch(
            "g4f.api.sandbox_apps.shutil.which", return_value="/usr/bin/xfce4-terminal"
        ):
            argv = sandbox_apps.terminal_argv(
                "/usr/local/bin/sandbox-cli", "litellm", "LiteLLM"
            )
        self.assertEqual(argv[0], "/usr/bin/xfce4-terminal")
        self.assertIn("--disable-server", argv)
        self.assertIn("--title=LiteLLM", argv)
        self.assertIn("--icon=g4f-sandbox-litellm", argv)
        self.assertIn("--command=/usr/local/bin/sandbox-cli litellm", argv)

class TestStartApp(unittest.TestCase):
    def setUp(self):
        sandbox_apps._processes.clear()
        self.addCleanup(sandbox_apps._processes.clear)

    def test_spawns_the_cli_detached(self):
        popen = MagicMock(return_value=_FakeProcess())
        with patch("g4f.api.sandbox_apps.cli_path", return_value="/usr/local/bin/sandbox-cli"), patch(
            "g4f.api.sandbox_apps.terminal_argv", return_value=None
        ), patch("g4f.api.sandbox_apps.subprocess.Popen", popen):
            self.assertTrue(start_app("litellm", "LiteLLM"))
        args, kwargs = popen.call_args
        self.assertEqual(args[0], ["/usr/local/bin/sandbox-cli", "litellm"])
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["stdin"], sandbox_apps.subprocess.DEVNULL)
        self.assertTrue(is_starting("litellm"))

    def test_spawns_the_cli_in_a_desktop_terminal(self):
        argv = [
            "/usr/bin/xfce4-terminal",
            "--disable-server",
            "--command=/usr/local/bin/sandbox-cli litellm",
        ]
        popen = MagicMock(return_value=_FakeProcess())
        with patch("g4f.api.sandbox_apps.cli_path", return_value="/usr/local/bin/sandbox-cli"), patch(
            "g4f.api.sandbox_apps.terminal_argv", return_value=argv
        ) as terminal, patch("g4f.api.sandbox_apps.subprocess.Popen", popen):
            self.assertTrue(start_app("litellm", "LiteLLM"))
        terminal.assert_called_once_with("/usr/local/bin/sandbox-cli", "litellm", "LiteLLM")
        args, kwargs = popen.call_args
        self.assertEqual(args[0], argv)
        self.assertTrue(kwargs["start_new_session"])
        self.assertEqual(kwargs["stdin"], sandbox_apps.subprocess.DEVNULL)
        self.assertTrue(is_starting("litellm"))

    def test_does_not_spawn_twice(self):
        popen = MagicMock(return_value=_FakeProcess())
        with patch("g4f.api.sandbox_apps.cli_path", return_value="/usr/local/bin/sandbox-cli"), patch(
            "g4f.api.sandbox_apps.subprocess.Popen", popen
        ):
            self.assertTrue(start_app("litellm", "LiteLLM"))
            self.assertFalse(start_app("litellm", "LiteLLM"))
        self.assertEqual(popen.call_count, 1)

    def test_skips_apps_without_a_tool(self):
        popen = MagicMock()
        with patch("g4f.api.sandbox_apps.cli_path", return_value="/usr/local/bin/sandbox-cli"), patch(
            "g4f.api.sandbox_apps.subprocess.Popen", popen
        ):
            self.assertFalse(start_app("desktop", "Desktop"))
        popen.assert_not_called()

    def test_skips_when_the_cli_is_missing(self):
        popen = MagicMock()
        with patch("g4f.api.sandbox_apps.cli_path", return_value=None), patch(
            "g4f.api.sandbox_apps.subprocess.Popen", popen
        ):
            self.assertFalse(start_app("litellm", "LiteLLM"))
        popen.assert_not_called()

    def test_reports_a_failed_spawn(self):
        with patch("g4f.api.sandbox_apps.cli_path", return_value="/usr/local/bin/sandbox-cli"), patch(
            "g4f.api.sandbox_apps.subprocess.Popen", side_effect=OSError("boom")
        ):
            self.assertFalse(start_app("litellm", "LiteLLM"))
        self.assertFalse(is_starting("litellm"))

    def test_writes_the_log_into_the_temp_dir(self):
        popen = MagicMock(return_value=_FakeProcess())
        with patch("g4f.api.sandbox_apps.cli_path", return_value="/usr/local/bin/sandbox-cli"), patch(
            "g4f.api.sandbox_apps.subprocess.Popen", popen
        ):
            start_app("litellm", "LiteLLM")
        log_path = os.path.join(tempfile.gettempdir(), "g4f-autostart-litellm.log")
        self.assertTrue(os.path.exists(log_path))
        os.remove(log_path)

class TestEnsureAppRunning(unittest.IsolatedAsyncioTestCase):
    async def test_returns_true_when_the_app_is_up(self):
        with patch("g4f.api.sandbox_apps.is_running", return_value=True), patch(
            "g4f.api.sandbox_apps.start_app"
        ) as start:
            self.assertTrue(
                await ensure_app_running("litellm", "LiteLLM", lambda: "http://x")
            )
        start.assert_not_called()

    async def test_returns_false_when_autostart_is_disabled(self):
        with patch.dict(os.environ, {"G4F_DOMAIN_AUTOSTART": "0"}), patch(
            "g4f.api.sandbox_apps.is_running", return_value=False
        ), patch("g4f.api.sandbox_apps.start_app") as start:
            self.assertFalse(
                await ensure_app_running("litellm", "LiteLLM", lambda: "http://x")
            )
        start.assert_not_called()

    async def test_returns_false_for_an_app_without_a_tool(self):
        with patch("g4f.api.sandbox_apps.is_running", return_value=False), patch(
            "g4f.api.sandbox_apps.start_app"
        ) as start:
            self.assertFalse(
                await ensure_app_running("desktop", "Desktop", lambda: "http://x")
            )
        start.assert_not_called()

    async def test_returns_true_once_the_app_comes_up(self):
        probes = iter([False, False, True])
        with patch("g4f.api.sandbox_apps.is_running", side_effect=lambda url: next(probes)), patch(
            "g4f.api.sandbox_apps.start_app", return_value=True
        ) as start, patch("g4f.api.sandbox_apps.asyncio.sleep"):
            self.assertTrue(
                await ensure_app_running("litellm", "LiteLLM", lambda: "http://x")
            )
        start.assert_called_once()

    async def test_returns_false_when_the_cli_cannot_be_spawned(self):
        with patch("g4f.api.sandbox_apps.is_running", return_value=False), patch(
            "g4f.api.sandbox_apps.start_app", return_value=False
        ), patch("g4f.api.sandbox_apps.asyncio.sleep") as sleep:
            self.assertFalse(
                await ensure_app_running("litellm", "LiteLLM", lambda: "http://x")
            )
        sleep.assert_not_called()

    async def test_waits_for_an_install_that_is_already_running(self):
        with patch("g4f.api.sandbox_apps.is_running", return_value=False), patch(
            "g4f.api.sandbox_apps.is_starting", return_value=True
        ), patch("g4f.api.sandbox_apps.start_app") as start, patch(
            "g4f.api.sandbox_apps.START_TIMEOUT", 0.0
        ):
            self.assertFalse(
                await ensure_app_running("litellm", "LiteLLM", lambda: "http://x")
            )
        start.assert_not_called()

    async def test_returns_false_when_the_app_never_comes_up(self):
        with patch("g4f.api.sandbox_apps.is_running", return_value=False), patch(
            "g4f.api.sandbox_apps.start_app", return_value=True
        ), patch("g4f.api.sandbox_apps.START_TIMEOUT", 0.0):
            self.assertFalse(
                await ensure_app_running("litellm", "LiteLLM", lambda: "http://x")
            )

class TestStartingResponse(unittest.TestCase):
    def test_payload(self):
        response = starting_response("litellm", "LiteLLM")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers["retry-after"], str(sandbox_apps.RETRY_AFTER))
        body = response.body.decode()
        self.assertIn("litellm_starting", body)
        self.assertIn("sandbox-cli litellm", body)

class TestProxyAutostart(unittest.TestCase):
    """The proxy answers "starting" while an app is installed and started."""

    def _client(self, app: str | None):
        proxy = LocalGatewayProxy(
            "/litellm",
            lambda: f"http://127.0.0.1:{_free_port()}",
            label="LiteLLM",
            app=app,
        )
        api = FastAPI()
        proxy.register(api)
        return TestClient(api)

    def test_reports_unavailable_without_an_app_name(self):
        response = self._client(None).get("/litellm/v1/models")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["type"], "litellm_unavailable")

    def test_reports_starting_while_the_install_runs(self):
        with patch(
            "g4f.api.local_gateway_proxy.ensure_app_running", return_value=False
        ), patch("g4f.api.local_gateway_proxy.is_starting", return_value=True):
            response = self._client("litellm").get("/litellm/v1/models")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["type"], "litellm_starting")
        self.assertEqual(response.headers["retry-after"], str(sandbox_apps.RETRY_AFTER))

    def test_reports_unavailable_when_the_app_cannot_be_started(self):
        with patch(
            "g4f.api.local_gateway_proxy.ensure_app_running", return_value=False
        ), patch("g4f.api.local_gateway_proxy.is_starting", return_value=False):
            response = self._client("litellm").get("/litellm/v1/models")
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["type"], "litellm_unavailable")

    def test_serves_the_request_when_the_app_comes_up(self):
        upstream = FastAPI()

        @upstream.get("/v1/models")
        async def models():
            return {"data": []}

        import threading
        import time

        import uvicorn

        port = _free_port()
        server = uvicorn.Server(
            uvicorn.Config(upstream, host="127.0.0.1", port=port, log_level="error")
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        try:
            for _ in range(100):
                if server.started:
                    break
                time.sleep(0.05)
            self.assertTrue(server.started, "upstream server did not start")

            # The app is down for the first attempt; starting it makes the
            # retried request reach the now-running upstream.
            target = {"port": _free_port()}

            def start(name, label, root_url):
                target["port"] = port
                return True

            proxy = LocalGatewayProxy(
                "/litellm",
                lambda: f"http://127.0.0.1:{target['port']}",
                label="LiteLLM",
                app="litellm",
            )
            api = FastAPI()
            proxy.register(api)
            with patch(
                "g4f.api.local_gateway_proxy.ensure_app_running", side_effect=start
            ) as ensure:
                response = TestClient(api).get("/litellm/v1/models")
            ensure.assert_called_once()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"data": []})
        finally:
            server.should_exit = True
            thread.join(timeout=5)