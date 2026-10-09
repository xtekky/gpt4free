"""
Tests for the Lightpanda browser installed by ``g4f-go browser install``.

Verifies that the installed binary is detected in the shared config
directory and preferred over Chrome whenever headless mode is on, that the
auto-started process is shut down again (Lightpanda does not implement the
CDP ``Browser.close`` command), and that the cookies from the .har files in
the cookies directory are loaded into every new session.
"""

import asyncio
import json
import os
import tempfile
import unittest
import unittest.mock
import urllib.request

import g4f.requests.cdp as cdp_module
from g4f.cookies import _parse_har_cookies, read_har_cookies
from g4f.requests.cdp import (
    CDPSession,
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


def _write_har(path: str, entries: list):
    with open(path, "w") as f:
        json.dump({"log": {"version": "1.2", "entries": entries}}, f)


def _har_entry(url: str, cookies=None, cookie_header=None):
    headers = [{"name": "Host", "value": url.split("/")[2]}]
    if cookie_header is not None:
        headers.append({"name": "Cookie", "value": cookie_header})
    entry = {"request": {"url": url, "headers": headers}}
    if cookies is not None:
        entry["request"]["cookies"] = cookies
    return entry


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


class TestHarCookieParsing(unittest.TestCase):
    """Parsing of .har files into CDP ready cookie objects."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def _har(self, name: str, entries: list) -> str:
        path = os.path.join(self.dir.name, name)
        _write_har(path, entries)
        return path

    def test_cookie_objects_keep_attributes(self):
        path = self._har("a.har", [_har_entry(
            "https://chatgpt.com/backend-api/",
            cookies=[{
                "name": "cf_clearance",
                "value": "abc",
                "domain": ".chatgpt.com",
                "path": "/",
                "expires": "2027-03-20T15:56:44.163Z",
                "httpOnly": True,
                "secure": True,
                "sameSite": "None",
            }],
        )])
        cookies = _parse_har_cookies(path)
        self.assertEqual(len(cookies), 1)
        cookie = cookies[0]
        self.assertEqual(cookie["name"], "cf_clearance")
        self.assertEqual(cookie["domain"], ".chatgpt.com")
        self.assertEqual(cookie["path"], "/")
        self.assertTrue(cookie["httpOnly"])
        self.assertTrue(cookie["secure"])
        self.assertEqual(cookie["sameSite"], "None")
        # Lightpanda rejects ISO-8601 strings — expires must be numeric.
        self.assertIsInstance(cookie["expires"], float)
        self.assertGreater(cookie["expires"], 0)

    def test_cookie_header_is_parsed_without_cookies_array(self):
        path = self._har("b.har", [_har_entry(
            "https://chat.qwen.ai/api/v2/chat/completions",
            cookie_header="alicfw_gfver=v1.200309.1; x-ap=eu-central-1",
        )])
        cookies = {c["name"]: c for c in _parse_har_cookies(path)}
        self.assertEqual(set(cookies), {"alicfw_gfver", "x-ap"})
        self.assertEqual(cookies["x-ap"]["value"], "eu-central-1")
        self.assertEqual(cookies["x-ap"]["domain"], "chat.qwen.ai")
        self.assertEqual(cookies["x-ap"]["path"], "/")

    def test_cookies_array_wins_over_cookie_header(self):
        path = self._har("c.har", [_har_entry(
            "https://chatgpt.com/",
            cookies=[{"name": "a", "value": "1", "domain": "chatgpt.com"}],
            cookie_header="stale=1",
        )])
        cookies = {c["name"] for c in _parse_har_cookies(path)}
        self.assertEqual(cookies, {"a"})

    def test_expired_cookies_are_dropped(self):
        path = self._har("d.har", [_har_entry(
            "https://chatgpt.com/",
            cookies=[
                {"name": "old", "value": "1", "domain": "chatgpt.com",
                 "expires": "2001-01-01T00:00:00.000Z"},
                {"name": "new", "value": "2", "domain": "chatgpt.com",
                 "expires": "2035-01-01T00:00:00.000Z"},
            ],
        )])
        self.assertEqual({c["name"] for c in _parse_har_cookies(path)}, {"new"})

    def test_session_cookie_has_no_expires(self):
        path = self._har("e.har", [_har_entry(
            "https://chatgpt.com/",
            cookies=[{"name": "s", "value": "1", "domain": "chatgpt.com",
                      "expires": "-1"}],
        )])
        self.assertNotIn("expires", _parse_har_cookies(path)[0])

    def test_duplicates_across_files_are_merged(self):
        self._har("f1.har", [_har_entry(
            "https://chatgpt.com/",
            cookies=[{"name": "a", "value": "old", "domain": "chatgpt.com"}],
        )])
        self._har("f2.har", [_har_entry(
            "https://chatgpt.com/",
            cookies=[{"name": "a", "value": "new", "domain": "chatgpt.com"}],
        )])
        cookies = read_har_cookies(self.dir.name)
        self.assertEqual(len(cookies), 1)
        self.assertEqual(cookies[0]["value"], "new")

    def test_broken_har_is_ignored(self):
        with open(os.path.join(self.dir.name, "broken.har"), "w") as f:
            f.write("{not json")
        self.assertEqual(read_har_cookies(self.dir.name), [])

    def test_missing_dir_returns_empty(self):
        self.assertEqual(
            read_har_cookies(os.path.join(self.dir.name, "nope")), []
        )


class TestCookieInjection(unittest.TestCase):
    """The CDP side of the injection, without a browser."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        _write_har(os.path.join(self.dir.name, "x.har"), [_har_entry(
            "https://chatgpt.com/",
            cookies=[{"name": "a", "value": "1", "domain": "chatgpt.com"}],
        )])

    def _session(self, calls: list) -> CDPSession:
        session = CDPSession(host="127.0.0.1", port=1, headless=True)

        async def call(method, _browser_level=False, **params):
            calls.append((method, params))
            return {}

        session.call = call
        return session

    def test_inject_uses_bulk_call(self):
        calls = []
        session = self._session(calls)
        count = asyncio.run(session.inject_har_cookies(self.dir.name))
        self.assertEqual(count, 1)
        self.assertEqual([m for m, _ in calls], ["Network.setCookies"])
        self.assertEqual(calls[0][1]["cookies"][0]["name"], "a")

    def test_inject_falls_back_to_single_calls(self):
        calls = []

        async def call(method, _browser_level=False, **params):
            calls.append(method)
            if method == "Network.setCookies":
                raise RuntimeError("not supported")
            return {}

        session = CDPSession(host="127.0.0.1", port=1, headless=True)
        session.call = call
        count = asyncio.run(session.inject_har_cookies(self.dir.name))
        self.assertEqual(count, 1)
        self.assertEqual(calls, ["Network.setCookies", "Network.setCookie"])

    def test_inject_without_har_files_is_a_noop(self):
        calls = []
        session = self._session(calls)
        empty = tempfile.TemporaryDirectory()
        self.addCleanup(empty.cleanup)
        self.assertEqual(asyncio.run(session.inject_har_cookies(empty.name)), 0)
        self.assertEqual(calls, [])


@unittest.skipUnless(LIGHTPANDA_PATH, "Lightpanda is not installed")
class TestLightpandaCookies(unittest.TestCase):
    """End-to-end: cookies from .har files reach a live Lightpanda session."""

    def setUp(self):
        _reset_shared_browser()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        _write_har(os.path.join(self.dir.name, "live.har"), [_har_entry(
            "https://chat.qwen.ai/",
            cookie_header="g4f_test_cookie=hello",
        )])
        self._patch = unittest.mock.patch(
            "g4f.cookies.CookiesConfig.cookies_dir", self.dir.name
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def tearDown(self):
        _reset_shared_browser()

    def _run(self, coro):
        """Run a coroutine that owns its sessions, closing them afterwards."""
        async def main():
            sessions = []

            async def new_session(port=None):
                session = CDPSession(host="127.0.0.1", port=port, headless=True)
                sessions.append(session)
                await session.start()
                return session

            try:
                return await coro(new_session)
            finally:
                for session in sessions:
                    try:
                        await session.close()
                    except Exception:
                        pass

        return asyncio.run(main())

    async def _cookies(self, session: CDPSession) -> dict:
        result = await session.call("Network.getAllCookies")
        return {c["name"]: c["value"] for c in result.get("cookies", [])}

    def test_cookies_are_injected_on_start(self):
        async def scenario(new_session):
            session = await new_session()
            return await self._cookies(session)

        self.assertEqual(self._run(scenario).get("g4f_test_cookie"), "hello")

    def test_every_session_gets_the_cookies(self):
        """Lightpanda keeps its cookie store per connection."""
        async def scenario(new_session):
            first = await new_session()
            second = await new_session(first.port)
            return await self._cookies(second)

        self.assertEqual(self._run(scenario).get("g4f_test_cookie"), "hello")

    def test_cookie_is_sent_to_the_site(self):
        async def scenario(new_session):
            session = await new_session()
            await session.call("Page.navigate", url="https://chat.qwen.ai/")
            return await session.evaluate_js("document.cookie")

        self.assertIn("g4f_test_cookie=hello", self._run(scenario) or "")


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
