"""
Tests for parallel CDP tab support.

Verifies that multiple CDPSession instances can share a
single browser process concurrently without killing each other when one
tab closes.
"""

import asyncio
import json
import os
import time
import urllib.request
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from g4f.requests.cdp import (
    CDPSession,
    acquire_shared_browser_ref,
    release_shared_browser_ref,
    get_shared_browser,
    _shared_browser_lock,
)
import g4f.requests.cdp as cdp_module


def _force_terminate_shared_browser():
    """Helper: cancel idle timer and terminate the shared browser immediately."""
    with _shared_browser_lock:
        if cdp_module._shared_browser_idle_timer:
            cdp_module._shared_browser_idle_timer.cancel()
            cdp_module._shared_browser_idle_timer = None
        cdp_module._shared_browser_refcount = 0
        if cdp_module._shared_browser_process:
            try:
                cdp_module._shared_browser_process.terminate()
                cdp_module._shared_browser_process.wait(timeout=5)
            except Exception:
                pass
            cdp_module._shared_browser_process = None
        cdp_module._shared_browser_port = None
        cdp_module._shared_browser_adopted = False


def _browser_alive(host: str, port: int) -> bool:
    """Check if the shared browser is reachable on the given port."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/json", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _count_targets(host: str, port: int) -> int:
    """Count open page-type targets on the browser."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/json", timeout=2) as r:
            targets = json.loads(r.read().decode("utf-8"))
            return sum(1 for t in targets if t.get("type") == "page")
    except Exception:
        return 0


class TestRefcount(unittest.TestCase):
    """Unit tests for the reference-counting helpers (no browser needed)."""

    def test_acquire_increment(self):
        before = cdp_module._shared_browser_refcount
        acquire_shared_browser_ref()
        self.assertEqual(cdp_module._shared_browser_refcount, before + 1)
        # Clean up
        release_shared_browser_ref()
        self.assertEqual(cdp_module._shared_browser_refcount, before)

    def test_release_returns_true_at_zero(self):
        # Ensure we start from a known state
        with _shared_browser_lock:
            cdp_module._shared_browser_refcount = 0
        self.assertTrue(release_shared_browser_ref())  # 0 -> 0, returns True
        # Should not go negative
        self.assertEqual(cdp_module._shared_browser_refcount, 0)

    def test_release_returns_false_when_others_active(self):
        with _shared_browser_lock:
            cdp_module._shared_browser_refcount = 0
        acquire_shared_browser_ref()
        acquire_shared_browser_ref()
        self.assertFalse(release_shared_browser_ref())  # 2 -> 1, not last
        self.assertTrue(release_shared_browser_ref())   # 1 -> 0, last ref
        # Should not go negative
        self.assertEqual(cdp_module._shared_browser_refcount, 0)


class TestBrowserIdleShutdown(unittest.TestCase):
    def setUp(self):
        _force_terminate_shared_browser()

    def tearDown(self):
        _force_terminate_shared_browser()

    def test_idle_timeout_configuration(self):
        for value, expected in (("", 60), ("2.5", 2.5), ("0", 0),
                                ("-1", 60), ("nan", 60), ("invalid", 60)):
            with self.subTest(value=value), patch.dict(os.environ, {"G4F_BROWSER_IDLE_TIMEOUT": value}):
                with patch.object(cdp_module.threading, "Timer") as timer:
                    release_shared_browser_ref()
                    self.assertEqual(timer.call_args.args[0], expected)

    def test_idle_shutdown_ignores_stale_timer_and_active_session(self):
        timer = MagicMock()
        with patch.object(cdp_module, "_close_browser_via_cdp") as close:
            cdp_module._shared_browser_port = 1234
            cdp_module._shared_browser_idle_timer = timer
            cdp_module._close_shared_browser_idle()
            close.assert_not_called()
            with patch.object(cdp_module.threading, "current_thread", return_value=timer):
                cdp_module._shared_browser_refcount = 1
                cdp_module._close_shared_browser_idle()
                close.assert_not_called()
                cdp_module._shared_browser_refcount = 0
                close.return_value = True
                cdp_module._close_shared_browser_idle()
                close.assert_called_once_with("127.0.0.1", 1234)
                self.assertIsNone(cdp_module._shared_browser_port)

    def test_idle_shutdown_leaves_external_browser_running(self):
        timer = MagicMock()
        cdp_module._shared_browser_port = 1234
        cdp_module._shared_browser_adopted = True
        cdp_module._shared_browser_idle_timer = timer
        with patch.object(cdp_module.threading, "current_thread", return_value=timer), \
             patch.object(cdp_module, "_close_browser_via_cdp") as close:
            cdp_module._close_shared_browser_idle()
            close.assert_not_called()

    def test_idle_shutdown_reaps_owned_process(self):
        timer = MagicMock()
        process = MagicMock()
        cdp_module._shared_browser_port = 1234
        cdp_module._shared_browser_process = process
        cdp_module._shared_browser_idle_timer = timer
        with patch.object(cdp_module.threading, "current_thread", return_value=timer), \
             patch.object(cdp_module, "_close_browser_via_cdp", return_value=True):
            cdp_module._close_shared_browser_idle()
        process.wait.assert_called_once_with(timeout=10)
        self.assertIsNone(cdp_module._shared_browser_process)


class TestBrowserStartupCleanup(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        _force_terminate_shared_browser()

    async def asyncTearDown(self):
        _force_terminate_shared_browser()

    async def test_failed_connection_releases_reference_once(self):
        session = CDPSession(port=1234)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "id": "test", "webSocketDebuggerUrl": "ws://localhost"
        }).encode()
        with patch.object(cdp_module.urllib.request, "urlopen", return_value=response), \
             patch.object(session, "connect", new_callable=AsyncMock, side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                await session.start()
            self.assertEqual(cdp_module._shared_browser_refcount, 0)
            self.assertIsNotNone(cdp_module._shared_browser_idle_timer)
            await session.close()
            self.assertEqual(cdp_module._shared_browser_refcount, 0)

    async def test_navigation_failure_closes_tab(self):
        from g4f.requests.cdp_browser import CDPBrowser

        session = MagicMock()
        session.start = AsyncMock()
        session.navigate = AsyncMock(side_effect=RuntimeError("navigation failed"))
        session.close = AsyncMock()
        browser = CDPBrowser()
        with patch("g4f.requests.cdp_browser.CDPSession", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "navigation failed"):
                await browser.get("https://example.com")
        session.close.assert_awaited_once()
        self.assertEqual(browser._tabs, [])

    async def test_cancelled_start_releases_reference(self):
        session = CDPSession(port=1234)
        with patch.object(cdp_module.urllib.request, "urlopen", side_effect=OSError), \
             patch.object(cdp_module.asyncio, "sleep", new_callable=AsyncMock,
                          side_effect=asyncio.CancelledError):
            with self.assertRaises(asyncio.CancelledError):
                await session.start()
        self.assertEqual(cdp_module._shared_browser_refcount, 0)
        self.assertIsNotNone(cdp_module._shared_browser_idle_timer)

    async def test_openai_auth_closes_tab_on_success_and_failure(self):
        from g4f.Provider.needs_auth.OpenaiChat import OpenaiChat

        for error in (None, RuntimeError("authentication failed")):
            with self.subTest(error=error):
                session = MagicMock()
                session.start = AsyncMock()
                session.navigate = AsyncMock()
                session.close = AsyncMock()
                with patch("g4f.requests.has_cdp", True), \
                     patch("g4f.requests.cdp_browser.CDPSession", return_value=session), \
                     patch.object(OpenaiChat, "_nodriver_auth_page",
                                  new_callable=AsyncMock, side_effect=error):
                    if error:
                        with self.assertRaisesRegex(RuntimeError, "authentication failed"):
                            await OpenaiChat.nodriver_auth()
                    else:
                        await OpenaiChat.nodriver_auth()
                session.close.assert_awaited_once()


@unittest.skipUnless(
    __import__("shutil").which("google-chrome")
    or __import__("shutil").which("chromium")
    or __import__("shutil").which("chromium-browser")
    or __import__("os").path.exists("/usr/bin/google-chrome")
    or __import__("os").path.exists("/usr/bin/chromium-browser"),
    "No Chrome/Chromium executable found",
)
class TestCDPSessionParallel(unittest.TestCase):
    """Integration tests that launch a real shared browser and open multiple tabs."""

    def setUp(self):
        _force_terminate_shared_browser()

    def tearDown(self):
        _force_terminate_shared_browser()

    def test_two_sessions_same_browser(self):
        """Two CDPSession instances should share the same browser port."""
        async def run():
            s1 = CDPSession(headless=True)
            s2 = CDPSession(headless=True)
            await s1.start()
            await s2.start()
            try:
                self.assertEqual(s1.port, s2.port)
                self.assertIsNotNone(s1.target_id)
                self.assertIsNotNone(s2.target_id)
                self.assertNotEqual(s1.target_id, s2.target_id)
                # Both should be able to navigate independently
                await s1.navigate("about:blank")
                await s2.navigate("about:blank")
                t1 = await s1.evaluate_js("document.title")
                t2 = await s2.evaluate_js("document.title")
                # about:blank has empty title
                self.assertEqual(t1, "")
                self.assertEqual(t2, "")
                # Browser should still be alive with 2 tabs
                self.assertTrue(_browser_alive(s1.host, s1.port))
            finally:
                await s2.close()
                await s1.close()

        asyncio.run(run())

    def test_closing_one_tab_keeps_browser_alive(self):
        """Closing one session must not kill the browser while another is active."""
        async def run():
            s1 = CDPSession(headless=True)
            s2 = CDPSession(headless=True)
            await s1.start()
            await s2.start()
            port = s1.port
            host = s1.host
            try:
                self.assertTrue(_browser_alive(host, port))
                # Close the second tab
                await s2.close()
                # Browser must still be alive because s1 is still open
                self.assertTrue(_browser_alive(host, port))
                # s1 should still work
                await s1.navigate("about:blank")
                self.assertTrue(_browser_alive(host, port))
            finally:
                await s1.close()
            # Now both closed — browser stays alive (idle timer) but no tabs
            # The browser is kept for reuse, so it should still be alive
            self.assertTrue(_browser_alive(host, port))

        asyncio.run(run())

    def test_concurrent_navigation(self):
        """Multiple sessions navigating concurrently should not interfere."""
        async def run():
            sessions = [CDPSession(headless=True) for _ in range(3)]
            await asyncio.gather(*[s.start() for s in sessions])
            try:
                # All should share the same port
                ports = {s.port for s in sessions}
                self.assertEqual(len(ports), 1)
                # Navigate all to different data URLs concurrently
                async def nav(s, i):
                    await s.navigate(f"data:text/html,<title>Tab{i}</title>")
                    return await s.evaluate_js("document.title")

                titles = await asyncio.gather(
                    *[nav(s, i) for i, s in enumerate(sessions)]
                )
                for i, title in enumerate(titles):
                    self.assertEqual(title, f"Tab{i}")
            finally:
                await asyncio.gather(*[s.close() for s in sessions])

        asyncio.run(run())

    def test_browser_reused_after_all_closed(self):
        """After all tabs close, the browser stays alive and a new session reuses it."""
        async def run():
            s1 = CDPSession(headless=True)
            await s1.start()
            port1 = s1.port
            await s1.close()
            # Browser should still be alive (idle timer keeps it for reuse)
            self.assertTrue(_browser_alive("127.0.0.1", port1))

            # New session should reuse the same browser
            s2 = CDPSession(headless=True)
            await s2.start()
            try:
                self.assertEqual(s2.port, port1)
                self.assertTrue(_browser_alive(s2.host, s2.port))
                await s2.navigate("about:blank")
            finally:
                await s2.close()

        asyncio.run(run())

    def test_browser_closes_after_configured_idle_timeout(self):
        async def run():
            with patch.dict(os.environ, {"G4F_BROWSER_IDLE_TIMEOUT": "0.1"}):
                session = CDPSession(headless=True)
                await session.start()
                port = session.port
                await session.close()
                for _ in range(100):
                    if not _browser_alive("127.0.0.1", port):
                        break
                    await asyncio.sleep(0.1)
                self.assertFalse(_browser_alive("127.0.0.1", port))

        asyncio.run(run())

if __name__ == "__main__":
    unittest.main()
