"""
Tests for parallel CDP tab support.

Verifies that multiple CDPSession instances can share a
single browser process concurrently without killing each other when one
tab closes.
"""

import asyncio
import json
import time
import urllib.request
import unittest

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

if __name__ == "__main__":
    unittest.main()
