from __future__ import annotations

"""
Aliyun Captcha V3 solver for the GLM (z.ai) provider.

Faithful Python port of the TypeScript reference implementation
``providers/glm/captcha-solver.ts``.

Loads the AliyunCaptcha.js SDK into a headless Chromium page with stealth
mitigations, then calls ``startTracelessVerification()`` to obtain a
``captcha_verify_param``. Tokens are cached for 45 seconds and the browser
instance is reused across solves.

The gpt4free stack standardises on ``zendriver`` (aliased as ``nodriver``) for
all browser automation, so this module uses the same CDP-based driver via
``get_nodriver`` instead of pulling in Playwright as an extra dependency.
Request interception is performed with the CDP ``Fetch`` domain, which is the
CDP equivalent of Playwright's ``page.route()`` used by the TS reference.
"""

import asyncio
import base64
import time
from pathlib import Path
from typing import Optional

from ... import debug

try:
    import zendriver as nodriver
    from zendriver import cdp
    has_nodriver = True
except ImportError:
    has_nodriver = False

from ...requests import get_nodriver


# z.ai Aliyun Captcha config (from window.AliyunCaptchaConfig on the page)
CAPTCHA_CONFIG = {
    "region": "sgp",
    "prefix": "no8xfe",
    "sceneId": "didk33e0",
}

# Bundled copy of the Aliyun captcha SDK — embedded directly in the page HTML,
# exactly like the TS reference (NOT loaded from alicdn via <script src>).
_BUNDLED_SDK_PATH = Path(__file__).parent / "AliyunCaptcha.js.txt"

TOKEN_TTL_S = 45           # captcha_verify_param is cached for 45 seconds
SOLVE_RETRIES = 3
SOLVE_TIMEOUT_MS = 40_000  # hard cap for a single solve attempt
SDK_LOAD_TIMEOUT_MS = 20_000

LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-blink-features=AutomationControlled",
    "--disable-features=ChromeWhatsNewUI",
    "--no-first-run",
    "--no-default-browser-check",
]

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
)

# Stealth init script — hides automation signals from the captcha SDK.
# Identical to the TS reference.
STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
window.chrome = { runtime: {}, loadTimes: () => ({}), csi: () => ({}), app: {} };
Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });
Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });
Object.defineProperty(navigator, 'maxTouchPoints', { get: () => 0 });
"""

# In-process cache for the solved captcha token.
_cached_token: dict = {"verify_param": None, "expires_at": 0.0}
# Serialises concurrent solves so we never launch more than one browser solve
# at the same time for the same process.
_solve_lock: Optional[asyncio.Lock] = None
# Reused browser instance (launched once, kept alive across solves).
_browser: Optional[object] = None
_browser_lock: Optional[asyncio.Lock] = None


def _get_solve_lock() -> asyncio.Lock:
    """Return a process-wide solve lock, creating it lazily."""
    global _solve_lock
    if _solve_lock is None:
        _solve_lock = asyncio.Lock()
    return _solve_lock


def _get_browser_lock() -> asyncio.Lock:
    """Return a process-wide browser lock, creating it lazily."""
    global _browser_lock
    if _browser_lock is None:
        _browser_lock = asyncio.Lock()
    return _browser_lock


def _load_bundled_sdk() -> str:
    """Return the bundled Aliyun captcha SDK source.

    The SDK is embedded directly in the page HTML (not loaded from alicdn),
    matching the TS reference exactly.
    """
    return _BUNDLED_SDK_PATH.read_text(encoding="utf-8")


def _build_page_html() -> str:
    """Build the minimal HTML page that hosts the Aliyun captcha SDK.

    The SDK is embedded directly via ``<script>...</script>`` — NOT loaded from
    alicdn via ``<script src>``. This matches the TS reference exactly and
    ensures the SDK's traceless verification is tied to the chat.z.ai origin
    served via request interception.
    """
    sdk = _load_bundled_sdk()
    # Escape </script> in the SDK source so it doesn't break the HTML.
    safe_sdk = sdk.replace("</script>", "<\\/script>")
    return f"""<!DOCTYPE html><html><head></head><body>
<div id="captcha-element"></div>
<button id="captcha-button"></button>
<script>{safe_sdk}</script>
</body></html>"""


async def _get_browser():
    """Return a reused browser instance, launching it once on first call.

    Mirrors the TS reference's ``getBrowser()`` which keeps a single
    ``browserPromise`` and reuses it across solves.
    """
    global _browser
    async with _get_browser_lock():
        if _browser is not None:
            try:
                # Check if the browser is still connected by opening a tab.
                test_tab = await _browser.get("about:blank")
                await test_tab.close()
                return _browser
            except Exception:
                _browser = None

        browser, _stop = await get_nodriver(
            user_data_dir="glm-captcha",
            browser_args=LAUNCH_ARGS,
        )
        _browser = browser
        return _browser


async def _intercept_and_fulfill(page, page_url: str, html: str) -> None:
    """Intercept navigation to ``page_url`` and serve ``html`` for the document.

    Uses the CDP ``Fetch`` domain. Only the top-level document request is
    fulfilled; every other request is allowed to continue to the network.

    This is the CDP equivalent of Playwright's ``page.route()`` used by the
    TS reference.
    """
    await page.send(cdp.fetch.enable(
        patterns=[cdp.fetch.RequestPattern(
            request_stage=cdp.fetch.RequestStage.REQUEST,
        )],
    ))

    async def on_request_paused(event: cdp.fetch.RequestPaused, page=None):
        request = event.request
        url = request.url
        # Only fulfil the top-level document navigation to chat.z.ai.
        is_document = (url == page_url or url == page_url.rstrip("/"))
        if is_document and event.resource_type == cdp.network.ResourceType.DOCUMENT:
            await page.send(cdp.fetch.fulfill_request(
                request_id=event.request_id,
                response_code=200,
                response_headers=[
                    cdp.fetch.HeaderEntry(name="Content-Type", value="text/html; charset=utf-8"),
                ],
                body=base64.b64encode(html.encode("utf-8")).decode("ascii"),
            ))
        else:
            await page.send(cdp.fetch.continue_request(request_id=event.request_id))

    page.add_handler(cdp.fetch.RequestPaused, on_request_paused)


async def _solve_once() -> str:
    """Solve the captcha once and return the captcha_verify_param string.

    Mirrors ``solveInBrowser()`` from the TS reference.
    """
    browser = await _get_browser()
    page_url = "https://chat.z.ai/"
    html = _build_page_html()

    # Open a new tab for this solve.
    tab = await browser.get("about:blank")

    try:
        # Install the Fetch interceptor before navigating so we catch the
        # initial document request.
        await _intercept_and_fulfill(tab, page_url, html)

        # Apply stealth mitigations before any page script runs.
        await tab.evaluate(STEALTH_INIT_SCRIPT, await_promise=False)

        # Navigate to the intercepted page.
        await tab.get(page_url)

        # Set the captcha config the same way the z.ai bundle does.
        await tab.evaluate(
            f"window.AliyunCaptchaConfig = {{region: {CAPTCHA_CONFIG['region']!r}, "
            f"prefix: {CAPTCHA_CONFIG['prefix']!r}}};",
            await_promise=False,
        )

        # Wait for the SDK to expose initAliyunCaptcha.
        # The SDK is embedded directly in the HTML, so it should be available
        # immediately after the page loads.
        waited = 0
        while True:
            ready = await tab.evaluate(
                "typeof window.initAliyunCaptcha === 'function'",
                await_promise=False,
            )
            if ready:
                break
            if waited >= SDK_LOAD_TIMEOUT_MS:
                raise TimeoutError("Aliyun captcha SDK failed to load")
            await asyncio.sleep(0.5)
            waited += 500

        # Solve the captcha — identical JS to the TS reference.
        cfg = CAPTCHA_CONFIG
        solve_js = """
        (async (cfg) => {
            return new Promise((resolve, reject) => {
                const timeout = setTimeout(
                    () => reject(new Error('Captcha solve timeout after ' + cfg.timeout + 'ms')),
                    cfg.timeout
                );
                window.initAliyunCaptcha({
                    SceneId: cfg.sceneId,
                    mode: 'popup',
                    region: cfg.region,
                    prefix: cfg.prefix,
                    language: 'en',
                    element: '#captcha-element',
                    button: '#captcha-button',
                    captchaLogoImg: '',
                    showErrorTip: false,
                    success: (param) => { clearTimeout(timeout); resolve(param); },
                    fail: (err) => { clearTimeout(timeout); reject(new Error('SDK fail: ' + JSON.stringify(err))); },
                    getInstance: (inst) => { inst.startTracelessVerification(); }
                });
            });
        })
        """

        param = await tab.evaluate(
            f"{solve_js}({{'region': {cfg['region']!r}, 'prefix': {cfg['prefix']!r}, "
            f"'sceneId': {cfg['sceneId']!r}, 'timeout': {SOLVE_TIMEOUT_MS}}})",
            await_promise=True,
        )
        if not isinstance(param, str) or not param:
            raise RuntimeError(f"Captcha solver returned an invalid token: {param!r}")
        debug.log("GLM captcha solved successfully")
        return param
    finally:
        # Close the tab but keep the browser alive for reuse.
        try:
            await tab.close()
        except Exception:
            pass


async def _solve_with_retry() -> str:
    """Solve the captcha with retry, matching SOLVE_RETRIES attempts."""
    last_err: Optional[Exception] = None
    for attempt in range(1, SOLVE_RETRIES + 1):
        try:
            debug.log(f"GLM captcha: solve attempt {attempt}/{SOLVE_RETRIES}")
            return await _solve_once()
        except Exception as err:
            last_err = err
            debug.log(f"GLM captcha: attempt {attempt} failed: {err}")
            if attempt < SOLVE_RETRIES:
                await asyncio.sleep(1)
    raise TimeoutError(
        f"GLM captcha solving failed after {SOLVE_RETRIES} attempts: {last_err}"
    ) from last_err


async def get_captcha_verify_param() -> str:
    """Return a fresh ``captcha_verify_param`` for the GLM API.

    A valid token is cached for ``TOKEN_TTL_S`` seconds (45s). Concurrent
    callers share a single solve to avoid spawning multiple browsers.
    """
    if _cached_token["verify_param"] and _cached_token["expires_at"] > time.time():
        return _cached_token["verify_param"]

    async with _get_solve_lock():
        # Re-check inside the lock — another coroutine may have just solved it.
        if _cached_token["verify_param"] and _cached_token["expires_at"] > time.time():
            return _cached_token["verify_param"]
        verify_param = await _solve_with_retry()
        _cached_token["verify_param"] = verify_param
        _cached_token["expires_at"] = time.time() + TOKEN_TTL_S
        return verify_param


def invalidate_captcha_token() -> None:
    """Force-invalidate the cached token.

    Call this after a 403 / FRONTEND_CAPTCHA error so the next request resolves
    a fresh token instead of reusing a rejected one.
    """
    _cached_token["verify_param"] = None
    _cached_token["expires_at"] = 0.0


def is_available() -> bool:
    """Whether the captcha solver can run (requires zendriver)."""
    return has_nodriver
