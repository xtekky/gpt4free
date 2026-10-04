import asyncio
import base64
import datetime
import json
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import time
from threading import Event, Lock
from unittest.mock import AsyncMock, MagicMock, patch

from g4f.Provider.Qwen import Qwen
from g4f.Provider.qwen.session import AUTH_URL, QwenAuth, cached_auth
from g4f.errors import CloudflareError, MissingAuthError, RateLimitError, ResponseError
from g4f.providers.response import JsonConversation
from .test_qwen_stream import QWEN_MODULE, Response, delta


def jwt(lifetime=900, subject="account-a"):
    payload = base64.urlsafe_b64encode(json.dumps({"sub": subject, "exp": int(time()) + lifetime, "jti": str(uuid.uuid4())}).encode()).decode().rstrip("=")
    return f"e30.{payload}.test"


class AuthResponse(Response):
    def __init__(self, payload=None, chunks=(), status=200, cookie=None, wait=None):
        super().__init__(chunks, payload)
        self.status, self.ok = status, status < 400
        self.wait = wait
        if cookie:
            self.headers["Set-Cookie"] = cookie

    async def __aenter__(self):
        if self.wait:
            await self.wait.wait()
        return self


def refreshed(token=None, **kwargs):
    return AuthResponse({"success": True, "data": {"access_token": token or jwt()}}, **kwargs)


BROWSER_HEADERS = {
    'user-agent': 'browser-agent', 'accept-language': 'en-US',
    'version': '0.3.12', 'source': 'web', 'timezone': 'browser-timezone',
    'sec-ch-ua': '"Browser";v="143"', 'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}


class TestQwenAuth(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        for field, value in (("_har_cookies", None), ("_har_headers", None), ("_har_cookie_records", None), ("_har_from_browser", False), ("_har_loaded_at", 0)):
            patcher = patch.object(Qwen, field, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def session(self, response):
        session = MagicMock()
        session.headers = {}
        session.get.return_value = response
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        patcher = patch("g4f.Provider.qwen.session.StreamSession", return_value=session)
        patcher.start()
        self.addCleanup(patcher.stop)
        return session

    async def test_proactive_refresh_and_cookie_rotation(self):
        auth = QwenAuth(jwt(100), "old-refresh")
        response = refreshed(cookie="refresh_token=new-refresh; Domain=.qwen.ai; Path=/; Secure; HttpOnly; Max-Age=2592000")
        session = self.session(response)
        await auth.ensure_valid(session, proxy="http://proxy.test")
        call = session.get.call_args
        self.assertEqual(call.args[0], AUTH_URL)
        self.assertEqual(call.kwargs["proxy"], "http://proxy.test")
        self.assertNotIn("Authorization", call.kwargs["headers"])
        self.assertEqual(call.kwargs["headers"]["cookie"], "refresh_token=old-refresh")
        self.assertEqual(call.kwargs["headers"]["version"], "0.3.12")
        self.assertEqual(call.kwargs["headers"]["source"], "web")
        self.assertIn("GMT", call.kwargs["headers"]["timezone"])
        self.assertIn("x-request-id", call.kwargs["headers"])
        self.assertFalse(call.kwargs["allow_redirects"])
        self.assertEqual(auth.request_headers({})["Cookie"], "refresh_token=new-refresh")
        self.assertGreater(auth.expires_at, time() + 800)
        await auth.ensure_valid(session)
        self.assertEqual(session.get.call_count, 1)

    async def test_concurrent_refresh_uses_one_request(self):
        ready = asyncio.Event()
        auth = QwenAuth(jwt(-1), "refresh")
        session = self.session(refreshed(wait=ready))
        tasks = [asyncio.create_task(auth.ensure_valid(session)) for _ in range(20)]
        await asyncio.sleep(0)
        ready.set()
        result = await asyncio.gather(*tasks)
        self.assertEqual(len(set(result)), 1)
        self.assertEqual(session.get.call_count, 1)

    async def test_concurrent_failed_refresh_does_not_fan_out(self):
        ready = asyncio.Event()
        auth = QwenAuth(jwt(-1), "refresh")
        session = self.session(AuthResponse({"success": False, "data": {"code": "temporarily_unavailable"}}, wait=ready))
        tasks = [asyncio.create_task(auth.ensure_valid(session)) for _ in range(10)]
        await asyncio.sleep(0)
        ready.set()
        result = await asyncio.gather(*tasks, return_exceptions=True)
        self.assertTrue(all(isinstance(e, ResponseError) for e in result))
        self.assertEqual(session.get.call_count, 1)

    async def test_rejected_old_snapshot_does_not_refresh_current_token(self):
        auth = QwenAuth(jwt(), "refresh")
        session = self.session(refreshed())
        await auth.ensure_valid(session, rejected_token="previous-token", recover=True)
        session.get.assert_not_called()

    async def test_clear_during_refresh_discards_response(self):
        ready = asyncio.Event()
        auth = QwenAuth(jwt(-1), "refresh")
        session = self.session(refreshed(wait=ready, cookie="refresh_token=new; Domain=.qwen.ai; Path=/"))
        task = asyncio.create_task(auth.ensure_valid(session))
        await asyncio.sleep(0)
        auth.clear()
        ready.set()
        with self.assertRaises(MissingAuthError):
            await task
        self.assertIsNone(auth.access_token)
        self.assertFalse(auth.can_refresh)

    async def test_cancelling_one_waiter_keeps_refresh_for_other_requests(self):
        ready = asyncio.Event()
        auth = QwenAuth(jwt(-1), "refresh")
        session = self.session(refreshed(wait=ready))
        first = asyncio.create_task(auth.ensure_valid(session))
        second = asyncio.create_task(auth.ensure_valid(session))
        await asyncio.sleep(0)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        ready.set()
        self.assertEqual(await second, auth.access_token)
        self.assertEqual(session.get.call_count, 1)

    async def test_explicit_persistence_exports_rotated_cookie_only_when_enabled(self):
        auth = QwenAuth(jwt(-1), "old-refresh")
        session = self.session(refreshed(cookie="refresh_token=new-refresh; Domain=.qwen.ai; Path=/; Secure"))
        with patch.object(Qwen, "_save_auth_har") as save:
            await Qwen._auth_context(auth_session=auth, persist_auth=True)
            await auth.ensure_valid(session)
        save.assert_called_once()
        self.assertEqual(save.call_args.args[0]["refresh_token"], "new-refresh")
        self.assertIn("Authorization", save.call_args.args[1])
        await Qwen._auth_context(auth_session=auth)
        self.assertIsNone(auth.on_update)

    async def test_browser_capture_is_not_discarded_by_har_ttl(self):
        with patch.object(Qwen, "_har_cookies", {"refresh_token": "live-refresh"}), patch.object(Qwen, "_har_headers", {}), patch.object(Qwen, "_har_from_browser", True), patch.object(Qwen, "_har_loaded_at", 0), patch.object(QWEN_MODULE, "get_har_files") as files:
            cookies, _ = Qwen._read_har()
        self.assertEqual(cookies["refresh_token"], "live-refresh")
        files.assert_not_called()

    async def test_files_cache_remains_shared_between_accounts(self):
        first, second = QwenAuth(jwt()), QwenAuth(jwt(subject="another-account"))
        first_file = {"id": "first-account-file"}
        payload = {"success": True, "data": {"file_id": "second-account-file", "file_url": "https://storage.example.com/file.png"}}
        session = self.session(AuthResponse({}))
        session.put.return_value = AuthResponse({})
        with patch.object(Qwen, "image_cache", True), patch.object(QWEN_MODULE, "ImagesCache", {}), patch.object(QWEN_MODULE, "to_bytes", return_value=b"test-image"), patch.object(QWEN_MODULE, "detect_file_type", return_value=(".png", "image/png")), patch.object(QWEN_MODULE, "get_oss_headers", return_value={}), patch.object(Qwen, "_api_json", new=AsyncMock(return_value=payload)) as upload:
            import hashlib
            key = hashlib.md5(b"test-image").hexdigest()
            QWEN_MODULE.ImagesCache[key] = first_file
            result_a = await Qwen.prepare_files([("image", "image.png")], session, auth=first)
            result_b = await Qwen.prepare_files([("image", "image.png")], session, auth=second)
        self.assertEqual(result_a[0]["id"], "first-account-file")
        self.assertEqual(result_b[0]["id"], "first-account-file")
        upload.assert_not_awaited()

    async def test_expired_refresh_session_requires_login(self):
        for status in (200, 401, 403):
            auth = QwenAuth(jwt(-1), "refresh")
            session = self.session(AuthResponse({"success": False, "data": {"code": "unauthorized"}}, status=status))
            with self.assertRaisesRegex(MissingAuthError, "sign in again"):
                await auth.ensure_valid(session)
            self.assertFalse(auth.can_refresh)
            self.assertIsNone(auth.access_token)
            with self.assertRaises(MissingAuthError):
                await auth.ensure_valid(session)
            self.assertEqual(session.get.call_count, 1)

    async def test_refresh_bad_request_preserves_reason_and_redacts_credentials(self):
        auth = QwenAuth(jwt(-1), "private-refresh")
        payload = {"success": False, "data": {"code": "Bad_Request", "details": "Invalid request header private-refresh"}}
        with self.assertRaisesRegex(ResponseError, "Bad_Request.*Invalid request header") as caught:
            await auth.ensure_valid(self.session(AuthResponse(payload)))
        self.assertNotIn("private-refresh", str(caught.exception))

    async def test_refresh_html_requires_browser_verification(self):
        response = AuthResponse({})
        response.headers = {"content-type": "text/html"}
        with self.assertRaises(CloudflareError):
            await QwenAuth(jwt(-1), "refresh").ensure_valid(self.session(response))

    async def test_quota_uses_the_refreshed_session(self):
        auth = QwenAuth(jwt(-1), "quota-refresh")
        session = self.session(refreshed())
        session.post.return_value = AuthResponse({"success": True, "data": {"id": "chat"}})
        with patch.object(QWEN_MODULE, "StreamSession", return_value=session), patch.object(Qwen, "_get_req_headers", new=AsyncMock(return_value={})):
            await Qwen.get_quota(auth_session=auth)
        self.assertEqual(session.post.call_args.kwargs["json"]["chat_mode"], "normal")
        self.assertEqual(session.post.call_args.kwargs["headers"]["Authorization"], f"Bearer {auth.access_token}")
        self.assertEqual(session.get.call_count, 1)

    async def test_expired_access_token_without_refresh_stops_locally(self):
        auth = QwenAuth(jwt(-1))
        session = self.session(refreshed())
        with self.assertRaisesRegex(MissingAuthError, "refresh_token"):
            await auth.ensure_valid(session)
        session.get.assert_not_called()

    async def test_empty_or_expired_refresh_cookie_does_not_become_guest(self):
        sessions = [QwenAuth(refresh_token=""), QwenAuth(cookies=[{"name": "refresh_token", "value": "expired", "expires": time() - 1}])]
        transport = self.session(refreshed())
        for auth in sessions:
            with self.assertRaisesRegex(MissingAuthError, "empty or expired"):
                await auth.ensure_valid(transport)
        transport.get.assert_not_called()

    async def test_refresh_only_session_and_expiry_from_server(self):
        auth = QwenAuth(refresh_token="refresh")
        response = refreshed(token="opaque-token")
        response.payload["data"]["expires_at"] = int(time()) + 900
        await auth.ensure_valid(self.session(response))
        self.assertEqual(auth.access_token, "opaque-token")
        self.assertTrue(auth.authenticated)

    async def test_invalid_refresh_payload_does_not_replace_token(self):
        for payload in ([], {"success": True, "data": {}}, {"success": True, "data": {"access_token": "no-expiry"}}, {"success": True, "data": {"access_token": jwt(-1)}}):
            auth = QwenAuth(jwt(-1), "refresh")
            old = auth.access_token
            with self.assertRaises(ResponseError):
                await auth.ensure_valid(self.session(AuthResponse(payload)))
            self.assertEqual(auth.access_token, old)

    async def test_cookie_domains_paths_expiry_and_external_hosts(self):
        auth = QwenAuth(jwt(), cookies=[
            {"name": "refresh_token", "value": "refresh", "domain": ".qwen.ai", "path": "/", "secure": True},
            {"name": "chat_cookie", "value": "chat", "domain": "chat.qwen.ai", "path": "/api/v2"},
            {"name": "old", "value": "old", "expires": time() - 1},
        ])
        self.assertNotIn("chat_cookie", auth.request_headers({}, AUTH_URL)["Cookie"])
        self.assertIn("chat_cookie=chat", auth.request_headers({}, "https://chat.qwen.ai/api/v2/chats/new")["Cookie"])
        self.assertNotIn("old=", auth.request_headers({}).get("Cookie", ""))
        external = auth.request_headers({"Cookie": "private", "Authorization": "private"}, "https://storage.example.com/file")
        self.assertNotIn("Cookie", external)
        self.assertNotIn("Authorization", external)
        auth.update_cookies(AuthResponse({}, cookie="refresh_token=; Domain=.qwen.ai; Path=/; Max-Age=0"), AUTH_URL)
        self.assertFalse(auth.can_refresh)

    async def test_har_iso_cookie_expiry_cannot_turn_into_guest_mode(self):
        for expiry in ("2020-01-01T00:00:00.000Z", "invalid", "NaN", "Infinity"):
            auth, _ = await Qwen._auth_context(cookies=[{
                "name": "refresh_token", "value": "expired", "domain": ".qwen.ai", "expires": expiry,
            }])
            self.assertTrue(auth.authenticated)
            self.assertFalse(auth.can_refresh)
            with self.assertRaises(MissingAuthError):
                await auth.ensure_valid(None)

    async def test_har_cookie_metadata_survives_import_export(self):
        future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
        auth = QwenAuth(cookies=[
            {"name": "refresh_token", "value": "refresh", "domain": ".qwen.ai", "expires": future.isoformat(), "httpOnly": True},
            {"name": "visible", "value": "public", "expires": -1, "httpOnly": False},
        ])
        self.assertTrue(auth.can_refresh)
        records = {c["name"]: c for c in auth.get_cookies()}
        self.assertEqual(records["refresh_token"]["expires"], int(future.timestamp()))
        self.assertTrue(records["refresh_token"]["httpOnly"])
        self.assertIsNone(records["visible"]["expires"])
        self.assertFalse(records["visible"]["httpOnly"])

    async def test_cookie_refresh_token_is_used_without_explicit_refresh_argument(self):
        token = jwt(100)
        with patch.object(Qwen, "_ensure_auth", new=AsyncMock()) as capture, patch.object(Qwen, "_read_har") as har:
            auth, headers = await Qwen._auth_context(token=token, cookies={"refresh_token": "cookie-only-refresh"})
        capture.assert_not_awaited()
        har.assert_not_called()
        self.assertEqual(auth.refresh_token, "cookie-only-refresh")
        self.assertNotIn("Cookie", headers)
        session = self.session(refreshed(cookie="refresh_token=rotated-cookie; Domain=.qwen.ai; Path=/; Secure; HttpOnly"))
        await auth.ensure_valid(session)
        self.assertEqual(session.get.call_args.kwargs["headers"]["cookie"], "refresh_token=cookie-only-refresh")
        self.assertEqual(auth.refresh_token, "rotated-cookie")
        self.assertNotEqual(auth.access_token, token)

    def test_refresh_cookie_accessor_respects_scope_expiry_deletion_and_clear(self):
        for record in (
            {"domain": "chat.qwen.ai"}, {"domain": ".qwen.ai", "path": "/other"},
            {"domain": ".qwen.ai", "expires": time() - 1},
        ):
            with self.subTest(record=record):
                auth = QwenAuth(cookies=[{"name": "refresh_token", "value": "unusable", **record}])
                self.assertIsNone(auth.refresh_token)
                self.assertFalse(auth.can_refresh)
        auth = QwenAuth(cookies=[{"name": "refresh_token", "value": "scoped", "domain": "auth.qwen.ai", "path": "/api/v2", "httpOnly": True}])
        self.assertEqual(auth.refresh_token, "scoped")
        self.assertNotIn("refresh_token", auth.request_headers({}).get("Cookie", ""))
        auth.update_cookies(AuthResponse({}, cookie="refresh_token=; Path=/api/v2; Max-Age=0"), AUTH_URL)
        self.assertIsNone(auth.refresh_token)
        auth.clear()
        self.assertIsNone(auth.refresh_token)

    def test_cookie_only_sessions_use_refresh_cookie_for_cache_identity(self):
        refresh = "cache-cookie-" + str(uuid.uuid4())
        first = QwenAuth(cookies={"refresh_token": refresh})
        second = QwenAuth(cookies=[{"name": "refresh_token", "value": refresh, "domain": ".qwen.ai"}])
        self.assertEqual(first.cache_id, second.cache_id)
        self.assertEqual(first.cache_id, QwenAuth(refresh_token=refresh).cache_id)
        self.assertIs(cached_auth(cookies={"refresh_token": refresh}), cached_auth(refresh_token=refresh))

    def test_refresh_cookie_received_from_server_marks_account_credentials(self):
        auth = QwenAuth(cookies={})
        self.assertFalse(auth.authenticated)
        auth.update_cookies(AuthResponse({}, cookie="refresh_token=received; Domain=.qwen.ai; Path=/; Secure; HttpOnly"), AUTH_URL)
        self.assertEqual(auth.refresh_token, "received")
        self.assertTrue(auth.authenticated)

    async def test_cached_session_keeps_rotated_credentials_and_isolates_accounts(self):
        first = cached_auth(jwt(-1), "account-a", proxy="proxy-a")
        await first.ensure_valid(self.session(refreshed(cookie="refresh_token=rotated; Domain=.qwen.ai; Path=/")))
        same = cached_auth(jwt(-1), "account-a", proxy="proxy-a")
        self.assertIs(first, same)
        self.assertIn("refresh_token=rotated", same.request_headers({})["Cookie"])
        self.assertIsNot(first, cached_auth(jwt(), "account-b", proxy="proxy-a"))
        self.assertIsNot(first, cached_auth(jwt(), "account-a", proxy="proxy-b"))

    async def test_explicit_credentials_do_not_read_other_accounts_har(self):
        with patch.object(Qwen, "_read_har") as har:
            auth, base = await Qwen._auth_context(token=jwt(), refresh_token="explicit-account")
        har.assert_not_called()
        self.assertNotIn("Authorization", base)
        self.assertNotIn("Cookie", base)
        self.assertIn("explicit-account", auth.request_headers({})["Cookie"])

    async def test_default_reuses_saved_session_without_opening_browser(self):
        token = jwt()
        with patch.object(Qwen, '_har_cookies', {'refresh_token': 'saved-account'}), patch.object(Qwen, '_har_headers', {'Authorization': f'Bearer {token}'}), patch.object(Qwen, '_har_from_browser', True), patch.object(QWEN_MODULE, 'has_cdp', False), patch.object(Qwen, '_read_cdp', new=AsyncMock()) as capture:
            auth, headers = await Qwen._auth_context()
        capture.assert_not_awaited()
        self.assertEqual(auth.access_token, token)
        self.assertIn('refresh_token=saved-account', auth.request_headers({}, AUTH_URL)['Cookie'])
        self.assertNotIn('Authorization', headers)

    async def test_first_capture_is_saved_and_reused_after_memory_reset(self):
        cookies = {'ssxmod_itna': 'existing-guest'}
        headers = {**BROWSER_HEADERS, 'bx-ua': 'sdk-ua', 'bx-umidtoken': 'sdk-uid'}
        with tempfile.TemporaryDirectory() as folder:
            files = lambda: [str(p) for p in Path(folder).glob('*.har')]
            with patch.object(QWEN_MODULE, 'get_cookies_dir', return_value=folder), patch.object(QWEN_MODULE, 'get_har_files', side_effect=files), patch.object(QWEN_MODULE, 'has_cdp', True), patch.object(Qwen, '_read_cdp', new=AsyncMock(return_value=(cookies, headers))) as capture:
                first, _ = await Qwen._auth_context()
                self.assertTrue((Path(folder) / 'chat.qwen.ai-cdp.har').is_file())
                second, _ = await Qwen._auth_context()
                capture.assert_awaited_once_with(None)
                self.assertEqual(first.fingerprint, second.fingerprint)
                self.assertEqual(first.get_cookies(), second.get_cookies())
                Qwen._har_cookies = Qwen._har_headers = Qwen._har_cookie_records = None
                Qwen._har_from_browser = False
                third, _ = await Qwen._auth_context()
                capture.assert_awaited_once_with(None)
                self.assertEqual(third.fingerprint['bx-ua'], 'sdk-ua')
                self.assertIn('ssxmod_itna=existing-guest', third.request_headers({})['Cookie'])

    async def test_automatic_capture_can_disable_persistence(self):
        with patch.object(Qwen, '_read_har', return_value=(None, None)), patch.object(QWEN_MODULE, 'has_cdp', True), patch.object(Qwen, '_read_cdp', new=AsyncMock(return_value=({'guest': 'cookie'}, BROWSER_HEADERS))) as capture, patch.object(Qwen, '_save_auth_har') as save:
            auth, _ = await Qwen._auth_context(persist_auth=False)
        capture.assert_awaited_once_with(None)
        save.assert_not_called()
        self.assertIsNone(auth.on_update)

    async def test_explicit_browser_capture_still_requires_cdp(self):
        with patch.object(QWEN_MODULE, 'has_cdp', False), patch.object(Qwen, '_read_har') as har:
            with self.assertRaisesRegex(MissingAuthError, 'available CDP browser'):
                await Qwen._auth_context(use_browser=True)
        har.assert_not_called()

    async def test_use_browser_false_keeps_automatic_lookup(self):
        with patch.object(Qwen, '_ensure_auth', new=AsyncMock()) as ensure, patch.object(Qwen, '_get_headers', return_value={}):
            auth, _ = await Qwen._auth_context(use_browser=False)
        ensure.assert_awaited_once_with(None, persist=True)
        self.assertFalse(auth.authenticated)

    async def test_default_browser_does_not_override_explicit_credentials_or_session(self):
        cases = [
            {"token": jwt()}, {"api_key": jwt()}, {"refresh_token": "explicit-refresh"},
            {"refresh_token": ""}, {"cookies": {"refresh_token": "cookie-refresh"}},
            {"cookies": {}}, {"auth_session": QwenAuth(jwt())}, {"auth_session": QwenAuth()},
        ]
        for kwargs in cases:
            with self.subTest(source=next(iter(kwargs))):
                with patch.object(Qwen, "_ensure_auth", new=AsyncMock()) as ensure, patch.object(Qwen, "_get_headers", return_value={}):
                    auth, _ = await Qwen._auth_context(**kwargs)
                ensure.assert_not_awaited()
                if "auth_session" in kwargs:
                    self.assertIs(auth, kwargs["auth_session"])

    async def test_conflicting_credential_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            await Qwen._auth_context(auth_session=QwenAuth(jwt()), token="another-account")
        with self.assertRaises(ValueError):
            await Qwen._auth_context(use_browser=True, refresh_token="another-account")

    async def test_fingerprint_waits_for_sdk_and_rejects_placeholder_tokens(self):
        browser = MagicMock()
        values = {'bx-ua': 'live-sdk-ua', 'bx-v': '2.5.37'}
        browser.evaluate_js = AsyncMock(side_effect=[None,
            {'bx-ua': 'defaultFY3_fyjs_not_initialized'}, values])
        with patch.object(QWEN_MODULE.asyncio, 'sleep', new=AsyncMock()) as sleep:
            result = await Qwen._read_browser_fingerprint(browser)
        self.assertEqual(result, values)
        self.assertEqual(sleep.await_count, 2)
        expression = browser.evaluate_js.call_args.args[0]
        self.assertIn('sdk.getUA(', expression)
        self.assertNotIn('getUidToken', expression)
        self.assertNotIn('bx-umidtoken', expression)
        self.assertNotIn('getFYModule', expression)
        self.assertIn('https://chat.qwen.ai/api/v2/chat/completions', expression)
        self.assertNotIn('fetch(', expression)
        self.assertNotIn('click()', expression)

    async def test_unready_fingerprint_fails_locally_without_seed_chat(self):
        browser = MagicMock()
        browser.evaluate_js = AsyncMock(return_value=None)
        with patch.object(QWEN_MODULE, 'time', side_effect=[0, 0, 31]), patch.object(QWEN_MODULE.asyncio, 'sleep', new=AsyncMock()):
            with self.assertRaisesRegex(MissingAuthError, 'fingerprint SDK'):
                await Qwen._read_browser_fingerprint(browser)
        browser.evaluate_js.assert_awaited_once()

    def browser(self, records, ready=True):
        browser = MagicMock()
        browser.__aenter__ = AsyncMock(return_value=browser)
        browser.__aexit__ = AsyncMock(return_value=False)
        browser.navigate = AsyncMock()
        browser.call = AsyncMock()
        browser.evaluate_js = AsyncMock(return_value=ready)
        browser.get_cookies_list = AsyncMock(return_value=records)
        return browser

    async def test_browser_account_uses_refresh_cookie_without_local_storage(self):
        records = [{'name': 'refresh_token', 'value': 'browser-refresh',
                    'domain': '.qwen.ai', 'path': '/', 'httpOnly': True}]
        browser = self.browser(records)
        with patch('g4f.requests.cdp.CDPSession', return_value=browser), patch.object(Qwen, '_read_browser_fingerprint', new=AsyncMock(return_value={'bx-ua': 'sdk-ua'})):
            cookies, headers = await Qwen._read_cdp()
        auth = QwenAuth(cookies=Qwen._har_cookie_records, headers=headers)
        self.assertEqual(cookies['refresh_token'], 'browser-refresh')
        self.assertTrue(auth.authenticated)
        self.assertEqual(auth.refresh_token, 'browser-refresh')
        self.assertNotIn('Authorization', headers)
        expressions = ' '.join(call.args[0] for call in browser.evaluate_js.call_args_list)
        self.assertNotIn('localStorage', expressions)
        self.assertNotIn('click()', expressions)
        browser.add_event_handler.assert_not_called()
        with patch.object(Qwen, '_read_cdp', new=AsyncMock(return_value=(cookies, headers))), patch.object(QWEN_MODULE, 'has_cdp', True), patch.object(Qwen, '_save_auth_har') as save:
            await Qwen._ensure_auth(force_cdp=True)
        save.assert_not_called()

    async def test_capture_accepts_ready_guest_without_seed_chat_or_login(self):
        records = [{'name': 'ssxmod_itna', 'value': 'guest-cookie', 'domain': 'chat.qwen.ai'}]
        browser = self.browser(records)
        with patch('g4f.requests.cdp.CDPSession', return_value=browser), patch.object(Qwen, '_read_browser_fingerprint', new=AsyncMock(return_value={'bx-ua': 'sdk-ua'})), patch.object(QWEN_MODULE.asyncio, 'sleep', new=AsyncMock()) as sleep:
            cookies, headers = await Qwen._read_cdp()
        sleep.assert_not_awaited()
        self.assertEqual(cookies, {'ssxmod_itna': 'guest-cookie'})
        self.assertEqual(Qwen._har_cookie_records, records)
        self.assertEqual(headers, {'bx-ua': 'sdk-ua'})
        self.assertFalse(QwenAuth(cookies=records).authenticated)
        browser.call.assert_not_awaited()
        browser.add_event_handler.assert_not_called()
        expressions = ' '.join(call.args[0] for call in browser.evaluate_js.call_args_list)
        self.assertNotIn('Hello', expressions)
        self.assertNotIn('click()', expressions)

    async def test_capture_reads_cookies_after_sdk_initialization(self):
        records = []
        browser = self.browser(records)
        async def fingerprint(session):
            self.assertIs(session, browser)
            records.append({'name': 'ssxmod_itna', 'value': 'initialized', 'domain': 'chat.qwen.ai'})
            return {'bx-ua': 'current-ua'}
        with patch('g4f.requests.cdp.CDPSession', return_value=browser), patch.object(Qwen, '_read_browser_fingerprint', new=AsyncMock(side_effect=fingerprint)):
            cookies, headers = await Qwen._read_cdp()
        self.assertEqual(cookies, {'ssxmod_itna': 'initialized'})
        self.assertEqual(headers, {'bx-ua': 'current-ua'})
        self.assertEqual(Qwen._har_cookie_records, records)
        browser.get_cookies_list.assert_awaited_once()

    async def test_capture_waits_for_guest_composer_to_load(self):
        browser = self.browser([])
        browser.evaluate_js.side_effect = [False, False, True]
        with patch('g4f.requests.cdp.CDPSession', return_value=browser), patch.object(Qwen, '_read_browser_fingerprint', new=AsyncMock(return_value={'bx-ua': 'sdk-ua'})) as fingerprint, patch.object(QWEN_MODULE.asyncio, 'sleep', new=AsyncMock()) as sleep:
            _, headers = await Qwen._read_cdp()
        self.assertEqual(sleep.await_count, 2)
        fingerprint.assert_awaited_once_with(browser)
        self.assertNotIn('Authorization', headers)

    async def test_har_cookie_array_and_latest_authorization(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "qwen.har"
            def entry(key):
                return {"request": {"url": "https://chat.qwen.ai/api/v2/chats/new", "headers": [{"name": "Authorization", "value": f"Bearer {key}"}], "cookies": [{"name": "refresh_token", "value": "har-refresh", "domain": ".qwen.ai", "path": "/"}]}}
            path.write_text(json.dumps({"log": {"entries": [entry("old"), entry("new")]}}))
            with patch.object(QWEN_MODULE, "get_har_files", return_value=[str(path)]), patch.object(Qwen, "_har_cookies", None), patch.object(Qwen, "_har_headers", None), patch.object(Qwen, "_har_cookie_records", None):
                cookies, headers = Qwen._read_har()
                self.assertEqual(cookies["refresh_token"], "har-refresh")
                self.assertEqual(headers["Authorization"], "Bearer new")

    async def test_latest_har_credentials_keep_their_own_account_snapshot(self):
        entries = [
            {"request": {"url": "https://chat.qwen.ai/api/v2/chat/completions", "headers": [
                {"name": "Authorization", "value": "Bearer older-account"},
                {"name": "bx-umidtoken", "value": "fingerprint"},
            ], "cookies": [{"name": "refresh_token", "value": "older-refresh"}]}},
            {"request": {"url": "https://auth.qwen.ai/api/v2/auths/refresh", "headers": [
                {"name": "Cookie", "value": "refresh_token=newer-refresh"},
            ], "cookies": []}},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "qwen.har"
            path.write_text(json.dumps({"log": {"entries": entries}}))
            with patch.object(QWEN_MODULE, "get_har_files", return_value=[str(path)]):
                cookies, headers = Qwen._read_har()
        self.assertEqual(cookies["refresh_token"], "newer-refresh")
        self.assertEqual(headers["bx-umidtoken"], "fingerprint")
        self.assertFalse(any(k.lower() == "authorization" for k in headers))

    async def test_latest_har_api_token_wins_over_older_completion_token(self):
        entries = [{"request": {"url": url, "headers": [
            {"name": "Authorization", "value": f"Bearer {token}"},
        ], "cookies": []}} for url, token in (
            ("https://chat.qwen.ai/api/v2/chat/completions", "old"),
            ("https://chat.qwen.ai/api/v1/auths", "new"),
            ("https://chat.qwen.ai.evil.test/api/v2/chat/completions", "foreign"),
            ("http://chat.qwen.ai/api/v2/chat/completions", "insecure"),
        )]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "qwen.har"
            path.write_text(json.dumps({"log": {"entries": entries}}))
            with patch.object(QWEN_MODULE, "get_har_files", return_value=[str(path)]):
                _, headers = Qwen._read_har()
        self.assertEqual(headers["Authorization"], "Bearer new")

    async def run_provider(self, auth, responses, media=None):
        session = self.session(AuthResponse({}))
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        session.post.side_effect = responses
        fresh = refreshed()
        session.get.side_effect = [AuthResponse({}), fresh]
        with patch.object(QWEN_MODULE, "StreamSession", return_value=session), patch.object(Qwen, "_get_req_headers", new=AsyncMock(return_value={})), patch.object(Qwen, "prepare_files", new=AsyncMock(return_value=[])) as upload:
            self.provider_session = session
            self.upload = upload
            self.output = []
            async for item in Qwen.create_async_generator(Qwen.default_model, [{"role": "user", "content": "test"}], auth_session=auth, conversation=JsonConversation(chat_id="chat", parent_id=None, cookies={}), media=media):
                self.output.append(item)
        return fresh

    async def test_auth_rejection_before_stream_retries_without_reupload(self):
        auth = QwenAuth(jwt(), "refresh")
        fresh = await self.run_provider(auth, [AuthResponse(chunks=[{"error": {"code": "unauthorized"}}]), AuthResponse(chunks=[delta("answer", "Answer")])], media=[("image", "file.png")])
        self.assertIn("Answer", self.output)
        self.assertEqual(self.provider_session.post.call_count, 2)
        self.upload.assert_awaited_once()
        first, second = self.provider_session.post.call_args_list
        self.assertEqual(first.kwargs["json"], second.kwargs["json"])
        self.assertNotEqual(first.kwargs["headers"]["Authorization"], second.kwargs["headers"]["Authorization"])
        self.assertEqual(auth.access_token, fresh.payload["data"]["access_token"])

    async def test_auth_failure_after_output_is_not_replayed(self):
        auth = QwenAuth(jwt(), "refresh")
        with self.assertRaises(MissingAuthError):
            await self.run_provider(auth, [AuthResponse(chunks=[delta("answer", "Started"), {"error": {"code": "unauthorized"}}])])
        self.assertIn("Started", self.output)
        self.assertEqual(self.provider_session.post.call_count, 1)
        self.assertEqual(self.provider_session.get.call_count, 1)

    async def test_second_auth_rejection_and_rate_limit_are_not_replayed(self):
        for code, error, count in (("unauthorized", MissingAuthError, 2), ("RateLimited", RateLimitError, 1)):
            auth = QwenAuth(jwt(), "refresh")
            with self.assertRaises(error):
                await self.run_provider(auth, [AuthResponse(chunks=[{"error": {"code": code}}]) for _ in range(count)])
            self.assertEqual(self.provider_session.post.call_count, count)

    async def test_captcha_requires_user_without_seed_chat(self):
        with self.assertRaises(CloudflareError):
            await self.run_provider(QwenAuth(jwt(), "refresh"), [AuthResponse(chunks=[{"error": {"code": "FAIL_SYS_USER_VALIDATE"}}])])
        self.assertEqual(self.provider_session.post.call_count, 1)

    async def test_baxia_json_challenge_is_typed_and_never_replayed(self):
        payload = {"ret": ["FAIL_SYS_USER_VALIDATE", "RGV587_ERROR::SM::Verification required"],
                   "data": {"url": "https://chat.qwen.ai/punish?x5secdata=private-challenge"}}
        with self.assertRaises(CloudflareError) as caught:
            await self.run_provider(QwenAuth(jwt(), "refresh"), [AuthResponse(payload)])
        self.assertIn("browser verification", str(caught.exception))
        self.assertNotIn("private-challenge", str(caught.exception))
        self.assertNotIn("https://", str(caught.exception))
        self.assertEqual(self.provider_session.post.call_count, 1)

    async def test_baxia_api_json_challenge_does_not_look_like_success(self):
        session = self.session(AuthResponse({}))
        session.post.return_value = AuthResponse({
            "ret": ["FAIL_SYS_USER_VALIDATE"], "data": {"url": "https://chat.qwen.ai/punish?x5secdata=private-challenge"},
        })
        with self.assertRaises(CloudflareError) as caught:
            await Qwen._api_json(session, "post", "https://chat.qwen.ai/api/v2/chats/new", QwenAuth(cookies={}))
        self.assertNotIn("private-challenge", str(caught.exception))
        self.assertEqual(session.post.call_count, 1)

    async def test_json_auth_recovery_updates_request_headers(self):
        auth = QwenAuth(jwt(), "refresh")
        session = self.session(refreshed())
        session.post.side_effect = [AuthResponse({"success": False, "data": {"code": "unauthorized"}}), AuthResponse({"success": True, "data": {"id": "chat"}})]
        result = await Qwen._api_json(session, "post", "https://chat.qwen.ai/api/v2/chats/new", auth, json={})
        self.assertEqual(result["data"]["id"], "chat")
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(session.get.call_count, 1)

    async def test_json_api_respects_empty_headers_without_mutating_defaults(self):
        session = self.session(AuthResponse({}))
        session.headers = {'X-Default': 'session-value'}
        session.get.return_value = AuthResponse({'success': True})
        headers = {}
        result = await Qwen._api_json(session, 'get', 'https://chat.qwen.ai/api/v2/configs', headers=headers)
        self.assertTrue(result['success'])
        self.assertEqual(session.get.call_args.kwargs['headers'], {})
        self.assertFalse(session.get.call_args.kwargs['allow_redirects'])
        self.assertEqual(headers, {})
        self.assertEqual(session.headers, {'X-Default': 'session-value'})

    async def test_json_auth_recovery_uses_cookie_rotated_by_rejected_request(self):
        auth = QwenAuth(jwt(), 'old-refresh')
        session = self.session(refreshed())
        session.post.side_effect = [
            AuthResponse({'error': {'code': 'unauthorized'}}, status=401,
                         cookie='refresh_token=server-rotated; Domain=.qwen.ai; Path=/; Secure'),
            AuthResponse({'success': True}),
        ]
        result = await Qwen._api_json(session, 'post', 'https://chat.qwen.ai/api/v2/chats/new', auth)
        self.assertTrue(result['success'])
        self.assertEqual(session.post.call_count, 2)
        self.assertEqual(session.get.call_args.kwargs['headers']['cookie'], 'refresh_token=server-rotated')

    async def test_json_rejection_deleting_refresh_cookie_stops_without_retry(self):
        auth = QwenAuth(jwt(), 'old-refresh')
        session = self.session(refreshed())
        session.post.return_value = AuthResponse({'error': {'code': 'unauthorized'}}, status=401,
            cookie='refresh_token=; Domain=.qwen.ai; Path=/; Max-Age=0; Secure')
        with self.assertRaises(MissingAuthError):
            await Qwen._api_json(session, 'post', 'https://chat.qwen.ai/api/v2/chats/new', auth)
        self.assertEqual(session.post.call_count, 1)
        session.get.assert_not_called()
        self.assertFalse(auth.can_refresh)

    async def test_json_api_invalid_payloads_are_not_retried(self):
        for payload in ([], None):
            with self.subTest(payload=payload):
                session = self.session(AuthResponse({}))
                session.post.return_value = AuthResponse(payload)
                with self.assertRaisesRegex(ResponseError, 'expected an object'):
                    await Qwen._api_json(session, 'post', 'https://chat.qwen.ai/api/v2/chats/new')
                self.assertEqual(session.post.call_count, 1)
        session = self.session(AuthResponse({}))
        response = AuthResponse({})
        response.json = AsyncMock(side_effect=ValueError('malformed body'))
        session.post.return_value = response
        with self.assertRaisesRegex(ResponseError, 'invalid JSON'):
            await Qwen._api_json(session, 'post', 'https://chat.qwen.ai/api/v2/chats/new')
        self.assertEqual(session.post.call_count, 1)

    async def test_json_api_only_retries_one_auth_rejection(self):
        for code, error, count in (('unauthorized', MissingAuthError, 2), ('RateLimited', RateLimitError, 1)):
            with self.subTest(code=code):
                auth = QwenAuth(jwt(), 'refresh')
                session = self.session(refreshed())
                session.post.side_effect = [AuthResponse({'error': {'code': code}}) for _ in range(count)]
                with self.assertRaises(error):
                    await Qwen._api_json(session, 'post', 'https://chat.qwen.ai/api/v2/chats/new', auth)
                self.assertEqual(session.post.call_count, count)
                self.assertEqual(session.get.call_count, count - 1)

    async def test_credentials_are_not_written_in_logs_or_repr(self):
        token = jwt(-1)
        auth = QwenAuth(token, "private-refresh")
        with patch("g4f.Provider.qwen.session.debug.log") as log:
            await auth.ensure_valid(self.session(refreshed()))
        output = repr(auth) + " ".join(str(call) for call in log.call_args_list)
        self.assertNotIn(token, output)
        self.assertNotIn("private-refresh", output)


class TestQwenAuthAcrossLoops(unittest.TestCase):
    def test_parallel_threads_and_event_loops_share_one_refresh(self):
        auth = QwenAuth(jwt(-1), "parallel-refresh")
        release, joined = Event(), Event()
        count_lock = Lock()
        waiters = 0
        original_wrap = asyncio.wrap_future
        def wrap(future, *args, **kwargs):
            nonlocal waiters
            if future is auth._refresh_future:
                with count_lock:
                    waiters += 1
                    if waiters == 8:
                        joined.set()
            return original_wrap(future, *args, **kwargs)
        class GatedResponse(AuthResponse):
            async def __aenter__(self):
                if not await asyncio.get_running_loop().run_in_executor(None, release.wait, 10):
                    raise TimeoutError("Refresh test gate timed out")
                return self
        response = GatedResponse({"success": True, "data": {"access_token": jwt()}},
                                 cookie="refresh_token=rotated; Domain=.qwen.ai; Path=/")
        transport = MagicMock()
        transport.__aenter__ = AsyncMock(return_value=transport)
        transport.__aexit__ = AsyncMock(return_value=False)
        transport.get.return_value = response
        with patch("g4f.Provider.qwen.session.StreamSession", return_value=transport), patch.object(asyncio, "wrap_future", side_effect=wrap):
            with ThreadPoolExecutor(max_workers=8) as workers:
                tasks = [workers.submit(asyncio.run, auth.ensure_valid(None)) for _ in range(8)]
                try:
                    self.assertTrue(joined.wait(10), "All event loops should join the same refresh")
                finally:
                    release.set()
                results = [task.result(timeout=10) for task in tasks]
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(transport.get.call_count, 1)
        self.assertIn("refresh_token=rotated", auth.request_headers({})["Cookie"])

    def test_rotated_session_survives_separate_asyncio_runs(self):
        refresh = "test-" + str(uuid.uuid4())
        initial = jwt(-1)
        transport = MagicMock()
        transport.__aenter__ = AsyncMock(return_value=transport)
        transport.__aexit__ = AsyncMock(return_value=False)
        transport.get.return_value = refreshed(cookie="refresh_token=rotated; Domain=.qwen.ai; Path=/")
        async def run():
            auth = cached_auth(initial, refresh)
            await auth.ensure_valid(transport)
            return auth
        with patch("g4f.Provider.qwen.session.StreamSession", return_value=transport):
            first = asyncio.run(run())
            second = asyncio.run(run())
        self.assertIs(first, second)
        self.assertIn("refresh_token=rotated", second.request_headers({})["Cookie"])
        self.assertEqual(transport.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
