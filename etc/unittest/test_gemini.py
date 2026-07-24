from __future__ import annotations

import asyncio
import importlib
import json
import unittest
from unittest.mock import AsyncMock, patch

from g4f.Provider.needs_auth.Gemini import (
    ACCOUNT_STATUS_AVAILABLE,
    ACCOUNT_STATUS_UNAUTHENTICATED,
    MODEL_HEADER_KEY,
    Gemini,
    _build_model_headers,
    _extract_gemini_error_code,
    _extract_reasoning,
    _has_authenticated_session,
    _is_xsrf_error,
    _iter_response_lines,
    _normalize_messages,
    _parse_account_models,
    _parse_google_frames,
    _resolve_gemini_conversation,
    _resolve_gemini_prompt,
    _resolve_model,
    Conversation,
)
from g4f.errors import MissingAuthError, ResponseError, ResponseStatusError
from g4f.models import ModelRegistry

GEMINI_MODULE = importlib.import_module("g4f.Provider.needs_auth.Gemini")


def build_account_response(status: int) -> tuple[str, dict[str, dict]]:
    body = [None] * 18
    body[14] = status
    body[15] = [
        ["fbb127bbb056c959", "Flash", "All-around help"],
        ["5bf011840784117a", "Thinking", "Solves complex problems"],
        ["9d8ca3786ebdfbea", "Pro", "Advanced math & code"],
    ]
    body[16] = []
    body[17] = []
    response = json.dumps([["wrb.fr", "otAQ7b", json.dumps(body)]])
    parsed_status, registry = _parse_account_models(response)
    assert parsed_status == status
    return response, registry


class GeminiHelpersTest(unittest.TestCase):
    def test_parse_account_models_and_availability(self):
        _, registry = build_account_response(ACCOUNT_STATUS_UNAUTHENTICATED)

        self.assertTrue(registry["fbb127bbb056c959"]["available"])
        self.assertFalse(registry["5bf011840784117a"]["available"])
        self.assertFalse(registry["9d8ca3786ebdfbea"]["available"])

    def test_missing_account_status_means_available(self):
        body = [None] * 18
        body[15] = [["e6fa609c3fa255c0", "Pro", "Advanced model"]]
        response = json.dumps([["wrb.fr", "otAQ7b", json.dumps(body)]])

        status, registry = _parse_account_models(response)

        self.assertEqual(status, ACCOUNT_STATUS_AVAILABLE)
        self.assertTrue(registry["e6fa609c3fa255c0"]["available"])

    def test_parse_utf16_length_prefixed_frame(self):
        raw = json.dumps([["wrb.fr", "rpc", "emoji: 😀"]], ensure_ascii=False)
        payload = f"\n{raw}\n"
        utf16_length = len(payload.encode("utf-16-le")) // 2

        frames, remaining = _parse_google_frames(f"{utf16_length}{payload}")

        self.assertEqual(frames[0][0], "wrb.fr")
        self.assertEqual(remaining, "")

    def test_model_header_capacity_fields(self):
        field_12 = json.loads(_build_model_headers("model", 4, 12)[MODEL_HEADER_KEY])
        field_13 = json.loads(_build_model_headers("model", 2, 13)[MODEL_HEADER_KEY])

        self.assertEqual(field_12[11], 4)
        self.assertIsNone(field_13[11])
        self.assertEqual(field_13[12], 2)

    def test_mode_categories_and_default_thinking_depths(self):
        expected = {
            "gemini-3.5-flash": (1, 4),
            "gemini-3.5-flash-thinking": (2, 0),
            "gemini-3.1-pro": (3, 4),
            "gemini-auto": (4, 4),
            "gemini-3.5-flash-thinking-lite": (5, 0),
            "gemini-flash-lite": (6, 4),
        }

        for requested, (mode, default_think) in expected.items():
            with self.subTest(model=requested):
                model, think = _resolve_model(requested)
                request = Gemini.build_request(
                    "test", "en", model, think, request_uuid="test-request"
                )
                self.assertEqual(request[79], mode)
                self.assertEqual(request[17], [[default_think]])

    def test_explicit_thinking_depths_are_preserved(self):
        for requested_depth in range(5):
            with self.subTest(depth=requested_depth):
                model, think = _resolve_model(
                    f"gemini-3.5-flash-thinking@think={requested_depth}"
                )
                request = Gemini.build_request(
                    "test", "en", model, think, request_uuid="test-request"
                )
                self.assertEqual(request[17], [[requested_depth]])

    def test_legacy_model_names_resolve_to_current_modes(self):
        expected = {
            "gemini-2.0": "gemini-3.5-flash",
            "gemini-2.0-flash": "gemini-3.5-flash",
            "gemini-2.0-flash-thinking": "gemini-3.5-flash-thinking",
            "gemini-2.0-flash-thinking-with-apps": "gemini-3.5-flash-thinking",
            "gemini-2.5-flash": "gemini-3.5-flash",
            "gemini-2.5-pro": "gemini-3.1-pro",
            "gemini-3.1-flash-lite": "gemini-flash-lite",
        }

        for legacy_name, current_name in expected.items():
            with self.subTest(model=legacy_name):
                resolved, _ = _resolve_model(legacy_name)
                self.assertEqual(resolved, current_name)

    def test_unknown_model_lists_supported_models(self):
        with self.assertRaises(ValueError) as context:
            _resolve_model("gemini-3.6-flash")

        message = str(context.exception)
        self.assertIn("Unknown Gemini model: gemini-3.6-flash", message)
        self.assertIn("Supported models:", message)
        self.assertIn("gemini-3.5-flash", message)

    def test_public_model_registry_exposes_current_models(self):
        self.assertEqual(ModelRegistry.get("gemini").name, "gemini-3.5-flash")
        for model in (
            "gemini-3.5-flash-thinking",
            "gemini-auto",
            "gemini-3.5-flash-thinking-lite",
            "gemini-flash-lite",
        ):
            with self.subTest(model=model):
                self.assertEqual(ModelRegistry.get(model).name, model)

    def test_prompt_only_requests_accept_missing_messages(self):
        messages = _normalize_messages(None)

        self.assertEqual(messages, [])
        self.assertEqual(
            _resolve_gemini_prompt(messages, "prompt-only request", None),
            "prompt-only request",
        )
        with self.assertRaises(TypeError):
            _normalize_messages({"role": "user"})

    def test_anonymous_followup_uses_full_message_history(self):
        messages = [
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second turn"},
        ]
        conversation = Conversation(
            "conversation-id",
            "response-id",
            "choice-id",
            "gemini-auto",
        )

        self.assertFalse(_has_authenticated_session({}))
        resolved = _resolve_gemini_conversation(
            conversation,
            "gemini-auto",
            {},
        )
        prompt = _resolve_gemini_prompt(messages, None, resolved)
        request = Gemini.build_request(
            prompt,
            "en",
            "gemini-auto",
            4,
            conversation=resolved,
            request_uuid="test-request",
        )

        self.assertIsNone(resolved)
        self.assertIn("first turn", prompt)
        self.assertIn("first answer", prompt)
        self.assertIn("second turn", prompt)
        self.assertEqual(request[2][:3], ["", "", ""])

    def test_authenticated_followup_keeps_conversation_handle(self):
        conversation = Conversation(
            "conversation-id",
            "response-id",
            "choice-id",
            "gemini-3.5-flash",
        )
        cookies = {"__Secure-1PSID": "session"}

        self.assertTrue(_has_authenticated_session(cookies))
        self.assertIs(
            _resolve_gemini_conversation(
                conversation,
                "gemini-3.5-flash",
                cookies,
            ),
            conversation,
        )
        self.assertIsNone(
            _resolve_gemini_conversation(
                conversation,
                "gemini-auto",
                cookies,
            )
        )

    def test_xsrf_response_is_retryable(self):
        error = ResponseStatusError(
            'Response 400: [["er",null,{"reason":"xsrf"}]]'
        )

        self.assertTrue(_is_xsrf_error(error, 400))
        self.assertFalse(_is_xsrf_error(error, 401))
        self.assertFalse(
            _is_xsrf_error(ResponseStatusError("Response 400"), 400)
        )

    def test_extract_reasoning_from_dedicated_field(self):
        candidate = [None] * 38
        candidate[37] = [["private reasoning"]]
        response = [None] * 5
        response[4] = [candidate]

        self.assertEqual(_extract_reasoning(response), "private reasoning")

    def test_extract_structured_error_code(self):
        frame = ["wrb.fr", None, None, None, None, [None, None, [[None, [1037]]]]]

        self.assertEqual(_extract_gemini_error_code(frame), 1037)

    def test_reject_silent_model_fallback(self):
        _, registry = build_account_response(ACCOUNT_STATUS_UNAUTHENTICATED)

        class ProbeGemini(Gemini):
            _account_status = ACCOUNT_STATUS_UNAUTHENTICATED
            _account_models = registry

        with self.assertRaises(MissingAuthError):
            ProbeGemini.validate_model_access("gemini-3.1-pro")
        ProbeGemini.validate_model_access("gemini-3.5-flash")
        ProbeGemini.validate_model_access(
            "gemini-3.1-pro", allow_model_fallback=True
        )

    def test_dynamic_headers_only_for_available_pro(self):
        _, registry = build_account_response(ACCOUNT_STATUS_AVAILABLE)

        class ProbeGemini(Gemini):
            _account_status = ACCOUNT_STATUS_AVAILABLE
            _account_models = registry

        pro_header = json.loads(
            ProbeGemini.get_model_headers("gemini-3.1-pro")[MODEL_HEADER_KEY]
        )
        self.assertEqual(pro_header[4], "9d8ca3786ebdfbea")
        self.assertEqual(ProbeGemini.get_model_headers("gemini-3.5-flash"), {})
        self.assertEqual(
            ProbeGemini.get_model_headers("gemini-3.5-flash-thinking"), {}
        )


class GeminiStreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_public_generator_retries_xsrf_with_prompt_only_request(self):
        class FakeResponse:
            def __init__(self, status, content=b"[]\n"):
                self.status = status
                self.headers = {}
                self.released = False
                self.content = self
                self._content = content

            async def __aenter__(self):
                return self

            async def __aexit__(self, _exc_type, _exc, _traceback):
                return False

            async def iter_any(self):
                yield self._content

            def release(self):
                self.released = True

        class FakeSession:
            def __init__(self, responses):
                self.responses = iter(responses)
                self.calls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, _exc_type, _exc, _traceback):
                return False

            async def post(self, _url, **kwargs):
                self.calls.append(
                    {
                        "data": dict(kwargs["data"]),
                        "params": dict(kwargs["params"]),
                        "cookies": dict(kwargs["cookies"]),
                        "cookies_object": kwargs["cookies"],
                    }
                )
                return next(self.responses)

        class ProbeGemini(Gemini):
            auto_refresh = False
            _cookies = None
            _metadata_cookie_key = None
            _metadata_auth_user = None
            _metadata_fetched_at = 0
            _account_status = None
            _account_models = {}
            _account_models_fetched_at = 0
            _snlm0e = None
            _sid = None

        first_response = FakeResponse(400)
        session = FakeSession([first_response, FakeResponse(200)])
        source_cookies = {
            "__Secure-1PSID": "session",
            "__Secure-1PSIDTS": "old",
        }
        metadata_fetches = 0

        async def fetch_metadata(_session, cookies, _auth_user=None):
            nonlocal metadata_fetches
            metadata_fetches += 1
            self.assertIsNot(cookies, source_cookies)
            ProbeGemini._snlm0e = f"token-{metadata_fetches}"
            ProbeGemini._sid = f"sid-{metadata_fetches}"
            ProbeGemini._bl = f"build-{metadata_fetches}"

        async def check_status(response):
            if response.status == 400:
                raise ResponseStatusError(
                    'Response 400: [["er",null,{"reason":"xsrf"}]]'
                )

        with patch.object(
            GEMINI_MODULE,
            "ClientSession",
            return_value=session,
        ):
            with patch.object(
                ProbeGemini,
                "fetch_snlm0e",
                new_callable=AsyncMock,
                side_effect=fetch_metadata,
            ) as fetch:
                with patch.object(
                    ProbeGemini,
                    "fetch_account_models",
                    new_callable=AsyncMock,
                ):
                    with patch.object(
                        ProbeGemini,
                        "upload_images",
                        new_callable=AsyncMock,
                        return_value=[],
                    ):
                        with patch.object(
                            GEMINI_MODULE,
                            "raise_for_status",
                            side_effect=check_status,
                        ):
                            generator = ProbeGemini.create_async_generator(
                                model="gemini-3.5-flash",
                                messages=None,
                                prompt="prompt-only request",
                                cookies=source_cookies,
                                max_retries=1,
                            )
                            await generator.__anext__()
                            await generator.aclose()

        self.assertEqual(fetch.await_count, 2)
        self.assertEqual(len(session.calls), 2)
        self.assertTrue(first_response.released)
        self.assertEqual(session.calls[0]["data"]["at"], "token-1")
        self.assertEqual(session.calls[0]["params"]["bl"], "build-1")
        self.assertEqual(session.calls[0]["params"]["f.sid"], "sid-1")
        self.assertEqual(session.calls[1]["data"]["at"], "token-2")
        self.assertEqual(session.calls[1]["params"]["bl"], "build-2")
        self.assertEqual(session.calls[1]["params"]["f.sid"], "sid-2")
        self.assertIs(
            session.calls[0]["cookies_object"],
            session.calls[1]["cookies_object"],
        )
        self.assertIsNot(session.calls[0]["cookies_object"], source_cookies)
        self.assertEqual(source_cookies["__Secure-1PSIDTS"], "old")
        request = json.loads(json.loads(session.calls[1]["data"]["f.req"])[1])
        self.assertEqual(request[0][0], "prompt-only request")

    async def test_auto_refresh_waits_before_rotating(self):
        with patch.object(
            GEMINI_MODULE.asyncio,
            "sleep",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError(),
        ) as sleep:
            with patch.object(
                GEMINI_MODULE,
                "rotate_1psidts",
                new_callable=AsyncMock,
            ) as rotate:
                with self.assertRaises(asyncio.CancelledError):
                    await Gemini.start_auto_refresh(
                        cookies={"__Secure-1PSID": "session"}
                    )

        sleep.assert_awaited_once_with(Gemini.refresh_interval)
        rotate.assert_not_awaited()

    async def test_auto_refresh_ignores_missing_cookies(self):
        with patch.object(
            GEMINI_MODULE,
            "rotate_1psidts",
            new_callable=AsyncMock,
        ) as rotate:
            await Gemini.start_auto_refresh(cookies=None)

            rotate.assert_not_awaited()

    async def test_auto_refresh_uses_cookie_snapshot(self):
        source = {
            "__Secure-1PSID": "session",
            "__Secure-1PSIDTS": "old",
        }
        observed = {}

        async def rotate(_url, cookies, _proxy):
            observed["same_object"] = cookies is source
            observed["sidts"] = cookies.get("__Secure-1PSIDTS")
            raise asyncio.CancelledError()

        previous_cookies = Gemini._cookies
        Gemini._cookies = None
        try:
            with patch.object(
                GEMINI_MODULE.asyncio,
                "sleep",
                new_callable=AsyncMock,
            ) as sleep:
                with patch.object(
                    GEMINI_MODULE,
                    "rotate_1psidts",
                    new_callable=AsyncMock,
                    side_effect=rotate,
                ) as rotate_mock:
                    with self.assertRaises(asyncio.CancelledError):
                        await Gemini.start_auto_refresh(cookies=source)

            sleep.assert_awaited_once_with(Gemini.refresh_interval)
            rotate_mock.assert_awaited_once()
            self.assertFalse(observed["same_object"])
            self.assertEqual(observed["sidts"], "old")
            self.assertEqual(source["__Secure-1PSIDTS"], "old")
        finally:
            Gemini._cookies = previous_cookies

    async def test_stream_idle_timeout(self):
        class SlowContent:
            async def iter_any(self):
                await asyncio.sleep(0.05)
                yield b"late\n"

        with self.assertRaises(ResponseError):
            async for _ in _iter_response_lines(SlowContent(), idle_timeout=0.01):
                pass

    async def test_stream_reassembles_split_unicode_line(self):
        data = '[["wrb.fr","rpc","😀"]]\n'.encode()

        class SplitContent:
            async def iter_any(self):
                for chunk in (data[:20], data[20:23], data[23:]):
                    yield chunk

        lines = [line async for line in _iter_response_lines(SplitContent(), 1)]
        self.assertEqual(json.loads(lines[0])[0][2], "😀")


if __name__ == "__main__":
    unittest.main()
