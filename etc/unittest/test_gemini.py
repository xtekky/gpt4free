from __future__ import annotations

import asyncio
import json
import unittest

from g4f.Provider.needs_auth.Gemini import (
    ACCOUNT_STATUS_AVAILABLE,
    ACCOUNT_STATUS_UNAUTHENTICATED,
    MODEL_HEADER_KEY,
    Gemini,
    _build_model_headers,
    _extract_gemini_error_code,
    _extract_reasoning,
    _iter_response_lines,
    _parse_account_models,
    _parse_google_frames,
    _resolve_model,
)
from g4f.errors import MissingAuthError, ResponseError
from g4f.models import ModelRegistry


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
