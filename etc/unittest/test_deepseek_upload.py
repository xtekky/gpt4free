from __future__ import annotations

import importlib
import unittest
from unittest.mock import AsyncMock, patch

from g4f.Provider.needs_auth.DeepSeek import (
    FILE_UPLOAD_ENDPOINT,
    FILE_UPLOAD_PATH,
    POW_CHALLENGE_ENDPOINT,
    DeepSeek,
    _build_upload_session_headers,
)


DEEPSEEK_MODULE = importlib.import_module("g4f.Provider.needs_auth.DeepSeek")


class FakeResponse:
    def __init__(self, payload, content_type="application/json"):
        self.status = 200
        self.headers = {"content-type": content_type}
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    async def json(self):
        return self.payload

    async def text(self):
        return str(self.payload)


class FakeSession:
    def __init__(self, *, post_responses=None, get_responses=None):
        self.post_responses = iter(post_responses or [])
        self.get_responses = iter(get_responses or [])
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return next(self.post_responses)

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return next(self.get_responses)


class FakeFormData:
    def __init__(self):
        self.fields = []

    def add_field(self, name, data=None, content_type=None, filename=None):
        self.fields.append(
            {
                "name": name,
                "data": data,
                "content_type": content_type,
                "filename": filename,
            }
        )


class FakePowSolver:
    def solve_challenge(self, challenge):
        return f"pow:{challenge['target_path']}"


class DeepSeekUploadTest(unittest.IsolatedAsyncioTestCase):
    async def test_upload_uses_current_endpoint_dedicated_pow_and_nested_file_id(self):
        challenge = {
            "algorithm": "DeepSeekHashV1",
            "challenge": "challenge",
            "salt": "salt",
            "difficulty": 1,
            "expire_at": 1,
            "signature": "signature",
            "target_path": FILE_UPLOAD_PATH,
        }
        session = FakeSession(
            post_responses=[
                FakeResponse({"data": {"biz_data": {"challenge": challenge}}}),
                FakeResponse({"code": 0, "data": {"biz_data": {"id": "file-123"}}}),
            ]
        )

        with patch.object(DEEPSEEK_MODULE, "FormData", FakeFormData):
            with patch.object(DEEPSEEK_MODULE, "DeepSeekPOW", return_value=FakePowSolver()):
                with patch.object(
                    DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock
                ):
                    result = await DeepSeek.upload_file(
                        session,
                        b'{"a":1}',
                        "sample.json",
                        thinking_enabled=False,
                    )

        self.assertEqual(result["file_id"], "file-123")
        self.assertEqual(session.post_calls[0][0], POW_CHALLENGE_ENDPOINT)
        self.assertEqual(
            session.post_calls[0][1]["json"], {"target_path": FILE_UPLOAD_PATH}
        )
        self.assertEqual(session.post_calls[1][0], FILE_UPLOAD_ENDPOINT)

        upload_kwargs = session.post_calls[1][1]
        self.assertEqual(upload_kwargs["headers"]["x-file-size"], "7")
        self.assertEqual(upload_kwargs["headers"]["x-model-type"], "default")
        self.assertEqual(upload_kwargs["headers"]["x-thinking-enabled"], "0")
        self.assertEqual(
            upload_kwargs["headers"]["x-ds-pow-response"],
            f"pow:{FILE_UPLOAD_PATH}",
        )
        self.assertNotIn("content-type", upload_kwargs["headers"])
        self.assertEqual(
            upload_kwargs["data"].fields,
            [
                {
                    "name": "file",
                    "data": b'{"a":1}',
                    "content_type": "application/json",
                    "filename": "sample.json",
                }
            ],
        )

    async def test_wait_for_file_parsing_polls_until_success(self):
        session = FakeSession(
            get_responses=[
                FakeResponse(
                    {"data": {"biz_data": {"files": [{"status": "PROCESSING"}]}}}
                ),
                FakeResponse(
                    {"data": {"biz_data": {"files": [{"status": "SUCCESS"}]}}}
                ),
            ]
        )

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            with patch.object(
                DEEPSEEK_MODULE.asyncio, "sleep", new_callable=AsyncMock
            ) as sleep:
                await DeepSeek.wait_for_file_parsed(
                    session, "file-123", timeout=5, poll_interval=0.01
                )

        self.assertEqual(len(session.get_calls), 2)
        self.assertEqual(
            session.get_calls[0][1]["params"], {"file_ids": "file-123"}
        )
        sleep.assert_awaited_once_with(0.01)

    def test_upload_session_does_not_inherit_json_or_chat_pow_headers(self):
        headers = {
            "authorization": "Bearer redacted",
            "content-type": "application/json",
            "x-ds-pow-response": "chat-pow",
            "x-client-platform": "web",
        }

        upload_headers = _build_upload_session_headers(headers)

        self.assertEqual(upload_headers["authorization"], "Bearer redacted")
        self.assertEqual(upload_headers["x-client-platform"], "web")
        self.assertNotIn("content-type", upload_headers)
        self.assertNotIn("x-ds-pow-response", upload_headers)


if __name__ == "__main__":
    unittest.main()
