from __future__ import annotations

import importlib
import unittest
from unittest.mock import AsyncMock, Mock, patch

from g4f.Provider.needs_auth.DeepSeek import (
    FILE_UPLOAD_ENDPOINT,
    FILE_UPLOAD_PATH,
    POW_CHALLENGE_ENDPOINT,
    DeepSeek,
    DeepSeekPOW,
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
    async def test_pow_rejects_unsupported_algorithm_before_solving(self):
        challenge = {
            "algorithm": "FutureHashV2",
            "challenge": "challenge",
            "salt": "salt",
            "difficulty": 1,
            "expire_at": 1,
            "signature": "signature",
            "target_path": FILE_UPLOAD_PATH,
        }
        session = FakeSession(
            post_responses=[
                FakeResponse({"data": {"biz_data": {"challenge": challenge}}})
            ]
        )

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            with patch.object(
                DEEPSEEK_MODULE,
                "DeepSeekPOW",
                side_effect=AssertionError("solver must not be created"),
            ):
                with self.assertRaisesRegex(RuntimeError, "FutureHashV2"):
                    await DeepSeek.create_pow_response(session, FILE_UPLOAD_PATH)

    async def test_pow_solver_runs_off_the_event_loop(self):
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
                FakeResponse({"data": {"biz_data": {"challenge": challenge}}})
            ]
        )

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            with patch.object(
                DEEPSEEK_MODULE.asyncio,
                "to_thread",
                new_callable=AsyncMock,
                return_value="pow-response",
            ) as to_thread:
                result = await DeepSeek.create_pow_response(
                    session, FILE_UPLOAD_PATH
                )

        self.assertEqual(result, "pow-response")
        to_thread.assert_awaited_once()

    def test_pow_rejects_missing_answer(self):
        solver = DeepSeekPOW.__new__(DeepSeekPOW)
        solver.hasher = Mock()
        solver.hasher.calculate_hash.return_value = None

        with self.assertRaisesRegex(RuntimeError, "no answer"):
            solver.solve_challenge(
                {
                    "algorithm": "DeepSeekHashV1",
                    "challenge": "challenge",
                    "salt": "salt",
                    "difficulty": 1,
                    "expire_at": 1,
                    "signature": "signature",
                    "target_path": FILE_UPLOAD_PATH,
                }
            )

    async def test_pow_exposes_inner_business_error(self):
        session = FakeSession(
            post_responses=[
                FakeResponse(
                    {
                        "data": {
                            "biz_code": 42901,
                            "biz_msg": "challenge rate limited",
                            "biz_data": None,
                        }
                    }
                )
            ]
        )

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            with self.assertRaisesRegex(RuntimeError, "challenge rate limited"):
                await DeepSeek.create_pow_response(session, FILE_UPLOAD_PATH)

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
                FakeResponse(
                    {
                        "data": {
                            "biz_code": 0,
                            "biz_msg": "",
                            "biz_data": {"id": "file-123"},
                        }
                    }
                ),
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

    async def test_upload_rejects_inner_business_error(self):
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
                FakeResponse(
                    {
                        "data": {
                            "biz_code": 40012,
                            "biz_msg": "file rejected",
                            "biz_data": {"id": "must-not-be-used"},
                        }
                    }
                ),
            ]
        )

        with patch.object(DEEPSEEK_MODULE, "FormData", FakeFormData):
            with patch.object(DEEPSEEK_MODULE, "DeepSeekPOW", return_value=FakePowSolver()):
                with patch.object(
                    DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock
                ):
                    with self.assertRaisesRegex(RuntimeError, "file rejected"):
                        await DeepSeek.upload_file(session, b'{"a":1}', "sample.json")

    async def test_upload_accepts_unicode_text_using_filename_mime(self):
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
                FakeResponse(
                    {
                        "data": {
                            "biz_code": 0,
                            "biz_data": {"id": "text-file"},
                        }
                    }
                ),
            ]
        )

        with patch.object(DEEPSEEK_MODULE, "FormData", FakeFormData):
            with patch.object(DEEPSEEK_MODULE, "DeepSeekPOW", return_value=FakePowSolver()):
                with patch.object(
                    DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock
                ):
                    result = await DeepSeek.upload_file(
                        session,
                        "مرحبا DeepSeek".encode("utf-8"),
                        "notes.txt",
                    )

        self.assertEqual(result["file_id"], "text-file")
        upload_form = session.post_calls[1][1]["data"]
        self.assertEqual(upload_form.fields[0]["content_type"], "text/plain")

    async def test_wait_for_file_parsing_polls_until_success(self):
        session = FakeSession(
            get_responses=[
                FakeResponse(
                    {
                        "data": {
                            "biz_code": 0,
                            "biz_data": {"files": [{"status": "PROCESSING"}]},
                        }
                    }
                ),
                FakeResponse(
                    {
                        "data": {
                            "biz_code": 0,
                            "biz_data": {"files": [{"status": "SUCCESS"}]},
                        }
                    }
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

    async def test_wait_for_file_parsing_rejects_inner_business_error(self):
        session = FakeSession(
            get_responses=[
                FakeResponse(
                    {
                        "data": {
                            "biz_code": 40404,
                            "biz_msg": "file not found",
                            "biz_data": None,
                        }
                    }
                )
            ]
        )

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            with self.assertRaisesRegex(RuntimeError, "file not found"):
                await DeepSeek.wait_for_file_parsed(session, "missing", timeout=0)

    async def test_wait_for_file_parsing_rejects_current_terminal_statuses(self):
        terminal_statuses = {
            "FAILED",
            "CONTENT_FILTER",
            "CONTENT_TOO_LONG",
            "CANCELLED",
            "CONTENT_EMPTY",
            "_CUSTOM_SYSTEM_ERROR_FAIL",
            "_CUSTOM_FROM_SHARE",
        }

        for status in terminal_statuses:
            with self.subTest(status=status):
                session = FakeSession(
                    get_responses=[
                        FakeResponse(
                            {
                                "data": {
                                    "biz_code": 0,
                                    "biz_data": {"files": [{"status": status}]},
                                }
                            }
                        )
                    ]
                )
                with patch.object(
                    DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock
                ):
                    with self.assertRaisesRegex(RuntimeError, status):
                        await DeepSeek.wait_for_file_parsed(
                            session, "file-123", timeout=0
                        )

    async def test_upload_files_processes_every_media_item(self):
        media = [(b"first", "first.txt"), (b"second", "second.txt")]
        with patch.object(
            DeepSeek,
            "upload_file",
            new_callable=AsyncMock,
            side_effect=[{"file_id": "file-1"}, {"file_id": "file-2"}],
        ) as upload_file:
            with patch.object(
                DeepSeek, "wait_for_file_parsed", new_callable=AsyncMock
            ) as wait_for_file_parsed:
                file_ids = await DeepSeek.upload_files(
                    object(),
                    media,
                    thinking_enabled=True,
                    model_type="default",
                )

        self.assertEqual(file_ids, ["file-1", "file-2"])
        self.assertEqual(upload_file.await_count, 2)
        self.assertEqual(wait_for_file_parsed.await_count, 2)

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
