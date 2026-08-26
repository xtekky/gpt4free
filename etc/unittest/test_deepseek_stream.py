from __future__ import annotations

import importlib
import json
import unittest
from unittest.mock import AsyncMock, patch

from g4f.Provider.needs_auth.DeepSeek import (
    CHAT_COMPLETION_ENDPOINT,
    CHAT_SESSION_CONTINUE_ENDPOINT,
    CHAT_SESSION_DELETE_ENDPOINT,
    CHAT_SESSION_RESUME_STREAM_ENDPOINT,
    DeepSeek,
    _build_chat_headers,
    _extract_chat_session_id,
    iter_deepseek_sse,
)
from g4f.errors import MissingAuthError, ResponseError
from g4f.providers.response import (
    FinishReason,
    JsonConversation,
    JsonRequest,
    Reasoning,
)


DEEPSEEK_MODULE = importlib.import_module("g4f.Provider.needs_auth.DeepSeek")


def sse_event(event: str, payload: dict) -> list[bytes]:
    return [
        f"event: {event}".encode(),
        f"data: {json.dumps(payload)}".encode(),
        b"",
    ]


class FakeStreamResponse:
    def __init__(
        self,
        lines: list[bytes] | None = None,
        *,
        payload: dict | None = None,
        content_type: str = "text/event-stream",
    ):
        self.status = 200
        self.headers = {"content-type": content_type}
        self.lines = lines or []
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    async def iter_lines(self):
        for line in self.lines:
            yield line

    async def json(self):
        return self.payload


class FakeStreamSession:
    def __init__(self, responses: list[FakeStreamResponse]):
        self.responses = iter(responses)
        self.post_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return next(self.responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False


class DeepSeekSSETest(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_authorization_survives_cookie_lookup(self):
        with (
            patch.object(DEEPSEEK_MODULE, "get_cookies", return_value={}),
            patch.object(DEEPSEEK_MODULE, "get_headers", return_value={}),
        ):
            generator = DeepSeek.create_async_generator(
                "deepseek-v3",
                [{"role": "user", "content": "hello"}],
                cookies=None,
                headers={"Authorization": "Bearer supplied"},
            )
            request = await generator.__anext__()
            await generator.aclose()

        self.assertIsInstance(request, JsonRequest)

    async def test_reasoning_effort_none_disables_r1_thinking(self):
        generator = DeepSeek.create_async_generator(
            "deepseek-r1",
            [{"role": "user", "content": "hello"}],
            cookies={},
            headers={"Authorization": "Bearer supplied"},
            reasoning_effort="none",
        )
        request = await generator.__anext__()
        await generator.aclose()

        self.assertFalse(request.get_dict()["thinking_enabled"])

    async def test_get_quota_reports_missing_auth_when_headers_are_unavailable(self):
        with (
            patch.object(DEEPSEEK_MODULE, "get_cookies", return_value={"sid": "x"}),
            patch.object(DEEPSEEK_MODULE, "get_headers", return_value=None),
        ):
            with self.assertRaises(MissingAuthError):
                await DeepSeek.get_quota()

    async def test_non_sse_resume_code_22_emits_full_message(self):
        response = FakeStreamResponse(
            payload={
                "data": {
                    "biz_code": 22,
                    "biz_msg": "resume returned full message",
                    "biz_data": {
                        "response": {
                            "message_id": 7,
                            "status": "FINISHED",
                            "fragments": [
                                {"type": "THINK", "content": "thought"},
                                {"type": "RESPONSE", "content": "answer"},
                            ],
                        }
                    },
                }
            },
            content_type="application/json",
        )
        session = FakeStreamSession([response])
        conversation = JsonConversation(parent_message_id=None)

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                )
            ]

        self.assertEqual(
            "".join(str(chunk) for chunk in chunks if isinstance(chunk, Reasoning)),
            "thought",
        )
        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "answer",
        )
        self.assertEqual(
            [chunk.reason for chunk in chunks if isinstance(chunk, FinishReason)],
            ["stop"],
        )
        self.assertEqual(conversation.parent_message_id, 7)

    async def test_finished_reasoning_only_stream_resumes_full_message_once(self):
        reasoning_only_response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [{"type": "THINK", "content": "thought"}],
                },
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        full_message_response = FakeStreamResponse(
            payload={
                "data": {
                    "biz_code": 22,
                    "biz_msg": "resume returned full message",
                    "biz_data": {
                        "response": {
                            "message_id": 4,
                            "status": "FINISHED",
                            "fragments": [
                                {"type": "THINK", "content": "thought"},
                                {"type": "RESPONSE", "content": "answer"},
                            ],
                        }
                    },
                }
            },
            content_type="application/json",
        )
        session = FakeStreamSession(
            [reasoning_only_response, full_message_response]
        )
        conversation = JsonConversation(parent_message_id=None)

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                    max_resume_attempts=3,
                )
            ]

        self.assertEqual(
            [url for url, _kwargs in session.post_calls],
            [CHAT_COMPLETION_ENDPOINT, CHAT_SESSION_RESUME_STREAM_ENDPOINT],
        )
        self.assertEqual(
            session.post_calls[1][1]["json"],
            {"chat_session_id": "session-1", "message_id": 4},
        )
        self.assertEqual(
            "".join(str(chunk) for chunk in chunks if isinstance(chunk, Reasoning)),
            "thought",
        )
        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "answer",
        )
        self.assertEqual(
            [chunk.reason for chunk in chunks if isinstance(chunk, FinishReason)],
            ["stop"],
        )
        self.assertEqual(conversation.parent_message_id, 4)

    async def test_finished_stream_raises_when_resumed_message_has_no_response(self):
        reasoning_only_response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {
                    "p": "response/fragments",
                    "o": "APPEND",
                    "v": [{"type": "THINK", "content": "thought"}],
                },
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        full_message_response = FakeStreamResponse(
            payload={
                "data": {
                    "biz_code": 22,
                    "biz_msg": "resume returned full message",
                    "biz_data": {
                        "response": {
                            "message_id": 4,
                            "status": "FINISHED",
                            "fragments": [
                                {"type": "THINK", "content": "thought"},
                            ],
                        }
                    },
                }
            },
            content_type="application/json",
        )
        session = FakeStreamSession(
            [reasoning_only_response, full_message_response]
        )
        conversation = JsonConversation(parent_message_id=None)
        chunks = []

        with (
            patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock),
            self.assertRaisesRegex(
                ResponseError,
                "DeepSeek finished without a response",
            ),
        ):
            async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
            ):
                chunks.append(chunk)

        self.assertEqual(
            [url for url, _kwargs in session.post_calls],
            [CHAT_COMPLETION_ENDPOINT, CHAT_SESSION_RESUME_STREAM_ENDPOINT],
        )
        self.assertEqual(
            "".join(str(chunk) for chunk in chunks if isinstance(chunk, Reasoning)),
            "thought",
        )
        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "",
        )

    async def test_finished_empty_stream_does_not_resume_when_disabled(self):
        response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession([response])
        conversation = JsonConversation(parent_message_id=None)

        with (
            patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock),
            self.assertRaisesRegex(
                ResponseError,
                "DeepSeek finished without a response",
            ),
        ):
            async for _chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                    max_resume_attempts=0,
            ):
                pass

        self.assertEqual(
            [url for url, _kwargs in session.post_calls],
            [CHAT_COMPLETION_ENDPOINT],
        )

    async def test_empty_response_resume_counts_toward_resume_limit(self):
        reasoning_only_response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        interrupted_resume = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
        )
        session = FakeStreamSession(
            [reasoning_only_response, interrupted_resume]
        )
        conversation = JsonConversation(parent_message_id=None)

        with (
            patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock),
            self.assertRaisesRegex(
                RuntimeError,
                "did not close normally after 1 resume attempt",
            ),
        ):
            async for _chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                    max_resume_attempts=1,
            ):
                pass

        self.assertEqual(
            [url for url, _kwargs in session.post_calls],
            [CHAT_COMPLETION_ENDPOINT, CHAT_SESSION_RESUME_STREAM_ENDPOINT],
        )

    async def test_full_message_without_response_always_raises(self):
        response = FakeStreamResponse(
            payload={
                "data": {
                    "biz_code": 22,
                    "biz_msg": "resume returned full message",
                    "biz_data": {
                        "response": {
                            "message_id": 4,
                            "status": "WIP",
                            "fragments": [
                                {"type": "THINK", "content": "thought"},
                            ],
                        }
                    },
                }
            },
            content_type="application/json",
        )
        session = FakeStreamSession([response])
        conversation = JsonConversation(parent_message_id=None)

        with (
            patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock),
            self.assertRaisesRegex(
                ResponseError,
                "DeepSeek finished without a response",
            ),
        ):
            async for _chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
            ):
                pass

    async def test_indexed_response_fragment_switches_from_reasoning(self):
        response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {
                    "v": {
                        "response": {
                            "message_id": 4,
                            "fragments": [
                                {"type": "THINK", "content": "thought"},
                            ],
                        }
                    }
                },
            )
            + sse_event(
                "message",
                {
                    "p": "response/fragments/-1",
                    "o": "APPEND",
                    "v": {"type": "RESPONSE", "content": ""},
                },
            )
            + sse_event(
                "message",
                {
                    "p": "response/fragments/-1",
                    "o": "SET",
                    "v": {"type": "RESPONSE", "content": "answer"},
                },
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession([response])
        conversation = JsonConversation(parent_message_id=None)

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                )
            ]

        self.assertEqual(len(session.post_calls), 1)
        self.assertEqual(
            "".join(str(chunk) for chunk in chunks if isinstance(chunk, Reasoning)),
            "thought",
        )
        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "answer",
        )

    async def test_indexed_content_uses_its_fragment_type(self):
        response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {
                    "v": {
                        "response": {
                            "message_id": 4,
                            "fragments": [
                                {"type": "THINK", "content": ""},
                                {"type": "RESPONSE", "content": ""},
                            ],
                        }
                    }
                },
            )
            + sse_event(
                "message",
                {
                    "p": "response/fragments/-1/type",
                    "o": "SET",
                    "v": "TEMPLATE_RESPONSE",
                },
            )
            + sse_event(
                "message",
                {
                    "p": "response/fragments/0/content",
                    "o": "APPEND",
                    "v": "thought",
                },
            )
            + sse_event(
                "message",
                {
                    "p": "response/fragments/1/content",
                    "o": "APPEND",
                    "v": "answer",
                },
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession([response])
        conversation = JsonConversation(parent_message_id=None)

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                )
            ]

        self.assertEqual(
            "".join(str(chunk) for chunk in chunks if isinstance(chunk, Reasoning)),
            "thought",
        )
        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "answer",
        )

    async def test_non_sse_business_error_exposes_biz_message(self):
        response = FakeStreamResponse(
            payload={
                "data": {
                    "biz_code": 40101,
                    "biz_msg": "authorization expired",
                    "biz_data": None,
                }
            },
            content_type="application/json",
        )
        session = FakeStreamSession([response])

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            with self.assertRaisesRegex(RuntimeError, "authorization expired"):
                _ = [
                    chunk
                    async for chunk in DeepSeek.iter_chat_stream(
                        session,
                        JsonConversation(parent_message_id=None),
                        {"chat_session_id": "session-1", "prompt": "prompt"},
                    )
                ]

    async def test_session_creation_exposes_biz_message(self):
        session = FakeStreamSession(
            [
                FakeStreamResponse(
                    payload={
                        "data": {
                            "biz_code": 50301,
                            "biz_msg": "session unavailable",
                            "biz_data": None,
                        }
                    },
                    content_type="application/json",
                )
            ]
        )
        generator = DeepSeek.create_async_generator(
            "deepseek-v3",
            [{"role": "user", "content": "hello"}],
            cookies={},
            headers={"Authorization": "Bearer supplied"},
        )

        await generator.__anext__()
        with (
            patch.object(DEEPSEEK_MODULE, "StreamSession", return_value=session),
            patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock),
        ):
            with self.assertRaisesRegex(RuntimeError, "session unavailable"):
                await generator.__anext__()

    async def test_delete_session_uses_one_observed_request(self):
        session = FakeStreamSession(
            [
                FakeStreamResponse(
                    payload={"data": {"biz_code": 0, "biz_data": {}}},
                    content_type="application/json",
                )
            ]
        )

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            deleted = await DeepSeek.delete_chat_session(
                session,
                "session-1",
                {"authorization": "Bearer redacted"},
            )

        self.assertTrue(deleted)
        self.assertEqual(
            session.post_calls,
            [
                (
                    CHAT_SESSION_DELETE_ENDPOINT,
                    {
                        "headers": {"authorization": "Bearer redacted"},
                        "json": {"chat_session_id": "session-1"},
                    },
                )
            ],
        )

    async def test_delete_session_business_error_returns_false_without_retry(self):
        session = FakeStreamSession(
            [
                FakeStreamResponse(
                    payload={
                        "data": {
                            "biz_code": 50001,
                            "biz_msg": "delete unavailable",
                            "biz_data": None,
                        }
                    },
                    content_type="application/json",
                )
            ]
        )

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            deleted = await DeepSeek.delete_chat_session(
                session,
                "session-1",
                {"authorization": "Bearer redacted"},
            )

        self.assertIs(deleted, False)
        self.assertEqual(len(session.post_calls), 1)

    def test_extracts_current_and_legacy_chat_session_ids(self):
        self.assertEqual(
            _extract_chat_session_id(
                {
                    "data": {
                        "biz_data": {
                            "chat_session": {"id": "current-session-id"}
                        }
                    }
                }
            ),
            "current-session-id",
        )
        self.assertEqual(
            _extract_chat_session_id(
                {"data": {"biz_data": {"id": "legacy-session-id"}}}
            ),
            "legacy-session-id",
        )

    async def test_parser_preserves_close_event(self):
        response = FakeStreamResponse(
            sse_event("message", {"v": "chunk"})
            + sse_event("update_session", {"updated_at": 1})
            + sse_event("close", {"auto_resume": False})
        )

        events = [event async for event in iter_deepseek_sse(response)]

        self.assertEqual(
            [event_type for event_type, _payload in events],
            ["message", "update_session", "close"],
        )

    async def test_incomplete_closed_stream_continues_same_message(self):
        first_response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {"p": "response/fragments/-1/content", "o": "APPEND", "v": "A"},
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "INCOMPLETE"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        continued_response = FakeStreamResponse(
            sse_event(
                "message",
                {"p": "response/fragments/-1/content", "o": "APPEND", "v": "B"},
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession([first_response, continued_response])
        conversation = JsonConversation(parent_message_id=None)

        with (
            patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock),
            patch.object(DEEPSEEK_MODULE.debug, "log") as log,
        ):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                    {"x-ds-pow-response": "completion-pow"},
                    auto_continue=True,
                    max_continue_attempts=3,
                    max_resume_attempts=3,
                )
            ]

        log_output = "\n".join(
            str(call.args[0]) for call in log.call_args_list if call.args
        )

        self.assertEqual("".join(map(str, chunks)), "AB")
        self.assertEqual(conversation.parent_message_id, 4)
        self.assertEqual(
            [url for url, _kwargs in session.post_calls],
            [CHAT_COMPLETION_ENDPOINT, CHAT_SESSION_CONTINUE_ENDPOINT],
        )
        self.assertEqual(
            session.post_calls[1][1]["json"],
            {
                "chat_session_id": "session-1",
                "message_id": 4,
                "fallback_to_resume": True,
            },
        )
        self.assertIn(
            "DeepSeekAuth: Stream status: status=INCOMPLETE source=patch "
            "operation=SET interpretation=requires_continue",
            log_output,
        )
        self.assertIn(
            "DeepSeekAuth: Stream closed: status=INCOMPLETE action=continue "
            "reason=incomplete_status finish_reason=none auto_resume=false "
            "click_behavior=none response_chars=1 reasoning_chars=0 "
            "message_id_present=true",
            log_output,
        )

    async def test_incomplete_without_operation_continues(self):
        incomplete_response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {"p": "response/fragments/-1/content", "o": "APPEND", "v": "A"},
            )
            + sse_event(
                "message",
                {"p": "response/status", "v": "INCOMPLETE"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        continued_response = FakeStreamResponse(
            sse_event(
                "message",
                {"p": "response/fragments/-1/content", "o": "APPEND", "v": "B"},
            )
            + sse_event(
                "message",
                {"p": "response/status", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession([incomplete_response, continued_response])
        conversation = JsonConversation(parent_message_id=None)

        with (
            patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock),
            patch.object(DEEPSEEK_MODULE.debug, "log") as log,
        ):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                    auto_continue=True,
                )
            ]

        log_output = "\n".join(
            str(call.args[0]) for call in log.call_args_list if call.args
        )

        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "AB",
        )
        self.assertEqual(
            [chunk.reason for chunk in chunks if isinstance(chunk, FinishReason)],
            ["stop"],
        )
        self.assertEqual(
            [url for url, _kwargs in session.post_calls],
            [CHAT_COMPLETION_ENDPOINT, CHAT_SESSION_CONTINUE_ENDPOINT],
        )
        self.assertIn(
            "DeepSeekAuth: Stream status: status=INCOMPLETE source=patch "
            "operation=none interpretation=requires_continue",
            log_output,
        )
        self.assertIn(
            "DeepSeekAuth: Stream closed: status=INCOMPLETE action=continue "
            "reason=incomplete_status finish_reason=none auto_resume=false "
            "click_behavior=none response_chars=1 reasoning_chars=0 "
            "message_id_present=true",
            log_output,
        )

    async def test_incomplete_snapshot_continues_without_prior_status_patch(self):
        incomplete_response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {
                    "v": {
                        "response": {
                            "message_id": 4,
                            "status": "INCOMPLETE",
                            "fragments": [
                                {"type": "RESPONSE", "content": "A"}
                            ],
                        }
                    }
                },
            )
            + sse_event("close", {"auto_resume": False})
        )
        continued_response = FakeStreamResponse(
            sse_event(
                "message",
                {"p": "response/fragments/-1/content", "o": "APPEND", "v": "B"},
            )
            + sse_event(
                "message",
                {"p": "response/status", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession([incomplete_response, continued_response])
        conversation = JsonConversation(parent_message_id=None)

        with (
            patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock),
            patch.object(DEEPSEEK_MODULE.debug, "log") as log,
        ):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                    auto_continue=True,
                )
            ]

        log_output = "\n".join(
            str(call.args[0]) for call in log.call_args_list if call.args
        )
        self.assertEqual("".join(map(str, chunks)), "AB")
        self.assertEqual(
            [url for url, _kwargs in session.post_calls],
            [CHAT_COMPLETION_ENDPOINT, CHAT_SESSION_CONTINUE_ENDPOINT],
        )
        self.assertIn(
            "DeepSeekAuth: Stream status: status=INCOMPLETE source=snapshot "
            "operation=none interpretation=requires_continue",
            log_output,
        )

    async def test_unclosed_stream_resumes_until_close_without_snapshot_duplicates(self):
        interrupted_response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {"p": "response/fragments/-1/content", "o": "APPEND", "v": "A"},
            )
        )
        resumed_but_unclosed_response = FakeStreamResponse(
            sse_event(
                "message",
                {
                    "v": {
                        "response": {
                            "message_id": 4,
                            "fragments": [
                                {"type": "RESPONSE", "content": "AB"}
                            ],
                        }
                    }
                },
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
        )
        finally_closed_response = FakeStreamResponse(
            sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession(
            [
                interrupted_response,
                resumed_but_unclosed_response,
                finally_closed_response,
            ]
        )
        conversation = JsonConversation(parent_message_id=None)

        with (
            patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock),
            patch.object(DEEPSEEK_MODULE.debug, "log") as log,
        ):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                    {"x-ds-pow-response": "completion-pow"},
                    max_continue_attempts=3,
                    max_resume_attempts=3,
                )
            ]

        log_output = "\n".join(
            str(call.args[0]) for call in log.call_args_list if call.args
        )

        self.assertEqual("".join(map(str, chunks)), "AB")
        self.assertEqual(
            [url for url, _kwargs in session.post_calls],
            [
                CHAT_COMPLETION_ENDPOINT,
                CHAT_SESSION_RESUME_STREAM_ENDPOINT,
                CHAT_SESSION_RESUME_STREAM_ENDPOINT,
            ],
        )
        for _url, kwargs in session.post_calls[1:]:
            self.assertEqual(
                kwargs["json"],
                {"chat_session_id": "session-1", "message_id": 4},
            )
        self.assertIn(
            "DeepSeekAuth: Stream ended without close: status=none "
            "message_id_present=true error=none",
            log_output,
        )
        self.assertIn(
            "DeepSeekAuth: Resuming interrupted response stream: "
            "action=resume_stream attempt=1",
            log_output,
        )

    async def test_unclosed_incomplete_stream_resumes_before_it_continues(self):
        interrupted_response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {"p": "response/fragments/-1/content", "o": "APPEND", "v": "A"},
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "INCOMPLETE"},
            )
        )
        resumed_response = FakeStreamResponse(
            sse_event(
                "message",
                {
                    "v": {
                        "response": {
                            "message_id": 4,
                            "status": "INCOMPLETE",
                            "fragments": [
                                {"type": "RESPONSE", "content": "A"}
                            ],
                        }
                    }
                },
            )
            + sse_event("update_session", {"v": "must-not-be-output"})
            + sse_event("close", {"auto_resume": False})
        )
        continued_response = FakeStreamResponse(
            sse_event(
                "message",
                {"p": "response/fragments/-1/content", "o": "APPEND", "v": "B"},
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession(
            [interrupted_response, resumed_response, continued_response]
        )
        conversation = JsonConversation(parent_message_id=None)

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                    {"x-ds-pow-response": "completion-pow"},
                    max_continue_attempts=3,
                    max_resume_attempts=3,
                )
            ]

        self.assertEqual("".join(map(str, chunks)), "AB")
        self.assertEqual(
            [url for url, _kwargs in session.post_calls],
            [
                CHAT_COMPLETION_ENDPOINT,
                CHAT_SESSION_RESUME_STREAM_ENDPOINT,
                CHAT_SESSION_CONTINUE_ENDPOINT,
            ],
        )

    async def test_terminal_statuses_are_exposed_as_finish_reasons(self):
        expected_reasons = {
            "FINISHED": "stop",
            "CONTENT_FILTER": "content_filter",
            "CONTEXT_LENGTH_EXCEEDED": "length",
            "INCOMPLETE": "incomplete",
            "WIP": "wip",
            "TIMEOUT": "timeout",
        }

        for status, expected_reason in expected_reasons.items():
            with self.subTest(status=status):
                response = FakeStreamResponse(
                    sse_event("ready", {"response_message_id": 4})
                    + sse_event(
                        "message",
                        {
                            "p": "response/fragments",
                            "o": "APPEND",
                            "v": [{"type": "RESPONSE", "content": "answer"}],
                        },
                    )
                    + sse_event(
                        "message",
                        {"p": "response/status", "o": "SET", "v": status},
                    )
                    + sse_event("close", {"auto_resume": False})
                )
                session = FakeStreamSession([response])
                conversation = JsonConversation(parent_message_id=None)

                with patch.object(
                    DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock
                ):
                    chunks = [
                        chunk
                        async for chunk in DeepSeek.iter_chat_stream(
                            session,
                            conversation,
                            {"chat_session_id": "session-1", "prompt": "prompt"},
                            auto_continue=False,
                        )
                    ]

                finish_reasons = [
                    chunk.reason for chunk in chunks if isinstance(chunk, FinishReason)
                ]
                self.assertEqual(finish_reasons, [expected_reason])
                self.assertEqual(len(session.post_calls), 1)

    async def test_fragment_types_separate_response_reasoning_and_metadata(self):
        response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {
                    "v": {
                        "response": {
                            "message_id": 4,
                            "fragments": [
                                {"type": "REQUEST", "content": "prompt"},
                                {"type": "THINK", "content": "think"},
                                {"type": "SEARCH", "content": "search"},
                                {"type": "TOOL_FIND", "content": "tool"},
                                {"type": "RESPONSE", "content": "answer"},
                                {
                                    "type": "TEMPLATE_RESPONSE",
                                    "content": "template",
                                },
                                {"type": "FILE", "content": "file"},
                                {"type": "TIP", "content": "tip"},
                                {"type": "READ_LINK", "content": "link"},
                                {"type": "UNKNOWN", "content": "unknown"},
                            ],
                        }
                    }
                },
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession([response])
        conversation = JsonConversation(parent_message_id=None)

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                )
            ]

        self.assertEqual(
            "".join(str(chunk) for chunk in chunks if isinstance(chunk, Reasoning)),
            "thinksearchtool",
        )
        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "answertemplate",
        )

    async def test_set_fragments_emits_only_snapshot_delta(self):
        response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {
                    "p": "response/fragments",
                    "o": "SET",
                    "v": [{"type": "RESPONSE", "content": "A"}],
                },
            )
            + sse_event(
                "message",
                {
                    "p": "response/fragments",
                    "o": "SET",
                    "v": [{"type": "RESPONSE", "content": "AB"}],
                },
            )
            + sse_event(
                "message",
                {"p": "response/status", "o": "SET", "v": "FINISHED"},
            )
            + sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession([response])
        conversation = JsonConversation(parent_message_id=None)

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                )
            ]

        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "AB",
        )

    async def test_batch_applies_append_fragments_and_status(self):
        response = FakeStreamResponse(
            sse_event("ready", {"response_message_id": 4})
            + sse_event(
                "message",
                {
                    "o": "BATCH",
                    "v": [
                        {
                            "p": "response/fragments",
                            "o": "APPEND",
                            "v": [
                                {
                                    "type": "TEMPLATE_RESPONSE",
                                    "content": "answer",
                                }
                            ],
                        },
                        {
                            "p": "response/status",
                            "o": "SET",
                            "v": "FINISHED",
                        },
                    ],
                },
            )
            + sse_event("close", {"auto_resume": False})
        )
        session = FakeStreamSession([response])
        conversation = JsonConversation(parent_message_id=None)

        with patch.object(DEEPSEEK_MODULE, "raise_for_status", new_callable=AsyncMock):
            chunks = [
                chunk
                async for chunk in DeepSeek.iter_chat_stream(
                    session,
                    conversation,
                    {"chat_session_id": "session-1", "prompt": "prompt"},
                )
            ]

        self.assertEqual(
            "".join(chunk for chunk in chunks if isinstance(chunk, str)),
            "answer",
        )
        self.assertEqual(
            [chunk.reason for chunk in chunks if isinstance(chunk, FinishReason)],
            ["stop"],
        )

    def test_chat_headers_forward_hif_but_drop_stale_pow(self):
        headers = _build_chat_headers(
            {
                "Authorization": "Bearer redacted",
                "X-Hif-Leim": "hif-redacted",
                "X-Hif-Dliq": "dliq-redacted",
                "X-Client-Version": "2.4.0",
                "X-Ds-Pow-Response": "stale-pow",
            },
            "Bearer redacted",
        )

        self.assertEqual(headers["x-hif-leim"], "hif-redacted")
        self.assertEqual(headers["x-hif-dliq"], "dliq-redacted")
        self.assertEqual(headers["x-client-version"], "2.4.0")
        self.assertEqual(headers["referer"], "https://chat.deepseek.com/a/chat/")
        self.assertNotIn("x-app-version", headers)
        self.assertNotIn("x-ds-pow-response", headers)

    def test_completion_payload_matches_current_web_contract(self):
        conversation = JsonConversation(parent_message_id=None)
        conversation.chat_session_id = "session-1"

        payload = DEEPSEEK_MODULE._build_completion_payload(
            conversation,
            prompt="hello",
            model_type="default",
            ref_file_ids=["file-1"],
            thinking_enabled=True,
            search_enabled=False,
        )

        self.assertEqual(
            payload,
            {
                "action": None,
                "chat_session_id": "session-1",
                "parent_message_id": None,
                "model_type": "default",
                "prompt": "hello",
                "ref_file_ids": ["file-1"],
                "thinking_enabled": True,
                "search_enabled": False,
                "preempt": False,
            },
        )

        conversation.parent_message_id = 9
        self.assertEqual(
            DEEPSEEK_MODULE._build_completion_payload(
                conversation,
                prompt="follow up",
                model_type="default",
                ref_file_ids=[],
                thinking_enabled=False,
                search_enabled=True,
            )["parent_message_id"],
            9,
        )


if __name__ == "__main__":
    unittest.main()
