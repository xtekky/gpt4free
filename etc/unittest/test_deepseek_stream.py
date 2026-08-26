from __future__ import annotations

import importlib
import json
import unittest
from unittest.mock import AsyncMock, patch

from g4f.Provider.needs_auth.DeepSeek import (
    CHAT_COMPLETION_ENDPOINT,
    CHAT_SESSION_CONTINUE_ENDPOINT,
    CHAT_SESSION_RESUME_STREAM_ENDPOINT,
    DeepSeek,
    _build_chat_headers,
    _extract_chat_session_id,
    iter_deepseek_sse,
)
from g4f.providers.response import FinishReason, JsonConversation, Reasoning


DEEPSEEK_MODULE = importlib.import_module("g4f.Provider.needs_auth.DeepSeek")


def sse_event(event: str, payload: dict) -> list[bytes]:
    return [
        f"event: {event}".encode(),
        f"data: {json.dumps(payload)}".encode(),
        b"",
    ]


class FakeStreamResponse:
    def __init__(self, lines: list[bytes]):
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self.lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback):
        return False

    async def iter_lines(self):
        for line in self.lines:
            yield line


class FakeStreamSession:
    def __init__(self, responses: list[FakeStreamResponse]):
        self.responses = iter(responses)
        self.post_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return next(self.responses)


class DeepSeekSSETest(unittest.IsolatedAsyncioTestCase):
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
                "X-Client-Version": "2.4.0",
                "X-Ds-Pow-Response": "stale-pow",
            },
            "Bearer redacted",
        )

        self.assertEqual(headers["x-hif-leim"], "hif-redacted")
        self.assertEqual(headers["x-client-version"], "2.4.0")
        self.assertNotIn("x-ds-pow-response", headers)


if __name__ == "__main__":
    unittest.main()
