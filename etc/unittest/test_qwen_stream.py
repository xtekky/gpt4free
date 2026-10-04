"""Qwen 0.3.12 stream contracts observed through the browser extension."""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import importlib
from pathlib import Path

import g4f.debug
from g4f.Provider.Qwen import Qwen
from g4f.errors import MissingAuthError, ResponseError, RateLimitError
from g4f.providers.response import FinishReason, ImageResponse, JsonConversation, Reasoning, Sources, Usage

g4f.debug.version_check = False
QWEN_MODULE = importlib.import_module("g4f.Provider.Qwen")


def delta(phase=None, content="", status="typing", **kwargs):
    return {"choices": [{"delta": {"phase": phase, "content": content, "status": status, **kwargs}}]}


class Response:
    def __init__(self, chunks=(), payload=None):
        self.chunks = chunks
        self.payload = payload
        self.headers = {"content-type": "application/json" if payload is not None else "text/event-stream"}

    async def iter_lines(self):
        yield b": keepalive"
        for chunk in self.chunks:
            yield b"data: " + json.dumps(chunk, ensure_ascii=False).encode()
            yield b""
        yield b"data: [DONE]"

    async def json(self):
        return self.payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False


class TestQwenStream(unittest.IsolatedAsyncioTestCase):
    async def read(self, chunks=(), payload=None):
        self.conversation = JsonConversation(chat_id="chat", parent_id=None)
        return [item async for item in Qwen._read_response(Response(chunks, payload), self.conversation, "test")]

    async def request(self, chunks=None, response=None, **kwargs):
        session = MagicMock()
        session.post.return_value = response or Response(chunks or [delta("answer", "Answer")])
        session.get.return_value = kwargs.pop("auth_response", Response(payload={}))
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        self.session = session
        conversation = kwargs.pop("conversation", JsonConversation(chat_id="chat", parent_id=None, cookies={}))
        status_error = kwargs.pop("status_error", None)
        with patch.object(QWEN_MODULE, "StreamSession", return_value=session), patch.object(Qwen, "_ensure_auth", new=AsyncMock()) as auth, patch.object(Qwen, "_get_headers", return_value={}) as headers, patch.object(Qwen, "_get_req_headers", new=AsyncMock(return_value={})), patch.object(Qwen, "raise_for_status", new=AsyncMock(side_effect=status_error)):
            result = [item async for item in Qwen.create_async_generator(Qwen.default_model, [{"role": "user", "content": "test"}], conversation=conversation, **kwargs)]
        return result, session, auth, headers

    async def test_captured_image_edit_stream(self):
        path = Path(__file__).with_name("fixtures") / "qwen_image_edit_stream.json"
        chunks = json.loads(path.read_text(encoding="utf-8"))
        result = await self.read(chunks)
        images = [x for x in result if isinstance(x, ImageResponse)]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].get_list(), ["https://example.com/display.png"])
        self.assertIn("red square", "".join(x for x in result if isinstance(x, str)))
        self.assertEqual(sum(isinstance(x, FinishReason) for x in result), 1)

    async def test_captured_search_stream(self):
        path = Path(__file__).with_name("fixtures") / "qwen_search_stream.json"
        chunks = json.loads(path.read_text(encoding="utf-8"))
        result = await self.read(chunks)
        answer = "".join(x for x in result if isinstance(x, str))
        self.assertIn("[37](https://www.python.org/downloads/source/)", answer)
        self.assertIn("[38](https://www.python.org/downloads/windows/)", answer)
        self.assertNotIn("[[37]]", answer)
        sources = next(x for x in result if isinstance(x, Sources)).list
        self.assertTrue(any(s["url"] == "https://www.python.org/downloads/" for s in sources))
        self.assertEqual(len(sources), len({s["url"] for s in sources}))

    async def test_split_citations_use_batch_reference_numbers(self):
        docs = [{"url": "https://example.com/first"}, {"url": "https://example.com/second"}]
        result = await self.read([
            delta("web_search", role="function", extra={"tool_result": {"docs": docs, "used_id": 39}}),
            delta("answer", "Source ["), delta("answer", "[3"), delta("answer", "7]"), delta("answer", "] and [[38]]."),
        ])
        self.assertEqual("".join(x for x in result if isinstance(x, str)), "Source [37](https://example.com/first) and [38](https://example.com/second).")

    async def test_unknown_and_incomplete_citations_are_preserved(self):
        result = await self.read([delta("answer", "A [[999]] and ["), delta("answer", "[37")])
        self.assertEqual("".join(x for x in result if isinstance(x, str)), "A [[999]] and [[37")

    async def test_extract_sources_and_non_citation_markdown(self):
        result = await self.read([
            delta("web_extractor", role="function", extra={"web_extract_info": [{"title": "Source", "url": "https://example.com/source"}]}),
            delta("answer", "["), delta("answer", "Markdown](https://example.com/source)"),
        ])
        self.assertEqual("".join(x for x in result if isinstance(x, str)), "[Markdown](https://example.com/source)")
        self.assertEqual(next(x for x in result if isinstance(x, Sources)).list[0]["url"], "https://example.com/source")

    async def test_explicit_thinking_modes_override_reasoning_effort(self):
        for mode, effort, expected in (("Thinking", "none", True), ("Auto", "none", True), ("Fast", "high", False), (" thinking ", "none", True), (None, "high", True), (None, "none", False)):
            with self.subTest(mode=mode, effort=effort):
                _, session, _, _ = await self.request(thinking_mode=mode, reasoning_effort=effort)
                features = session.post.call_args.kwargs["json"]["messages"][0]["feature_config"]
                self.assertEqual(features["thinking_enabled"], expected)
                if expected:
                    self.assertEqual(features["auto_thinking"], mode in ("Auto", None))

    async def test_invalid_thinking_mode_fails_before_auth(self):
        with patch.object(Qwen, "_ensure_auth", new=AsyncMock()) as auth:
            with self.assertRaisesRegex(ValueError, "thinking_mode"):
                await anext(Qwen.create_async_generator(Qwen.default_model, [{"role": "user", "content": "test"}], thinking_mode="invalid"))
            auth.assert_not_awaited()

    async def test_search_flags_respect_explicit_choice(self):
        for chat_type, mode, search in (("search", "Fast", True), ("search", "Thinking", False), ("t2t", "Thinking", False)):
            with self.subTest(chat_type=chat_type, mode=mode, search=search):
                _, session, _, _ = await self.request(chat_type=chat_type, thinking_mode=mode, auto_search=search)
                features = session.post.call_args.kwargs["json"]["messages"][0]["feature_config"]
                self.assertEqual(features["auto_search"], search)

    async def test_api_key_auth_and_explicit_token_precedence(self):
        for kwargs, expected in (({"api_key": "test-key"}, "test-key"), ({"api_key": "test-key", "token": "test-token"}, "test-token")):
            with self.subTest(kwargs=kwargs):
                _, session, auth, headers = await self.request(**kwargs)
                auth.assert_not_awaited()
                headers.assert_called_once_with(expected)
                self.assertEqual(session.post.call_args.kwargs["json"]["chat_mode"], "normal")

    async def test_quota_uses_api_key(self):
        session = MagicMock()
        session.post.return_value = Response(payload={"success": True, "data": {}})
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        with patch.object(QWEN_MODULE, "StreamSession", return_value=session), patch.object(Qwen, "_ensure_auth", new=AsyncMock()) as auth, patch.object(Qwen, "_get_headers", return_value={}) as headers, patch.object(Qwen, "_get_req_headers", new=AsyncMock(return_value={})), patch.object(Qwen, "raise_for_status", new=AsyncMock()):
            await Qwen.get_quota(api_key="test-key")
        auth.assert_not_awaited()
        headers.assert_called_once_with("test-key")
        self.assertEqual(session.post.call_args.kwargs["json"]["chat_mode"], "normal")

    def test_model_loading_uses_api_key(self):
        client = MagicMock()
        client.get.return_value.ok = False
        with patch.object(QWEN_MODULE, "has_curl_cffi", True), patch.object(QWEN_MODULE, "curl_cffi", client, create=True), patch.object(Qwen, "_models_loaded", False), patch.object(Qwen, "_get_headers", return_value={}) as headers:
            Qwen.get_models(api_key="test-key")
        headers.assert_called_once_with("test-key")
        client.get.assert_called_once_with("https://chat.qwen.ai/api/models", headers={})

    async def test_followup_uses_assistant_parent_id(self):
        conversation = JsonConversation(chat_id="chat", parent_id="previous-assistant", cookies={})
        _, session, _, _ = await self.request([{"response.created": {"response_id": "new-assistant"}}, delta("answer", "Answer")], conversation=conversation)
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(payload["parent_id"], "previous-assistant")
        self.assertEqual(payload["messages"][0]["parentId"], "previous-assistant")
        self.assertEqual(conversation.parent_id, "new-assistant")

    async def test_midtoken_is_not_written_to_logs(self):
        response = MagicMock()
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=False)
        response.text = AsyncMock(return_value="umx.wu('test-private-midtoken')")
        session = MagicMock()
        session.headers = {}
        session.get.return_value = response
        with patch.object(Qwen, "_midtoken", None), patch.object(Qwen, "_midtoken_uses", 0), patch.object(QWEN_MODULE.debug, "log") as log:
            headers = await Qwen._get_req_headers(session)
        self.assertEqual(headers["bx-umidtoken"], "test-private-midtoken")
        self.assertNotIn("test-private-midtoken", " ".join(str(call) for call in log.call_args_list))

    async def test_captured_browser_stream(self):
        path = Path(__file__).with_name("fixtures") / "qwen_image_stream.json"
        chunks = json.loads(path.read_text(encoding="utf-8"))
        result = await self.read(chunks)
        images = [x for x in result if isinstance(x, ImageResponse)]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].get_list(), ["https://example.com/display.png"])
        answer = "".join(x for x in result if isinstance(x, str))
        self.assertIn("I have generated the image", answer)
        reasoning = "".join(x.token for x in result if isinstance(x, Reasoning))
        self.assertEqual(reasoning.count("I will generate a visual representation"), 1)
        self.assertEqual(sum(isinstance(x, FinishReason) for x in result), 1)
        self.assertEqual(next(x for x in result if isinstance(x, Usage)).get_dict()["total_tokens"], 1863)

    async def test_observed_image_tool_then_answer(self):
        # URLs and IDs are synthetic; the wire shape is from the live image test.
        chunks = [
            {"response.created": {"chat_id": "chat", "parent_id": "user", "response_id": "assistant", "response_index": "0"}},
            {**delta("thinking_summary", extra={"summary_thought": {"content": ["I will generate an image."]}}), "usage": {"input_tokens": 1561, "output_tokens": 70}},
            delta("thinking_summary", status="finished"),
            delta("image_gen_tool", function_call={"name": "image_gen", "arguments": "{\"prompt\":\"circle\"}"}),
            {**delta("image_gen_tool", status="finished", role="function", extra={"image_list": [{"image": "https://example.com/display.png"}], "tool_result": [{"image": "https://example.com/original.png"}], "display_position": "answer"}), "usage": {}},
            delta("answer", "تم إنشاء الصورة.", role="assistant"),
            {**delta("answer", status="finished"), "usage": {"input_tokens": 1793, "output_tokens": 70, "output_tokens_details": {"reasoning_tokens": 36}}},
        ]
        result = await self.read(chunks)
        self.assertEqual(self.conversation.parent_id, "assistant")
        self.assertEqual([x.get_list() for x in result if isinstance(x, ImageResponse)], [["https://example.com/display.png"]])
        self.assertEqual([x for x in result if isinstance(x, str)], ["تم إنشاء الصورة."])
        self.assertLess(next(i for i, x in enumerate(result) if isinstance(x, ImageResponse)), next(i for i, x in enumerate(result) if isinstance(x, str)))
        self.assertEqual(sum(isinstance(x, FinishReason) for x in result), 1)
        usage = next(x for x in result if isinstance(x, Usage))
        self.assertEqual(usage.get_dict()["total_tokens"], 1863)
        self.assertEqual(usage.get_dict()["reasoning_tokens"], 36)

    async def test_summary_snapshots_are_not_duplicated(self):
        result = await self.read([
            delta("thinking_summary", extra={"summary_thought": {"content": ["Plan"]}}),
            delta("thinking_summary", extra={"summary_thought": {"content": ["Plan"]}}),
            delta("thinking_summary", extra={"summary_thought": {"content": ["Plan more"]}}),
            delta("thinking_summary", status="finished"),
            delta("answer", "Answer"),
            delta("thinking_summary", extra={"summary_thought": {"content": ["Plan"]}}),
        ])
        self.assertEqual([x.token for x in result if isinstance(x, Reasoning)], ["Plan", " more", "Plan"])
        self.assertIn("Answer", result)

    async def test_thinking_end_and_skip_think(self):
        result = await self.read([
            delta("think", "Reason"),
            delta("think", status="finished"),
            delta(content="Answer"),
            delta("think", "More reasoning"),
            {"response.info": {"action": "skip_think", "response_id": "assistant"}},
            delta(content="Skipped"),
        ])
        self.assertEqual([x.token for x in result if isinstance(x, Reasoning)], ["Reason", "More reasoning"])
        self.assertEqual([x for x in result if isinstance(x, str)], ["Answer", "Skipped"])

    async def test_image_edit_and_tool_result_fallback(self):
        for phase in ("image_edit", "image_edit_tool", "image_gen", "generate_image"):
            with self.subTest(phase=phase):
                result = await self.read([delta(phase, status="finished", extra={"tool_result": [{"image": "https://example.com/image.png"}]})])
                self.assertEqual(next(x for x in result if isinstance(x, ImageResponse)).get_list(), ["https://example.com/image.png"])

    async def test_image_content_and_deduplication(self):
        event = delta("image_gen", [{"image": "https://example.com/image.png"}, {"image": "https://example.com/image.png"}])
        result = await self.read([event, event, delta("image_gen", status="finished"), delta("answer", "After image")])
        self.assertEqual(sum(isinstance(x, ImageResponse) for x in result), 1)
        self.assertIn("After image", result)
        self.assertIsInstance(result[-1], FinishReason)

    async def test_generated_file_path_uses_chat_download_route(self):
        result = await self.read([delta("generate_image", status="finished", extra={"tool_result": {"file_path": "/images/blue circle.png"}})])
        self.assertEqual(next(x for x in result if isinstance(x, ImageResponse)).get_list(), ["https://chat.qwen.ai/api/v2/chat/chat/images/blue%20circle.png"])

    async def test_tool_arguments_and_heartbeat_are_not_answer_text(self):
        result = await self.read([
            delta("think", "Reason"),
            delta("KeepAlive", "heartbeat"),
            delta("image_gen_tool", '{"prompt":"circle"}'),
            delta("web_search", "search query", role="function"),
            delta("answer", "Answer"),
        ])
        self.assertEqual([x for x in result if isinstance(x, str)], ["Answer"])

    async def test_sources_from_root_and_search_tool(self):
        source = {"url": "https://example.com/page", "title": "Title"}
        result = await self.read([
            {"sources": [source]},
            delta("web_search", role="function", extra={"web_search_info": [source, {"link": "https://example.com/other", "title": "Other"}]}),
            delta("answer", "Answer"),
        ])
        self.assertEqual(len(next(x for x in result if isinstance(x, Sources)).list), 2)

    async def test_empty_usage_does_not_erase_previous_usage(self):
        result = await self.read([{"usage": {"input_tokens": 12, "output_tokens": 4}}, {"usage": {}}, {"usage": None}])
        self.assertEqual(next(x for x in result if isinstance(x, Usage)).get_dict()["total_tokens"], 16)

    async def test_error_shapes_are_not_swallowed(self):
        for error, message in (({"code": "Bad_Request", "details": "Rejected"}, "Rejected"), ({"code": "Bad_Request", "message": "Rejected"}, "Rejected"), ({"code": "Bad_Request"}, "Bad_Request"), ("Rejected", "Rejected")):
            with self.subTest(error=error):
                with self.assertRaisesRegex(ResponseError, message):
                    await self.read([{"error": error}])

    async def test_phase_error_propagates(self):
        with self.assertRaisesRegex(ResponseError, "Rejected"):
            await self.read([delta("image_edit_tool", status="error", extra={"error": "Rejected"})])

    async def test_stopped_and_done_end_the_stream(self):
        for event in ({"response.stopped": {"response_id": "assistant"}}, {"done": True}):
            with self.subTest(event=event):
                result = await self.read([delta("answer", "Before"), event, delta("answer", "After")])
                self.assertEqual([x for x in result if isinstance(x, str)], ["Before"])
                self.assertIsInstance(result[-1], FinishReason)

    async def test_non_stream_json_completion(self):
        result = await self.read(payload={"choices": [{"message": {"role": "assistant", "content": "Answer"}, "finish_reason": "length"}], "usage": {"prompt_tokens": 2, "completion_tokens": 3}})
        self.assertIn("Answer", result)
        self.assertEqual(result[-1].reason, "length")

    async def test_json_envelope_and_errors(self):
        self.assertIn("Answer", await self.read(payload={"success": True, "data": {"content": "Answer"}}))
        with self.assertRaisesRegex(ResponseError, "Rejected"):
            await self.read(payload={"success": False, "data": {"code": "Bad_Request", "details": "Rejected"}})
        for payload in ({"success": True, "data": {}}, {"error": "Rejected"}):
            with self.subTest(payload=payload):
                with self.assertRaises(ResponseError):
                    await self.read(payload=payload)

    async def test_rate_limit_json_is_not_replayed_and_preserves_details(self):
        response = Response(payload={"success": False, "data": {"code": "RateLimited", "details": "Image generation quota exhausted"}})
        with patch.object(Qwen, "_midtoken", "original-fingerprint"), patch.object(Qwen, "_har_headers", {"bx-umidtoken": "har-fingerprint"}), patch.object(Qwen, "prepare_files", new=AsyncMock(return_value=[{"id": "uploaded"}])) as upload, patch.object(QWEN_MODULE.asyncio, "sleep", new=AsyncMock()) as sleep:
            with self.assertRaisesRegex(RateLimitError, "RateLimited: Image generation quota exhausted"):
                await self.request(response=response, media=[("https://example.com/image.png", "image.png")])
            self.assertEqual(self.session.post.call_count, 1)
            upload.assert_awaited_once()
            sleep.assert_not_awaited()
            self.assertEqual(Qwen._midtoken, "original-fingerprint")
            self.assertEqual(Qwen._har_headers["bx-umidtoken"], "har-fingerprint")

    async def test_rate_limit_sse_and_other_limit_codes(self):
        for code in ("RateLimited", "ParallelLimited", "quotaLimited"):
            with self.subTest(code=code):
                with self.assertRaisesRegex(RateLimitError, "Try later"):
                    await self.request([{"error": {"code": code, "details": "Try later"}}])
                self.assertEqual(self.session.post.call_count, 1)

    async def test_string_rate_limit_error(self):
        with self.assertRaisesRegex(RateLimitError, "RateLimited"):
            await self.read([{"error": "RateLimited: Guest limit exceeded"}])

    async def test_http_429_is_not_replayed(self):
        with patch.object(QWEN_MODULE.asyncio, "sleep", new=AsyncMock()) as sleep:
            with self.assertRaisesRegex(RateLimitError, "Response 429"):
                with patch.object(Qwen, "_read_response", side_effect=RateLimitError("Response 429: quota exhausted")):
                    await self.request()
            self.assertEqual(self.session.post.call_count, 1)
            sleep.assert_not_awaited()

    async def test_legacy_runtime_rate_limit_is_not_replayed(self):
        with patch.object(Qwen, "_read_response", side_effect=RuntimeError("RateLimited: model quota exhausted")):
            with self.assertRaisesRegex(RateLimitError, "model quota exhausted"):
                await self.request()
        self.assertEqual(self.session.post.call_count, 1)

    async def test_captcha_error_keeps_separate_recovery_type(self):
        with self.assertRaisesRegex(RuntimeError, "FAIL_SYS_USER_VALIDATE"):
            await self.read(payload={"success": False, "data": {"code": "FAIL_SYS_USER_VALIDATE", "details": "Verification required"}})

    async def test_expired_token_stops_before_chat_and_upload(self):
        response = Response(payload={"success": False, "data": {"code": "unauthorized", "details": "Token has expired, please log in again."}})
        with patch.object(Qwen, "prepare_files", new=AsyncMock()) as upload:
            with self.assertRaisesRegex(MissingAuthError, "Token has expired"):
                await self.request(token="expired-test-token", auth_response=response, media=[("https://example.com/image.png", "image.png")])
            self.session.post.assert_not_called()
            upload.assert_not_awaited()

    async def test_unauthorized_stream_is_an_auth_error(self):
        with self.assertRaisesRegex(MissingAuthError, "Token has expired"):
            await self.read([{"error": {"code": "unauthorized", "details": "Token has expired"}}])

    async def test_http_auth_failure_is_not_ignored(self):
        with self.assertRaises(MissingAuthError):
            await self.request(token="expired-test-token", status_error=MissingAuthError("Response 401"))
        self.session.post.assert_not_called()

    async def test_json_envelope_preserves_usage(self):
        result = await self.read(payload={"success": True, "usage": {"input_tokens": 2, "output_tokens": 3}, "data": {"content": "Answer"}})
        self.assertEqual(next(x for x in result if isinstance(x, Usage)).get_dict()["total_tokens"], 5)

    async def test_null_metadata_and_response_id_fallback(self):
        result = await self.read([None, {"choices": None}, {"choices": [{"delta": None}]}, {"response.created": None}, {"response_id": "assistant", **delta("answer", "Answer", extra=None)}])
        self.assertIn("Answer", result)
        self.assertEqual(self.conversation.parent_id, "assistant")

    async def test_create_generator_reads_stream_and_json(self):
        for stream in (True, False):
            with self.subTest(stream=stream):
                response = Response([delta("answer", "Answer")]) if stream else Response(payload={"choices": [{"message": {"content": "Answer"}}]})
                session = unittest.mock.MagicMock()
                session.post.return_value = response
                session.__aenter__ = AsyncMock(return_value=session)
                session.__aexit__ = AsyncMock(return_value=False)
                with patch.object(QWEN_MODULE, "StreamSession", return_value=session), patch.object(Qwen, "_ensure_auth", new=AsyncMock()), patch.object(Qwen, "_get_headers", return_value={}), patch.object(Qwen, "_get_req_headers", new=AsyncMock(return_value={})), patch.object(Qwen, "raise_for_status", new=AsyncMock()):
                    result = [item async for item in Qwen.create_async_generator(Qwen.default_model, [{"role": "user", "content": "test"}], conversation=JsonConversation(chat_id="chat", parent_id=None, cookies={}), stream=stream)]
                self.assertIn("Answer", result)
                self.assertEqual(session.post.call_args.kwargs["json"]["stream"], stream)
                self.assertIsInstance(result[-1], FinishReason)


if __name__ == "__main__":
    unittest.main()
