import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from g4f.Provider.Qwen import Qwen
from g4f.Provider.qwen.session import QwenAuth
from g4f.errors import RateLimitError, ResponseError
from g4f.providers.response import JsonConversation
from .test_qwen_auth import AuthResponse, jwt
from .test_qwen_stream import QWEN_MODULE, delta


LIMIT = {"code": "RateLimited", "details": "You've reached the guest chat limit. Log in to continue."}


def chat(chat_id, cookie=None):
    return AuthResponse({"success": True, "data": {"id": chat_id}}, cookie=cookie)


class TestQwenGuest(unittest.IsolatedAsyncioTestCase):
    async def run_provider(self, responses, auth=None, conversation=None, media=None):
        self.auth = auth or QwenAuth(cookies={"exhausted_guest": "old"})
        self.old_cache_id = self.auth.cache_id
        self.sessions = []
        for replies in responses:
            session = MagicMock()
            session.headers = {}
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=False)
            session.post.side_effect = replies
            session.get.return_value = AuthResponse({})
            self.sessions.append(session)
        self.output, self.upload_cache_ids = [], []

        async def upload(*args, auth, **kwargs):
            self.upload_cache_ids.append(auth.cache_id)
            return [{"id": f"upload-{len(self.upload_cache_ids)}"}]

        with patch.object(QWEN_MODULE, "StreamSession", side_effect=self.sessions) as transport, patch.object(Qwen, "_get_headers", return_value={}), patch.object(Qwen, "_get_req_headers", new=AsyncMock(return_value={})), patch.object(QWEN_MODULE, "generate_cookies", return_value={"ssxmod_itna": "fresh", "ssxmod_itna2": "fresh2"}), patch.object(Qwen, "prepare_files", new=AsyncMock(side_effect=upload)) as uploads:
            self.transport, self.uploads = transport, uploads
            async for item in Qwen.create_async_generator(
                Qwen.default_model, [{"role": "user", "content": "test"}],
                auth_session=self.auth, conversation=conversation, media=media,
                chat_type="image_edit" if media else "t2t",
            ):
                self.output.append(item)

    async def test_guest_retry_preserves_cookies_and_shared_upload_cache(self):
        conversation = JsonConversation(chat_id='old-chat', parent_id='old-parent', cookies={'conversation_guest': 'old'})
        auth = QwenAuth(cookies={'exhausted_guest': 'old'})
        shared_file = {'id': 'cached-file'}
        with patch.dict(QWEN_MODULE.ImagesCache, {'image-hash': shared_file}), patch.object(Qwen, '_midtoken', 'old-midtoken'):
            await self.run_provider([
                [AuthResponse(payload={'success': False, 'data': LIMIT})],
                [chat('retry-chat'), AuthResponse(chunks=[{'response.created': {'response_id': 'assistant'}}, delta('answer', 'Answer')])],
            ], auth=auth, conversation=conversation, media=[('image', 'image.png')])
            self.assertIs(QWEN_MODULE.ImagesCache['image-hash'], shared_file)
            self.assertIsNone(Qwen._midtoken)
        self.assertIn('Answer', self.output)
        self.assertEqual(self.transport.call_count, 2)
        self.sessions[0].__aexit__.assert_awaited_once()
        self.assertEqual(len(set(self.upload_cache_ids)), 1)
        self.assertEqual(auth.cache_id, self.old_cache_id)
        first = self.sessions[0].post.call_args.kwargs['json']
        second = self.sessions[1].post.call_args.kwargs['json']
        self.assertEqual(first['chat_id'], 'old-chat')
        self.assertEqual(second['chat_id'], 'retry-chat')
        self.assertIsNone(second['parent_id'])
        self.assertEqual(second['messages'][0]['chat_type'], 'image_edit')
        for call in self.sessions[1].post.call_args_list:
            self.assertIn('exhausted_guest=old', call.kwargs['headers']['Cookie'])
        result = next(item for item in self.output if isinstance(item, JsonConversation))
        self.assertEqual(result.chat_id, 'retry-chat')

    async def test_sse_guest_limit_is_retried(self):
        await self.run_provider([
            [chat("old-chat"), AuthResponse(chunks=[{"error": LIMIT}])],
            [chat("fresh-chat"), AuthResponse(chunks=[delta("answer", "Answer")])],
        ])
        self.assertIn("Answer", self.output)
        self.assertEqual(self.transport.call_count, 2)

    async def test_limit_during_chat_creation_is_retried(self):
        await self.run_provider([
            [AuthResponse({"success": False, "data": LIMIT})],
            [chat("fresh-chat"), AuthResponse(chunks=[delta("answer", "Answer")])],
        ])
        self.assertIn("Answer", self.output)
        self.assertEqual(self.auth.cache_id, self.old_cache_id)

    async def test_second_guest_limit_stops(self):
        with self.assertRaisesRegex(RateLimitError, "guest chat limit"):
            await self.run_provider([
                [chat("old-chat"), AuthResponse(chunks=[{"error": LIMIT}])],
                [chat("fresh-chat"), AuthResponse(chunks=[{"error": LIMIT}])],
            ])
        self.assertEqual(self.transport.call_count, 2)

    async def test_guest_limit_after_output_is_not_replayed(self):
        for started in (delta("answer", "Started"), {"response.created": {"response_id": "assistant"}}):
            with self.subTest(started=started):
                with self.assertRaises(RateLimitError):
                    await self.run_provider([[chat("old-chat"), AuthResponse(chunks=[started, {"error": LIMIT}])]])
                self.assertEqual(self.transport.call_count, 1)
                self.assertEqual(self.auth.cache_id, self.old_cache_id)

    async def test_guest_quota_response_error_retries_once_without_reset(self):
        error = {'code': 'quota_limit', 'details': 'Guest quota response'}
        await self.run_provider([
            [chat('old-chat'), AuthResponse(chunks=[{'error': error}])],
            [AuthResponse(chunks=[delta('answer', 'Answer')])],
        ])
        self.assertIn('Answer', self.output)
        self.assertEqual(self.transport.call_count, 2)
        self.assertEqual(self.auth.cache_id, self.old_cache_id)
        self.assertEqual(self.sessions[1].post.call_args.kwargs['json']['chat_id'], 'old-chat')

    async def test_quota_response_error_stops_after_retry_or_output(self):
        error = {'code': 'quota_limit', 'details': 'Guest quota response'}
        with self.assertRaises(ResponseError):
            await self.run_provider([
                [chat('old-chat'), AuthResponse(chunks=[{'error': error}])],
                [AuthResponse(chunks=[{'error': error}])],
            ])
        self.assertEqual(self.transport.call_count, 2)
        with self.assertRaises(ResponseError):
            await self.run_provider([[chat('old-chat'), AuthResponse(chunks=[delta('answer', 'Started'), {'error': error}])]])
        self.assertEqual(self.transport.call_count, 1)

    async def test_general_guest_quota_is_not_replayed(self):
        for code in ("RateLimited", "ParallelLimited", "quotaLimited"):
            with self.subTest(code=code):
                error = {"code": code, "details": "Image generation quota exhausted"}
                with self.assertRaises(RateLimitError):
                    await self.run_provider([[chat("old-chat"), AuthResponse(chunks=[{"error": error}])]])
                self.assertEqual(self.transport.call_count, 1)
                self.assertEqual(self.auth.cache_id, self.old_cache_id)

    async def test_authenticated_guest_limit_does_not_reset_account(self):
        auth = QwenAuth(jwt(), "keep-refresh")
        with self.assertRaises(RateLimitError):
            await self.run_provider([[chat("account-chat"), AuthResponse(chunks=[{"error": LIMIT}])]], auth=auth)
        self.assertEqual(self.transport.call_count, 1)
        self.assertTrue(auth.authenticated)
        self.assertEqual(auth.cache_id, self.old_cache_id)
        self.assertIn("refresh_token=keep-refresh", auth.request_headers({}, "https://auth.qwen.ai")["Cookie"])

    def test_reset_cannot_reopen_closed_or_authenticated_session(self):
        closed = QwenAuth()
        closed.clear()
        for auth in (closed, QwenAuth(jwt()), QwenAuth(refresh_token="")):
            with self.assertRaises(ValueError):
                auth.reset_guest({"guest": "new"})
