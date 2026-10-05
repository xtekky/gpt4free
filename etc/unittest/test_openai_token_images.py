import asyncio
import base64
import importlib
import json
import time
import unittest
from unittest.mock import AsyncMock, patch

from g4f.errors import MissingAuthError, ResponseError
from g4f.Provider.needs_auth.OpenaiAccount import OpenaiAccount, Conversation as AccountConversation
from g4f.Provider.needs_auth.OpenaiChat import OpenaiChat, Conversation, OpenAISources, ContentReferences
from g4f.Provider.openai.auth import parse_access_token
from g4f.Provider.openai.images import download_image, poll_images, turn_messages
from g4f.providers.response import AuthResult, ImageResponse, ImagePreview


def jwt(claims=None):
    if claims is None:
        claims = {'exp': time.time() + 3600}
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip('=')
    return f'e30.{encoded}.signature'


def image_record(status='finished_successfully'):
    return {'current_node': 'image-node', 'mapping': {
        'old-image': {'parent': None, 'message': {'id': 'old', 'status': 'finished_successfully', 'content': {
            'parts': [{'content_type': 'image_asset_pointer', 'asset_pointer': 'sediment://file_old'}]}}},
        'user-node': {'parent': 'old-image', 'message': {'id': 'requested-user', 'author': {'role': 'user'}}},
        'image-node': {'parent': 'user-node', 'message': {'id': 'image-message', 'author': {'role': 'tool'},
            'status': status, 'metadata': {}, 'content': {'content_type': 'multimodal_text', 'parts': [
                {'content_type': 'image_asset_pointer', 'asset_pointer': 'sediment://file_current'}]}}},
    }}


class Response:
    def __init__(self, data, status=200):
        self.data, self.status = data, status
        self.ok = status < 400
        self.headers = {'content-type': 'application/json'}

    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def json(self): return self.data


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


class TestAccessToken(unittest.TestCase):
    def test_plain_and_bearer_tokens(self):
        token = jwt()
        for value in (token, f'Bearer {token}', f'bearer {token}'):
            self.assertEqual(parse_access_token(value)[0], token)

    def test_invalid_tokens_fail_without_leaking_input(self):
        for token in ('sensitive-invalid-token', 'e30.invalid.signature', jwt({}), jwt({'exp': True}),
                      jwt({'exp': '2099'}), jwt({'exp': float('inf')}), jwt({'exp': time.time() - 1}),
                      {'accessToken': jwt()}, None):
            with self.subTest(token_type=type(token).__name__):
                with self.assertRaises(MissingAuthError) as error:
                    parse_access_token(token)
                self.assertNotIn('sensitive-', str(error.exception))

    def test_rejected_token_clears_stale_authorization(self):
        with patch.object(OpenaiChat, '_headers', {'authorization': 'Bearer old', 'Authorization': 'Bearer old'}), \
             patch.object(OpenaiChat, '_api_key', 'old'), patch.object(OpenaiChat, '_expires', 9999999999):
            self.assertFalse(OpenaiChat._set_api_key('sensitive-invalid-token'))
            self.assertIsNone(OpenaiChat._api_key)
            self.assertIsNone(OpenaiChat._expires)
            self.assertNotIn('authorization', OpenaiChat._headers)
            self.assertNotIn('Authorization', OpenaiChat._headers)


class TestTokenImages(unittest.IsolatedAsyncioTestCase):
    async def test_direct_token_bypasses_cache_for_both_providers(self):
        token = jwt()
        for provider in (OpenaiChat, OpenaiAccount):
            async def create(cls, model, messages, auth_result, **kwargs):
                yield auth_result.api_key
            with self.subTest(provider=provider.__name__), \
                 patch.object(provider, 'create_authed', classmethod(create)), \
                 patch.object(provider, 'get_auth_result', side_effect=AssertionError('cache read')), \
                 patch.object(provider, 'write_cache_file', side_effect=AssertionError('cache write')):
                chunks = [chunk async for chunk in provider.create_async_generator('auto', [], api_key=token)]
                self.assertEqual(chunks, [token])

    async def test_on_auth_async_accepts_direct_token(self):
        token = jwt()
        for provider in (OpenaiChat, OpenaiAccount):
            chunks = [chunk async for chunk in provider.on_auth_async(api_key=token)]
            self.assertEqual(chunks[0].api_key, token)

    async def test_stream_timeout_closes_generator(self):
        closed = []
        async def create(cls, model, messages, auth_result, **kwargs):
            try:
                await asyncio.sleep(1)
                yield 'late'
            finally:
                closed.append(True)
        with patch.object(OpenaiAccount, 'create_authed', classmethod(create)):
            with self.assertRaises(asyncio.TimeoutError):
                [chunk async for chunk in OpenaiAccount.create_async_generator('auto', [], api_key=jwt(), stream_timeout=0.01)]
        self.assertEqual(closed, [True])

    async def test_completed_tool_image_without_task_id(self):
        session = Session([Response(image_record()), Response({'download_url': 'https://chatgpt.com/image.png'})])
        images = [image async for image in poll_images(session, AuthResult(headers={}), 'chat', 'requested-user')]
        self.assertEqual(type(images[0]), ImageResponse)
        self.assertEqual(images[0].get_list(), ['https://chatgpt.com/image.png'])
        self.assertNotIn('file_old', str(session.calls))

    async def test_other_turn_and_cycles_are_not_used(self):
        record = image_record()
        self.assertEqual(turn_messages(record, 'unrelated-user'), [])
        record['mapping']['user-node']['parent'] = 'image-node'
        self.assertEqual(turn_messages(record, 'unrelated-user'), [])

    async def test_in_progress_and_404_are_polled(self):
        session = Session([
            Response(image_record('in_progress')), Response(image_record()), Response({}, 404),
            Response(image_record()), Response({'download_url': 'https://chatgpt.com/image.png'}),
        ])
        images = [image async for image in poll_images(session, AuthResult(headers={}), 'chat', 'requested-user', poll_interval=0)]
        self.assertEqual(len(images), 1)
        self.assertEqual(len(session.calls), 5)

    async def test_refusal_is_not_empty_success(self):
        record = image_record()
        record['mapping']['image-node']['message'].update(
            author={'role': 'assistant'}, recipient='all', content={'content_type': 'text', 'parts': ['Unavailable']})
        with self.assertRaisesRegex(ResponseError, 'without generating'):
            [chunk async for chunk in poll_images(Session([Response(record)]), AuthResult(headers={}), 'chat', 'requested-user')]

    async def test_wait_is_bounded(self):
        with self.assertRaises(TimeoutError):
            [chunk async for chunk in poll_images(Session([]), AuthResult(headers={}), 'chat', 'requested-user', timeout=0)]

    async def test_slow_request_respects_deadline(self):
        class SlowResponse(Response):
            async def __aenter__(self):
                await asyncio.sleep(1)
                return self
        with self.assertRaises(asyncio.TimeoutError):
            [chunk async for chunk in poll_images(Session([SlowResponse(image_record())]), AuthResult(headers={}),
                                                 'chat', 'requested-user', timeout=0.01)]

    async def test_asset_auth_rejection_propagates(self):
        with self.assertRaises(MissingAuthError):
            await download_image(Session([Response({'error': {'message': 'Unauthorized'}}, 401)]),
                                 AuthResult(headers={}), 'file-service://file_test', '', None, 'finished_successfully')

    async def test_invalid_asset_path_is_rejected(self):
        with self.assertRaises(ResponseError):
            await download_image(Session([]), AuthResult(headers={}), 'sediment://../invalid', '', 'chat', 'finished_successfully')

    async def test_file_service_without_status_is_final(self):
        image = await download_image(Session([Response({'download_url': 'https://chatgpt.com/image.png'})]),
                                     AuthResult(headers={}), 'file-service://file_test', '', None, None)
        self.assertEqual(type(image), ImageResponse)

    async def test_stream_snapshots_return_final_image_or_preview(self):
        for status, result_type in [('finished_successfully', ImageResponse), ('in_progress', ImagePreview)]:
            message = image_record(status)['mapping']['image-node']['message']
            line = b'data: ' + json.dumps({'v': {'conversation_id': 'chat', 'message': message}}).encode()
            chunks = [chunk async for chunk in OpenaiChat.iter_messages_line(
                Session([Response({'download_url': 'https://chatgpt.com/image.png'})]), AuthResult(headers={}),
                line, Conversation(), OpenAISources([]), ContentReferences())]
            self.assertEqual(len(chunks), 1)
            self.assertEqual(type(chunks[0]), result_type)

    async def test_user_uploaded_image_is_not_a_generated_image(self):
        message = image_record()['mapping']['image-node']['message']
        message['author']['role'] = 'user'
        line = b'data: ' + json.dumps({'v': {'conversation_id': 'chat', 'message': message}}).encode()
        chunks = [chunk async for chunk in OpenaiChat.iter_messages_line(
            Session([]), AuthResult(headers={}), line, Conversation(), OpenAISources([]), ContentReferences())]
        self.assertEqual(chunks, [])

    async def test_text_finish_keeps_stream_open_for_image(self):
        module = importlib.import_module(OpenaiChat.__module__)
        lines = [
            {'v': {'conversation_id': 'chat', 'message': {'author': {'role': 'assistant'}, 'recipient': 'all'}}},
            {'v': [{'p': '/message/metadata', 'v': {'finish_details': {'type': 'stop'}}}]},
            {'v': {'message': image_record()['mapping']['image-node']['message']}},
        ]
        class Transport(Session):
            done = False
            status = 200
            ok = True
            def __init__(self):
                super().__init__([Response({}), Response({'download_url': 'https://chatgpt.com/image.png'})])
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            def post(self, url, **kwargs):
                if url.endswith('/prepare'): return Response({'conduit_token': 'conduit'})
                if url.endswith('/chat-requirements'): return Response({'token': 'requirements'})
                return self
            async def iter_lines(self):
                for line in lines:
                    yield b'data: ' + json.dumps(line).encode()
                self.done = True
                yield b'data: [DONE]'
                raise AssertionError('Reading beyond DONE')
        transport = Transport()
        async def no_poll(*args, **kwargs):
            raise AssertionError('A streamed final image must not need polling')
            yield
        with patch.object(module, 'StreamSession', return_value=transport), \
             patch.object(OpenaiChat, '_update_request_args'), \
             patch.object(OpenaiChat, 'upload_files', AsyncMock(return_value=None)), \
             patch.object(module, 'poll_images', no_poll):
            images = []
            async for chunk in OpenaiChat.create_authed('gpt-image', [{'role': 'user', 'content': 'Draw a robot'}],
                                                      OpenaiChat._explicit_auth({'api_key': jwt()})):
                if isinstance(chunk, ImageResponse):
                    self.assertFalse(transport.done)
                    images.append(chunk)
        self.assertTrue(transport.done)
        self.assertEqual(len(images), 1)

    async def test_account_image_request_and_prepare_have_picture_hint(self):
        module = importlib.import_module(OpenaiAccount.__module__)
        class Transport(Response):
            def get(self, *args, **kwargs): return Response({})
            def post(self, url, **kwargs):
                self.request = kwargs['json']
                return self
            async def iter_lines(self):
                yield b'data: [DONE]'
                raise AssertionError('Reading beyond DONE')
        transport = Transport({})
        image = ImageResponse(['https://chatgpt.com/image.png'], '')
        async def poll(*args, **kwargs): yield image
        with patch.object(module, 'StreamSession', return_value=transport), \
             patch.object(OpenaiAccount, '_get_requirements', AsyncMock(return_value=('token', None))), \
             patch.object(OpenaiAccount, '_prepare_conversation', AsyncMock(return_value='conduit')) as prepare, \
             patch.object(module, 'poll_images', poll):
            chunks = [chunk async for chunk in OpenaiAccount.create_authed(
                'gpt-image', [{'role': 'user', 'content': 'Draw a robot'}], OpenaiAccount._explicit_auth({'api_key': jwt()}))]
        self.assertIn(image, chunks)
        self.assertEqual(transport.request['model'], 'auto')
        self.assertEqual(transport.request['system_hints'], ['picture_v2'])
        self.assertEqual(prepare.call_args.args[4], ['picture_v2'])

    async def test_account_continuation_does_not_mutate_previous_conversation(self):
        conversation = AccountConversation('auto')
        conversation.finish_reason = 'stop'
        conversation.recipient = 'tool'
        module = importlib.import_module(OpenaiAccount.__module__)
        class Transport(Response):
            def get(self, *args, **kwargs): return Response({})
            def post(self, *args, **kwargs): return self
            async def iter_lines(self): yield b'data: [DONE]'
        with patch.object(module, 'StreamSession', return_value=Transport({})), \
             patch.object(OpenaiAccount, '_get_requirements', AsyncMock(return_value=('token', None))), \
             patch.object(OpenaiAccount, '_prepare_conversation', AsyncMock(return_value='conduit')):
            chunks = [chunk async for chunk in OpenaiAccount.create_authed(
                'auto', [{'role': 'user', 'content': 'hello'}], OpenaiAccount._explicit_auth({'api_key': jwt()}),
                conversation=conversation)]
        returned = next(chunk for chunk in chunks if isinstance(chunk, AccountConversation))
        self.assertIsNot(returned, conversation)
        self.assertEqual(returned.recipient, 'all')
        self.assertEqual(conversation.recipient, 'tool')

    async def test_requirements_auth_rejection_is_not_retried(self):
        class Transport:
            calls = 0
            def post(self, *args, **kwargs):
                self.calls += 1
                return Response({'error': {'message': 'Verification required'}}, 403)
        transport = Transport()
        auth = OpenaiAccount._explicit_auth({'api_key': jwt()})
        with self.assertRaises(ResponseError):
            await OpenaiAccount._get_requirements(transport, auth.headers, auth)
        self.assertEqual(transport.calls, 1)

    async def test_temporary_account_image_fails_before_transport(self):
        with self.assertRaises(ValueError):
            [chunk async for chunk in OpenaiAccount.create_authed('gpt-image', [], AuthResult(), temporary=True)]


if __name__ == '__main__':
    unittest.main()
