import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from g4f import ChatCompletion
from g4f.Provider.ChatGPT import ChatGPT


class TestChatGPTTimeout(unittest.IsolatedAsyncioTestCase):
    async def test_reply_timeout(self):
        cases = [
            ({}, "Hello", 180),
            ({"timeout": None}, "Hello", 180),
            ({"timeout": 0}, "Hello", 60),
            ({"timeout": 300}, "Hello", 300),
            ({"timeout": None}, "x" * 4000, 260),
            ({"timeout": None}, "x" * 40000, 1800),
            ({"timeout": 2000}, "Hello", 1800),
        ]
        for kwargs, prompt, expected_timeout in cases:
            with self.subTest(kwargs=kwargs, prompt_length=len(prompt)):
                browser = MagicMock()
                browser.__aenter__ = AsyncMock(return_value=browser)
                browser.call = AsyncMock()
                browser.navigate = AsyncMock()
                browser.wait_for_event = AsyncMock(return_value={
                    "requestId": "request-id",
                    "request": {
                        "url": "https://chatgpt.com/conversation/updates",
                        "postData": "prompt=Hello",
                    },
                })
                browser.get_cookies = AsyncMock(return_value={})

                response = MagicMock()
                response.__aenter__ = AsyncMock(return_value=response)
                response.text = AsyncMock(return_value='<p data-assistant-stream-block="">Answer</p>')
                transport = MagicMock()
                transport.__aenter__ = AsyncMock(return_value=transport)
                transport.post.return_value = response

                with patch("g4f.Provider.ChatGPT.CDPSession", return_value=browser), patch(
                    "g4f.Provider.ChatGPT.StreamSession", return_value=transport
                ) as stream_session:
                    chunks = [chunk async for chunk in ChatCompletion.create_async(
                        model=ChatGPT.default_model,
                        provider=ChatGPT,
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                        **kwargs,
                    )]

                self.assertEqual(chunks[0], "Answer")
                self.assertEqual(stream_session.call_args.kwargs["timeout"], expected_timeout)
