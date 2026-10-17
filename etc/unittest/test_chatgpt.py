import importlib
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from g4f import ChatCompletion, debug
from g4f.Provider.ChatGPT import ChatGPT, Conversation
from g4f.providers.response import FinishReason

CHATGPT_MODULE = importlib.import_module("g4f.Provider.ChatGPT")

class TestChatGPT(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _browser_mock():
        # evaluate_js is called in a fixed sequence: composer check,
        # typing, send click, poller kickoff, reply polls, id extraction.
        browser = MagicMock()
        browser.__aenter__ = AsyncMock(return_value=browser)
        browser.navigate = AsyncMock()
        browser.evaluate_js = AsyncMock(side_effect=[
            True,   # composer appeared
            None,   # prompt typed
            True,   # send button clicked
            None,   # poller started
            {"text": "Answer", "done": False},
            {"text": "Answer!", "done": True},
            {"conversation_id": "conversation-id", "message_id": "message-id"},
        ])
        return browser

    async def test_page_driven_reply(self):
        browser = self._browser_mock()

        with patch.object(CHATGPT_MODULE, "CDPSession", return_value=browser), patch.object(
            debug, "version_check", False
        ):
            chunks = [chunk async for chunk in ChatCompletion.create_async(
                model=ChatGPT.default_model,
                provider=ChatGPT,
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
            )]

        browser.navigate.assert_awaited_with("https://chatgpt.com")
        self.assertEqual(chunks[0], "Answer")
        self.assertEqual("".join(c for c in chunks if isinstance(c, str)), "Answer!")
        self.assertEqual(chunks[-1].reason, "stop")
        conversation = chunks[-2]
        self.assertEqual(conversation.conversation_id, "conversation-id")
        self.assertEqual(conversation.message_id, "message-id")

    async def test_continue_conversation(self):
        browser = self._browser_mock()
        conversation = Conversation(ChatGPT.default_model)
        conversation.conversation_id = "conversation-id"
        conversation.message_id = "message-id"

        with patch.object(CHATGPT_MODULE, "CDPSession", return_value=browser), patch.object(
            debug, "version_check", False
        ):
            chunks = [chunk async for chunk in ChatCompletion.create_async(
                model=ChatGPT.default_model,
                provider=ChatGPT,
                messages=[{"role": "user", "content": "Hello again"}],
                stream=True,
                conversation=conversation,
            )]

        browser.navigate.assert_awaited_with("https://chatgpt.com/c/conversation-id")
        self.assertEqual("".join(c for c in chunks if isinstance(c, str)), "Answer!")
        self.assertEqual(chunks[-2], conversation)

    def test_reply_timeout(self):
        cases = [
            (None, "Hello", 180),
            (0, "Hello", 60),
            (300, "Hello", 300),
            (None, "x" * 4000, 260),
            (None, "x" * 40000, 1800),
            (2000, "Hello", 1800),
        ]
        for timeout, prompt, expected_timeout in cases:
            with self.subTest(timeout=timeout, prompt_length=len(prompt)):
                self.assertEqual(ChatGPT._reply_timeout(timeout, prompt), expected_timeout)
