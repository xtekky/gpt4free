"""Tests for the browsing APIs, live feeds, triggers and notifications."""

import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from g4f.browsing import (
    NotificationStore,
    list_apis,
    parse_feed,
    parse_trigger_prompt,
    feed_to_conversation,
    search,
)
from g4f.browsing import triggers as trigger_module
from g4f.browsing.browse import extract_text

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Example News</title>
    <link>https://example.com/</link>
    <description>An example feed</description>
    <item>
      <guid>post-2</guid>
      <title>Second post</title>
      <link>https://example.com/second</link>
      <description>&lt;p&gt;Newer &amp;amp; better&lt;/p&gt;</description>
      <pubDate>Tue, 02 Jan 2024 12:00:00 GMT</pubDate>
    </item>
    <item>
      <guid>post-1</guid>
      <title>First post</title>
      <link>https://example.com/first</link>
      <description>Older post</description>
      <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Example</title>
  <entry>
    <id>tag:example.com,2024:1</id>
    <title>Atom entry</title>
    <link href="https://example.com/atom-entry"/>
    <summary>Atom summary</summary>
    <updated>2024-03-01T10:00:00Z</updated>
    <author><name>Ada</name></author>
  </entry>
</feed>
"""


class TestCatalog(unittest.TestCase):
    def test_free_apis_are_available_without_keys(self):
        apis = {api["id"]: api for api in list_apis()}
        self.assertTrue(apis["rss"]["available"])
        self.assertTrue(apis["wikipedia"]["available"])
        self.assertFalse(apis["tavily"]["available"])
        self.assertEqual(apis["tavily"]["pricing"], "free-tier")
        self.assertEqual(apis["exa"]["pricing"], "paid")

    def test_credentials_are_never_serialized(self):
        with patch.dict(os.environ, {"TAVILY_API_KEY": "super-secret"}, clear=False):
            apis = {api["id"]: api for api in list_apis()}
        tavily = apis["tavily"]
        self.assertTrue(tavily["available"])
        self.assertTrue(tavily["credential_set"])
        self.assertNotIn("super-secret", json.dumps(tavily))

    def test_searxng_requires_instance_url(self):
        with patch.dict(os.environ, {"G4F_SEARXNG_URL": ""}, clear=False):
            apis = {api["id"]: api for api in list_apis()}
        self.assertFalse(apis["searxng"]["available"])


class TestFeeds(unittest.TestCase):
    def test_parse_rss_and_normalise_items(self):
        feed = parse_feed(RSS, "https://example.com/feed.xml")
        self.assertEqual(feed["title"], "Example News")
        self.assertEqual([item["title"] for item in feed["items"]], ["Second post", "First post"])
        self.assertEqual(feed["items"][0]["summary"], "Newer & better")
        self.assertEqual(feed["items"][0]["published"], "2024-01-02T12:00:00+00:00")

    def test_parse_atom(self):
        feed = parse_feed(ATOM, "https://example.com/atom.xml")
        item = feed["items"][0]
        self.assertEqual(item["title"], "Atom entry")
        self.assertEqual(item["link"], "https://example.com/atom-entry")
        self.assertEqual(item["author"], "Ada")
        self.assertTrue(item["published"].startswith("2024-03-01T10:00:00"))

    def test_feed_becomes_live_conversation_in_chronological_order(self):
        conversation = feed_to_conversation(parse_feed(RSS, "https://example.com/feed.xml"))
        self.assertEqual(conversation["type"], "feed")
        self.assertEqual(conversation["feedUrl"], "https://example.com/feed.xml")
        self.assertEqual(len(conversation["items"]), 2)
        # Oldest first so the newest item is at the bottom of a chat.
        self.assertIn("First post", conversation["items"][0]["content"])
        self.assertIn("Second post", conversation["items"][1]["content"])
        self.assertEqual(len(conversation["feedSeen"]), 2)

    def test_invalid_feed_raises(self):
        with self.assertRaises(ValueError):
            parse_feed("<html>not a feed</html>")


class TestBrowse(unittest.TestCase):
    def test_extract_text_strips_scripts_and_tags(self):
        text = extract_text("<html><script>bad()</script><h1>Hi</h1><p>there</p></html>", max_words=20)
        self.assertIn("Hi", text)
        self.assertIn("there", text)
        self.assertNotIn("bad()", text)


class TestTriggers(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ, {"G4F_BROWSING_STORE": os.path.join(self.tmp.name, "triggers.json")}
        )
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_parse_prompt_detects_feed_keywords_and_interval(self):
        draft = parse_trigger_prompt(
            'Notify me when the "OpenAI" launch is mentioned, every 10 minutes',
            {"url": "https://example.com/feed.xml"},
        )
        self.assertEqual(draft["source"]["type"], "rss")
        self.assertEqual(draft["source"]["url"], "https://example.com/feed.xml")
        self.assertEqual(draft["condition"]["keywords"], ["OpenAI"])
        self.assertEqual(draft["interval"], 600)
        self.assertTrue(draft["enabled"])

    def test_create_and_delete_trigger(self):
        trigger = trigger_module.create_trigger(
            {
                "prompt": "Watch https://example.com/feed.xml and tell me about AI",
                "source": {"url": "https://example.com/feed.xml"},
            }
        )
        self.assertEqual(len(trigger_module.list_triggers()), 1)
        self.assertIsNotNone(trigger_module.get_trigger(trigger["id"]))
        self.assertTrue(trigger_module.delete_trigger(trigger["id"]))
        self.assertEqual(trigger_module.list_triggers(), [])

    async def test_run_trigger_emits_once_and_deduplicates(self):
        trigger = trigger_module.create_trigger(
            {
                "prompt": "Watch https://example.com/feed.xml for AI",
                "source": {"url": "https://example.com/feed.xml"},
                "condition": {"keywords": []},
            }
        )
        items = [
            {
                "id": "item-1",
                "title": "AI news",
                "summary": "Something happened",
                "link": "https://example.com/1",
            }
        ]
        with patch.object(
            trigger_module, "_source_items", new=AsyncMock(return_value=items)
        ):
            first = await trigger_module.run_trigger(trigger["id"])
            second = await trigger_module.run_trigger(trigger["id"])

        self.assertEqual(len(first["new_items"]), 1)
        self.assertEqual(len(first["notifications"]), 1)
        self.assertEqual(second["new_items"], [])
        self.assertEqual(second["notifications"], [])
        self.assertEqual(first["notifications"][0]["kind"], "trigger")

    async def test_run_trigger_records_source_errors(self):
        trigger = trigger_module.create_trigger({"prompt": "Watch AI news"})
        with patch.object(
            trigger_module, "_source_items", new=AsyncMock(side_effect=ValueError("boom"))
        ):
            result = await trigger_module.run_trigger(trigger["id"])
        self.assertIn("boom", result["error"])
        self.assertEqual(trigger_module.get_trigger(trigger["id"])["last_error"], "ValueError: boom")


class TestNotifications(unittest.TestCase):
    def test_add_list_and_mark_read(self):
        store = NotificationStore(max_items=2)
        first = store.add("One", body="first")
        store.add("Two")
        store.add("Three")
        self.assertEqual(store.unread_count(), 2)
        # Oldest entries are evicted.
        self.assertEqual([item["title"] for item in store.list()], ["Three", "Two"])
        self.assertEqual(store.mark_read(first["id"]), 0)
        self.assertEqual(store.mark_read(), 2)
        self.assertEqual(store.unread_count(), 0)

    def test_trigger_notifications_are_available_to_endpoints(self):
        from g4f.browsing import notifications

        notifications.clear()
        item = notifications.add("Finished", body="Your message is ready", kind="chat")
        self.assertEqual(notifications.list()[0]["id"], item["id"])
        notifications.clear()


if __name__ == "__main__":
    unittest.main()
