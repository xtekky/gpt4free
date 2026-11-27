import os
import sys
import unittest
from unittest.mock import AsyncMock, patch

DISCORD_BOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "projects", "discord-bot")
)
if DISCORD_BOT_DIR not in sys.path:
    sys.path.insert(0, DISCORD_BOT_DIR)

from buffer_client import BufferClient


class TestBufferClient(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_channel_ids_skip_discovery(self):
        client = BufferClient(api_key="test-key", channel_ids=["ch1", "ch2"])
        with patch.object(client, "_graphql", new=AsyncMock()) as mock_gql:
            channels = await client.get_target_channel_ids()
        self.assertEqual(channels, ["ch1", "ch2"])
        mock_gql.assert_not_called()

    async def test_auto_discovers_active_channels(self):
        client = BufferClient(api_key="test-key")
        responses = [
            {"account": {"organizations": [{"id": "org1", "name": "Org"}]}},
            {
                "channels": [
                    {"id": "ch_active", "name": "X", "service": "twitter", "isQueuePaused": False},
                    {"id": "ch_paused", "name": "IG", "service": "instagram", "isQueuePaused": True},
                ]
            },
        ]
        with patch.object(client, "_graphql", new=AsyncMock(side_effect=responses)):
            channels = await client.get_target_channel_ids()
            # Second call should use cached discovery
            channels_cached = await client.get_target_channel_ids()
        self.assertEqual(channels, ["ch_active"])
        self.assertEqual(channels_cached, ["ch_active"])

    async def test_create_post_with_images_and_fallback_text(self):
        client = BufferClient(api_key="test-key", channel_ids=["ch1"], mode="shareNow")
        mock_gql = AsyncMock(
            return_value={
                "createPost": {
                    "post": {"id": "p1", "text": "📢 Update", "dueAt": None}
                }
            }
        )
        with patch.object(client, "_graphql", new=mock_gql):
            posts = await client.create_post(
                text="",
                image_urls=["https://cdn.example.com/a.png", "ftp://ignored.example.com/b.png"],
            )

        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0]["id"], "p1")
        _, kwargs = mock_gql.call_args
        post_input = kwargs["variables"]["input"]
        self.assertEqual(post_input["text"], "📢 Update")
        self.assertEqual(post_input["channelId"], "ch1")
        self.assertEqual(post_input["mode"], "shareNow")
        self.assertEqual(
            post_input["assets"],
            [{"image": {"url": "https://cdn.example.com/a.png"}}],
        )

    async def test_create_post_raises_on_mutation_error(self):
        client = BufferClient(api_key="test-key", channel_ids=["ch1"])
        mock_gql = AsyncMock(
            return_value={"createPost": {"message": "Channel not found"}}
        )
        with patch.object(client, "_graphql", new=mock_gql):
            with self.assertRaises(RuntimeError) as ctx:
                await client.create_post(text="Hello")
        self.assertIn("Channel not found", str(ctx.exception))

    async def test_create_post_applies_channel_specific_rules_on_discovery(self):
        client = BufferClient(api_key="test-key")
        long_text = ("This is a long announcement sentence for testing. " * 8).strip()
        responses = [
            {"account": {"organizations": [{"id": "org1", "name": "Org"}]}},
            {
                "channels": [
                    {"id": "fb_grp", "name": "FB Group", "service": "facebook", "type": "group", "isQueuePaused": False},
                    {"id": "tw_ch", "name": "X", "service": "twitter", "type": "profile", "isQueuePaused": False},
                    {"id": "ig_ch", "name": "IG", "service": "instagram", "type": "profile", "isQueuePaused": False},
                ]
            },
            {"createPost": {"post": {"id": "p_fb", "text": long_text, "dueAt": None}}},
            {"createPost": {"post": {"id": "p_tw", "text": "part1", "dueAt": None}}},
            {"createPost": {"post": {"id": "p_ig", "text": long_text, "dueAt": None}}},
        ]
        mock_gql = AsyncMock(side_effect=responses)
        with patch.object(client, "_graphql", new=mock_gql):
            posts = await client.create_post(
                text=long_text,
                image_urls=["https://cdn.example.com/img.png"],
            )

        self.assertEqual([p["id"] for p in posts], ["p_fb", "p_tw", "p_ig"])

        fb_input = mock_gql.call_args_list[2].kwargs["variables"]["input"]
        self.assertEqual(fb_input["schedulingType"], "notification")

        tw_input = mock_gql.call_args_list[3].kwargs["variables"]["input"]
        self.assertEqual(tw_input["schedulingType"], "automatic")
        self.assertLessEqual(len(tw_input["text"]), 280)
        thread = tw_input["metadata"]["twitter"]["thread"]
        self.assertGreater(len(thread), 1)
        self.assertTrue(all(len(item["text"]) <= 280 for item in thread))
        self.assertEqual(thread[0]["assets"], [{"image": {"url": "https://cdn.example.com/img.png"}}])
        self.assertNotIn("assets", tw_input)

        ig_input = mock_gql.call_args_list[4].kwargs["variables"]["input"]
        self.assertEqual(ig_input["schedulingType"], "notification")
        self.assertEqual(
            ig_input["metadata"]["instagram"],
            {"type": "post", "shouldShareToFeed": True},
        )

    async def test_create_post_fallback_retries_for_explicit_channels(self):
        client = BufferClient(api_key="test-key", channel_ids=["fb_grp", "tw_ch", "ig_ch"])
        long_text = ("Announcement update with enough words to exceed 280 chars. " * 7).strip()
        responses = [
            {"createPost": {"message": "Facebook Group channels require notification scheduling. Use notification scheduling instead."}},
            {"createPost": {"post": {"id": "p_fb", "text": long_text, "dueAt": None}}},
            {"createPost": {"message": "Invalid post: Twitter / X posts cannot exceed 280 characters."}},
            {"createPost": {"message": "Threads are not supported on this plan"}},
            {"createPost": {"post": {"id": "p_tw", "text": long_text[:280], "dueAt": None}}},
            {"createPost": {"message": "Invalid post: Instagram posts require a type (post, story, or reel)."}},
            {"createPost": {"post": {"id": "p_ig", "text": long_text, "dueAt": None}}},
        ]
        mock_gql = AsyncMock(side_effect=responses)
        with patch.object(client, "_graphql", new=mock_gql):
            posts = await client.create_post(text=long_text)

        self.assertEqual([p["id"] for p in posts], ["p_fb", "p_tw", "p_ig"])
        tw_truncated_input = mock_gql.call_args_list[4].kwargs["variables"]["input"]
        self.assertLessEqual(len(tw_truncated_input["text"]), 280)
        self.assertNotIn("metadata", tw_truncated_input)


if __name__ == "__main__":
    unittest.main()
