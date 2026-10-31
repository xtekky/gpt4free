from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.dev_pipeline.freeflow import rank_models
from tools.dev_pipeline.status import (
    BOARD_BEGIN,
    STATE_BEGIN,
    StatusError,
    StatusStore,
)


class StatusStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "AGENTS.md"
        self.path.write_text("# AGENTS.md\n\nProject notes.\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_add_save_and_reload(self) -> None:
        store = StatusStore("gpt4free", self.path)
        self.assertTrue(store.created)
        task = store.add_task(
            "Fix the widget",
            project="gpt4free",
            priority="P1",
            acceptance="Widget works",
        )
        self.assertEqual(task["id"], "G4F-001")
        store.save()

        text = self.path.read_text(encoding="utf-8")
        self.assertIn(STATE_BEGIN, text)
        self.assertIn(BOARD_BEGIN, text)
        self.assertIn("Fix the widget", text)

        reloaded = StatusStore("gpt4free", self.path)
        self.assertFalse(reloaded.created)
        self.assertEqual(len(reloaded.tasks), 1)
        self.assertEqual(reloaded.tasks[0]["status"], "pending")

    def test_priority_and_status_updates(self) -> None:
        store = StatusStore("gpt4free", self.path)
        low = store.add_task("Low", priority="P3")
        high = store.add_task("High", priority="P0")
        selected = store.select_task()
        self.assertEqual(selected["id"], high["id"])

        store.update_task(high["id"], status="in-progress", increment_attempt=True)
        self.assertEqual(store.get_task(high["id"])["attempts"], 1)
        selected = store.select_task()
        self.assertEqual(selected["id"], low["id"])

        store.update_task(low["id"], status="blocked")
        self.assertIsNone(store.select_task())
        retried = store.select_task(retry_blocked=True)
        self.assertEqual(retried["id"], low["id"])

    def test_board_escapes_pipes_and_newlines(self) -> None:
        store = StatusStore("gpt4free", self.path)
        store.add_task("Line one\nline two | pipe", acceptance="a | b")
        store.save()
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("Line one line two \\| pipe", text)
        self.assertIn("a \\| b", text)

    def test_invalid_state_raises(self) -> None:
        self.path.write_text(
            f"# AGENTS.md\n\n{STATE_BEGIN}\n```json\n{{not json}}\n```\n"
            "<!-- DEV-PIPELINE:STATE:END -->\n",
            encoding="utf-8",
        )
        with self.assertRaises(StatusError):
            StatusStore("gpt4free", self.path)

    def test_state_json_is_valid(self) -> None:
        store = StatusStore("gpt4free", self.path)
        store.add_task("A task")
        store.save()
        text = self.path.read_text(encoding="utf-8")
        start = text.index(STATE_BEGIN) + len(STATE_BEGIN)
        end = text.index("<!-- DEV-PIPELINE:STATE:END -->")
        block = text[start:end].strip()
        block = block.strip()
        if block.startswith("```json"):
            block = block[len("```json"):]
        if block.endswith("```"):
            block = block[:-3]
        block = block.strip()
        payload = json.loads(block)
        self.assertEqual(payload["schema"], 1)
        self.assertEqual(len(payload["tasks"]), 1)


class ModelRankingTests(unittest.TestCase):
    def test_preferred_model_comes_first(self) -> None:
        models = [
            "ovh-free/Meta-Llama-3_3-70B-Instruct",
            "deepseek-v4.1-flash-free",
            "glm-5.3-flash-free",
        ]
        ranked = rank_models(models, ["deepseek-v4.1-flash-free", "glm-5.3-flash-free"])
        self.assertEqual(ranked[0], "deepseek-v4.1-flash-free")
        self.assertEqual(ranked[1], "glm-5.3-flash-free")

    def test_alias_matches_full_id(self) -> None:
        ranked = rank_models(
            ["ovh-free/Qwen3.5-397B-A17B", "ovh-free/Meta-Llama-3_3-70B-Instruct"],
            ["qwen3.5-397b", "llama-3.3-70b"],
        )
        self.assertEqual(ranked[0], "ovh-free/Qwen3.5-397B-A17B")


if __name__ == "__main__":
    unittest.main()
