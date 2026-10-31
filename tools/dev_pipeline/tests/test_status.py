from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.dev_pipeline.cli import _pr_command
from tools.dev_pipeline.config import ConfigError, load_config
from tools.dev_pipeline.freeflow import rank_models
from tools.dev_pipeline.runner import AgentRunner, _message_text, _summarise_tool_args
from tools.dev_pipeline.status import (
    BOARD_BEGIN,
    STATE_BEGIN,
    StatusError,
    StatusStore,
)
from tools.dev_pipeline.util import run_command_streaming


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

    def test_disabled_model_is_removed(self) -> None:
        ranked = rank_models(
            [
                "nvidia/nemotron-3.5-lightning:free",
                "cohere/north-mini-code:free",
            ],
            ["nvidia/nemotron-3.5-lightning:free", "north-mini-code"],
            ["nvidia/nemotron-3.5-lightning:free"],
        )
        self.assertEqual(ranked, ["cohere/north-mini-code:free"])

    def test_disabled_alias_matches_full_id(self) -> None:
        ranked = rank_models(
            [
                "nvidia/nemotron-3.5-lightning:free",
                "cohere/north-mini-code:free",
            ],
            [],
            ["nemotron-3.5-lightning-free"],
        )
        self.assertEqual(ranked, ["cohere/north-mini-code:free"])


class DisabledModelConfigTests(unittest.TestCase):
    def _config(self, payload):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "dev-pipeline.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_config(Path(temp.name), path)

    def test_disabled_models_load(self) -> None:
        config = self._config(
            {"disabled_models": ["nvidia/nemotron-3.5-lightning:free"]}
        )
        self.assertEqual(
            config.disabled_models, ["nvidia/nemotron-3.5-lightning:free"]
        )

    def test_disabled_models_default_to_empty(self) -> None:
        self.assertEqual(self._config({}).disabled_models, [])

    def test_disabled_models_must_be_strings(self) -> None:
        with self.assertRaises(ConfigError):
            self._config({"disabled_models": [123]})


class FallbackModelTests(unittest.TestCase):
    def _config(self, payload):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "dev-pipeline.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_config(Path(temp.name), path)

    def test_fallback_models_load(self) -> None:
        config = self._config(
            {"fallback_models": ["pollinations/deepseek/deepseek-v4.1-flash-1017"]}
        )
        self.assertEqual(
            config.fallback_models,
            ["pollinations/deepseek/deepseek-v4.1-flash-1017"],
        )

    def test_fallback_models_default_to_empty(self) -> None:
        self.assertEqual(self._config({}).fallback_models, [])

    def test_fallback_models_must_be_strings(self) -> None:
        with self.assertRaises(ConfigError):
            self._config({"fallback_models": [123]})

    def test_provider_qualified_model_ref_is_kept(self) -> None:
        runner = AgentRunner.__new__(AgentRunner)
        self.assertEqual(
            runner.model_ref("pollinations/deepseek/deepseek-v4.1-flash-1017"),
            "pollinations/deepseek/deepseek-v4.1-flash-1017",
        )
        self.assertEqual(
            runner.model_ref("ovh-free/gpt-oss-120b"),
            "freeflow/ovh-free/gpt-oss-120b",
        )
        self.assertEqual(
            runner.model_ref("big-pickle"),
            "freeflow/big-pickle",
        )


class StreamingProgressTests(unittest.TestCase):
    def test_streaming_forwards_lines(self) -> None:
        lines = []
        result = run_command_streaming(
            [sys.executable, "-c", "print('one'); print('two')"],
            Path("."),
            timeout=10,
            on_line=lines.append,
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(lines, ["one", "two"])

    def test_tool_summary_uses_command(self) -> None:
        self.assertEqual(
            _summarise_tool_args({"command": "pwd", "other": 1}),
            "pwd",
        )

    def test_message_text_joins_text_blocks(self) -> None:
        self.assertEqual(
            _message_text(
                {
                    "content": [
                        {"type": "thinking", "thinking": "skip"},
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ]
                }
            ),
            "hello world",
        )


class PrCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=self.root, text=True, capture_output=True
        )

    def _init_repo(self) -> None:
        self._git("init", "-q")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "Test")
        (self.root / "README.md").write_text("x\n", encoding="utf-8")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "init")

    def _run(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = _pr_command(SimpleNamespace(root=self.root))
        return code, out.getvalue(), err.getvalue()

    def test_rejects_non_git_directory(self) -> None:
        code, _, err = self._run()
        self.assertEqual(code, 1)
        self.assertIn("not a git repository", err)

    def test_rejects_missing_remote(self) -> None:
        self._init_repo()
        code, _, err = self._run()
        self.assertEqual(code, 1)
        self.assertIn("no git remote", err)

    def test_rejects_non_github_remote(self) -> None:
        self._init_repo()
        self._git("remote", "add", "origin", "https://gitlab.com/x/y.git")
        code, _, err = self._run()
        self.assertEqual(code, 1)
        self.assertIn("no GitHub remote", err)

    def test_rejects_missing_gh(self) -> None:
        self._init_repo()
        self._git("remote", "add", "origin", "https://github.com/x/y.git")
        with mock.patch("tools.dev_pipeline.cli.shutil.which", return_value=None):
            code, _, err = self._run()
        self.assertEqual(code, 1)
        self.assertIn("gh (GitHub CLI) not found", err)

    def _fake_gh(self, existing: bool = False) -> str:
        remote_temp = tempfile.TemporaryDirectory()
        self.addCleanup(remote_temp.cleanup)
        bin_dir = Path(remote_temp.name) / "bin"
        bin_dir.mkdir()
        gh = bin_dir / "gh"
        view_status = "0" if existing else "1"
        gh.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "auth" ]; then exit 0; fi\n'
            'if [ "$1" = "pr" ] && [ "$2" = "view" ]; then\n'
            f"  echo https://github.com/x/y/pull/1\n"
            f"  exit {view_status}\n"
            "fi\n"
            "echo https://github.com/x/y/pull/1\n",
            encoding="utf-8",
        )
        gh.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        return env

    def _add_github_remote(self) -> Path:
        remote_temp = tempfile.TemporaryDirectory()
        self.addCleanup(remote_temp.cleanup)
        bare = Path(remote_temp.name) / "github.com" / "x" / "y.git"
        bare.parent.mkdir(parents=True)
        subprocess.run(["git", "init", "--bare", "-q", str(bare)], check=True)
        self._git("remote", "add", "origin", str(bare))
        return bare

    def test_creates_branch_commit_pushes_and_opens_pr(self) -> None:
        self._init_repo()
        bare = self._add_github_remote()
        (self.root / "change.txt").write_text("change\n", encoding="utf-8")
        env = self._fake_gh(existing=False)
        with mock.patch.dict(os.environ, env, clear=False):
            code, out, err = self._run()
        self.assertEqual(code, 0, err)
        self.assertIn("created branch pipeline/pr-", out)
        self.assertIn("pushed pipeline/pr-", out)
        self.assertIn("PR created successfully", out)
        branches = self._git("branch", "--list", "pipeline/pr-*").stdout
        self.assertIn("pipeline/pr-", branches)
        log = self._git("log", "-1", "--pretty=%s").stdout.strip()
        self.assertEqual(log, "pipeline: prepare PR")
        remote_branches = subprocess.run(
            ["git", "--git-dir", str(bare), "for-each-ref", "--format=%(refname:short)"],
            text=True,
            capture_output=True,
        ).stdout
        self.assertIn("pipeline/pr-", remote_branches)

    def test_updates_existing_pr(self) -> None:
        self._init_repo()
        self._add_github_remote()
        (self.root / "change.txt").write_text("change\n", encoding="utf-8")
        env = self._fake_gh(existing=True)
        with mock.patch.dict(os.environ, env, clear=False):
            code, out, err = self._run()
        self.assertEqual(code, 0, err)
        self.assertIn("PR updated successfully", out)

    def test_falls_back_to_writable_fork_when_origin_is_read_only(self) -> None:
        self._init_repo()
        # A read-only upstream URL still looks like GitHub but cannot be pushed.
        self._git(
            "remote",
            "add",
            "origin",
            str(self.root / "missing" / "github.com" / "xtekky" / "gpt4free.git"),
        )
        remote_temp = tempfile.TemporaryDirectory()
        self.addCleanup(remote_temp.cleanup)
        fork = Path(remote_temp.name) / "github.com" / "g4f-copilot" / "gpt4free.git"
        fork.parent.mkdir(parents=True)
        subprocess.run(["git", "init", "--bare", "-q", str(fork)], check=True)
        self._git("remote", "add", "fork", str(fork))

        (self.root / "change.txt").write_text("main\n", encoding="utf-8")
        env = self._fake_gh(existing=False)
        with mock.patch.dict(os.environ, env, clear=False):
            code, out, err = self._run()
        self.assertEqual(code, 0, err)
        self.assertIn("pushed pipeline/pr-", out)
        self.assertIn("to fork", out)
        remote_branches = subprocess.run(
            ["git", "--git-dir", str(fork), "for-each-ref", "--format=%(refname:short)"],
            text=True,
            capture_output=True,
        ).stdout
        self.assertIn("pipeline/pr-", remote_branches)


if __name__ == "__main__":
    unittest.main()
