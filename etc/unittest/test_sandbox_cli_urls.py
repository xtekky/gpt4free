"""URL listings are opt-in, not part of every sandbox tool launch."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


CLI = Path(__file__).resolve().parents[2] / "bin" / "sandbox-cli"


class TestSandboxCliUrls(unittest.TestCase):
    def test_launch_does_not_print_public_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "g4f-go"
            fake.write_text('#!/bin/sh\nprintf "tool launched: %s\\n" "$*"\n', encoding="utf-8")
            fake.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{directory}:{env['PATH']}"
            env["G4F_SANDBOX_ENV"] = str(Path(directory) / "missing-settings")
            env.pop("G4F_PROXY", None)
            result = subprocess.run(
                [str(CLI), "g4f-go", "help"], env=env, capture_output=True,
                text=True, timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "tool launched: help\n")

    def test_polli_asset_launch_does_not_print_public_urls(self):
        text = (CLI.parents[1] / "g4f-go" / "polli" / "assets" / "sandbox-cli").read_text(
            encoding="utf-8"
        )
        start = text.index("launch() {")
        body = text[start:text.index("\n}\n", start)]
        self.assertNotIn("public_urls", body)

    def test_urls_command_still_prints_list(self):
        result = subprocess.run(
            [str(CLI), "urls"], capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Public Urls", result.stdout)
        self.assertIn("http://g4f.g4f.local/", result.stdout)


if __name__ == "__main__":
    unittest.main()
