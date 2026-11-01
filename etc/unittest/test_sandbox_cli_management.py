"""sandbox-cli forwards launcher/package management to g4f-go."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


CLI = Path(__file__).resolve().parents[2] / "bin" / "sandbox-cli"


class TestSandboxCliManagement(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        self.out = self.directory / "args"
        fake = self.directory / "g4f-go"
        fake.write_text(
            '#!/bin/sh\nprintf "%s\\n" "$@" > "$G4F_CLI_ARGS"\n',
            encoding="utf-8",
        )
        fake.chmod(0o755)
        self.env = os.environ.copy()
        self.env["PATH"] = f"{self.directory}:{self.env['PATH']}"
        self.env["G4F_SANDBOX_ENV"] = str(self.directory / "missing-settings")
        self.env["G4F_CLI_ARGS"] = str(self.out)
        self.env.pop("G4F_PROXY", None)

    def tearDown(self):
        self._tmp.cleanup()

    def run_cli(self, *args):
        return subprocess.run(
            [str(CLI), *args], env=self.env, capture_output=True,
            text=True, timeout=10,
        )

    def forwarded(self):
        return self.out.read_text(encoding="utf-8").splitlines()

    def test_update_forwards_flags(self):
        result = self.run_cli("update", "--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.forwarded(), ["update", "--check"])

    def test_upgrade_forwards_to_g4f_go(self):
        result = self.run_cli("upgrade")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.forwarded(), ["upgrade"])

    def test_uninstall_forwards_flags(self):
        result = self.run_cli("uninstall", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.forwarded(), ["uninstall", "--dry-run"])

    def test_uninstall_forwards_a_package_name(self):
        result = self.run_cli("uninstall", "g4f")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.forwarded(), ["uninstall", "g4f"])

    def test_help_does_not_call_g4f_go(self):
        result = self.run_cli("update", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sandbox-cli update", result.stdout)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
