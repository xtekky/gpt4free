"""Launchers installed under the user's home are published on the system PATH."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


CLI = Path(__file__).resolve().parents[2] / "bin" / "sandbox-cli"


def run_cli(arguments, home, system_bin, path_prefix=None):
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["SANDBOX_SYSTEM_BIN_DIR"] = str(system_bin)
    env["G4F_SANDBOX_ENV"] = str(home / "missing-settings")
    env.pop("G4F_PROXY", None)
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}:{env['PATH']}"
    return subprocess.run(
        [str(CLI), *arguments], env=env, capture_output=True, text=True,
        timeout=10,
    )


class TestSandboxCliSystemPath(unittest.TestCase):
    def make_launcher(self, home, name="opencode", directory=".opencode/bin"):
        launcher = home / directory / name
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        launcher.chmod(0o755)
        return launcher

    def test_exit_publishes_user_launchers(self):
        # `list` exits without launching a tool, so this covers the EXIT hook.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home, system_bin = root / "home", root / "system-bin"
            home.mkdir()
            system_bin.mkdir()
            launcher = self.make_launcher(home)
            result = run_cli(["list"], home, system_bin)
            link = system_bin / "opencode"
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(link.is_symlink(), result.stderr)
            self.assertEqual(link.resolve(), launcher.resolve())

    def test_launch_publishes_before_exec(self):
        # exec() skips the EXIT hook, so launch() must publish first.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home, system_bin, fake_path = (
                root / "home", root / "system-bin", root / "fake-path",
            )
            home.mkdir()
            system_bin.mkdir()
            fake_path.mkdir()
            launcher = self.make_launcher(home, "pi", ".local/bin")
            fake = fake_path / "g4f-go"
            fake.write_text('#!/bin/sh\nprintf "tool launched\\n"\n', encoding="utf-8")
            fake.chmod(0o755)
            result = run_cli(["g4f-go", "help"], home, system_bin, fake_path)
            link = system_bin / "pi"
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "tool launched\n")
            self.assertTrue(link.is_symlink(), result.stderr)
            self.assertEqual(link.resolve(), launcher.resolve())

    def test_system_binary_with_the_same_name_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home, system_bin = root / "home", root / "system-bin"
            home.mkdir()
            system_bin.mkdir()
            self.make_launcher(home)
            real = system_bin / "opencode"
            real.write_text("system binary\n", encoding="utf-8")
            result = run_cli(["list"], home, system_bin)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(real.is_symlink())
            self.assertEqual(real.read_text(encoding="utf-8"), "system binary\n")


if __name__ == "__main__":
    unittest.main()
