"""The `nodejs` tool installs the newest Node.js release, not Debian's older one."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "bin" / "sandbox-cli"
COPIES = (
    CLI,
    ROOT / "projects" / "android" / "app" / "src" / "main" / "assets"
    / "container" / "bin" / "sandbox-cli",
    ROOT / "g4f-go" / "polli" / "assets" / "sandbox-cli",
)
MENU_ENTRY = ROOT / "docker" / "sandbox-menu" / "g4f-sandbox-nodejs.desktop"


class TestSandboxCliNodejs(unittest.TestCase):
    def run_cli(self, *arguments):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            env = os.environ.copy()
            env["HOME"] = str(home)
            env["G4F_SANDBOX_ENV"] = str(home / "missing-settings")
            env["SANDBOX_SYSTEM_BIN_DIR"] = str(home / "bin")
            env.pop("G4F_PROXY", None)
            (home / "bin").mkdir()
            return subprocess.run(
                [str(CLI), *arguments], env=env, capture_output=True,
                text=True, timeout=10,
            )

    def test_list_publishes_nodejs_in_the_g4f_group(self):
        result = self.run_cli("list")
        self.assertEqual(result.returncode, 0, result.stderr)
        group = None
        entries = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            if not line.startswith(" "):
                group = line.strip()
            elif group == "g4f":
                entries[line.strip().split()[0]] = line.strip()
        self.assertIn("nodejs", entries)
        self.assertIn("newest Node.js", entries["nodejs"])

    def test_help_and_usage_mention_nodejs(self):
        result = self.run_cli("help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sandbox-cli nodejs", result.stdout)

    def test_installer_uses_current_release_with_apt_fallback(self):
        source = CLI.read_text(encoding="utf-8")
        for needle in (
            'NODEJS_SETUP_URL="${NODEJS_SETUP_URL:-https://deb.nodesource.com/setup_current.x}"',
            "install_nodejs_latest()",
            'bash -c \'set -o pipefail; curl -fsSL "$1" | bash -\'',
            "apt-get install -y -qq nodejs",
            "falling back to Debian's nodejs/npm packages",
            "nodejs)",
            'launch "Node.js (latest)" node',
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, source)

        block = source[source.index("ensure_node() {"):source.index("ensure_npm()")]
        self.assertIn("install_nodejs_latest", block)
        self.assertIn("apt-get install -y -qq nodejs npm", block)

    def test_every_cli_copy_has_the_nodejs_tool(self):
        contents = [path.read_bytes() for path in COPIES]
        self.assertEqual(len(set(contents)), 1, "sandbox-cli copies are out of sync")
        self.assertIn(b"install_nodejs_latest()", contents[0])

    def test_desktop_menu_entry_opens_the_installer(self):
        content = MENU_ENTRY.read_text(encoding="utf-8")
        self.assertIn("--disable-server", content)
        self.assertIn('--command="/usr/local/bin/sandbox-cli nodejs"', content)
        self.assertIn("Name=Latest Node.js", content)
        readme = (ROOT / "docker" / "sandbox-menu" / "README.md").read_text(
            encoding="utf-8")
        self.assertIn("| `nodejs`", readme)


if __name__ == "__main__":
    unittest.main()
