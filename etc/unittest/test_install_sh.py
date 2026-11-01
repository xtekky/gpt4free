from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / "bin" / "install.sh"


@unittest.skipUnless(INSTALL.exists(), "bin/install.sh is not present")
class TestInstallSh(unittest.TestCase):
    def test_termux_installs_android_release_into_prefix_bin(self):
        """Termux installs into $PREFIX/bin, which is already on PATH."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prefix = tmp_path / "usr"
            (prefix / "bin").mkdir(parents=True)
            fakebin = tmp_path / "fakebin"
            fakebin.mkdir()

            archive = fakebin / "release.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("g4f-go", "#!/bin/sh\necho fake g4f-go\n")

            (fakebin / "uname").write_text(
                "#!/bin/sh\n"
                'case "$1" in\n'
                "  -s) echo Linux ;;\n"
                "  -m) echo aarch64 ;;\n"
                "  -o) echo Android ;;\n"
                "  *) echo Linux ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            (fakebin / "curl").write_text(
                "#!/bin/sh\n"
                'out=""\n'
                'while [ $# -gt 0 ]; do\n'
                '  case "$1" in\n'
                '    -o) out="$2"; shift 2 ;;\n'
                "    *) shift ;;\n"
                "  esac\n"
                "done\n"
                '[ -n "$out" ] && cp "${FAKE_RELEASE:?}" "$out"\n'
                "exit 0\n",
                encoding="utf-8",
            )
            for script in (fakebin / "uname", fakebin / "curl"):
                script.chmod(0o755)

            env = {
                "PATH": f"{fakebin}{os.pathsep}{os.environ.get('PATH', '')}",
                "HOME": str(tmp_path / "home"),
                "TERMUX_VERSION": "0.118",
                "PREFIX": str(prefix),
                "FAKE_RELEASE": str(archive),
            }
            env.pop("G4F_GO_PREFIX", None)

            result = subprocess.run(
                ["bash", str(INSTALL), "--version", "v0.1.0", "--no-verify"],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("android/arm64", result.stdout)
            installed = prefix / "bin" / "g4f-go"
            self.assertTrue(installed.exists(), result.stdout)
            self.assertIn("fake g4f-go", installed.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
