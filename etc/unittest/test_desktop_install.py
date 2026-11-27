"""Guard the shared desktop installer and generated Polli command assets."""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class TestSharedDesktopInstaller(unittest.TestCase):
    def test_installer_is_reused_by_polli_and_android(self):
        installer = (ROOT / "bin" / "install-desktop.sh").read_text(encoding="utf-8")
        self.assertTrue(installer.startswith("#!/bin/sh"))
        for needle in ("apt-get update -qq", "apt-get install -y --no-install-recommends",
                       "G4F_DESKTOP_REPAIR", "G4F_DESKTOP_NOVNC_ASSETS",
                       '"$NOVNC_ASSETS/vnc.html"', "rm -rf /usr/share/novnc",
                       "dpkg --configure -a"):
            with self.subTest(needle=needle):
                self.assertIn(needle, installer)

        polli = (ROOT / "bin" / "polli-sandbox-bootstrap.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("/usr/local/bin/install-desktop.sh", polli)
        self.assertIn("G4F_DESKTOP_UPDATE_ATTEMPTS=1", polli)

        android = (ROOT / "projects" / "android" / "app" / "src" / "main"
                   / "assets" / "container" / "scripts" / "provision-desktop.sh")
        text = android.read_text(encoding="utf-8")
        self.assertIn("/g4f-bin/install-desktop.sh", text)
        self.assertIn(
            "G4F_DESKTOP_PACKAGES='xfce4 tigervnc-standalone-server novnc "
            "dbus-x11 procps'", text
        )
        self.assertNotIn(" xterm'", text)
        catalog = json.loads((ROOT / "projects" / "android" / "app" / "src"
                              / "main" / "assets" / "container" / "catalog.json")
                             .read_text(encoding="utf-8"))
        self.assertNotIn("xterm", catalog["g4f"]["desktop_packages"])

        cli = (ROOT / "bin" / "sandbox-cli").read_text(encoding="utf-8")
        self.assertIn("/usr/local/bin/install-desktop.sh", cli)

    def test_go_embed_assets_are_generated_from_bin(self):
        sync = (ROOT / "g4f-go" / "sync-assets.sh").read_text(encoding="utf-8")
        self.assertIn('cp "$REPO/bin/$name" "$DEST/$name"', sync)
        for name in ("sandbox-cli", "polli-sandbox-bootstrap.sh",
                     "install-desktop.sh", "patch-xfce-menu.py",
                     "sandbox-discover", "sandbox-docs", "start-browser"):
            with self.subTest(name=name):
                self.assertIn(name, sync)
                self.assertTrue((ROOT / "bin" / name).is_file())

        assets = (ROOT / "g4f-go" / "polli" / "assets.go").read_text(encoding="utf-8")
        self.assertIn("//go:embed assets/install-desktop.sh", assets)
        self.assertIn("generated from bin/", assets)
        self.assertNotIn("tracked snapshots", assets)

        remote = (ROOT / "g4f-go" / "polli" / "remote.go").read_text(encoding="utf-8")
        self.assertIn('{"usr/local/bin/install-desktop.sh", installDesktopScript, 0o755}',
                      remote)

        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/g4f-go/polli/assets/", gitignore)
        self.assertNotIn("!/g4f-go/polli/assets/", gitignore)


if __name__ == "__main__":
    unittest.main()
