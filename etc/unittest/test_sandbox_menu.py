from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MENU_DIR = ROOT / "docker" / "sandbox-menu"


def _exec_line(entry: Path) -> str:
    for line in entry.read_text(encoding="utf-8").splitlines():
        if line.startswith("Exec="):
            return line[len("Exec="):]
    raise AssertionError(f"{entry.name} has no Exec line")


def _categories(entry: Path) -> set[str]:
    for line in entry.read_text(encoding="utf-8").splitlines():
        if line.startswith("Categories="):
            return set(line[len("Categories="):].strip(";").split(";"))
    raise AssertionError(f"{entry.name} has no Categories line")


def _load_menu_patch():
    path = ROOT / "bin" / "patch-xfce-menu.py"
    spec = importlib.util.spec_from_file_location("patch_xfce_menu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSandboxMenuEntries(unittest.TestCase):
    def test_terminal_entries_open_their_own_window(self):
        """Each menu entry must not join an existing terminal as a tab."""
        entries = sorted(MENU_DIR.glob("g4f-sandbox-*.desktop"))
        self.assertTrue(entries)
        for entry in entries:
            argv = _exec_line(entry).split()
            if argv[0] != "xfce4-terminal":
                continue
            self.assertIn(
                "--disable-server",
                argv,
                f"{entry.name} would open as a tab in an existing terminal window",
            )

    def test_files_and_editors_share_one_menu_category(self):
        """Editors live in the Files menu instead of a separate Editors menu."""
        for name in ("files", "nemo", "kate", "code-server"):
            categories = _categories(MENU_DIR / f"g4f-sandbox-{name}.desktop")
            self.assertIn(
                "X-G4F-Files",
                categories,
                f"{name} is missing the merged Files & Editors category",
            )
            self.assertNotIn(
                "X-G4F-Editor",
                categories,
                f"{name} still uses the obsolete Editors category",
            )

        patch = _load_menu_patch()
        menu_names = [name for name, _ in patch.MENUS]
        self.assertNotIn("Editors", menu_names)
        self.assertIn("Files", menu_names)
        self.assertIn("Editors", patch.REMOVE_MENUS)

        files_menu = dict(patch.MENUS)["Files"]
        self.assertIn("<Category>X-G4F-Files</Category>", files_menu)
        self.assertNotIn("X-G4F-Editor", files_menu)

    def test_old_editors_menu_is_removed(self):
        """Re-running the patch upgrades a menu file from an older image."""
        patch = _load_menu_patch()
        old_menu = """    <Menu>
        <Name>Editors</Name>
        <Directory>g4f-sandbox-editors.directory</Directory>
        <Include>
            <Category>X-G4F-Editor</Category>
        </Include>
    </Menu>
"""
        src, removed = patch.remove_named_menus(old_menu, patch.REMOVE_MENUS)
        self.assertEqual(removed, ["Editors"])
        self.assertNotIn("<Name>Editors</Name>", src)


if __name__ == "__main__":
    unittest.main()
