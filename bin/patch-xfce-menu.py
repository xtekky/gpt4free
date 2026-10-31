#!/usr/bin/env python3
"""Add the g4f sandbox categories to the Xfce applications menu.

The sandbox .desktop entries carry X-G4F-* categories instead of Development,
so they need their own top-level menus. Shipping them as a merged menu file
does not work: garcon copies a merged file's elements with
g_node_traverse(..., depth=2, G_IN_ORDER), which never descends to the nested
<Menu> nodes, so only the merged root's direct children are inserted and the
menus themselves are silently dropped.

Instead the stock /etc/xdg/menus/xfce-applications.menu is patched directly,
inserting the menus just before its <DefaultMergeDirs/> node (the last child of
the root menu). The stock Internet (network) and Development menus are removed
at the same time, so the Applications menu contains only the sandbox's own
categories plus the remaining desktop menus. Idempotent: re-running is a no-op.
"""

import re
import sys

MENU_PATH = "/etc/xdg/menus/xfce-applications.menu"
ANCHOR = "    <DefaultMergeDirs/>"

# Stock top-level menus that are not useful in the sandbox image. Xfce labels
# the menu whose XML name is "Network" as "Internet" in its locale files.
# "Editors" is not stock: it is the separate menu older images used before
# files and editors were merged, and re-running the patch removes it.
REMOVE_MENUS = ("Internet", "Network", "Development", "Editors")

MENUS = [
    (
        "g4f",
        """    <Menu>
        <Name>g4f</Name>
        <Directory>g4f-sandbox-g4f.directory</Directory>
        <Include>
            <Category>X-G4F-Core</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Browsers",
        """    <Menu>
        <Name>Browsers</Name>
        <Directory>g4f-sandbox-browsers.directory</Directory>
        <Include>
            <Category>X-G4F-Browser</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Files",
        """    <Menu>
        <Name>Files</Name>
        <Directory>g4f-sandbox-files.directory</Directory>
        <Include>
            <Category>X-G4F-Files</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Coding Agents",
        """    <Menu>
        <Name>Coding Agents</Name>
        <Directory>g4f-sandbox-coding-agents.directory</Directory>
        <Include>
            <Category>X-G4F-Agent</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Agent Skills",
        """    <Menu>
        <Name>Agent Skills</Name>
        <Directory>g4f-sandbox-skills.directory</Directory>
        <Include>
            <Category>X-G4F-Skills</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Gateways",
        """    <Menu>
        <Name>Gateways</Name>
        <Directory>g4f-sandbox-gateways.directory</Directory>
        <Include>
            <Category>X-G4F-Gateway</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Local AI",
        """    <Menu>
        <Name>Local AI</Name>
        <Directory>g4f-sandbox-local-ai.directory</Directory>
        <Include>
            <Category>X-G4F-LocalAI</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Assistants",
        """    <Menu>
        <Name>Assistants</Name>
        <Directory>g4f-sandbox-assistants.directory</Directory>
        <Include>
            <Category>X-G4F-Assistant</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Chat UIs",
        """    <Menu>
        <Name>Chat UIs</Name>
        <Directory>g4f-sandbox-chat-uis.directory</Directory>
        <Include>
            <Category>X-G4F-ChatUI</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Writing",
        """    <Menu>
        <Name>Writing</Name>
        <Directory>g4f-sandbox-writing.directory</Directory>
        <Include>
            <Category>X-G4F-Writing</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Media",
        """    <Menu>
        <Name>Media</Name>
        <Directory>g4f-sandbox-media.directory</Directory>
        <Include>
            <Category>X-G4F-Media</Category>
        </Include>
    </Menu>
""",
    ),
    (
        "Sandboxes",
        """    <Menu>
        <Name>Sandboxes</Name>
        <Directory>g4f-sandbox-sandboxes.directory</Directory>
        <Include>
            <Category>X-G4F-Sandboxes</Category>
        </Include>
    </Menu>
""",
    ),
]

def _menu_pattern(name: str) -> re.Pattern[str]:
    """Match one top-level <Menu> block by its XML name."""
    # Top-level menus are indented by four spaces; the non-greedy body reaches
    # the first closing tag with the same indentation, so nested menus (the
    # Settings submenu) are not cut in half.
    return re.compile(
        rf"^    <Menu>\s*\n\s*<Name>{re.escape(name)}</Name>.*?^    </Menu>\n",
        re.MULTILINE | re.DOTALL,
    )


def remove_named_menus(src: str, names: tuple[str, ...]) -> tuple[str, list[str]]:
    """Drop the top-level <Menu> blocks named in *names*."""
    removed: list[str] = []
    for name in names:
        src, count = _menu_pattern(name).subn("", src)
        if count:
            removed.append(name)
    return src, removed


def drop_outdated_menus(src: str) -> tuple[str, list[str]]:
    """Remove generated menus whose block no longer matches MENUS.

    This upgrades a menu file patched by an older image when a menu's include
    list or directory changes in a later release. Matching blocks stay
    untouched, so re-running the script remains a no-op.
    """
    outdated: list[str] = []
    for name, xml in MENUS:
        pattern = _menu_pattern(name)
        match = pattern.search(src)
        if match and match.group(0).strip() != xml.strip():
            src = src[: match.start()] + src[match.end():]
            outdated.append(name)
    return src, outdated


def main() -> int:
    with open(MENU_PATH, encoding="utf-8") as fh:
        src = fh.read()

    src, removed = remove_named_menus(src, REMOVE_MENUS)
    src, outdated = drop_outdated_menus(src)

    # Each menu is checked separately so an already-patched file still gains
    # menus added in a later release.
    missing = [
        (name, xml)
        for name, xml in MENUS
        if f"<Name>{name}</Name>" not in src
    ]
    if not missing and not removed:
        print(f"{MENU_PATH}: already patched")
        return 0

    if missing:
        if ANCHOR not in src:
            print(f"{MENU_PATH}: anchor {ANCHOR!r} not found", file=sys.stderr)
            return 1
        block = "\n".join(xml.rstrip("\n") for _, xml in missing) + "\n\n"
        src = src.replace(ANCHOR, block + ANCHOR, 1)

    with open(MENU_PATH, "w", encoding="utf-8") as fh:
        fh.write(src)

    changes = []
    added = [name for name, _ in missing if name not in outdated]
    if added:
        changes.append("added " + ", ".join(added))
    if outdated:
        changes.append("updated " + ", ".join(outdated))
    if removed:
        changes.append("removed " + ", ".join(removed))
    print(f"{MENU_PATH}: patched ({'; '.join(changes)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())