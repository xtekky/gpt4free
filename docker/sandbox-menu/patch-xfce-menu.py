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
the root menu). Idempotent: re-running is a no-op.
"""

import sys

MENU_PATH = "/etc/xdg/menus/xfce-applications.menu"
ANCHOR = "    <DefaultMergeDirs/>"

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
        "Editors",
        """    <Menu>
        <Name>Editors</Name>
        <Directory>g4f-sandbox-editors.directory</Directory>
        <Include>
            <Category>X-G4F-Editor</Category>
        </Include>
    </Menu>
""",
    ),
]

def main() -> int:
    with open(MENU_PATH, encoding="utf-8") as fh:
        src = fh.read()

    # Each menu is checked separately so an already-patched file still gains
    # menus added in a later release.
    missing = [
        (name, xml)
        for name, xml in MENUS
        if f"<Name>{name}</Name>" not in src
    ]
    if not missing:
        print(f"{MENU_PATH}: already patched")
        return 0

    if ANCHOR not in src:
        print(f"{MENU_PATH}: anchor {ANCHOR!r} not found", file=sys.stderr)
        return 1

    block = "\n".join(xml.rstrip("\n") for _, xml in missing) + "\n\n"
    with open(MENU_PATH, "w", encoding="utf-8") as fh:
        fh.write(src.replace(ANCHOR, block + ANCHOR, 1))

    print(f"{MENU_PATH}: patched ({', '.join(name for name, _ in missing)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())