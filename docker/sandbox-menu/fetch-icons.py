#!/usr/bin/env python3
"""Install a per-tool icon for every sandbox desktop menu entry.

Every docker/sandbox-menu/g4f-sandbox-*.desktop names its icon as
g4f-sandbox-<tool>. This script materializes those icons in the hicolor theme
from the tool's own website -- its favicon, its logo asset, or its GitHub
avatar -- so the Applications menu and the terminal windows it opens show the
vendor's mark instead of the shared g4f glyph.

Candidates are tried in order and the first usable one wins. A candidate is
usable when curl reports success and the payload is a real image:

  * SVG is kept as SVG, but only when it parses and is roughly square --
    wordmarks such as aider.chat's 200x60 logo are rejected.
  * Raster images are normalized to a 128x128 PNG through Pillow. GTK icon
    themes load png/svg/xpm only, so the JPEG sources (GitHub avatars, some
    favicons) have to be converted.

A tool whose candidates all fail falls back to the shared g4f-sandbox-cli.svg,
so a vendor URL going away never breaks the image build.

Runs at image build time, after Pillow is installed:

    fetch-icons.py [--dest /usr/share/icons/hicolor] [--only TOOL]
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from PIL import Image

SIZE = 128
MAX_BYTES = 4 * 1024 * 1024
SVG_NS = "{http://www.w3.org/2000/svg}"
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Entries that are g4f's own (the bundled CLI, the sandbox settings, the remote
# desktop) have no vendor site for this script to fetch. The g4f-go, settings
# and reset entries keep the shared g4f-sandbox-cli.svg; remote-desktop ships
# its own icon, copied from projects/remote-desktop/web/icon.svg at build time.
SHARED = {"g4f-go", "settings", "reset", "remote-desktop"}


def google(domain: str) -> str:
    """Google's favicon service: the only source that works for most vendors."""
    return f"https://www.google.com/s2/favicons?domain={domain}&sz={SIZE}"


def github(org: str) -> str:
    """GitHub org avatar, for projects whose site serves no usable icon."""
    return f"https://github.com/{org}.png?size={SIZE}"


# tool id -> candidate URLs, best first. Direct logo assets come first (crisp
# at any size), then the favicon service, then the GitHub avatar.
ICONS: dict[str, list[str]] = {
    # Browsers
    "chrome": ["https://www.google.com/chrome/static/images/chrome-logo.svg"],
    "lightpanda": [google("lightpanda.io")],
    # Coding agents
    "opencode": ["https://opencode.ai/favicon.svg"],
    "kilocode": [google("kilocode.ai")],
    "antigravity": [google("antigravity.google")],
    "gemini-cli": [
        "https://www.gstatic.com/lamda/images/"
        "gemini_favicon_f069958c85030456e93de685481c559f160ea06b.png"
    ],
    "qwencode": [google("qwen.ai")],
    "claude-code": [google("claude.ai")],
    "codex": [google("openai.com")],
    "copilot": ["https://github.githubassets.com/favicons/favicon.svg"],
    "cline": [google("cline.bot")],
    "amp": [google("ampcode.com")],
    "goose": [google("goose-docs.ai")],
    "aider": [github("Aider-AI")],
    "cursor": [google("cursor.com")],
    "crush": [google("charm.sh")],
    "continue-cli": [google("continue.dev")],
    "grok": [google("x.ai")],
    "droid": [google("factory.ai")],
    "openhands": [github("All-Hands-AI")],
    "plandex": [github("plandex-ai")],
    "hermes": [google("nousresearch.com")],
    "kiro": [google("kiro.dev")],
    "kimi": [google("kimi.com")],
    "roomote": [google("roomote.dev")],
    "auggie": [google("augmentcode.com")],
    "qoder": [google("qoder.com")],
    "iflow": [google("iflow.cn")],
    "vibe": [google("mistral.ai")],
    "sweagent": [google("swe-agent.com")],
    # Gateways
    "omniroute": [google("omniroute.online")],
    "litellm": [google("litellm.ai")],
    "proxy": [google("mitmproxy.org")],
    "tunnel": [google("cloudflare.com")],
    # Local AI
    "ollama": ["https://ollama.com/public/ollama.png", google("ollama.com")],
    "llamacpp": [
        "https://github.com/ggml-org/llama.cpp/raw/master/media/"
        "llama1-icon-transparent.png"
    ],
    "lmstudio": [google("lmstudio.ai")],
    "jan": [google("jan.ai")],
    # Chat UIs
    "librechat": [google("librechat.ai")],
    "lobechat": [google("lobehub.com")],
    "msty": [google("msty.ai")],
    "sillytavern": [github("SillyTavern")],
    "cptr": [github("open-webui"), google("openwebui.com")],
    # Editors
    "code-server": [google("coder.com"), github("coder")],
    # Assistants
    "openclaw": ["https://openclaw.ai/favicon.svg"],
    "fabric": [github("danielmiessler")],
    "sgpt": [github("TheR1D")],
    "gptme": ["https://gptme.org/media/icon.svg"],
    "llm": [github("simonw")],
    "assistant-ui": [google("assistant-ui.com"), github("assistant-ui")],
}


def fetch(url: str) -> bytes | None:
    """Download url, or return None when curl reports any failure.

    curl -f turns the HTTP status into the exit code, which is what makes the
    favicon services' 404-with-a-placeholder-image responses detectable.
    """
    try:
        proc = subprocess.run(
            [
                "curl", "-fsSL",
                "--max-time", "30",
                "--retry", "2", "--retry-delay", "1",
                "--max-filesize", str(MAX_BYTES),
                "-A", UA,
                url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout


def svg_ratio(data: bytes) -> float | None:
    """width/height of an SVG, or None when it is not a usable icon."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    if root.tag != SVG_NS + "svg":
        return None

    width = height = None
    view_box = root.get("viewBox")
    if view_box:
        parts = view_box.replace(",", " ").split()
        if len(parts) == 4:
            try:
                width, height = float(parts[2]), float(parts[3])
            except ValueError:
                width = height = None

    if not width or not height:
        def number(value: str | None) -> float | None:
            if not value:
                return None
            try:
                return float(value.strip().removesuffix("px"))
            except ValueError:
                return None

        width, height = number(root.get("width")), number(root.get("height"))

    if not width or not height:
        return 1.0
    return width / height


def install(data: bytes, name: str, png_dir: str, svg_dir: str) -> bool:
    """Write data as <name>.svg or <name>.png; False when it is not an icon."""
    png_path = os.path.join(png_dir, name + ".png")
    svg_path = os.path.join(svg_dir, name + ".svg")
    for stale in (png_path, svg_path):
        if os.path.exists(stale):
            os.remove(stale)

    head = data[:2048]
    if head.lstrip()[:1] == b"<" and b"<svg" in head:
        ratio = svg_ratio(data)
        if ratio is None or not 0.75 <= ratio <= 1.33:
            return False
        with open(svg_path, "wb") as handle:
            handle.write(data)
        return True

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        image = image.convert("RGBA")
    except Exception:
        return False

    image.thumbnail((SIZE, SIZE), RESAMPLE)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.paste(image, ((SIZE - image.width) // 2, (SIZE - image.height) // 2), image)
    canvas.save(png_path, "PNG", optimize=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default="/usr/share/icons/hicolor",
                        help="hicolor theme root (default: %(default)s)")
    parser.add_argument("--only", action="append", metavar="TOOL",
                        help="fetch just this tool (repeatable)")
    args = parser.parse_args()

    png_dir = os.path.join(args.dest, "128x128", "apps")
    svg_dir = os.path.join(args.dest, "scalable", "apps")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)

    fallback = os.path.join(svg_dir, "g4f-sandbox-cli.svg")
    if not os.path.isfile(fallback):
        print(f"fetch-icons: missing fallback icon {fallback}", file=sys.stderr)
        return 1

    tools = sorted(ICONS)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - set(ICONS) - SHARED
        if unknown:
            print(f"fetch-icons: unknown tool(s): {', '.join(sorted(unknown))}",
                  file=sys.stderr)
            return 1
        tools = [tool for tool in tools if tool in wanted]

    fell_back = []
    for tool in tools:
        name = f"g4f-sandbox-{tool}"
        for url in ICONS[tool]:
            data = fetch(url)
            if data and install(data, name, png_dir, svg_dir):
                print(f"  {name:<32} {url}")
                break
        else:
            fell_back.append(tool)
            shutil.copyfile(fallback, os.path.join(svg_dir, name + ".svg"))
            print(f"  {name:<32} fallback: g4f-sandbox-cli.svg")

    print(f"fetch-icons: {len(tools) - len(fell_back)}/{len(tools)} icons "
          f"from vendor sites")
    if fell_back:
        print("fetch-icons: fell back to the g4f icon for: "
              + ", ".join(fell_back))
    return 0


if __name__ == "__main__":
    sys.exit(main())