"""
System tray integration for g4f.

Starts the g4f API server in a background thread and places an icon in the
system tray with quick-access menu items.

Requirements:
    pip install pystray pillow
"""

from __future__ import annotations

import threading
import webbrowser
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# SVG source for the tray icon (https://g4f.dev/dist/img/g4f.svg)
_G4F_SVG = """\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#6e48aa"/>
      <stop offset="100%" style="stop-color:#4776E6"/>
    </linearGradient>
  </defs>
  <circle cx="256" cy="256" r="256" fill="url(#bgGrad)"/>
  <text x="256" y="328" font-family="Arial,Helvetica,sans-serif" font-size="194" font-weight="900"
        fill="#f8f9fa" text-anchor="middle" letter-spacing="-5">G4F</text>
</svg>"""


def _make_icon():
    """
    Return the g4f tray icon as a PIL Image (64×64 RGBA).

    Rendering priority:
    1. cairosvg  — renders the official g4f SVG to PNG then loads with Pillow.
    2. Pillow-only fallback — simple programmatic circle + text icon.
    """
    try:
        from PIL import Image
        import io
    except ImportError:
        return None

    # --- attempt SVG rendering via cairosvg ---
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(
            bytestring=_G4F_SVG.encode(),
            output_width=16,
            output_height=16,
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        pass

    # --- Pillow-only fallback ---
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        return None

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Gradient-ish background (approximate purple→blue)
    draw.ellipse([4, 4, size - 4, size - 4], fill=(110, 72, 170, 255))

    try:
        font = ImageFont.truetype("arial.ttf", 72)
    except Exception:
        font = ImageFont.load_default()

    text = "G4F"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(((size - text_w) // 2, (size - text_h) // 2), text, fill=(248, 249, 250, 255), font=font)

    return img


def run_tray(
    port: int = 1337,
    host: str = "0.0.0.0",
    debug: bool = False,
    no_autostart: bool = False,
):
    """
    Launch the g4f system tray application.

    Parameters
    ----------
    port : int
        Port for the API server (default 1337).
    host : str
        Bind host for the API server (default 0.0.0.0).
    debug : bool
        Enable verbose logging.
    no_autostart : bool
        If True, the API server is NOT started automatically on launch.
    """
    try:
        import pystray
    except ImportError:
        raise ImportError(
            "pystray is required for system tray support. "
            "Install it with: pip install pystray pillow"
        )

    from g4f.api import AppConfig, run_api

    browser_url = f"http://127.0.0.1:{port}"

    # ------------------------------------------------------------------ #
    # Server management                                                    #
    # ------------------------------------------------------------------ #
    _server_thread: Optional[threading.Thread] = None
    _server_running = threading.Event()

    def _start_server():
        nonlocal _server_thread
        if _server_running.is_set():
            return
        _server_running.set()
        AppConfig.set_config(gui=True)

        def _run():
            try:
                run_api(
                    bind=f"{host}:{port}",
                    port=None,
                    debug=debug,
                )
            except Exception as exc:
                logger.error("API server error: %s", exc)
            finally:
                _server_running.clear()

        _server_thread = threading.Thread(target=_run, daemon=True, name="g4f-api")
        _server_thread.start()

    def _stop_server():
        # uvicorn does not expose a clean stop; we clear the flag so the UI
        # reflects the intent.  The daemon thread will be killed on process exit.
        _server_running.clear()

    # ------------------------------------------------------------------ #
    # Menu callbacks                                                       #
    # ------------------------------------------------------------------ #
    def on_open_browser(pathname: str = ""):
        webbrowser.open(f"{browser_url}{pathname}")

    def on_toggle_server(icon, item):
        if _server_running.is_set():
            _stop_server()
        else:
            _start_server()
        icon.update_menu()

    def on_quit(icon, item):
        _stop_server()
        icon.stop()

    # ------------------------------------------------------------------ #
    # Dynamic menu                                                         #
    # ------------------------------------------------------------------ #
    def server_label(item) -> str:
        return "Stop Server" if _server_running.is_set() else "Start Server"

    def server_enabled(item) -> bool:
        return True

    menu = pystray.Menu(
        pystray.MenuItem("Open in Browser", on_open_browser, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(server_label, on_toggle_server, enabled=server_enabled),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Show logs", lambda: on_open_browser("/logs")),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    icon_image = _make_icon()
    if icon_image is None:
        # Fallback: 1×1 transparent image (pystray still works, icon may be blank)
        try:
            from PIL import Image
            icon_image = Image.new("RGBA", (1, 1))
        except ImportError:
            raise ImportError(
                "pillow is required for system tray support. "
                "Install it with: pip install pystray pillow"
            )

    tray_icon = pystray.Icon(
        name="g4f",
        icon=icon_image,
        title="g4f AI Server",
        menu=menu,
    )

    # Auto-start the server unless opted out
    if not no_autostart:
        _start_server()

    tray_icon.run()


def _tray_main():
    """
    Standalone entry point installed as the ``g4f-tray`` console script.
    Parses CLI arguments and delegates to :func:`run_tray`.
    """
    import argparse
    from g4f.config import DEFAULT_PORT

    parser = argparse.ArgumentParser(
        description="Run g4f as a system tray application",
        prog="g4f-tray",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for the API server (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host for the API server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Do not start the API server automatically on launch.",
    )
    args = parser.parse_args()
    run_tray(
        port=args.port,
        host=args.host,
        debug=args.debug,
        no_autostart=args.no_autostart,
    )


if __name__ == "__main__":
    _tray_main()
