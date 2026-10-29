"""
System tray integration for g4f.

Starts the g4f API server as a child process and places an icon in the system
tray with quick-access menu items. Because the server runs in its own process,
it can be stopped and started again without restarting the tray, which is what
the "Restart Server" item uses.

Requirements:
    pip install "g4f[tray]"
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import webbrowser
from typing import Callable, Mapping, Optional, Sequence, Union

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

    font = None
    for name in (
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
        "arial.ttf",
        "Arial.ttf",
    ):
        try:
            font = ImageFont.truetype(name, 92)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    text = "G4F"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((size - text_w) // 2, (size - text_h) // 2 - bbox[1]),
        text,
        fill=(248, 249, 250, 255),
        font=font,
    )

    return img


EnvSource = Union[Mapping[str, str], Callable[[], Mapping[str, str]]]


class _ServerManager:
    """Start, stop and restart the g4f API as a child process.

    A thread/lock pair guards the process handle. The server is started in its
    own session so stopping it also stops the Python child (and its reloader)
    that g4f-go may have spawned.
    """

    def __init__(
        self,
        command: Sequence[str],
        env: Optional[EnvSource] = None,
    ) -> None:
        self.command = list(command)
        self._env = env
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._waiter: Optional[threading.Thread] = None

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        # Only a value produced for this start should enable API-key auth; a
        # stale inherited key would otherwise survive clearing the setting.
        env.pop("G4F_API_KEY", None)
        extra = self._env() if callable(self._env) else self._env
        if extra:
            env.update(
                {
                    key: str(value)
                    for key, value in extra.items()
                    if value is not None
                }
            )
        return env

    def is_running(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        with self._lock:
            if self.is_running():
                return False
            try:
                process = subprocess.Popen(
                    self.command,
                    env=self._environment(),
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as exc:
                logger.error("could not start the g4f server: %s", exc)
                return False
            self._process = process
            self._waiter = threading.Thread(
                target=self._wait,
                args=(process,),
                daemon=True,
                name="g4f-api-waiter",
            )
            self._waiter.start()
            logger.info("g4f server started (pid %s)", process.pid)
            return True

    def _wait(self, process: subprocess.Popen) -> None:
        try:
            code = process.wait()
        except Exception:  # pragma: no cover - defensive
            code = None
        logger.info("g4f server exited (code %s)", code)

    def _signal(self, process: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(os.getpgid(process.pid), sig)
        except (AttributeError, ProcessLookupError, PermissionError, OSError):
            try:
                if sig == signal.SIGKILL:
                    process.kill()
                else:
                    process.terminate()
            except OSError:
                pass

    def stop(self, timeout: float = 10.0) -> bool:
        with self._lock:
            process = self._process
            if process is None:
                return False
            self._process = None
        if process.poll() is None:
            logger.info("stopping g4f server (pid %s)", process.pid)
            self._signal(process, signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("g4f server did not stop; killing it")
                self._signal(process, signal.SIGKILL)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        return True

    def restart(self) -> bool:
        self.stop()
        return self.start()


def run_tray(
    port: int = 1337,
    host: str = "0.0.0.0",
    debug: bool = False,
    no_autostart: bool = False,
    server_command: Optional[Sequence[str]] = None,
    server_env: Optional[EnvSource] = None,
    api_key: Optional[str] = None,
    on_update_package: Optional[Callable[[], object]] = None,
    on_change_package: Optional[Callable[[], object]] = None,
    package_label: str = "g4f package",
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
    server_command : sequence of str, optional
        Command used to run the API server. Defaults to
        ``sys.executable -m g4f api --port <port>``. The sandbox desktop passes
        ``g4f-go api`` here so the package manager and server stay in sync.
    server_env : mapping or callable, optional
        Extra environment for the server. A callable is consulted on every
        start, so settings changed through the menu take effect on restart.
    api_key : str, optional
        Shorthand for ``server_env={"G4F_API_KEY": api_key}``. The key is
        passed to the server; the tray intentionally has no key editor, so a
        host application can keep the key in its own settings UI.
    on_update_package, on_change_package : callable, optional
        Menu actions. They normally open a terminal prompt and return when the
        user closes it; the server is restarted afterwards. Omit an action to
        hide its menu item.
    package_label : str
        Name used by the package menu items.
    """
    try:
        import pystray
    except ImportError:
        raise ImportError(
            "pystray is required for system tray support. "
            "Install it with: pip install \"g4f[tray]\""
        )

    if server_command is None:
        server_command = [
            sys.executable,
            "-m",
            "g4f",
            "api",
            "--port",
            str(port),
            "--bind",
            f"{host}:{port}",
        ]
        if debug:
            server_command.append("--debug")

    env_source = server_env
    if api_key is not None and env_source is None:
        env_source = {"G4F_API_KEY": api_key}
    elif api_key is not None:
        base_env = env_source

        def _env_with_key():
            values = dict(base_env() if callable(base_env) else base_env)
            values["G4F_API_KEY"] = api_key
            return values

        env_source = _env_with_key

    manager = _ServerManager(server_command, env=env_source)
    browser_url = f"http://127.0.0.1:{port}"

    # ------------------------------------------------------------------ #
    # Menu callbacks                                                       #
    # ------------------------------------------------------------------ #
    def on_open_browser(pathname: str = ""):
        webbrowser.open(f"{browser_url}{pathname}")

    def refresh_menu(icon) -> None:
        try:
            icon.update_menu()
        except Exception:  # pragma: no cover - backend may be shutting down
            pass

    def run_action(icon, action: Callable[[], object]) -> None:
        """Run a blocking menu action off the UI thread, then restart."""

        def worker() -> None:
            try:
                result = action()
            except Exception:
                logger.exception("tray action failed")
                return
            if result is not False:
                manager.restart()
            refresh_menu(icon)

        threading.Thread(target=worker, daemon=True, name="g4f-tray-action").start()

    def on_toggle_server(icon, item):
        if manager.is_running():
            manager.stop()
        else:
            manager.start()
        refresh_menu(icon)

    def on_restart_server(icon, item):
        threading.Thread(target=manager.restart, daemon=True).start()
        refresh_menu(icon)

    def on_quit(icon, item):
        manager.stop()
        icon.stop()

    def on_update_package_click(icon, item):
        if on_update_package is not None:
            run_action(icon, on_update_package)

    def on_change_package_click(icon, item):
        if on_change_package is not None:
            run_action(icon, on_change_package)

    # ------------------------------------------------------------------ #
    # Dynamic menu                                                         #
    # ------------------------------------------------------------------ #
    def server_label(item) -> str:
        return "Stop Server" if manager.is_running() else "Start Server"

    items = [
        pystray.MenuItem(
            "Open in Browser", lambda icon, item: on_open_browser(), default=True
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(server_label, on_toggle_server),
        pystray.MenuItem("Restart Server", on_restart_server),
    ]
    if on_update_package is not None or on_change_package is not None:
        items.append(pystray.Menu.SEPARATOR)
        if on_update_package is not None:
            items.append(
                pystray.MenuItem(
                    f"Update {package_label}", on_update_package_click
                )
            )
        if on_change_package is not None:
            items.append(
                pystray.MenuItem(
                    f"Change {package_label} Version...", on_change_package_click
                )
            )
    items.extend(
        [
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Show Logs", lambda icon, item: on_open_browser("/logs")
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", on_quit),
        ]
    )
    menu = pystray.Menu(*items)

    icon_image = _make_icon()
    if icon_image is None:
        # Fallback: 1×1 transparent image (pystray still works, icon may be blank)
        try:
            from PIL import Image

            icon_image = Image.new("RGBA", (1, 1))
        except ImportError:
            raise ImportError(
                "pillow is required for system tray support. "
                "Install it with: pip install \"g4f[tray]\""
            )

    tray_icon = pystray.Icon(
        name="g4f",
        icon=icon_image,
        title="g4f AI Server",
        menu=menu,
    )

    # Stop the server when the container/session sends SIGTERM, instead of
    # leaving a child process behind.
    def handle_signal(signum, frame):
        logger.info("received signal %s, stopping g4f", signum)
        manager.stop()
        tray_icon.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handle_signal)
        except (ValueError, OSError):  # not the main thread
            pass

    # Auto-start the server unless opted out
    if not no_autostart:
        manager.start()

    try:
        tray_icon.run()
    finally:
        manager.stop()


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
        "--port",
        "-p",
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
        "--debug",
        "-d",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Do not start the API server automatically on launch.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Require this API key (G4F_API_KEY) for the server.",
    )
    args = parser.parse_args()
    run_tray(
        port=args.port,
        host=args.host,
        debug=args.debug,
        no_autostart=args.no_autostart,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    _tray_main()
