"""Command line entry point: ``python -m remote_desktop``."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser

from .app import create_app, share_urls
from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remote-desktop",
        description="Share this screen with a phone using the browser's native screencast API.",
    )
    parser.add_argument("--host", help="interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, help="port to listen on (default: 8765)")
    parser.add_argument("--token", help="require this access token from clients")
    parser.add_argument("--public-url", help="URL to advertise instead of the detected LAN address")
    parser.add_argument("--log-level", help="uvicorn log level (default: info)")
    parser.add_argument("--no-input", action="store_true", help="do not inject pointer and keyboard events")
    parser.add_argument("--no-auto-control", action="store_true", help="viewers must request control explicitly")
    parser.add_argument("--open-browser", action="store_true", help="open the host page in the default browser")
    parser.add_argument("--turn-url", help="TURN relay URL(s), comma separated (e.g. turn:turn.example.com:3478)")
    parser.add_argument("--turn-secret", help="shared secret for time-limited TURN credentials (TURN REST / coturn use-auth-secret)")
    parser.add_argument("--stun-url", help="STUN URL(s), comma separated (empty string disables STUN)")
    return parser


def print_banner(settings: Settings) -> None:
    urls = share_urls(settings)
    print()
    print("  Remote Desktop is running")
    print("  " + "-" * 46)
    print(f"  Host page (this computer) : http://127.0.0.1:{settings.port}/host")
    for url in urls:
        print(f"  Phone (same Wi-Fi)        : {url}/view")
    if settings.token:
        print(f"  Access token              : {settings.token}")
    print(f"  ICE servers               : {describe_ice(settings)}")
    print("  " + "-" * 46)
    print("  Open the host page, press Share, then scan the QR code on your phone.")
    print()

def describe_ice(settings: Settings) -> str:
    """One-line summary of the relays the browsers will be handed."""
    urls = [
        url
        for entry in settings.ice_servers()
        for url in (entry.get("urls") if isinstance(entry.get("urls"), list) else [entry.get("urls")])
        if url
    ]
    if not urls:
        return "none - LAN only, mobile data will not work"
    return ", ".join(urls)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_args(args).clamp()
    settings.resolve_token()

    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    import uvicorn

    app = create_app(settings)
    print_banner(settings)

    if settings.open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{settings.port}/host")).start()

    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
