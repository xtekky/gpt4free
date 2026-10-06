"""Command line entry point: ``python -m remote_desktop``."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Settings
from .server import build_hub, create_app, lan_ip, qr_ascii


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="remote-desktop",
        description="Share this desktop's screen with a phone and control it from there.",
    )
    parser.add_argument("--host", help="interface to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, help="port to listen on (default: 8765)")
    parser.add_argument("--token", help="shared access token (default: generated)")
    parser.add_argument("--fps", type=int, help="target frames per second (default: 12)")
    parser.add_argument("--quality", type=int, help="JPEG quality 10-95 (default: 60)")
    parser.add_argument("--max-width", type=int, help="downscale frames to this width")
    parser.add_argument("--monitor", type=int, help="monitor index, 1 is primary")
    parser.add_argument("--no-input", action="store_true", help="view only, no control")
    parser.add_argument(
        "--no-auto-control",
        action="store_true",
        help="do not hand control to the first client automatically",
    )
    parser.add_argument("--no-cursor", action="store_true", help="hide the mouse cursor")
    parser.add_argument("--log-level", help="debug, info, warning or error")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_args(args)
    settings.resolve_token()

    logging.basicConfig(
        level=getattr(logging, str(settings.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    import uvicorn

    app = create_app(settings, hub=build_hub(settings))
    url = f"http://{lan_ip()}:{settings.port}/?token={settings.token}"

    print()
    print("  Remote Desktop Agent")
    print(f"  Local URL : http://127.0.0.1:{settings.port}/?token={settings.token}")
    print(f"  Phone URL : {url}")
    print(f"  Token     : {settings.token}")
    print(f"  Input     : {'enabled' if settings.allow_input else 'disabled'}")
    print()
    code = qr_ascii(url)
    if code:
        print(code)
        print()
    print("  Scan the code with your phone, or open the URL above.")
    print("  Press Ctrl+C to stop.")
    print()
    sys.stdout.flush()

    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
