from __future__ import annotations

"""
This module defines the command-line interface (CLI) entrypoint for g4f,
including:
- API server mode
- GUI mode
- Client mode
- MCP (Model Context Protocol) mode

It provides argument parsers for each mode and executes the appropriate
runtime function depending on the CLI arguments.
"""

import argparse
from argparse import ArgumentParser

# Local imports (within g4f package)
from .client import get_parser, run_client_args
from ..requests import BrowserConfig
from ..gui.run import gui_parser, run_gui_args
from ..config import DEFAULT_PORT, DEFAULT_TIMEOUT, DEFAULT_STREAM_TIMEOUT
from .. import Provider
from .. import cookies


# --------------------------------------------------------------
#  API PARSER
# --------------------------------------------------------------
def get_api_parser() -> ArgumentParser:
    """
    Creates and returns the argument parser used for:
        g4f api ...
    """
    api_parser = ArgumentParser(description="Run the API and GUI")

    api_parser.add_argument(
        "--bind",
        default=None,
        help=f"The bind address (default: 0.0.0.0:{DEFAULT_PORT})."
    )
    api_parser.add_argument(
        "--port", "-p",
        default=None,
        help=f"Port for the API server (default: {DEFAULT_PORT})."
    )
    api_parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable verbose logging."
    )

    # Deprecated GUI flag but kept for compatibility
    api_parser.add_argument(
        "--gui", "-g",
        default=None,
        action="store_true",
        help="(deprecated) Use --no-gui instead."
    )

    api_parser.add_argument(
        "--no-gui", "-ng",
        default=False,
        action="store_true",
        help="Run API without the GUI."
    )

    api_parser.add_argument(
        "--model",
        default=None,
        help="Default model for chat completion (incompatible with reload/workers)."
    )

    # Providers for chat completion
    api_parser.add_argument(
        "--provider",
        choices=[p.__name__ for p in Provider.__providers__ if p.working],
        default=None,
        help="Default provider for chat completion."
    )

    # Providers for image generation
    api_parser.add_argument(
        "--media-provider",
        choices=[
            p.__name__ for p in Provider.__providers__
            if p.working and bool(getattr(p, "image_models", False))
        ],
        default=None,
        help="Default provider for image generation."
    )

    api_parser.add_argument(
        "--proxy",
        default=None,
        help="Default HTTP proxy."
    )

    api_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes."
    )

    api_parser.add_argument(
        "--disable-colors",
        action="store_true",
        help="Disable colorized output."
    )

    api_parser.add_argument(
        "--ignore-cookie-files",
        action="store_true",
        help="Do not read .har or cookie files."
    )

    api_parser.add_argument(
        "--g4f-api-key",
        type=str,
        default=None,
        help="Authentication key for your API."
    )

    api_parser.add_argument(
        "--ignored-providers",
        nargs="+",
        choices=[p.__name__ for p in Provider.__providers__ if p.working],
        default=[],
        help="Providers to ignore during request processing."
    )

    api_parser.add_argument(
        "--cookie-browsers",
        nargs="+",
        choices=[browser.__name__ for browser in cookies.BROWSERS],
        default=[],
        help="Browsers to fetch cookies from."
    )

    api_parser.add_argument("--reload", action="store_true", help="Enable hot reload.")
    api_parser.add_argument("--demo", action="store_true", help="Enable demo mode.")

    api_parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Default request timeout in seconds."
    )

    api_parser.add_argument(
        "--stream-timeout",
        type=int,
        default=DEFAULT_STREAM_TIMEOUT,
        help="Default streaming timeout in seconds."
    )

    api_parser.add_argument("--ssl-keyfile", type=str, default=None, help="SSL key file.")
    api_parser.add_argument("--ssl-certfile", type=str, default=None, help="SSL cert file.")
    api_parser.add_argument("--log-config", type=str, default=None, help="Path to log config.")
    api_parser.add_argument("--access-log", action="store_true", default=True, help="Enable access logging.")
    api_parser.add_argument("--no-access-log", dest="access_log", action="store_false", help="Disable access logging.")

    api_parser.add_argument(
        "--browser-port",
        type=int,
        help="Port for browser automation tool."
    )

    api_parser.add_argument(
        "--browser-host",
        type=str,
        default="127.0.0.1",
        help="Host for browser automation tool."
    )

    return api_parser


# --------------------------------------------------------------
#  API RUNNER
# --------------------------------------------------------------
def run_api_args(args):
    """
    Runs the API server using the parsed CLI arguments.
    """
    from g4f.api import AppConfig, run_api

    # Apply configuration
    AppConfig.set_config(
        ignore_cookie_files=args.ignore_cookie_files,
        ignored_providers=args.ignored_providers,
        g4f_api_key=args.g4f_api_key,
        provider=args.provider,
        media_provider=args.media_provider,
        proxy=args.proxy,
        model=args.model,
        gui=not args.no_gui,
        demo=args.demo,
        timeout=args.timeout,
        stream_timeout=args.stream_timeout
    )

    # Browser automation config
    if args.browser_port:
        BrowserConfig.port = args.browser_port
        BrowserConfig.host = args.browser_host

    # Custom cookie browsers
    if args.cookie_browsers:
        cookies.BROWSERS = [cookies[b] for b in args.cookie_browsers]

    # Launch server
    run_api(
        bind=args.bind,
        port=args.port,
        debug=args.debug,
        workers=args.workers,
        use_colors=not args.disable_colors,
        reload=args.reload,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        log_config=args.log_config,
        access_log=args.access_log,
    )


# --------------------------------------------------------------
#  MCP PARSER
# --------------------------------------------------------------
def get_mcp_parser() -> ArgumentParser:
    """
    Parser for:
        g4f mcp ...
    """
    mcp_parser = ArgumentParser(description="Run the MCP (Model Context Protocol) server")
    mcp_parser.add_argument("--debug", "-d", action="store_true", help="Enable verbose logging.")
    mcp_parser.add_argument("--http", action="store_true", help="Use HTTP instead of stdio.")
    mcp_parser.add_argument("--host", default="0.0.0.0", help="HTTP server host.")
    mcp_parser.add_argument("--port", type=int, default=8765, help="HTTP server port.")
    mcp_parser.add_argument("--origin", type=str, default=None, help="CORS origin.")
    return mcp_parser


def run_mcp_args(args):
    """
    Runs the MCP server with the chosen transport method.
    """
    from ..mcp.server import main as mcp_main
    mcp_main(
        http=args.http,
        host=args.host,
        port=args.port,
        origin=args.origin
    )


# --------------------------------------------------------------
#  MAIN ENTRYPOINT
# --------------------------------------------------------------
def main():
    """
    Main entry function exposed via CLI (e.g. g4f).
    Handles selecting: api / gui / client / mcp
    """
    parser = argparse.ArgumentParser(description="Run gpt4free", exit_on_error=False)

    # Create sub-commands
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run g4f in.")
    subparsers.add_parser("api", parents=[get_api_parser()], add_help=False)
    subparsers.add_parser("gui", parents=[gui_parser()], add_help=False)
    subparsers.add_parser("client", parents=[get_parser()], add_help=False)
    subparsers.add_parser("mcp", parents=[get_mcp_parser()], add_help=False)

    try:
        args = parser.parse_args()

        # Mode routing
        if args.mode == "api":
            run_api_args(args)
        elif args.mode == "gui":
            run_gui_args(args)
        elif args.mode == "client":
            run_client_args(args)
        elif args.mode == "mcp":
            run_mcp_args(args)
        else:
            # No mode provided
            raise argparse.ArgumentError(
                None,
                "No valid mode specified. Use 'api', 'gui', 'client', or 'mcp'."
            )

    except argparse.ArgumentError:
        # Fallback chain:
        # 1. Try client mode
        try:
            run_client_args(
                get_parser(exit_on_error=False).parse_args(),
                exit_on_error=False
            )
        except argparse.ArgumentError:
            # 2. Try API mode with default arguments
            run_api_args(get_api_parser().parse_args())
