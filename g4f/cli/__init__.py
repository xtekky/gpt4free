from __future__ import annotations

import argparse
from argparse import ArgumentParser
from .client import get_parser, run_client_args

from g4f import Provider
from g4f.gui.run import gui_parser, run_gui_args
import g4f.cookies

def get_api_parser():
    api_parser = ArgumentParser(description="Run the API and GUI")
    api_parser.add_argument("--bind", default=None, help="The bind string. (Default: 0.0.0.0:1337)")
    api_parser.add_argument("--port", "-p", default=None, help="Change the port of the server.")
    api_parser.add_argument("--debug", "-d", action="store_true", help="Enable verbose logging.")
    api_parser.add_argument("--gui", "-g", default=None, action="store_true", help="Start also the gui.")
    api_parser.add_argument("--model", default=None, help="Default model for chat completion. (incompatible with --reload and --workers)")
    api_parser.add_argument("--provider", choices=[provider.__name__ for provider in Provider.__providers__ if provider.working],
                            default=None, help="Default provider for chat completion. (incompatible with --reload and --workers)")
    api_parser.add_argument("--media-provider", choices=[provider.__name__ for provider in Provider.__providers__ if provider.working and bool(getattr(provider, "image_models", False))],
                            default=None, help="Default provider for image generation. (incompatible with --reload and --workers)"),
    api_parser.add_argument("--proxy", default=None, help="Default used proxy. (incompatible with --reload and --workers)")
    api_parser.add_argument("--workers", type=int, default=None, help="Number of workers.")
    api_parser.add_argument("--disable-colors", action="store_true", help="Don't use colors.")
    api_parser.add_argument("--ignore-cookie-files", action="store_true", help="Don't read .har and cookie files. (incompatible with --reload and --workers)")
    api_parser.add_argument("--g4f-api-key", type=str, default=None, help="Sets an authentication key for your API. (incompatible with --reload and --workers)")
    api_parser.add_argument("--ignored-providers", nargs="+", choices=[provider.__name__ for provider in Provider.__providers__ if provider.working],
                            default=[], help="List of providers to ignore when processing request. (incompatible with --reload and --workers)")
    api_parser.add_argument("--cookie-browsers", nargs="+", choices=[browser.__name__ for browser in g4f.cookies.BROWSERS],
                            default=[], help="List of browsers to access or retrieve cookies from. (incompatible with --reload and --workers)")
    api_parser.add_argument("--reload", action="store_true", help="Enable reloading.")
    api_parser.add_argument("--demo", action="store_true", help="Enable demo mode.")
    api_parser.add_argument("--timeout", type=int, default=600, help="Default timeout for requests in seconds. (incompatible with --reload and --workers)")
    api_parser.add_argument("--ssl-keyfile", type=str, default=None, help="Path to SSL key file for HTTPS.")
    api_parser.add_argument("--ssl-certfile", type=str, default=None, help="Path to SSL certificate file for HTTPS.")
    api_parser.add_argument("--log-config", type=str, default=None, help="Custom log config.")
	
    return api_parser

def run_api_args(args):
    from g4f.api import AppConfig, run_api

    AppConfig.set_config(
        ignore_cookie_files=args.ignore_cookie_files,
        ignored_providers=args.ignored_providers,
        g4f_api_key=args.g4f_api_key,
        provider=args.provider,
        media_provider=args.media_provider,
        proxy=args.proxy,
        model=args.model,
        gui=args.gui,
        demo=args.demo,
        timeout=args.timeout,
    )
    if args.cookie_browsers:
        g4f.cookies.BROWSERS = [g4f.cookies[browser] for browser in args.cookie_browsers]
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
    )

def main():
    parser = argparse.ArgumentParser(description="Run gpt4free", exit_on_error=False)
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run the g4f in.")
    subparsers.add_parser("api", parents=[get_api_parser()], add_help=False)
    subparsers.add_parser("gui", parents=[gui_parser()], add_help=False)
    subparsers.add_parser("client", parents=[get_parser()], add_help=False)

    try:
        args = parser.parse_args()
        if args.mode == "api":
            run_api_args(args)
        elif args.mode == "gui":
            run_gui_args(args)
        elif args.mode == "client":
            run_client_args(args)
        else:
            parser.print_help()
            exit(1)
    except argparse.ArgumentError:
        run_client_args(get_parser().parse_args())