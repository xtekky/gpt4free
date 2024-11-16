from __future__ import annotations

import argparse

from g4f import Provider
from g4f.gui.run import gui_parser, run_gui_args

def main():
    parser = argparse.ArgumentParser(description="Run gpt4free")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run the g4f in.")
    api_parser = subparsers.add_parser("api")
    api_parser.add_argument("--bind", default="0.0.0.0:1337", help="The bind string.")
    api_parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    api_parser.add_argument("--model", default=None, help="Default model for chat completion. (incompatible with --reload and --workers)")
    api_parser.add_argument("--provider", choices=[provider.__name__ for provider in Provider.__providers__ if provider.working],
                            default=None, help="Default provider for chat completion. (incompatible with --reload and --workers)")
    api_parser.add_argument("--image-provider", choices=[provider.__name__ for provider in Provider.__providers__ if provider.working and hasattr(provider, "image_models")],
                            default=None, help="Default provider for image generation. (incompatible with --reload and --workers)"),
    api_parser.add_argument("--proxy", default=None, help="Default used proxy. (incompatible with --reload and --workers)")
    api_parser.add_argument("--workers", type=int, default=None, help="Number of workers.")
    api_parser.add_argument("--disable-colors", action="store_true", help="Don't use colors.")
    api_parser.add_argument("--ignore-cookie-files", action="store_true", help="Don't read .har and cookie files. (incompatible with --reload and --workers)")
    api_parser.add_argument("--g4f-api-key", type=str, default=None, help="Sets an authentication key for your API. (incompatible with --reload and --workers)")
    api_parser.add_argument("--ignored-providers", nargs="+", choices=[provider.__name__ for provider in Provider.__providers__ if provider.working],
                            default=[], help="List of providers to ignore when processing request. (incompatible with --reload and --workers)")
    api_parser.add_argument("--reload", action="store_true", help="Enable reloading.")
    subparsers.add_parser("gui", parents=[gui_parser()], add_help=False)

    args = parser.parse_args()
    if args.mode == "api":
        run_api_args(args)
    elif args.mode == "gui":
        run_gui_args(args)
    else:
        parser.print_help()
        exit(1)

def run_api_args(args):
    from g4f.api import AppConfig, run_api

    AppConfig.set_config(
        ignore_cookie_files=args.ignore_cookie_files,
        ignored_providers=args.ignored_providers,
        g4f_api_key=args.g4f_api_key,
        provider=args.provider,
        image_provider=args.image_provider,
        proxy=args.proxy,
        model=args.model
    )
    run_api(
        bind=args.bind,
        debug=args.debug,
        workers=args.workers,
        use_colors=not args.disable_colors,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
