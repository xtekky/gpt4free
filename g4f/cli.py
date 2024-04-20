import argparse

from g4f import Provider
from g4f.gui.run import gui_parser, run_gui_args

def main():
    parser = argparse.ArgumentParser(description="Run gpt4free")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run the g4f in.")
    api_parser = subparsers.add_parser("api")
    api_parser.add_argument("--bind", default="0.0.0.0:1337", help="The bind string.")
    api_parser.add_argument("--debug", action="store_true", help="Enable verbose logging.")
    api_parser.add_argument("--workers", type=int, default=None, help="Number of workers.")
    api_parser.add_argument("--disable_colors", action="store_true", help="Don't use colors.")
    api_parser.add_argument("--ignored-providers", nargs="+", choices=[provider for provider in Provider.__map__],
                            default=[], help="List of providers to ignore when processing request.")
    subparsers.add_parser("gui", parents=[gui_parser()], add_help=False)

    args = parser.parse_args()
    if args.mode == "api":
        import g4f.api
        g4f.api.api.set_list_ignored_providers(
            args.ignored_providers
        )
        g4f.api.run_api(
            bind=args.bind,
            debug=args.debug,
            workers=args.workers,
            use_colors=not args.disable_colors
        )
    elif args.mode == "gui":
        run_gui_args(args)
    else:
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()
