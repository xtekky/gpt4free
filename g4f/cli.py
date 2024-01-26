import argparse
from enum import Enum

import g4f
from g4f import Provider

from g4f.gui.run import gui_parser, run_gui_args

def run_gui(args):
    print("Running GUI...")

def main():
    IgnoredProviders = Enum("ignore_providers", {key: key for key in Provider.__all__})
    parser = argparse.ArgumentParser(description="Run gpt4free")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run the g4f in.")
    api_parser=subparsers.add_parser("api")
    api_parser.add_argument("--bind", default="0.0.0.0:1337", help="The bind string.")
    api_parser.add_argument("--debug", type=bool, default=False, help="Enable verbose logging")
    api_parser.add_argument("--ignored-providers", nargs="+", choices=[provider.name for provider in IgnoredProviders],
                            default=[], help="List of providers to ignore when processing request.")
    subparsers.add_parser("gui", parents=[gui_parser()], add_help=False)

    args = parser.parse_args()
    if args.mode == "api":
        from g4f.api import Api
        controller=Api(engine=g4f, debug=args.debug, list_ignored_providers=args.ignored_providers)
        controller.run(args.bind)
    elif args.mode == "gui":
        run_gui_args(args)
    else:
        parser.print_help()
        exit(1)

if __name__ == "__main__":
    main()
