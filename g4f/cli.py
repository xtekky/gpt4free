import argparse

from g4f.api import run_api
from g4f.gui.run import gui_parser, run_gui_args


def run_gui(args):
    print("Running GUI...")


def main():
    parser = argparse.ArgumentParser(description="Run gpt4free")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run the g4f in.")
    subparsers.add_parser("api")
    subparsers.add_parser("gui", parents=[gui_parser()], add_help=False)

    args = parser.parse_args()
    if args.mode == "api":
        run_api()
    elif args.mode == "gui":
        run_gui_args(args)
    else:
        parser.print_help()
        exit(1)


if __name__ == "__main__":
    main()
