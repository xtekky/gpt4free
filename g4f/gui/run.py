from argparse import ArgumentParser
from g4f.gui import run_gui
from g4f import Provider
from enum import Enum


def gui_parser():
    IgnoredProviders = Enum("ignore_providers", {
        key: key for key in Provider.__all__})
    parser = ArgumentParser(description="Run the GUI")
    parser.add_argument("-host", type=str, default="0.0.0.0", help="hostname")
    parser.add_argument("-port", type=int, default=80, help="port")
    parser.add_argument("-debug", action="store_true", help="debug mode")
    parser.add_argument("-ignored-providers", nargs="+", choices=[
        provider.name for provider in IgnoredProviders],
        default=[], help="List of providers to ignore when processing request.")
    return parser


def run_gui_args(args, apiStatus=False):
    host = args.host
    port = args.port
    debug = args.debug
    ignored_providers = args.ignored_providers
    run_gui(host, port, debug, apiStatus, ignored_providers)


if __name__ == "__main__":
    parser = gui_parser()
    args = parser.parse_args()
    run_gui_args(args)
