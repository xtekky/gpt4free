from argparse import ArgumentParser

from g4f.gui import run_gui


def gui_parser():
    parser = ArgumentParser(description="Run the GUI")
    parser.add_argument("-host", type=str, default="0.0.0.0", help="hostname")
    parser.add_argument("-port", type=int, default=8080, help="port")
    parser.add_argument("-debug", action="store_true", help="debug mode")
    return parser


def run_gui_args(args):
    host = args.host
    port = args.port
    debug = args.debug
    run_gui(host, port, debug)


if __name__ == "__main__":
    parser = gui_parser()
    args = parser.parse_args()
    run_gui_args(args)
