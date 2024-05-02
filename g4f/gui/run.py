from .gui_parser import gui_parser
from ..cookies import read_cookie_files
import g4f.debug

def run_gui_args(args):
    if args.debug:
        g4f.debug.logging = True
    if not args.ignore_cookie_files:
        read_cookie_files()
    from g4f.gui import run_gui
    host = args.host
    port = args.port
    debug = args.debug
    run_gui(host, port, debug)

if __name__ == "__main__":
    parser = gui_parser()
    args = parser.parse_args()
    run_gui_args(args)