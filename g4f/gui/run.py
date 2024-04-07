from .gui_parser import gui_parser

def run_gui_args(args):
    from g4f.gui import run_gui
    host = args.host
    port = args.port
    debug = args.debug
    run_gui(host, port, debug)

if __name__ == "__main__":
    parser = gui_parser()
    args = parser.parse_args()
    run_gui_args(args)