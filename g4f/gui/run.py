from g4f.gui import run_gui
from argparse import ArgumentParser


if __name__ == '__main__':
    
    parser = ArgumentParser(description='Run the GUI')
    
    parser.add_argument('-host', type=str, default='0.0.0.0', help='hostname')
    parser.add_argument('-port', type=int, default=80, help='port')
    parser.add_argument('-debug', action='store_true', help='debug mode')

    args = parser.parse_args()
    port = args.port
    host = args.host
    debug = args.debug
    
    run_gui(host, port, debug)