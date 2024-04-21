from argparse import ArgumentParser

def gui_parser():
    parser = ArgumentParser(description="Run the GUI")
    parser.add_argument("-host", type=str, default="0.0.0.0", help="hostname")
    parser.add_argument("-port", type=int, default=8080, help="port")
    parser.add_argument("-debug", action="store_true", help="debug mode")
    parser.add_argument("--ignore-cookie-files", action="store_true", help="Don't read .har and cookie files.")
    return parser