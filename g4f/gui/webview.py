import webview
from functools import partial
from platformdirs import user_config_dir

from g4f.gui import run_gui
from g4f.gui.run import gui_parser
import g4f.version
import g4f.debug

def run_webview(host: str = "0.0.0.0", port: int = 8080, debug: bool = True):
    webview.create_window(f"g4f - {g4f.version.utils.current_version}", f"http://{host}:{port}/")
    if debug:
        g4f.debug.logging = True
    webview.start(
        partial(run_gui, host, port),
        private_mode=False,
        storage_path=user_config_dir("g4f-webview"),
        debug=debug
    )

if __name__ == "__main__":
    parser = gui_parser()
    args = parser.parse_args()
    run_webview(args.host, args.port, args.debug)